"""Perangkat hitung Mode CCTV: GPU (ONNX Runtime + DirectML) atau CPU (OpenVINO, seperti Mode Pameran).

- "cpu": selalu CPU. GPU tidak disentuh.
- "otomatis": GPU tercepat hanya bila sudah lolos uji kesetaraan (models/gpu_kesetaraan.json, latih/banding_gpu.py)
  dan lebih cepat dari CPU. Selain itu CPU.
- "gpu": GPU tercepat walau belum lolos uji kesetaraan (dengan peringatan). Bila GPU tidak ada, CPU dengan alasannya.

Setiap adaptor DirectML dicoba dan diukur, lalu yang tercepat dipakai (di laptop gaming, adaptor #0 bisa jadi GPU
bawaan prosesor, bukan GPU gaming). Kapasitas tetap dibatasi penilaian di CPU (±30 ms per kamera): GPU secepat apa pun
hanya mempercepat langkah pose.
"""
from __future__ import annotations

import contextlib
import io
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .autotune import pipeline_fps
from .config import MODELS_DIR

HITUNG = ("otomatis", "gpu", "cpu")
KESETARAAN = MODELS_DIR / "gpu_kesetaraan.json"
DML = "DmlExecutionProvider"
_UKUR_ID = "__ukur__"
_clock = time.perf_counter


@dataclass
class Pilihan:
    jenis: str                # "gpu" | "cpu"
    pose: object              # punya detect(cam_id, frame), reset(cam_id), forget(cam_id)
    nama: str
    ms: float | None          # median milidetik per gambar (pose saja)
    pesan: str = ""
    device_id: int | None = None


def nama_gpu() -> list[str]:
    """Nama kartu grafis Windows (Win32_VideoController), tanpa "Microsoft Basic". [] di sistem lain."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [n.strip() for n in out.splitlines() if n.strip() and "Microsoft Basic" not in n]


def dml_tersedia() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return DML in ort.get_available_providers()


def baca_kesetaraan(nama: list[str], imgsz: int) -> bool | None:
    """Hasil uji kesetaraan GPU untuk kartu grafis `nama` (nama_gpu()) dan resolusi pose `imgsz`: True/False, atau None
    bila belum diuji di komputer ini (folder SAHABAT disalin dari laptop lain), di resolusi lain, atau berkasnya rusak."""
    try:
        data = json.loads(Path(KESETARAAN).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("perangkat"), dict):
        return None
    if sorted(data["perangkat"].get("nama") or []) != sorted(nama) or data.get("imgsz") != imgsz:
        return None
    lolos = data.get("lolos")
    return lolos if isinstance(lolos, bool) else None


def buat_gpu(device_id: int, model_path=None, imgsz: int = 416):
    """OnnxPose di adaptor DirectML ke-device_id. Melempar RuntimeError bila adaptor itu tidak bisa dipakai.

    ONNX Runtime tidak melempar galat untuk adaptor yang tidak ada: ia diam-diam pindah ke CPU (dan mencetak "EP Error").
    Karena itu penyedia sesi diperiksa sesudah dibuat.
    """
    from .pose_onnx import ONNX_MODEL, OnnxPose
    with contextlib.redirect_stdout(io.StringIO()):
        pose = OnnxPose(model_path or ONNX_MODEL, providers=(DML, "CPUExecutionProvider"),
                        provider_options=[{"device_id": int(device_id)}, {}], nama=f"GPU (DirectML #{device_id})",
                        imgsz=int(imgsz))
    if DML not in pose.session.get_providers():
        raise RuntimeError(f"adaptor #{device_id} tidak bisa dipakai DirectML")
    return pose


def gpu_factory_bawaan(model_path=None, imgsz: int = 416):
    """Pembuat OnnxPose GPU pada resolusi pose `imgsz` (sama dengan CPU: resolusi tidak pernah diturunkan diam-diam),
    atau (None, alasan) bila paket DirectML atau model ONNX tidak ada."""
    from .pose_onnx import ONNX_MODEL
    if not dml_tersedia():
        return None, "paket GPU (onnxruntime-directml) belum terpasang"
    path = Path(model_path or ONNX_MODEL)
    if not path.exists():
        return None, f"model {path.name} belum ada"
    return (lambda device_id: buat_gpu(device_id, path, imgsz)), ""


def ukur(pose, frame, n: int = 5) -> float:
    """Median detik per gambar sesudah 1 pemanasan. Pelacak ukur dibuang sesudahnya."""
    try:
        pose.detect(_UKUR_ID, frame)
        times = []
        for _ in range(n):
            t = _clock()
            pose.detect(_UKUR_ID, frame)
            times.append(_clock() - t)
        return statistics.median(times)
    finally:
        forget = getattr(pose, "forget", None)
        if forget is not None:
            forget(_UKUR_ID)


def _gpu_tercepat(gpu_factory, frame, gpu_ids):
    best = None
    for device_id in gpu_ids:
        try:
            pose = gpu_factory(device_id)
            detik = ukur(pose, frame)
        except Exception:                       # adaptor tidak ada, driver menolak, kehabisan memori: lewati
            continue
        if best is None or detik < best[1]:
            best = (pose, detik, device_id)
    return best


def pilih(mode: str, *, cpu_factory, gpu_factory=None, frame, gpu_ids=range(4), lolos=None,
          alasan_gpu: str = "") -> Pilihan:
    if mode not in HITUNG:
        raise ValueError(f"Perangkat hitung tidak dikenal: {mode}")

    def cpu(pesan: str = "", ms: float | None = None) -> Pilihan:
        pose = cpu_factory()
        if ms is None:
            ms = ukur(pose, frame) * 1000
        return Pilihan("cpu", pose, getattr(pose, "nama", "CPU (OpenVINO)"), ms, pesan)

    if mode == "cpu":
        return cpu()
    if mode == "otomatis" and lolos is not True:
        pesan = "" if gpu_factory is None else "GPU belum lolos uji kesetaraan, Mode CCTV memakai CPU."
        return cpu(pesan)
    best = _gpu_tercepat(gpu_factory, frame, gpu_ids) if gpu_factory is not None else None
    if best is None:
        alasan = alasan_gpu or "tidak ada adaptor DirectML yang bisa dipakai"
        return cpu(f"GPU tidak tersedia ({alasan}), Mode CCTV memakai CPU." if mode == "gpu" else "")
    pose, detik, device_id = best
    gpu = Pilihan("gpu", pose, getattr(pose, "nama", "GPU (DirectML)"), detik * 1000, device_id=device_id)
    if mode == "gpu":
        if lolos is not True:
            gpu.pesan = "GPU dipakai walau belum lolos uji kesetaraan; hasilnya bisa sedikit berbeda dari CPU."
        return gpu
    fallback = cpu_factory()
    cpu_ms = ukur(fallback, frame) * 1000
    if gpu.ms < cpu_ms:
        return gpu
    return Pilihan("cpu", fallback, getattr(fallback, "nama", "CPU (OpenVINO)"), cpu_ms,
                   "GPU tidak lebih cepat dari CPU di komputer ini, Mode CCTV memakai CPU.")


def kapasitas(detik_per_gambar: float, rate_hz: float = 10.0) -> int:
    """Banyak kamera penuh yang sanggup dinilai rate_hz kali per detik (pose + penilaian CPU ±30 ms)."""
    if detik_per_gambar <= 0:
        return 0
    return int(pipeline_fps(1.0 / detik_per_gambar) // rate_hz)
