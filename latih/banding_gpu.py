"""Uji kesetaraan GPU: apakah pose dari GPU (ONNX Runtime + DirectML) memberi hasil deteksi yang tidak lebih buruk?

    python -m latih.banding_gpu ekstrak    jejak pose ulang dengan GPU untuk data uji (bisa dilanjutkan bila terhenti)
    python -m latih.banding_gpu ukur       ukur penilai aplikasi pada jejak CPU dan GPU, lalu tulis hasilnya

Data uji sama dengan baris "aplikasi" 10 Hz di latih.laju_rendah:
- jatuh: CAUCAFall subjek 9-10 dan Le2i bagian uji (F1);
- kekerasan dan alarm keliru per jam: rekaman sesi 17 November 2026.
Model, ambang, dan aturan tidak diubah. Hanya sumber titik tubuhnya yang berganti: jejak CPU (OpenVINO) yang dipakai
selama ini, atau jejak GPU baru dari video yang sama.

Hasil: models/gpu_kesetaraan.json ({"lolos", "tanggal", "ringkas", ...}) dibaca Mode CCTV. Pilihan "Otomatis" baru
memakai GPU bila lolos: F1 jatuh, F1 kekerasan, dan kekerasan tertangkap tidak lebih rendah, dan alarm keliru per jam
tidak lebih tinggi daripada CPU. Hasilnya hanya berlaku untuk kartu grafis tempat uji dijalankan dan resolusi 416
(jejak CPU dibuat pada 416): di komputer lain atau resolusi lain, Mode CCTV menganggapnya "belum diuji".
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path("D:/sahabat-data")
P3 = "putaran3"
SESSION = "webcam/sesi-20261117-101936"
OUT_TRACES = DATA / "banding-gpu"
OUT = Path(__file__).resolve().parent.parent / "docs" / "uji" / "banding-gpu"
IMGSZ = 416                 # resolusi jejak CPU data uji


LATENSI_TOLERANSI_S = 0.1     # satu penilaian (10 Hz): alarm jatuh GPU tidak boleh lebih lambat dari ini


def lolos(cpu: dict, gpu: dict) -> bool:
    """GPU tidak lebih buruk dari CPU di semua angka gerbang, termasuk waktu peringatan jatuh."""
    c, g = cpu["kekerasan_sesi17"], gpu["kekerasan_sesi17"]
    lat_c, lat_g = cpu["jatuh"].get("latensi_median"), gpu["jatuh"].get("latensi_median")
    late = lat_c is not None and (lat_g is None or lat_g > lat_c + LATENSI_TOLERANSI_S)
    return (gpu["jatuh"]["f1"] >= cpu["jatuh"]["f1"] and g["f1"] >= c["f1"] and g["tp"] >= c["tp"]
            and g["alarm_keliru_per_jam"] <= c["alarm_keliru_per_jam"] and not late)


def ringkas(cpu: dict, gpu: dict) -> str:
    c, g = cpu["kekerasan_sesi17"], gpu["kekerasan_sesi17"]
    return (f"CPU → GPU: F1 jatuh {cpu['jatuh']['f1']} → {gpu['jatuh']['f1']}, F1 kekerasan {c['f1']} → {g['f1']}, "
            f"kekerasan tertangkap {c['tp']} → {g['tp']}, alarm keliru/jam {c['alarm_keliru_per_jam']} → "
            f"{g['alarm_keliru_per_jam']}")


def gpu_path(original: Path) -> Path | None:
    """Jejak GPU untuk jejak CPU `original` (tata letak folder sama, di bawah OUT_TRACES)."""
    try:
        rel = Path(original).resolve().relative_to(DATA.resolve())
    except ValueError:
        return None
    return OUT_TRACES / rel


def redirect(load, used: set):
    """PoseTrace.load yang membaca jejak GPU. Jejak yang belum diekstrak menghentikan pengukuran (bukan diam-diam
    memakai jejak CPU)."""
    def patched(path):
        mirror = gpu_path(Path(path))
        if mirror is None:
            return load(path)
        if not mirror.is_file():
            raise FileNotFoundError(f"Jejak GPU {mirror} belum ada. Jalankan: python -m latih.banding_gpu ekstrak")
        used.add(mirror)
        return load(mirror)
    return patched


def jobs():
    """(jejak CPU asli, sumber frame) untuk semua data uji."""
    from sahabat.config import Config
    from .latih_jatuh_baru import annotation
    rate = Config().rate_hz
    root = DATA / P3
    videos = {v.stem: v for v in (root / "cauca").rglob("*.avi")}
    for name, subject, *_ in annotation(root):
        if subject in (9, 10):
            yield root / "cauca_416" / f"{name}.npz", ("video", videos[name], rate)
    for row in json.loads((root / "le2i-selection.json").read_text(encoding="utf-8")):
        video = root / "le2i" / row["scene"] / (row["name"] + row["ext"])
        yield root / "le2i_416" / (row["key"].replace("/", "__") + ".npz"), ("video", video, 10)
    yield DATA / SESSION / "jejak_416.npz", ("session", DATA / SESSION)


def _frames(spec):
    from sahabat.recorder import read_session
    from .putar import video_ticks
    if spec[0] == "video":
        return video_ticks(spec[1], spec[2])
    return read_session(spec[1])


class _Single:
    """OnnxPose untuk satu video: pelacak baru tiap video, seperti YoloPose.reset() pada ekstraksi CPU."""

    def __init__(self, pose) -> None:
        self.pose = pose

    def detect(self, frame):
        return self.pose.detect("video", frame)


def extract(device_id: int | None = None) -> int:
    from sahabat import perangkat_hitung as ph
    from .putar import trace
    factory, why = ph.gpu_factory_bawaan(imgsz=IMGSZ)
    if factory is None:
        print("GPU tidak tersedia:", why)
        return 1
    if device_id is None:
        frame = np.zeros((480, 640, 3), np.uint8)
        best = ph._gpu_tercepat(factory, frame, range(4))
        if best is None:
            print("Tidak ada adaptor DirectML yang bisa dipakai.")
            return 1
        pose, _, device_id = best
    else:
        pose = factory(device_id)
    print("GPU:", pose.nama, ph.nama_gpu(), flush=True)
    todo = list(jobs())
    started = time.monotonic()
    for i, (original, spec) in enumerate(todo, 1):
        target = gpu_path(original)
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        pose.reset("video")
        tr = trace(_frames(spec), _Single(pose))
        if not len(tr.t):
            raise ValueError(f"Tidak ada frame dari {spec[1]}")
        temporary = target.with_suffix(".part.npz")
        tr.save(temporary)
        temporary.replace(target)
        print(f"{i}/{len(todo)} {original.name} ({len(tr.t)} tick, {time.monotonic() - started:.0f} dtk)", flush=True)
    (OUT_TRACES / "perangkat.json").write_text(json.dumps({"gpu": pose.nama, "nama": ph.nama_gpu(),
                                                          "device_id": device_id}, ensure_ascii=False),
                                               encoding="utf-8")
    return 0


def compare_traces() -> dict:
    """Seberapa mirip jejak GPU dengan jejak CPU, per tick: jumlah orang sama, dan selisih titik tubuh (piksel)."""
    from .putar import PoseTrace
    same = total = 0
    diffs = []
    for original, _ in jobs():
        a, b = PoseTrace.load(original), PoseTrace.load(gpu_path(original))
        for pa, pb in zip(a.people, b.people):
            total += 1
            same += len(pa) == len(pb)
            ia, ib = {p.track_id: p for p in pa}, {p.track_id: p for p in pb}
            for k in ia.keys() & ib.keys():
                ok = (ia[k].conf > 0.5) & (ib[k].conf > 0.5)
                if ok.any():
                    diffs.append(float(np.abs(ia[k].kp[ok] - ib[k].kp[ok]).mean()))
    return {"tick": total, "jumlah_orang_sama": round(same / total, 4) if total else None,
            "selisih_titik_median_px": round(float(np.median(diffs)), 2) if diffs else None}


def measure() -> int:
    from . import laju_rendah
    from .putar import PoseTrace
    missing = [gpu_path(o) for o, _ in jobs() if not gpu_path(o).is_file()]
    if missing:
        print(f"{len(missing)} jejak GPU belum ada (misalnya {missing[0]}). Jalankan dulu: "
              "python -m latih.banding_gpu ekstrak")
        return 1
    cpu = laju_rendah.measure_rate(1)
    print("CPU", json.dumps(cpu, ensure_ascii=False), flush=True)
    original = PoseTrace.load
    used: set = set()
    PoseTrace.load = classmethod(lambda cls, path: redirect(lambda p: original.__func__(cls, p), used)(path))
    try:
        gpu = laju_rendah.measure_rate(1)
    finally:
        PoseTrace.load = original
    expected = {gpu_path(o) for o, _ in jobs()}
    if used != expected:
        raise RuntimeError(f"Pengukuran GPU memakai {len(used)} dari {len(expected)} jejak GPU.")
    print("GPU", json.dumps(gpu, ensure_ascii=False), flush=True)
    info_path = OUT_TRACES / "perangkat.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    result = {"lolos": lolos(cpu, gpu), "tanggal": time.strftime("%Y-%m-%d"), "ringkas": ringkas(cpu, gpu),
              "imgsz": IMGSZ, "perangkat": info, "jejak": compare_traces(), "cpu": cpu, "gpu": gpu}
    from sahabat.perangkat_hitung import KESETARAAN
    KESETARAAN.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hasil.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print()
    print(("LOLOS" if result["lolos"] else "TIDAK LOLOS") + ":", result["ringkas"])
    print("Jejak:", json.dumps(result["jejak"], ensure_ascii=False))
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["ekstrak"]:
        device = int(argv[2]) if len(argv) > 2 and argv[1] == "--gpu" else None
        return extract(device)
    if argv[:1] == ["ukur"]:
        return measure()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
