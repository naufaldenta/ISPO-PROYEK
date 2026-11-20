"""Perekam uji: menyimpan frame yang dinilai (tanpa kerangka) beserta cap waktunya.

Mati secara bawaan. Dinyalakan hanya lewat "Mulai SAHABAT (rekam uji).bat" (python -m sahabat --rekam),
supaya sesi uji bisa diputar ulang berkali-kali saat menyetel ambang (lihat latih/putar.py).

Satu sesi = satu folder di data/rekaman/, berisi potongan 0001.avi, 0002.avi, ... (MJPG: tetap terbaca
walau jendela aplikasi ditutup paksa), 0001.txt, ... (satu cap waktu per frame), dan tanda.jsonl
(tanda adegan yang ditekan pencatat), dan meta.json (jenis rekaman "adegan" atau "biasa", nama kamera, posisi
kamera; dibaca latih.kamera_tinggi sekolah). Rekaman berisi wajah orang: hapus foldernya setelah selesai.
"""
from __future__ import annotations

import json
import queue
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

SEGMENT_S = 60.0
MIN_FREE_BYTES = 2 * 1024 ** 3      # berhenti merekam sebelum disk benar-benar penuh
NOMINAL_FPS = 10.0                  # hanya untuk pemutar video biasa; waktu sebenarnya ada di berkas .txt
MARKS_FILE = "tanda.jsonl"
META_FILE = "meta.json"
_EPS = 1e-6


def _free_bytes(folder: Path) -> int:
    return shutil.disk_usage(folder).free


class Recorder:
    def __init__(self, root: Path, *, name: str | None = None, segment_s: float = SEGMENT_S,
                 min_free: int = MIN_FREE_BYTES, free_bytes: Callable[[Path], int] = _free_bytes,
                 queue_size: int = 50, start: bool = True, meta: dict | None = None) -> None:
        self.folder = Path(root) / (name or datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.segment_s = segment_s
        self.min_free = min_free
        self.dropped = 0                     # frame yang dilewati karena penulis tertinggal
        self.active = False
        self._free_bytes = free_bytes
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._writer: cv2.VideoWriter | None = None
        self._times = None
        self._segment = 0
        self._segment_t0 = 0.0
        self._size: tuple[int, int] | None = None
        self._stopped_for_space = False
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
        except OSError:                      # folder hanya-baca: aplikasi tetap jalan, hanya tanpa rekaman
            print("Peringatan: rekaman uji tidak bisa disimpan; lanjut tanpa merekam.")
            return
        self.active = True
        if meta is not None:
            try:
                (self.folder / META_FILE).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
        if start:
            self.start()

    def start(self) -> None:
        if self.active and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="perekam", daemon=True)
            self._thread.start()

    # ---- dipanggil dari mesin: tidak boleh menunggu disk ---------------------------

    def write(self, t: float, frame: np.ndarray) -> None:
        if not self.active:
            return
        try:
            self._queue.put_nowait(("frame", t, frame))
        except queue.Full:
            self.dropped += 1

    def mark(self, t: float, label: str) -> bool:
        """Tanda adegan dari pencatat (misalnya "jatuh"). False bila tidak sedang merekam."""
        if not self.active:
            return False
        try:
            self._queue.put_nowait(("mark", t, label))
        except queue.Full:
            return False
        return True

    def close(self) -> None:
        """Menulis sisa antrean lalu menutup berkas."""
        was_active, self.active = self.active, False
        if self._thread is not None:
            self._queue.put(("stop", 0.0, None))
            self._thread.join(timeout=10.0)
            self._thread = None
        elif was_active:
            self._release()

    # ---- utas penulis -------------------------------------------------------------

    def _run(self) -> None:
        while True:
            kind, t, payload = self._queue.get()
            if kind == "stop":
                break
            try:
                if kind == "frame":
                    self._write_frame(t, payload)
                else:
                    with open(self.folder / MARKS_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"t": t, "label": payload}) + "\n")
            except (OSError, cv2.error):
                print("Peringatan: rekaman uji gagal ditulis; perekaman dihentikan.")
                self.active = False
        self._release()

    def _write_frame(self, t: float, frame: np.ndarray) -> None:
        if self._stopped_for_space:
            return
        size = (frame.shape[1], frame.shape[0])
        if self._writer is None or size != self._size or t - self._segment_t0 >= self.segment_s - _EPS:
            self._release()
            if self._free_bytes(self.folder) < self.min_free:
                print("Peringatan: disk hampir penuh; perekaman uji dihentikan.")
                self._stopped_for_space = True
                self.active = False
                return
            self._segment += 1
            stem = self.folder / f"{self._segment:04d}"
            writer = cv2.VideoWriter(str(stem.with_suffix(".avi")), cv2.VideoWriter_fourcc(*"MJPG"),
                                     NOMINAL_FPS, size)
            if not writer.isOpened():
                writer.release()
                raise OSError("penulis video tidak bisa dibuka")
            self._writer, self._size, self._segment_t0 = writer, size, t
            self._times = open(stem.with_suffix(".txt"), "a", encoding="utf-8")
        self._writer.write(frame)
        self._times.write(repr(float(t)) + "\n")
        self._times.flush()

    def _release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._times is not None:
            self._times.close()
            self._times = None


def read_session(folder: Path) -> Iterator[tuple[float, np.ndarray]]:
    """(cap waktu, frame) seluruh sesi, berurutan. Potongan yang terpotong dibaca sejauh yang utuh."""
    for video in sorted(Path(folder).glob("[0-9]*.avi")):
        try:
            times = [float(line) for line in video.with_suffix(".txt").read_text(encoding="utf-8").split()]
        except (OSError, ValueError):
            continue
        cap = cv2.VideoCapture(str(video))
        try:
            for t in times:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                yield t, frame
        finally:
            cap.release()


def read_marks(folder: Path) -> list[tuple[float, str]]:
    try:
        lines = (Path(folder) / MARKS_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    marks = []
    for line in lines:
        try:
            rec = json.loads(line)
            marks.append((float(rec["t"]), str(rec["label"])))
        except (ValueError, KeyError, TypeError):
            continue
    return marks
