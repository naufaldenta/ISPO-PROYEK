"""Sumber video: webcam atau file. Satu utas latar menyimpan frame terbaru dan menyambung ulang sendiri."""
from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

from .devices import camera_routes

CAMERA_SIZE = (640, 480)
MAX_BAD_READS = 15
STALE_S = 1.5              # frame lebih tua dari ini dianggap kamera terputus (driver bisa macet tanpa melapor)


def _open_camera(index: int, backend: int = cv2.CAP_DSHOW) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
    return cap


def find_camera(max_index: int | None = None) -> int | None:
    """Indeks kamera pertama yang benar-benar memberi frame."""
    routes = camera_routes(None) if max_index is None else [(i, cv2.CAP_DSHOW) for i in range(max_index + 1)]
    for index, backend in routes:
        cap = _open_camera(index, backend)
        try:
            if cap.isOpened() and cap.read()[0]:
                return index + backend    # OpenCV CAP_ANY encode backend pada ratusan/ribuan
        finally:
            cap.release()
    return None


class VideoSource:
    def __init__(self, spec: int | str | None, retry_s: float = 2.0, flip: bool = False) -> None:
        self.spec = spec                    # camera:... = ID perangkat, int = indeks lama, str lain = video
        # Sebagian driver webcam mengirim gambar tercermin. flip=True membaliknya kiri-kanan sebelum gambar dipakai
        # siapa pun (pose, perekam, layar). Model tidak terpengaruh: cirinya sama untuk adegan dan cerminannya.
        self.flip = bool(flip)
        self.retry_s = retry_s
        self.connected = False
        self._latest: tuple[float, np.ndarray] | None = None
        self._times: deque[float] = deque(maxlen=120)      # cap waktu frame terakhir, untuk mengukur laju kamera
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._changed = threading.Event()
        self.version = 0
        self.error = ""
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="sumber-video", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._changed.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._drop()

    def select(self, spec: int | str | None) -> None:
        """Utas capture sendiri yang melepas perangkat lama; frame yang sedang dibaca dibuang."""
        with self._lock:
            if spec == self.spec:
                return
            self.spec = spec
            self.version += 1
            self.connected, self._latest = False, None
            self._times.clear()
            self.error = "Menghubungkan kamera pilihan…"
        self._changed.set()

    def packet(self):
        """Versi input dan frame dibaca bersama agar mesin tidak mencampur dua kamera."""
        with self._lock:
            got = self._latest if self.connected else None
            if got is not None and time.monotonic() - got[0] > STALE_S:
                got = None
            return self.version, got

    def read(self) -> tuple[float, np.ndarray] | None:
        """(cap waktu monotonic, frame BGR) terbaru; None bila tidak tersambung."""
        with self._lock:
            if not self.connected or self._latest is None:
                return None
            if time.monotonic() - self._latest[0] > STALE_S:
                return None
            return self._latest

    @property
    def fps(self) -> float:
        """Laju kamera sebenarnya dalam 2 detik terakhir; 0 bila tidak tersambung atau belum terukur."""
        with self._lock:
            if not self.connected or len(self._times) < 2:
                return 0.0
            recent = [t for t in self._times if self._times[-1] - t <= 2.0]
            span = recent[-1] - recent[0]
            return (len(recent) - 1) / span if span > 0 else 0.0

    def _flips(self, spec) -> bool:
        """Hanya kamera yang dibalik; video uji diputar apa adanya."""
        return self.flip and not (isinstance(spec, str) and not spec.startswith("camera:"))

    def _drop(self) -> None:
        with self._lock:
            self.connected = False
            self._latest = None
            self._times.clear()

    def _open(self) -> cv2.VideoCapture | None:
        spec = self.spec
        if isinstance(spec, str) and not spec.startswith("camera:"):
            cap = cv2.VideoCapture(spec)
            if cap.isOpened():
                return cap
            cap.release()
            return None
        for index, backend in camera_routes(spec):
            if self._stop.is_set() or spec != self.spec:
                return None
            cap = None
            try:
                cap = _open_camera(index, backend)
                if cap.isOpened():
                    for _ in range(5):
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            return cap
                        time.sleep(0.02)
            except cv2.error:
                pass                                # driver ini gagal; coba backend lain untuk perangkat yang sama
            if cap is not None:
                cap.release()
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._changed.clear()
            version, spec = self.version, self.spec
            is_file = isinstance(spec, str) and not spec.startswith("camera:")
            try:
                cap = self._open()
                self.error = "" if cap is not None else "Belum bisa dibuka. Periksa sambungan, izin kamera, atau aplikasi lain yang memakai kamera."
            except Exception as exc:
                cap = None
                self.error = "Kamera tidak bisa dibuka: " + str(exc)
            if cap is None:
                self._drop()
                self._changed.wait(self.retry_s)
                continue
            frame_gap = 0.0
            if is_file:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_gap = 1.0 / fps if 1.0 <= fps <= 120.0 else 1.0 / 25.0
            bad = 0
            try:
                while not self._stop.is_set() and version == self.version:
                    started = time.monotonic()
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        if is_file and cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                            ok, frame = cap.read()             # file habis: ulang dari awal
                    if not ok or frame is None:
                        bad += 1
                        if bad >= (1 if is_file else MAX_BAD_READS):
                            break
                        self._stop.wait(0.02)
                        continue
                    bad = 0
                    if self._flips(spec):
                        frame = cv2.flip(frame, 1)
                    with self._lock:
                        if version != self.version:
                            break
                        now = time.monotonic()
                        self._latest = (now, frame)
                        self._times.append(now)
                        self.connected = True
                    if frame_gap:
                        self._changed.wait(max(0.0, frame_gap - (time.monotonic() - started)))
            except cv2.error as exc:
                self.error = "Sambungan kamera terputus: " + str(exc)
            finally:
                cap.release()
            self._drop()
            if version == self.version:
                self._changed.wait(self.retry_s)
