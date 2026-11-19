"""Sumber video: webcam, file, atau kamera jaringan (IP camera, DVR/NVR, MJPEG). Satu utas latar menyimpan frame
terbaru dan menyambung ulang sendiri.

Kamera CCTV dari panel Kamera dipilih sebagai "ipcam:<id>"; alamat lengkap dengan kata sandi hanya dibangun saat
membuka stream (KameraStore.resolve) dan tidak pernah disimpan di objek ini. Semua pesan galat disamarkan.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Callable

import cv2
import numpy as np

from .devices import camera_routes
from .kamera import IPCAM_PREFIX, Stream
from .merek import SKEMA_BOLEH, samarkan

# RTSP lewat TCP: UDP sering putus atau patah-patah di WiFi sekolah. Diisi sebelum stream pertama dibuka.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
# Log FFmpeg bisa mencetak alamat stream lengkap dengan kata sandi: dibungkam, galat dilaporkan lewat self.error.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
NET_TIMEOUT_MS = 5000      # batas buka dan baca stream jaringan, supaya utas tidak macet bila kamera mati
NETWORK_MESSAGE = "Kamera CCTV belum bisa dibuka. Periksa kabel/WiFi, alamat, dan kata sandi di panel Kamera."
UNKNOWN_IPCAM = "Kamera CCTV ini tidak ada di daftar. Buka panel Kamera."
OPEN_FAILED = "Belum bisa dibuka. Periksa sambungan, izin kamera, atau aplikasi lain yang memakai kamera."
WRONG_PASSWORD = "Kata sandi kamera CCTV salah. Perbaiki di panel Kamera (Ubah), lalu Simpan."
NET_STALL_S = 5.0          # stream jaringan tanpa frame selama ini dianggap putus (kabel/daya kamera mati)
NET_RETRY_MAX_S = 60.0     # jeda coba ulang kamera jaringan naik dua kali lipat sampai batas ini
CAPTURE_SIZE = (1920, 1080)  # capture card keluaran HDMI DVR: resolusi penuh supaya ubin tidak kecil


def _rtsp_probe(url: str, timeout_s: float = 3.0) -> str:
    from .cctv_uji import rtsp_probe        # impor di sini: cctv_uji juga mengimpor modul ini
    return rtsp_probe(url, timeout_s=timeout_s)

CAMERA_SIZE = (640, 480)
MAX_BAD_READS = 15
STALE_S = 1.5              # frame lebih tua dari ini dianggap kamera terputus (driver bisa macet tanpa melapor)


def _open_camera(index: int, backend: int = cv2.CAP_DSHOW, size: tuple[int, int] = CAMERA_SIZE) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    return cap


def source_kind(spec) -> str:
    """webcam (int/None/camera:), cctv (ipcam:), jaringan (rtsp/http), atau file."""
    if spec is None or isinstance(spec, int) or (isinstance(spec, str) and spec.startswith("camera:")):
        return "webcam"
    if spec.startswith(IPCAM_PREFIX):
        return "cctv"
    scheme = spec.split("://", 1)[0].lower() if "://" in spec else ""
    return "jaringan" if scheme in SKEMA_BOLEH else "file"


def open_stream(url: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, NET_TIMEOUT_MS,
                                                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, NET_TIMEOUT_MS])


def crop_tile(frame: np.ndarray, grid: int, ubin: int) -> np.ndarray:
    """Satu ubin dari tampilan mosaik DVR (grid x grid, ubin dihitung dari kiri atas, mulai 0)."""
    h, w = frame.shape[:2]
    row, col = divmod(ubin, grid)
    y0, y1 = h * row // grid, h * (row + 1) // grid
    x0, x1 = w * col // grid, w * (col + 1) // grid
    return frame[y0:y1, x0:x1]


def _checked(cap: cv2.VideoCapture) -> cv2.VideoCapture | None:
    if cap.isOpened():
        return cap
    cap.release()
    return None


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
    def __init__(self, spec: int | str | None, retry_s: float = 2.0, flip: bool = False,
                 resolve: Callable[[str], Stream | None] | None = None) -> None:
        # camera:... = ID perangkat, int = indeks lama, ipcam:... = kamera CCTV, rtsp/http = stream, lain = video
        self.spec = spec
        self._resolve = resolve             # None = daftar kamera bersama (sahabat.kamera.default_store)
        self._opened_file = False
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
            # stream jaringan bisa sedang menunggu batas baca 5 dtk
            self._thread.join(timeout=7.0 if source_kind(self.spec) in ("jaringan", "cctv") else 3.0)
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

    def reopen(self) -> None:
        """Membuka ulang sumber yang sama (misalnya alamat atau kata sandi kamera CCTV baru saja diubah)."""
        with self._lock:
            self.version += 1
            self.connected, self._latest = False, None
            self._times.clear()
            self.error = "Menghubungkan ulang kamera…"
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
        """Hanya webcam yang dibalik; video uji dan kamera CCTV diputar apa adanya."""
        return self.flip and source_kind(spec) == "webcam"

    def _resolver(self) -> Callable[[str], Stream | None]:
        if self._resolve is None:
            from .kamera import default_store
            self._resolve = default_store().resolve
        return self._resolve

    def _drop(self) -> None:
        with self._lock:
            self.connected = False
            self._latest = None
            self._times.clear()

    def _open(self) -> cv2.VideoCapture | None:
        return self._open_source(self.spec)[0]

    def _open_network(self, url: str):
        """(capture | None, pesan, berhenti mencoba). Kata sandi salah tidak dicoba terus, karena kamera
        Hikvision/Dahua mengunci akun atau IP laptop setelah beberapa kali gagal login."""
        if url.lower().startswith("rtsp://"):
            probe = _rtsp_probe(url, 3.0)
            if probe == "sandi":
                return None, WRONG_PASSWORD, True
            if probe == "tak_terjangkau":
                return None, NETWORK_MESSAGE, False
        cap = _checked(open_stream(url))
        return cap, ("" if cap is not None else NETWORK_MESSAGE), False

    def _open_source(self, spec):
        """(capture | None, ubin DVR | None, pesan galat bila gagal, berhenti mencoba sampai sumber diubah)."""
        kind = source_kind(spec)
        if kind == "cctv":
            stream = self._resolver()(spec)
            if stream is None:
                return None, None, UNKNOWN_IPCAM, False
            if stream.galat:
                return None, stream.potong, stream.galat, True
            if source_kind(stream.url) == "webcam":                # capture card keluaran HDMI DVR
                cap = self._open_webcam(stream.url, spec, CAPTURE_SIZE)
                return cap, stream.potong, ("" if cap is not None else OPEN_FAILED), False
            if source_kind(stream.url) == "file":                  # kamera "video (simulasi)": diputar seperti file
                self._opened_file = True
                cap = _checked(cv2.VideoCapture(stream.url))
                return cap, stream.potong, ("" if cap is not None else OPEN_FAILED), False
            cap, message, park = self._open_network(stream.url)
            return cap, stream.potong, message, park
        if kind == "jaringan":
            cap, message, park = self._open_network(spec)
            return cap, None, message, park
        if kind == "file":
            cap = _checked(cv2.VideoCapture(spec))
            return cap, None, ("" if cap is not None else OPEN_FAILED), False
        cap = self._open_webcam(spec, spec)
        return cap, None, ("" if cap is not None else OPEN_FAILED), False

    def _open_webcam(self, device, spec, size: tuple[int, int] = CAMERA_SIZE) -> cv2.VideoCapture | None:
        """device = ID kamera yang dibuka; spec = pilihan sumber (berbeda untuk capture card dari daftar CCTV)."""
        for index, backend in camera_routes(device):
            if self._stop.is_set() or spec != self.spec:
                return None
            cap = None
            try:
                # webcam memakai pemanggilan lama (ukuran bawaan); hanya capture card meminta ukuran lain
                cap = _open_camera(index, backend) if size == CAMERA_SIZE else _open_camera(index, backend, size)
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

    def _retry_wait(self, network: bool, failures: int) -> float:
        if not network:
            return self.retry_s
        return min(self.retry_s * (2 ** max(0, failures - 1)), NET_RETRY_MAX_S)

    def _run(self) -> None:
        failures = 0                                    # gagal berturut-turut (jeda kamera jaringan bertambah)
        last_version = self.version
        while not self._stop.is_set():
            self._changed.clear()
            version, spec = self.version, self.spec
            if version != last_version:
                failures, last_version = 0, version
            kind = source_kind(spec)
            self._opened_file = False                   # diisi _open_source bila ipcam: menunjuk ke berkas video
            tile, park = None, False
            try:
                if kind in ("cctv", "jaringan"):
                    cap, tile, message, park = self._open_source(spec)
                else:                                   # jalur lama lewat _open() (dipakai juga oleh tes)
                    cap = self._open()
                    message = "" if cap is not None else OPEN_FAILED
                self.error = samarkan(message)
            except Exception as exc:
                cap = None
                self.error = samarkan("Kamera tidak bisa dibuka: " + str(exc))
            is_file = kind == "file" or self._opened_file
            network = kind in ("jaringan", "cctv") and not self._opened_file
            if cap is None:
                self._drop()
                failures += 1
                if park:                                # sandi salah/rusak: tunggu sampai sumber diubah
                    while not self._changed.is_set() and not self._stop.is_set():
                        self._changed.wait(1.0)
                else:
                    self._changed.wait(self._retry_wait(network, failures))
                continue
            frame_gap = 0.0
            if is_file:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_gap = 1.0 / fps if 1.0 <= fps <= 120.0 else 1.0 / 25.0
            bad = 0
            last_ok = time.monotonic()
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
                        if network and time.monotonic() - last_ok >= NET_STALL_S:
                            break                               # stream macet: tiap baca menunggu batas 5 dtk
                        self._stop.wait(0.02)
                        continue
                    bad = 0
                    last_ok = time.monotonic()
                    failures = 0
                    if self._flips(spec):
                        frame = cv2.flip(frame, 1)
                    if tile is not None:
                        frame = np.ascontiguousarray(crop_tile(frame, *tile))
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
                self.error = samarkan("Sambungan kamera terputus: " + str(exc))
            finally:
                cap.release()
            self._drop()
            if version == self.version:
                failures += 1
                self._changed.wait(self._retry_wait(network, failures))
