import time

import cv2
import numpy as np
import pytest

from sahabat.source import VideoSource


@pytest.fixture
def tiny_video(tmp_path):
    """Video 20 frame 160x120, 30 fps (±0,7 detik)."""
    path = tmp_path / "kecil.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120))
    for i in range(20):
        writer.write(np.full((120, 160, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path


def wait_frame(source, after=None, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = source.read()
        if got is not None and (after is None or got[0] > after):
            return got
        time.sleep(0.01)
    raise AssertionError("tidak ada frame baru")


def test_file_source_gives_frames_with_increasing_time(tiny_video):
    s = VideoSource(str(tiny_video))
    s.start()
    try:
        a = wait_frame(s)
        b = wait_frame(s, after=a[0])
        assert s.connected and b[0] > a[0] and a[1].shape == (120, 160, 3)
    finally:
        s.stop()


def test_file_source_loops(tiny_video):
    s = VideoSource(str(tiny_video))
    s.start()
    try:
        first = wait_frame(s)
        time.sleep(1.5)                                   # video hanya ±0,7 dtk
        assert s.connected and wait_frame(s, after=first[0])[0] > first[0] + 1.0
    finally:
        s.stop()


def test_missing_camera_is_disconnected_not_crashed():
    s = VideoSource(99, retry_s=0.2)
    s.start()
    try:
        time.sleep(0.6)
        assert s.connected is False and s.read() is None
    finally:
        s.stop()


def test_missing_file_is_disconnected_not_crashed(tmp_path):
    s = VideoSource(str(tmp_path / "tidak-ada.mp4"), retry_s=0.2)
    s.start()
    try:
        time.sleep(0.4)
        assert s.connected is False and s.read() is None
    finally:
        s.stop()


def test_stop_is_quick_and_idempotent(tiny_video):
    s = VideoSource(str(tiny_video))
    s.start()
    wait_frame(s)
    t0 = time.monotonic()
    s.stop()
    s.stop()
    assert time.monotonic() - t0 < 2.0 and s.connected is False


def test_source_reports_its_own_frame_rate(tiny_video):
    s = VideoSource(str(tiny_video))
    assert s.fps == 0.0
    s.start()
    try:
        wait_frame(s)
        time.sleep(1.5)
        assert 20.0 <= s.fps <= 40.0                      # video uji 30 FPS
    finally:
        s.stop()
    assert s.fps == 0.0


def test_stale_frame_counts_as_disconnected():
    s = VideoSource(0)                                      # tidak dijalankan: keadaan diisi langsung
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    s._latest, s.connected = (time.monotonic(), frame), True
    assert s.read() is not None
    s._latest = (time.monotonic() - 5.0, frame)             # driver macet: frame terakhir sudah basi
    assert s.read() is None


def test_selected_camera_falls_back_to_same_device_other_backend(monkeypatch):
    from sahabat import source
    calls = []
    class Capture:
        def __init__(self, good):
            self.good, self.released = good, False
        def isOpened(self): return True
        def read(self): return (True, np.zeros((10, 10, 3), np.uint8)) if self.good else (False, None)
        def release(self): self.released = True
    bad, good = Capture(False), Capture(True)
    monkeypatch.setattr(source, "camera_routes", lambda spec: [(7, cv2.CAP_DSHOW), (2, cv2.CAP_MSMF)])
    def opener(index, backend):
        calls.append((index, backend))
        return bad if backend == cv2.CAP_DSHOW else good
    monkeypatch.setattr(source, "_open_camera", opener)
    assert VideoSource("camera:usb")._open() is good
    assert bad.released and calls == [(7, cv2.CAP_DSHOW), (2, cv2.CAP_MSMF)]


def test_backend_exception_still_tries_second_route(monkeypatch):
    from sahabat import source
    class Capture:
        def isOpened(self): return True
        def read(self): return True, np.zeros((10, 10, 3), np.uint8)
    good = Capture()
    monkeypatch.setattr(source, "camera_routes", lambda spec: [(0, cv2.CAP_DSHOW), (1, cv2.CAP_MSMF)])
    def open_camera(index, backend):
        if backend == cv2.CAP_DSHOW:
            raise cv2.error("driver gagal")
        return good
    monkeypatch.setattr(source, "_open_camera", open_camera)
    assert VideoSource("camera:usb")._open() is good


def test_switch_discards_inflight_frame_and_releases_old_capture(monkeypatch):
    import threading
    from sahabat import source
    entered, release = threading.Event(), threading.Event()
    class Capture:
        def __init__(self, value): self.value, self.closed = value, False
        def read(self):
            if self.value == 1:
                entered.set()
                release.wait(2)
            time.sleep(0.01)
            return True, np.full((10, 10, 3), self.value, np.uint8)
        def release(self): self.closed = True
    old, new = Capture(1), Capture(2)
    s = VideoSource(0, retry_s=0.01)
    monkeypatch.setattr(s, "_open", lambda: old if s.spec == 0 else new)
    s.start()
    try:
        assert entered.wait(1)
        s.select(1)
        assert s.read() is None
        release.set()
        _, frame = wait_frame(s)
        assert frame[0, 0, 0] == 2 and old.closed
        assert s.version == 1
    finally:
        release.set()
        s.stop()


def _fake_camera(monkeypatch, s):
    class Capture:
        def read(self):
            time.sleep(0.01)
            frame = np.zeros((2, 3, 3), np.uint8)
            frame[:, 0] = 255                      # kolom kiri putih
            return True, frame
        def release(self): pass
    monkeypatch.setattr(s, "_open", lambda: Capture())


def test_camera_can_be_flipped_left_right_and_switched_live(monkeypatch):
    s = VideoSource(0, retry_s=0.01, flip=True)
    _fake_camera(monkeypatch, s)
    s.start()
    try:
        _, frame = wait_frame(s)
        assert frame[0, 2, 0] == 255 and frame[0, 0, 0] == 0      # kiri pindah ke kanan
        s.flip = False
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and s.read()[1][0, 0, 0] != 255:
            time.sleep(0.02)
        assert s.read()[1][0, 0, 0] == 255
    finally:
        s.stop()


def test_video_files_are_never_flipped(tiny_video):
    plain, flipped = VideoSource(str(tiny_video)), VideoSource(str(tiny_video), flip=True)
    for s in (plain, flipped):
        s.start()
    try:
        a, b = wait_frame(plain)[1], wait_frame(flipped)[1]
        assert a.shape == b.shape
        assert flipped._flips(str(tiny_video)) is False and flipped._flips("camera:usb") is True
    finally:
        plain.stop()
        flipped.stop()


# ---- kamera jaringan, CCTV, dan ubin DVR ----
import sahabat.source as source_module  # noqa: E402
from sahabat.kamera import Stream  # noqa: E402
from sahabat.source import crop_tile, source_kind  # noqa: E402
from tests.helpers import wait_for  # noqa: E402


def test_kinds():
    assert source_kind(None) == source_kind(0) == source_kind("camera:usb") == "webcam"
    assert source_kind("rtsp://h/x") == source_kind("http://h:8080/video") == "jaringan"
    assert source_kind("ipcam:k1") == "cctv" and source_kind("C:/v/a.mp4") == "file"


def test_reads_mjpeg_stream_without_file_pacing(mjpeg_server):
    src = VideoSource(mjpeg_server.url)
    src.start()
    try:
        assert wait_for(lambda: src.read() is not None, 5)
        assert wait_for(lambda: src.fps > 12, 5)               # file dipacu, stream jaringan tidak
    finally:
        src.stop()


def test_ipcam_resolves_through_store_and_crops_tile(mjpeg_server):
    src = VideoSource("ipcam:k1", resolve=lambda spec: Stream(mjpeg_server.url, (2, 3)))
    src.start()
    try:
        assert wait_for(lambda: src.read() is not None, 5)
        assert src.read()[1].shape == (180, 320, 3)
    finally:
        src.stop()


def test_unknown_ipcam_sets_message_and_never_flips():
    src = VideoSource("ipcam:hilang", flip=True, resolve=lambda spec: None)
    src.start()
    try:
        assert wait_for(lambda: "tidak ada di daftar" in src.error, 3)
    finally:
        src.stop()
    assert not VideoSource("rtsp://h/x", flip=True)._flips("rtsp://h/x")
    assert not VideoSource("ipcam:k1", flip=True)._flips("ipcam:k1")


def test_unreachable_camera_times_out_and_stops():              # Review Focus 2
    src = VideoSource("ipcam:k1", retry_s=0.2,
                      resolve=lambda s: Stream("rtsp://admin:rahasia@127.0.0.1:1/x", None))
    src.start()
    assert wait_for(lambda: src.error not in ("", "Menghubungkan kamera pilihan…"), 8)
    assert "rahasia" not in src.error
    t = time.monotonic()
    src.stop()
    assert time.monotonic() - t < 7 and src.read() is None


def test_error_text_is_masked(monkeypatch):                     # Review Focus 3
    def boom(url):
        raise cv2.error("tidak bisa membuka " + url)
    monkeypatch.setattr(source_module, "open_stream", boom)
    src = VideoSource("ipcam:k1", retry_s=0.2, resolve=lambda s: Stream("http://admin:rahasia@h/x", None))
    src.start()
    try:
        assert wait_for(lambda: "••••" in src.error, 3)
        assert "rahasia" not in src.error
    finally:
        src.stop()


def test_tile_crop_with_uneven_size():                          # Review Focus 4
    frame = np.zeros((768, 1366, 3), np.uint8)
    tiles = [crop_tile(frame, 3, i) for i in range(9)]
    assert sum(t.shape[1] for t in tiles[:3]) == 1366 and sum(t.shape[0] for t in tiles[::3]) == 768


# ---- perbaikan review akhir ----
class _NetCapture:
    """Capture tiruan: beberapa frame bagus, lalu tiap baca gagal setelah menunggu (stream macet)."""

    def __init__(self, good=3, stall_s=1.0):
        self.good, self.stall_s, self.reads = good, stall_s, 0

    def isOpened(self):
        return True

    def read(self):
        self.reads += 1
        if self.reads <= self.good:
            time.sleep(0.02)
            return True, np.zeros((36, 64, 3), np.uint8)
        time.sleep(self.stall_s)
        return False, None

    def release(self):
        pass


def test_stalled_network_stream_reconnects_quickly(monkeypatch):     # Final #5
    opens = []
    monkeypatch.setattr(source_module, "open_stream", lambda url: opens.append(time.monotonic()) or _NetCapture())
    src = VideoSource("http://cam/video", retry_s=0.1)
    src.start()
    try:
        assert wait_for(lambda: len(opens) >= 2, 9)               # lama: 15 baca x 1 dtk = 15 dtk sebelum buka ulang
        assert opens[1] - opens[0] < 8
    finally:
        src.stop()


def test_network_retry_backs_off(monkeypatch):                         # Final #4
    opens = []

    class Closed:
        def isOpened(self):
            return False

        def release(self):
            pass
    monkeypatch.setattr(source_module, "open_stream", lambda url: opens.append(1) or Closed())
    src = VideoSource("http://cam/video", retry_s=0.2)
    src.start()
    time.sleep(3.2)
    src.stop()
    assert 2 <= len(opens) <= 6                                      # tanpa jeda bertambah: sekitar 15 kali


def test_wrong_rtsp_password_stops_retrying_until_changed(monkeypatch):   # Final #4
    probes, opens = [], []
    monkeypatch.setattr(source_module, "_rtsp_probe", lambda url, timeout_s=3.0: probes.append(url) or "sandi")
    monkeypatch.setattr(source_module, "open_stream", lambda url: opens.append(url) or _NetCapture())
    src = VideoSource("ipcam:k1", retry_s=0.1, resolve=lambda s: Stream("rtsp://admin:xyzzy@cam/x", None))
    src.start()
    try:
        assert wait_for(lambda: "Kata sandi" in src.error, 3)
        time.sleep(1.0)
        assert len(probes) == 1 and opens == [] and "xyzzy" not in src.error
        src.reopen()                                                  # guru memperbaiki sandi lalu menyimpan
        assert wait_for(lambda: len(probes) == 2, 3)
    finally:
        src.stop()


def test_unreadable_password_is_not_tried(monkeypatch):                # Final #4
    opens = []
    monkeypatch.setattr(source_module, "open_stream", lambda url: opens.append(url) or _NetCapture())
    src = VideoSource("ipcam:k1", retry_s=0.1, resolve=lambda s: Stream("", None, "Kata sandi perlu diisi ulang."))
    src.start()
    try:
        assert wait_for(lambda: src.error == "Kata sandi perlu diisi ulang.", 3)
        time.sleep(0.5)
        assert opens == []
    finally:
        src.stop()


def test_reopen_bumps_version_and_reconnects(monkeypatch):            # Final #3
    opens = []
    monkeypatch.setattr(source_module, "open_stream", lambda url: opens.append(url) or _NetCapture(good=10 ** 6))
    src = VideoSource("http://cam/video", retry_s=0.1)
    src.start()
    try:
        assert wait_for(lambda: src.read() is not None, 3)
        version = src.version
        src.reopen()
        assert src.version == version + 1
        assert wait_for(lambda: len(opens) == 2, 3)
    finally:
        src.stop()


def test_capture_card_asks_full_hd_but_webcam_stays_640(monkeypatch):  # Final #8
    sizes = {}

    class Cap:
        def __init__(self, index, backend):
            self.index = index

        def isOpened(self):
            return True

        def set(self, prop, value):
            sizes.setdefault(self.index, {})[prop] = value

        def read(self):
            time.sleep(0.02)
            return True, np.zeros((10, 10, 3), np.uint8)

        def release(self):
            pass
    monkeypatch.setattr(source_module.cv2, "VideoCapture", Cap)
    monkeypatch.setattr(source_module, "camera_routes", lambda spec: [(1, 700)] if spec == "camera:dvr" else [(0, 700)])
    card = VideoSource("ipcam:k1", resolve=lambda s: Stream("camera:dvr", (2, 0)))
    assert card._open_source("ipcam:k1")[0] is not None
    webcam = VideoSource("camera:usb")
    assert webcam._open() is not None
    assert sizes[1][cv2.CAP_PROP_FRAME_WIDTH] == 1920 and sizes[1][cv2.CAP_PROP_FRAME_HEIGHT] == 1080
    assert sizes[0][cv2.CAP_PROP_FRAME_WIDTH] == 640
