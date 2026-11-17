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
