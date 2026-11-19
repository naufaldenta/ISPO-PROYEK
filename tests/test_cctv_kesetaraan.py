"""Gerbang kesetaraan Mode CCTV di tingkat mesin: jejak pose sungguhan diputar ulang lewat CctvRunner (Engine asli,
model terpasang, penjadwal, zona bawaan) dan lewat Engine Mode Pameran langsung; alarmnya harus sama."""
from pathlib import Path

import numpy as np
import pytest

from latih.putar import PoseTrace
from sahabat.cctv import CctvRunner
from sahabat.config import Config
from sahabat.engine import Engine
from sahabat.server import FALL_MODEL, VIOLENCE_MODEL
from tests.test_cctv import FakeClock, write_store

P3 = Path("D:/sahabat-data/putaran3/le2i_416")
SESSION = Path("D:/sahabat-data/webcam/sesi-20261117-101936/jejak_416.npz")
CLIPS = [P3 / "Coffee_room_01__video_22.npz", P3 / "Coffee_room_02__video_49.npz", P3 / "Coffee_room_01__video_43.npz"]


class Replay:
    """Frame tiruan yang membawa nomor tick (2 piksel) dan cukup berubah tiap tick supaya gerak murah terdeteksi."""

    def __init__(self, trace):
        self.trace = trace
        self.frame = np.zeros((trace.frame_h, trace.frame_w, 3), np.uint8)

    def frame_for(self, i):
        self.frame[0, 0, 0], self.frame[0, 0, 1] = i % 256, i // 256
        self.frame[100:300, 100:400] = 0 if i % 2 else 200
        return self.frame

    def people(self, frame):
        return self.trace.people[int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])]


class PameranPose:
    def __init__(self, replay):
        self.replay = replay

    def detect(self, frame):
        return self.replay.people(frame)


class SharedReplay:
    def __init__(self, replay, clock):
        self.replay, self.clock = replay, clock

    def detect(self, cam_id, frame):
        self.clock.advance(0.001)
        return self.replay.people(frame)

    def forget(self, cam_id):
        pass


class Source:
    def __init__(self, replay, clock):
        self.replay, self.clock, self.version, self.error = replay, clock, 0, ""
        self.t = np.asarray(replay.trace.t)

    def start(self):
        pass

    def stop(self):
        pass

    def reopen(self):
        pass

    def _now(self):
        i = int(np.searchsorted(self.t, self.clock.now, side="right")) - 1
        return max(0, min(i, len(self.t) - 1))

    def packet(self):
        i = self._now()
        return 0, (float(self.t[i]), self.replay.frame_for(i))

    def read(self):
        return self.packet()[1]

    fps = 0.0


class Events:
    wants_frames = False
    keep_clips = False

    def __init__(self):
        self.got = []

    def frame(self, *a, **k):
        pass

    def event(self, t, jenis, mode, ids, sumber="model", **kw):
        self.got.append((round(float(t), 1), jenis))

    def close(self):
        pass


def pameran(trace):
    replay, events = Replay(trace), Events()
    engine = Engine(Config(), None, PameranPose(replay), None, False, VIOLENCE_MODEL, FALL_MODEL, None, events)
    for i, t in enumerate(trace.t):
        engine.step(float(t), replay.frame_for(i))
    return events.got


def cctv(trace, tmp_path, monkeypatch):
    import sahabat.predlog as predlog

    class NoLog:
        def __init__(self, *a, **k):
            pass

        def write(self, *a):
            pass

        def close(self):
            pass
    monkeypatch.setattr(predlog, "PredLog", NoLog)
    clock = FakeClock()
    clock.now = float(trace.t[0])
    replay, events = Replay(trace), Events()
    store = write_store(tmp_path, ["k1"])
    runner = CctvRunner(Config(), store, SharedReplay(replay, clock), events,
                        source_factory=lambda cam: Source(replay, clock), clock=clock, sleep=clock.advance,
                        start_thread=False)
    runner.start()
    end = float(trace.t[-1]) + 0.05
    while clock.now < end:
        clock.advance(max(runner._cycle(clock.now), 0.001))
    return events.got


def _same(a, b, tol=0.3):
    return len(a) == len(b) and all(x[1] == y[1] and abs(x[0] - y[0]) <= tol for x, y in zip(a, b))


@pytest.mark.skipif(not all(p.exists() for p in CLIPS), reason="butuh jejak pose Le2i")
@pytest.mark.parametrize("clip", CLIPS, ids=lambda p: p.stem)
def test_fall_alarms_match_pameran(clip, tmp_path, monkeypatch):
    trace = PoseTrace.load(clip)
    expected = pameran(trace)
    got = cctv(trace, tmp_path, monkeypatch)
    assert any(j == "jatuh" for _, j in expected)                     # klip ini memang beralarm di Mode Pameran
    assert _same(expected, got), (expected, got)


@pytest.mark.skipif(not SESSION.exists(), reason="butuh jejak sesi 17 November")
def test_violence_alarms_match_pameran_on_session_recording(tmp_path, monkeypatch):
    full = PoseTrace.load(SESSION)
    t = np.asarray(full.t)
    pick = np.where((t >= 2220.0) & (t <= 2330.0))[0]                 # ulangan kekerasan K1 pertama
    trace = PoseTrace(t[pick], full.frame_h, full.frame_w, [full.people[i] for i in pick])
    expected = pameran(trace)
    got = cctv(trace, tmp_path, monkeypatch)
    assert any(j == "kekerasan" for _, j in expected)
    assert _same(expected, got), (expected, got)
