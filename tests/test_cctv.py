"""CctvRunner: banyak kamera, penjadwal penuh/siaga/lambat, alarm per kamera. Jam, sumber, dan mesin tiruan."""
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from sahabat.cctv import CctvError, CctvRunner
from sahabat.config import Config
from sahabat.engine import BAR_TEXT, Snapshot
from sahabat.kamera import KameraStore

RED = ("jatuh", "kekerasan")


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, s):
        self.now += s


class FakeSource:
    def __init__(self, clock, connected=True):
        self.clock, self.connected, self.moving = clock, connected, False
        self.version, self.error = 0, ""
        self.started = self.stopped = False
        self.reopened = 0
        self._still = np.full((180, 320, 3), 90, np.uint8)
        self._rng = np.random.default_rng(1)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def reopen(self):
        self.reopened += 1

    def _frame(self):
        if self.moving:
            return self._rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
        return self._still

    def packet(self):
        return self.version, ((self.clock.now, self._frame()) if self.connected else None)

    def read(self):
        return (self.clock.now, self._frame()) if self.connected else None


class FakeEngine:
    def __init__(self, clock, cost):
        self.clock, self.cost = clock, cost
        self.steps, self.step_times = [], []
        self.people, self.bar, self.alarm_seq = 0, "aman", 0
        self.zone, self.jpeg_args = None, []
        self.log = None
        self.alarm = SimpleNamespace(active=None)

    def raise_alarm(self, bar):
        self.bar, self.alarm_seq = bar, self.alarm_seq + 1
        self.alarm.active = object()

    def step(self, t, frame, input_version=None):
        self.clock.advance(self.cost)
        if frame is None:
            if self.bar not in RED:
                self.bar = "kamera_terputus"
            return
        self.steps.append(t)
        self.step_times.append(self.clock.now)
        if self.bar == "kamera_terputus":
            self.bar = "aman"

    def snapshot(self):
        recent = [x for x in self.step_times if self.clock.now - x <= 2.0]
        fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 and recent[-1] > recent[0] else 0.0
        return Snapshot(self.bar, BAR_TEXT[self.bar], round(fps, 1), self.people, False, self.alarm_seq, ())

    def acknowledge(self):
        was = self.bar in RED
        if was:
            self.bar, self.alarm.active = "aman", None
        return was

    def set_zone(self, points):
        self.zone = points
        return points

    def latest_jpeg(self, width=None, quality=75):
        self.jpeg_args.append((width, quality))
        return b"jpeg"


class FakeShared:
    def __init__(self, clock, cost=0.05):
        self.clock, self.cost, self.forgotten = clock, cost, []

    def detect(self, cam_id, frame):
        self.clock.advance(self.cost)
        return []

    def forget(self, cam_id):
        self.forgotten.append(cam_id)


def write_store(tmp_path, ids, names=None):
    items = [{"id": i, "nama": (names or {}).get(i, "Kamera " + i), "jenis": "alamat",
              "alamat": f"rtsp://10.0.0.{n + 1}/x", "pengguna": "", "sandi": "", "aktif": True, "prioritas": False,
              "zona": None, "potong": None} for n, i in enumerate(ids)]
    (tmp_path / "kamera.json").write_text(json.dumps({"kamera": items}), encoding="utf-8")
    return KameraStore(tmp_path / "kamera.json")


@pytest.fixture
def fake_clock_runner(tmp_path):
    def make(ids, people=None, names=None, connected=True, step_cost_s=0.005, connect_timeout_s=10.0, start=True):
        clock = FakeClock()
        store = write_store(tmp_path, ids, names)
        engines, sources = {}, {}

        def source_factory(cam):
            src = FakeSource(clock, connected)
            sources[cam["id"]] = src
            return src

        def engine_factory(cam, source, pose):
            eng = FakeEngine(clock, step_cost_s)
            eng.people = (people or {}).get(cam["id"], 0)
            engines[cam["id"]] = eng
            return eng

        runner = CctvRunner(Config(), store, FakeShared(clock), riwayat=None, engine_factory=engine_factory,
                            source_factory=source_factory, clock=clock, wall=lambda: 1_800_000_000.0,
                            connect_timeout_s=connect_timeout_s, sleep=clock.advance, start_thread=False)
        runner.store, runner.sources, runner.sources_created = store, sources, sources

        def run_for(seconds):
            end = clock.now + seconds
            while clock.now < end:
                clock.advance(max(runner._cycle(clock.now), 0.001))
        runner.run_for = run_for
        if start:
            runner.start()
        return runner, engines
    return make


def test_person_camera_runs_full_rate_and_empty_camera_waits(fake_clock_runner):
    runner, engines = fake_clock_runner(["k1", "k2"], people={"k1": 1, "k2": 0})
    runner.run_for(2.0)
    assert 17 <= len(engines["k1"].steps) <= 22 and len(engines["k2"].steps) <= 2
    states = {c["id"]: c["status"] for c in runner.snapshot()["kamera"]}
    assert states == {"k1": "penuh", "k2": "siaga"}


def test_idle_camera_wakes_within_half_second_on_motion(fake_clock_runner):
    runner, engines = fake_clock_runner(["k1"], people={"k1": 0})
    runner.run_for(1.0)
    runner.sources["k1"].moving = True
    engines["k1"].people = 1
    before = len(engines["k1"].steps)
    runner.run_for(0.5)
    assert len(engines["k1"].steps) - before >= 2


def test_full_camera_stays_full_for_hold_time(fake_clock_runner):
    runner, engines = fake_clock_runner(["k1"], people={"k1": 1})
    runner.run_for(1.0)
    engines["k1"].people = 0
    runner.run_for(4.0)
    assert runner.snapshot()["kamera"][0]["status"] == "penuh"
    runner.run_for(1.5)
    assert runner.snapshot()["kamera"][0]["status"] == "siaga"


def test_oldest_alarm_shown_first_and_ack_is_per_camera(fake_clock_runner):      # Review Focus 1
    runner, engines = fake_clock_runner(["k1", "k2"], names={"k1": "Lorong A", "k2": "Kelas 7B"})
    engines["k2"].raise_alarm("jatuh")
    runner.run_for(0.3)
    engines["k1"].raise_alarm("kekerasan")
    runner.run_for(0.3)
    s = runner.snapshot()
    assert s["alarm_list"] == ["k2", "k1"] and s["alarm_kamera"] == "k2" and s["bar"] == "jatuh"
    assert s["text"].startswith("JATUH TERDETEKSI — Kelas 7B — ") and re.search(r"— \d\d\.\d\d\.\d\d$", s["text"])
    assert s["alarm_seq"] == 2 and runner.has_unacked() and runner.has_unacked("k1")
    assert runner.acknowledge() and runner.snapshot()["alarm_list"] == ["k1"]
    assert runner.acknowledge_all() == 1 and runner.snapshot()["bar"] == "aman" and not runner.has_unacked()


def test_all_cameras_lost_stays_in_cctv_and_keeps_alarm(fake_clock_runner):     # Review Focus 2
    runner, engines = fake_clock_runner(["k1", "k2"])
    engines["k1"].raise_alarm("jatuh")
    runner.run_for(0.3)
    for src in runner.sources.values():
        src.connected = False
    runner.run_for(1.0)
    s = runner.snapshot()
    assert s["aktif"] and {c["status"] for c in s["kamera"]} == {"terputus"} and s["bar"] == "jatuh"
    runner.acknowledge_all()
    runner.run_for(0.2)
    assert runner.snapshot()["bar"] == "kamera_terputus"


def test_over_capacity_marks_slow_and_alarm_still_sounds(fake_clock_runner):
    runner, engines = fake_clock_runner(["k1", "k2", "k3"], people={"k1": 1, "k2": 1, "k3": 1}, step_cost_s=0.05)
    engines["k3"].raise_alarm("kekerasan")
    runner.run_for(4.0)
    s = runner.snapshot()
    assert s["lambat"] and any(c["status"] == "lambat" for c in s["kamera"])
    assert s["bar"] == "kekerasan" and s["text"].endswith("(dinilai lambat)")


def test_priority_camera_keeps_full_rate_when_overloaded(fake_clock_runner):
    runner, engines = fake_clock_runner(["k1", "k2", "k3"], people={"k1": 1, "k2": 1, "k3": 1}, step_cost_s=0.05)
    runner.store.set_flags("k3", prioritas=True)
    runner.reload()
    runner.run_for(4.0)
    status = {c["id"]: c["status"] for c in runner.snapshot()["kamera"]}
    assert status["k3"] == "penuh" and "lambat" in (status["k1"], status["k2"])


def test_start_measures_capacity(fake_clock_runner):
    runner, _ = fake_clock_runner(["k1"])
    assert runner.snapshot()["kapasitas"] == 1           # 50 ms pose -> 1/(0,05+0,03) = 12,5 penilaian/dtk
    assert "__ukur__" in runner.shared_pose.forgotten


def test_start_fails_cleanly_without_connected_camera(fake_clock_runner):
    runner, _ = fake_clock_runner(["k1"], connected=False, connect_timeout_s=0.2, start=False)
    with pytest.raises(CctvError, match="Belum ada kamera CCTV yang tersambung"):
        runner.start()
    assert all(src.stopped for src in runner.sources_created.values()) and not runner.snapshot()["aktif"]


def test_start_without_active_camera(fake_clock_runner):
    runner, _ = fake_clock_runner([], start=False)
    with pytest.raises(CctvError, match="Tidak ada kamera CCTV aktif"):
        runner.start()


def test_reload_adds_removes_and_reopens(fake_clock_runner):
    runner, engines = fake_clock_runner(["k1", "k2"])
    runner.store.set_flags("k2", aktif=False)
    runner.reload()
    assert [c["id"] for c in runner.snapshot()["kamera"]] == ["k1"] and runner.sources_created["k2"].stopped
    runner.store.save({"id": "k1", "nama": "Baru", "jenis": "alamat", "alamat": "rtsp://10.0.0.9/y"})
    runner.reload()
    assert runner.snapshot()["kamera"][0]["nama"] == "Baru" and runner.sources["k1"].reopened == 1
    runner.store.set_flags("k2", aktif=True)
    runner.reload()
    assert [c["id"] for c in runner.snapshot()["kamera"]] == ["k1", "k2"]


def test_zone_and_small_jpeg_are_forwarded(fake_clock_runner):
    runner, engines = fake_clock_runner(["k1"])
    runner.set_zone("k1", [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    assert engines["k1"].zone == [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
    assert runner.latest_jpeg("k1", small=True) == b"jpeg" and engines["k1"].jpeg_args[-1] == (480, 60)
    assert runner.latest_jpeg("tidakada") is None


def test_stop_releases_everything(fake_clock_runner):
    runner, _ = fake_clock_runner(["k1", "k2"])
    runner.stop()
    assert all(s.stopped for s in runner.sources.values()) and not runner.snapshot()["aktif"]
    assert {"k1", "k2"} <= set(runner.shared_pose.forgotten)


# ---- perbaikan review akhir ----
def test_default_engine_never_inherits_the_pameran_zone(tmp_path, monkeypatch):       # Final #3
    from dataclasses import replace
    import sahabat.predlog as predlog

    class NoLog:
        def __init__(self, *a, **k):
            pass

        def write(self, *a):
            pass

        def close(self):
            pass
    monkeypatch.setattr(predlog, "PredLog", NoLog)
    store = write_store(tmp_path, ["k1"])
    cfg = replace(Config(), zone=[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])        # zona Mode Pameran
    runner = CctvRunner(cfg, store, FakeShared(FakeClock()), riwayat=None, start_thread=False)
    engine = runner._default_engine(store.active()[0], FakeSource(FakeClock()), None)
    assert engine.zone is None


def test_empty_camera_list_is_not_shown_as_safe(fake_clock_runner):                    # Final #5
    runner, engines = fake_clock_runner(["k1"])
    runner.store.set_flags("k1", aktif=False)
    runner.reload()
    assert runner.snapshot()["bar"] == "kamera_terputus"


def test_password_only_change_reopens_the_camera(fake_clock_runner):                  # Final #4
    runner, engines = fake_clock_runner(["k1"])
    runner.store.save({"id": "k1", "nama": "Kamera k1", "jenis": "alamat", "alamat": "rtsp://10.0.0.1/x",
                       "pengguna": "admin", "sandi": "lama"})
    runner.reload()
    first = runner.sources["k1"].reopened
    runner.store.save({"id": "k1", "nama": "Kamera k1", "jenis": "alamat", "alamat": "rtsp://10.0.0.1/x",
                       "pengguna": "admin", "sandi": "baru"})
    runner.reload()
    assert runner.sources["k1"].reopened == first + 1


def test_starved_camera_is_reported_slow_not_full(fake_clock_runner):                 # Final #1
    """Laptop lambat (110 ms per penilaian) + satu kamera prioritas: kamera lain yang ada orangnya tidak boleh
    tetap berstatus "dinilai penuh" dengan angka FPS lama, dan kamera siaga tetap diperiksa."""
    runner, engines = fake_clock_runner(["k1", "k2", "k3"], people={"k1": 1, "k2": 1, "k3": 0}, step_cost_s=0.11)
    runner.store.set_flags("k1", prioritas=True)
    runner.reload()
    runner.run_for(1.0)
    k3_first = len(engines["k3"].steps)
    runner.run_for(5.5)
    status = {c["id"]: (c["status"], c["fps"]) for c in runner.snapshot()["kamera"]}
    assert status["k2"][0] == "lambat" and status["k2"][1] < 8
    assert k3_first >= 1 and len(engines["k3"].steps) > k3_first          # pengaman 5 dtk tetap jalan
