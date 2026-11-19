"""Pergantian Mode Pameran / Mode CCTV dan API per kamera."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sahabat import server
from sahabat.cctv import CctvError, ModeControl
from sahabat.config import Config
from sahabat.jaringan import COOKIE
from sahabat.kamera import KameraStore
from sahabat.server import create_app
from tests.test_jaringan import FakeLan
from tests.test_server import StubEngine


class ModeSource:
    def __init__(self):
        self.stopped = self.started_again = False
        self.spec, self.version, self.flip, self.error = None, 0, False, ""

    def read(self):
        return None

    def stop(self):
        self.stopped = True

    def start(self):
        self.started_again = True


class ModeEngine(StubEngine):
    def __init__(self):
        super().__init__()
        self.paused = False
        self.source = ModeSource()
        self.alarm = SimpleNamespace(active=None)
        self.alarm_active = False
        self.cfg = Config()
        self.zone = None

    def set_zone(self, points):
        self.zone = points
        return points

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    @property
    def alarm_active(self):
        return self.alarm.active is not None

    @alarm_active.setter
    def alarm_active(self, on):
        self.alarm.active = object() if on else None


class FakeRunner:
    def __init__(self, start_error=None):
        self.start_error, self.active = start_error, False
        self.started = self.stopped = self.reloaded = self.acked_all = False
        self.acked, self.zones, self.small_requests, self.alarms = [], [], [], []

    def start(self):
        if self.start_error:
            raise CctvError(self.start_error)
        self.started = self.active = True
        return self.snapshot()

    def stop(self):
        self.stopped, self.active = True, False

    def reload(self):
        self.reloaded = True

    def fake_alarms(self, ids):
        self.alarms = list(ids)

    def snapshot(self):
        cams = [{"id": i, "nama": "Kamera " + i, "bar": "jatuh" if i in self.alarms else "aman", "text": "",
                 "fps": 10.0, "people": 0, "status": "penuh" if i == "k1" else "siaga", "prioritas": False,
                 "zona": None, "galat": ""} for i in ("k1", "k2")]
        return {"aktif": self.active, "bar": "jatuh" if self.alarms else "aman",
                "text": "JATUH TERDETEKSI — Kamera k2 — 10.42.13" if self.alarms else "AMAN", "fps": 10.0,
                "people": 1, "alarm_seq": len(self.alarms), "alarm_kamera": self.alarms[0] if self.alarms else None,
                "alarm_list": list(self.alarms), "kapasitas": 1, "lambat": False, "kamera": cams}

    def has_unacked(self, cam_id=None):
        return bool(self.alarms) if cam_id is None else cam_id in self.alarms

    def acknowledge(self, cam_id=None):
        self.acked.append(cam_id)
        return True

    def acknowledge_all(self):
        self.acked_all = True
        return len(self.alarms)

    def set_zone(self, cam_id, points):
        self.zones.append((cam_id, points))
        return points

    def latest_jpeg(self, cam_id, small=False):
        self.small_requests.append((cam_id, small))
        return b"\xff\xd8JPEG"


@pytest.fixture
def mode_env():
    def make(start_error=None):
        engine = ModeEngine()
        runner = FakeRunner(start_error)
        remembered = []
        mode = ModeControl(engine, lambda: runner, remember=remembered.append)
        mode.remembered = remembered
        return mode, engine, runner
    return make


def _store(tmp_path):
    items = [{"id": i, "nama": "Kamera " + i, "jenis": "alamat", "alamat": f"rtsp://10.0.0.{n}/x", "pengguna": "",
              "sandi": "", "aktif": True, "prioritas": False, "zona": None, "potong": None}
             for n, i in enumerate(("k1", "k2"), 1)]
    (tmp_path / "kamera.json").write_text(json.dumps({"kamera": items}), encoding="utf-8")
    return KameraStore(tmp_path / "kamera.json")


@pytest.fixture
def app_cctv(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "remember_zone", lambda zone: None)
    engine, runner = ModeEngine(), FakeRunner()
    mode = ModeControl(engine, lambda: runner, remember=lambda on: None)
    mode.set_cctv(True)
    app = create_app(engine, lan=FakeLan(), kamera=_store(tmp_path), mode=mode)
    return app, runner


def test_switch_to_cctv_and_back_restores_webcam(mode_env):                       # Review Focus 4
    mode, engine, runner = mode_env()
    st = mode.set_cctv(True)
    assert st["hidup"] and engine.paused and engine.source.stopped and runner.started
    assert st["pesan"] == ("Mode CCTV aktif: 2 kamera, 1 dinilai penuh. "
                           "Komputer ini sanggup menilai 1 kamera penuh sekaligus")
    mode.set_cctv(False)
    assert not engine.paused and engine.source.started_again and runner.stopped
    assert mode.remembered == [True, False] and not mode.status()["hidup"]


def test_switch_refused_while_alarm_unacked(mode_env):
    mode, engine, runner = mode_env()
    engine.alarm_active = True
    with pytest.raises(ValueError, match="Periksa alarm"):
        mode.set_cctv(True)
    engine.alarm_active = False
    mode.set_cctv(True)
    runner.fake_alarms(["k1"])
    with pytest.raises(ValueError, match="Periksa alarm"):
        mode.set_cctv(False)


def test_failed_start_falls_back_to_pameran(mode_env):
    mode, engine, runner = mode_env(start_error="Belum ada kamera CCTV yang tersambung. Periksa panel Kamera.")
    with pytest.raises(ValueError, match="Belum ada kamera"):
        mode.set_cctv(True)
    assert not engine.paused and engine.source.started_again and mode.cctv is None and mode.remembered == []


def test_ws_and_ack_route_to_cameras(app_cctv):
    app, runner = app_cctv
    runner.fake_alarms(["k2", "k1"])
    with TestClient(app) as c:
        with c.websocket_connect("/ws") as ws:
            m = ws.receive_json()
        assert m["cctv"]["aktif"] and m["cctv"]["alarm_kamera"] == "k2" and m["bar"] == "jatuh"
        assert m["text"].startswith("JATUH TERDETEKSI — Kamera k2") and m["alarm_seq"] == 2
        assert c.post("/api/ack", json={"kamera": "semua"}).json()["ok"] and runner.acked_all
        assert c.post("/api/ack", json={"kamera": "k1"}).json()["ok"] and runner.acked[-1] == "k1"
        assert c.post("/api/ack").json()["ok"] and runner.acked[-1] is None
        assert c.get("/api/cctv").json()["hidup"] is True


def test_pameran_state_without_cctv(tmp_path):
    engine = ModeEngine()
    mode = ModeControl(engine, lambda: FakeRunner(), remember=lambda on: None)
    with TestClient(create_app(engine, kamera=_store(tmp_path), mode=mode)) as c:
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json()["cctv"] == {"aktif": False}
        assert c.post("/api/ack").json()["ok"] and engine.ack_calls == 1


def test_switch_endpoint_reports_errors(tmp_path):
    engine = ModeEngine()
    mode = ModeControl(engine, lambda: FakeRunner("Tidak ada kamera CCTV aktif."), remember=lambda on: None)
    with TestClient(create_app(engine, kamera=_store(tmp_path), mode=mode)) as c:
        r = c.post("/api/cctv", json={"hidup": True})
        assert r.status_code == 400 and r.json()["error"] == "Tidak ada kamera CCTV aktif."
        assert c.post("/api/cctv", json={"hidup": "ya"}).status_code == 400


def test_phone_sees_small_slow_tiles_but_cannot_switch(app_cctv):                 # Review Focus 5
    app, runner = app_cctv
    with TestClient(app, client=("192.168.1.50", 5000)) as p:
        p.cookies.set(COOKIE, "ABCD2345")
        assert p.post("/api/cctv", json={"hidup": False}).status_code == 403
        assert p.post("/api/kamera/k1/tanda", json={"aktif": False}).status_code == 403
        assert p.post("/api/zona", json={"titik": None, "kamera": "k1"}).status_code == 403
        assert p.post("/api/ack", json={"kamera": "k1"}).status_code == 200
        assert p.get("/api/cctv").status_code == 200
    assert server._video_gap(local=False, small=True) == 0.2 and server._video_gap(local=True, small=True) == 0.2
    assert server._video_gap(local=False, small=False) == 0.1 and server._video_gap(local=True, small=False) == 0.05


def test_tile_video_comes_from_the_camera(app_cctv):
    app, runner = app_cctv
    with TestClient(app) as c:
        with c.stream("GET", "/video?kamera=k1&kecil=1") as r:
            first = next(r.iter_bytes())
        assert b"JPEG" in first and runner.small_requests[0] == ("k1", True)


def test_zone_per_camera_is_saved(app_cctv, tmp_path):
    app, runner = app_cctv
    pts = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
    with TestClient(app) as c:
        r = c.post("/api/zona", json={"titik": pts, "kamera": "k2"})
        assert r.json()["ok"] and runner.zones[-1][0] == "k2"
    assert KameraStore(tmp_path / "kamera.json").get_public("k2")["zona"] == pts


def test_camera_with_unacked_alarm_cannot_be_removed(app_cctv):                    # Review Focus 3
    app, runner = app_cctv
    runner.fake_alarms(["k1"])
    with TestClient(app) as c:
        assert c.delete("/api/kamera/k1").status_code == 409
        assert c.post("/api/kamera/k1/tanda", json={"aktif": False}).status_code == 409
        r = c.post("/api/kamera/k2/tanda", json={"aktif": False, "prioritas": True})
        assert r.json()["ok"] and r.json()["kamera"]["prioritas"] is True and runner.reloaded
        runner.reloaded = False
        assert c.post("/api/kamera", json={"nama": "C", "jenis": "alamat", "alamat": "rtsp://h/z"}).json()["ok"]
        assert runner.reloaded


def test_message_when_computer_cannot_do_one_full_camera(mode_env):
    mode, engine, runner = mode_env()
    original = runner.snapshot
    runner.snapshot = lambda: {**original(), "kapasitas": 0}
    st = mode.set_cctv(True)
    assert st["pesan"].endswith("Komputer ini belum sanggup menilai satu kamera penuh 10 kali per detik")


# ---- perbaikan review akhir ----
def test_any_start_error_restores_pameran(mode_env):                                   # Final #6
    mode, engine, runner = mode_env()

    def broken_start():
        raise RuntimeError("OpenVINO gagal")
    runner.start = broken_start
    with pytest.raises(ValueError, match="Mode CCTV gagal dinyalakan"):
        mode.set_cctv(True)
    assert not engine.paused and engine.source.started_again and runner.stopped and mode.cctv is None
    engine2 = ModeEngine()

    def no_model():
        raise FileNotFoundError("model belum ada")
    mode2 = ModeControl(engine2, no_model, remember=lambda on: None)
    with pytest.raises(ValueError, match="Mode CCTV gagal dinyalakan"):
        mode2.set_cctv(True)
    assert not engine2.paused and engine2.source.started_again


def test_last_active_camera_cannot_be_removed_while_cctv_runs(app_cctv, tmp_path):     # Final #5
    app, runner = app_cctv
    with TestClient(app) as c:
        assert c.post("/api/kamera/k2/tanda", json={"aktif": False}).json()["ok"]
        assert c.post("/api/kamera/k1/tanda", json={"aktif": False}).status_code == 409
        assert c.delete("/api/kamera/k1").status_code == 409
        assert c.post("/api/kamera/k1/tanda", json={"prioritas": True}).json()["ok"]


def test_tile_snapshot_is_a_single_short_response(app_cctv):                            # Final #2
    """Ubin grid memakai satu JPEG per permintaan, bukan stream: 16 ubin tidak menghabiskan 6 sambungan browser."""
    app, runner = app_cctv
    with TestClient(app) as c:
        r = c.get("/video/gambar?kamera=k2&kecil=1")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content.startswith(b"\xff\xd8")
        assert r.headers["cache-control"] == "no-store" and runner.small_requests[-1] == ("k2", True)
        assert c.get("/video/gambar?kamera=tidakada&kecil=1").status_code in (200, 404)
