"""Pergantian Mode Pameran / Mode CCTV dan API per kamera."""
import json
from types import SimpleNamespace

import numpy as np
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
                           "CPU: sanggup menilai 1 kamera penuh sekaligus")
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


def test_status_names_the_compute_device(mode_env):
    mode, engine, runner = mode_env()
    original = runner.snapshot
    mode.set_cctv(True)
    runner.snapshot = lambda: {**original(), "kapasitas": 3, "perangkat": "GPU (DirectML #1)", "galat_gpu": "",
                               "pesan_perangkat": ""}
    assert "GPU: DirectML #1, sanggup menilai 3 kamera penuh sekaligus" in mode.status()["pesan"]
    runner.snapshot = lambda: {**original(), "kapasitas": 1, "perangkat": "CPU",
                               "galat_gpu": "GPU bermasalah (rusak), Mode CCTV pindah ke CPU.", "pesan_perangkat": ""}
    st = mode.status()["pesan"]
    assert "CPU: sanggup menilai 1 kamera penuh sekaligus" in st and st.endswith("Mode CCTV pindah ke CPU.")
    runner.snapshot = lambda: {**original(), "kapasitas": 1, "perangkat": "CPU", "galat_gpu": "GPU bermasalah (x), Mode CCTV pindah ke CPU.",
                               "pesan_perangkat": "Catatan satu."}
    assert mode.status()["pesan"].endswith("sekaligus. Catatan satu. GPU bermasalah (x), Mode CCTV pindah ke CPU.")
    runner.snapshot = lambda: {**original(), "kapasitas": 2, "perangkat": "GPU (DirectML #0)", "galat_gpu": "",
                               "pesan_perangkat": "GPU dipakai walau belum lolos uji kesetaraan; hasilnya bisa sedikit berbeda dari CPU."}
    assert "belum lolos uji kesetaraan" in mode.status()["pesan"]


class _Pose:
    def __init__(self, ms, clock, nama):
        self.ms, self.clock, self.nama = ms, clock, nama

    def detect(self, cam_id, frame):
        self.clock.t += self.ms / 1000
        return []

    def forget(self, cam_id):
        pass


def test_choose_compute_caches_and_rechooses(monkeypatch):
    from sahabat import perangkat_hitung as ph
    clock = SimpleNamespace(t=0.0)
    monkeypatch.setattr(ph, "_clock", lambda: clock.t)
    made, saved = [], []

    def gpu(device_id):
        if device_id != 1:
            raise RuntimeError("tidak ada")
        made.append(device_id)
        return _Pose(9, clock, "GPU (DirectML #1)")
    cpu_pose = _Pose(60, clock, "CPU (OpenVINO)")
    cfg, cache, frame = Config(hitung="otomatis"), {}, np.zeros((8, 8, 3), np.uint8)
    kw = dict(cpu_factory=lambda: cpu_pose, gpu_factory=gpu, lolos=True, save=lambda ms, i: saved.append((ms, i)))
    first = server.choose_compute(cfg, frame, cache, **kw)
    assert first.jenis == "gpu" and first.device_id == 1 and saved == [(9.0, 1)]
    assert server.choose_compute(cfg, frame, cache, **kw) is first and made == [1]      # model dibuat sekali
    cfg.hitung = "cpu"
    assert server.choose_compute(cfg, frame, cache, **kw).pose is cpu_pose
    cfg.hitung = "otomatis"
    again = server.choose_compute(cfg, frame, cache, **kw)
    cache["gagal"] = True
    assert server.choose_compute(cfg, frame, cache, **kw) is not again                   # GPU bermasalah: pilih ulang
    cfg.hitung = "ngawur"
    assert server.choose_compute(cfg, frame, cache, **kw).jenis == "gpu"                 # nilai rusak = otomatis


def test_runner_without_chosen_pose_releases_cleanly(tmp_path):
    from sahabat.cctv import CctvRunner
    store = _store(tmp_path)
    src = SimpleNamespace(start=lambda: None, stop=lambda: None, read=lambda: None)
    runner = CctvRunner(Config(), store, None, None, engine_factory=lambda cam, s, p: SimpleNamespace(log=None),
                        source_factory=lambda cam: src, connect_timeout_s=0.0, sleep=lambda s: None,
                        start_thread=False)
    with pytest.raises(CctvError):
        runner.start()


def test_compute_choice_is_saved_and_laptop_only(app_cctv, monkeypatch):
    from sahabat import perangkat_hitung as ph
    app, runner = app_cctv
    saved = []
    monkeypatch.setattr(server, "remember_hitung", saved.append)
    monkeypatch.setattr(ph, "nama_gpu", lambda: ["AMD Radeon(TM) Vega 8 Graphics"])
    monkeypatch.setattr(ph, "dml_tersedia", lambda: False)
    monkeypatch.setattr(ph, "baca_kesetaraan", lambda nama, imgsz: None)
    runner.perangkat = "GPU (DirectML #1)"
    with TestClient(app) as c:
        info = c.get("/api/cctv/hitung").json()
        assert info["hitung"] in ("otomatis", "gpu", "cpu") and info["gpu"] == ["AMD Radeon(TM) Vega 8 Graphics"]
        assert info["dml"] is False and info["lolos"] is None and isinstance(info["model_onnx"], bool)
        assert info["aktif"] == "GPU (DirectML #1)"
        r = c.post("/api/cctv/hitung", json={"hitung": "gpu"})
        assert r.json()["ok"] and saved == ["gpu"] and "Berlaku saat Mode CCTV dinyalakan berikutnya" in r.json()["pesan"]
        assert c.get("/api/cctv/hitung").json()["hitung"] == "gpu"
        assert c.post("/api/cctv/hitung", json={"hitung": "tpu"}).status_code == 400
    with TestClient(app, client=("192.168.1.50", 5000)) as p:
        p.cookies.set(COOKIE, "ABCD2345")
        assert p.post("/api/cctv/hitung", json={"hitung": "cpu"}).status_code == 403
    assert saved == ["gpu"]


def test_compute_choice_while_cctv_off_says_it_applies_on_next_start(mode_env, monkeypatch, tmp_path):
    mode, engine, runner = mode_env()
    monkeypatch.setattr(server, "remember_hitung", lambda v: None)
    app = create_app(engine, lan=FakeLan(), kamera=_store(tmp_path), mode=mode)
    with TestClient(app) as c:
        r = c.post("/api/cctv/hitung", json={"hitung": "cpu"}).json()
        assert r["ok"] and engine.cfg.hitung == "cpu" and "Berlaku saat Mode CCTV dinyalakan" in r["pesan"]
        assert c.get("/api/cctv/hitung").json()["aktif"] is None


def test_pameran_position_is_saved_and_swaps_models(mode_env, monkeypatch, tmp_path):     # Bagian 5
    from sahabat.profil_kamera import LABEL, PERINGATAN
    mode, engine, runner = mode_env()
    saved, swaps = [], []
    monkeypatch.setattr(server, "remember_posisi", saved.append)
    engine.ganti_model = lambda v, f: swaps.append((v.name, f.name))
    app = create_app(engine, lan=FakeLan(), kamera=_store(tmp_path), mode=mode)
    with TestClient(app) as c:
        got = c.get("/api/posisi").json()
        assert got["posisi"] == "dada" and got["peringatan"] == ""
        assert got["pilihan"] == [{"value": k, "label": v} for k, v in LABEL.items()]
        r = c.post("/api/posisi", json={"posisi": "tinggi"}).json()
        assert r["ok"] and r["posisi"] == "tinggi" and r["peringatan"] == PERINGATAN
        assert saved == ["tinggi"] and engine.cfg.posisi_kamera == "tinggi"
        assert swaps == [("violence_model.joblib", "fall_model.joblib")]
        assert c.post("/api/posisi", json={"posisi": "atap"}).status_code == 400
    with TestClient(app, client=("192.168.1.50", 5000)) as p:                              # Review Focus: HP
        p.cookies.set(COOKIE, "ABCD2345")
        assert p.post("/api/posisi", json={"posisi": "dada"}).status_code == 403
        assert p.get("/api/posisi").json()["posisi"] == "tinggi"
    assert saved == ["tinggi"]


def test_close_position_sets_time_confirmation_and_back(mode_env, monkeypatch, tmp_path):
    from sahabat.profil_kamera import CONFIRM_DEKAT_S, PERINGATAN_DEKAT
    mode, engine, _ = mode_env()
    calls = []
    monkeypatch.setattr(server, "remember_posisi", lambda value: None)
    engine.ganti_model = lambda v, f: None
    engine.atur_profil = lambda window, dekat=False: calls.append((window, dekat))
    app = create_app(engine, lan=FakeLan(), kamera=_store(tmp_path), mode=mode)
    with TestClient(app) as c:
        r = c.post("/api/posisi", json={"posisi": "dekat"}).json()
        assert r["ok"] and r["posisi"] == "dekat" and r["peringatan"] == PERINGATAN_DEKAT
        assert c.post("/api/posisi", json={"posisi": "dada"}).json()["peringatan"] == ""
    assert calls == [(CONFIRM_DEKAT_S, True), (None, False)]


def test_camera_list_warns_for_high_position_without_high_model(tmp_path):                # Bagian 5
    from sahabat.profil_kamera import PERINGATAN
    store = _store(tmp_path)
    item = store.get_public("k1")
    store.save({"id": "k1", "nama": item["nama"], "jenis": "alamat", "alamat": "rtsp://10.0.0.1/x", "posisi": "tinggi"})
    engine = ModeEngine()
    app = create_app(engine, lan=FakeLan(), kamera=store, mode=ModeControl(engine, FakeRunner, lambda on: None))
    with TestClient(app) as c:
        rows = {r["id"]: r for r in c.get("/api/kamera").json()["kamera"]}
    assert rows["k1"]["posisi"] == "tinggi" and rows["k1"]["peringatan"] == PERINGATAN
    assert rows["k2"]["posisi"] == "dada" and rows["k2"]["peringatan"] == ""


def test_assistant_api_for_pameran_and_cctv_cameras(mode_env, tmp_path):                  # Bagian 5
    from sahabat.asisten_kamera import WAIT, Asisten
    mode, engine, runner = mode_env()
    engine.asisten = Asisten()
    app = create_app(engine, lan=FakeLan(), kamera=_store(tmp_path), mode=mode)
    with TestClient(app) as c:
        assert c.get("/api/asisten").json()["saran"] == [WAIT]
        mode.set_cctv(True)
        runner.asisten = lambda cam_id: {"saran": ["dari " + cam_id]} if cam_id == "k1" else None
        assert c.get("/api/asisten?kamera=k1").json() == {"ok": True, "saran": ["dari k1"]}
        assert c.get("/api/asisten?kamera=k9").status_code == 404
    with TestClient(app, client=("192.168.1.50", 5000)) as p:                              # HP boleh membaca
        p.cookies.set(COOKIE, "ABCD2345")
        assert p.get("/api/asisten?kamera=k1").json()["saran"] == ["dari k1"]


def test_requests_survive_cctv_switching_off_mid_request(tmp_path):                     # minor B2 #11
    engine, runner = ModeEngine(), FakeRunner()
    runner.active = True

    class FlakyMode:
        """Mode CCTV dimatikan tepat sesudah server memeriksanya."""
        def __init__(self):
            self.reads = 0

        @property
        def cctv(self):
            self.reads += 1
            return runner if self.reads == 1 else None

        @property
        def hidup(self):
            return self.cctv is not None

        def status(self):
            return {"hidup": False}
    app = create_app(engine, lan=FakeLan(), kamera=_store(tmp_path), mode=FlakyMode())
    with TestClient(app) as c:
        assert c.get("/video/gambar?kamera=k1").status_code in (200, 404)


def test_compute_choice_is_redone_when_the_equivalence_result_changes(monkeypatch):     # minor B4 #10
    from sahabat import perangkat_hitung as ph
    clock = SimpleNamespace(t=0.0)
    monkeypatch.setattr(ph, "_clock", lambda: clock.t)
    cpu_pose = _Pose(60, clock, "CPU (OpenVINO)")
    gpu = lambda device_id: _Pose(9, clock, f"GPU (DirectML #{device_id})")          # noqa: E731
    cfg, cache, frame = Config(hitung="otomatis"), {}, np.zeros((8, 8, 3), np.uint8)
    kw = dict(cpu_factory=lambda: cpu_pose, gpu_factory=gpu, save=lambda ms, i: None)
    assert server.choose_compute(cfg, frame, cache, lolos=True, **kw).jenis == "gpu"
    assert server.choose_compute(cfg, frame, cache, lolos=False, **kw).jenis == "cpu"


def test_pameran_camera_reopens_only_when_its_stream_changes(tmp_path):                 # minor B5 #9
    engine = ModeEngine()
    engine.source.spec = "ipcam:k1"
    reopened = []
    engine.source.reopen = lambda: reopened.append(1)
    store = _store(tmp_path)
    app = create_app(engine, lan=FakeLan(), kamera=store, mode=ModeControl(engine, FakeRunner, lambda on: None))
    form = {"id": "k1", "nama": "Kamera k1", "jenis": "alamat", "alamat": "rtsp://10.0.0.1/x", "pengguna": ""}
    with TestClient(app) as c:
        assert c.post("/api/kamera", json={**form, "posisi": "tinggi"}).json()["ok"]
        assert reopened == []                                  # hanya posisi: tidak menyambung ulang
        assert c.post("/api/kamera", json={**form, "posisi": "tinggi", "alamat": "rtsp://10.0.0.7/x"}).json()["ok"]
        assert reopened == [1]


def test_compute_info_shows_the_last_gpu_measurement(app_cctv, monkeypatch):           # minor B4 #11c
    from sahabat import perangkat_hitung as ph
    app, runner = app_cctv
    monkeypatch.setattr(ph, "nama_gpu", lambda: [])
    monkeypatch.setattr(ph, "dml_tersedia", lambda: True)
    with TestClient(app) as c:
        info = c.get("/api/cctv/hitung").json()
    assert "terakhir" in info and set(info["terakhir"]) == {"ms", "gpu_id"}


def test_recording_meta_follows_position_changes(mode_env, monkeypatch, tmp_path):     # minor B5 #6
    mode, engine, runner = mode_env()
    monkeypatch.setattr(server, "remember_posisi", lambda v: None)
    updates = []
    engine.recorder = SimpleNamespace(active=True, meta={"jenis": "biasa", "kamera": "0", "posisi": "dada"},
                                      perbarui_meta=lambda **kw: updates.append(kw))
    app = create_app(engine, lan=FakeLan(), kamera=_store(tmp_path), mode=mode)
    with TestClient(app) as c:
        c.post("/api/posisi", json={"posisi": "tinggi"})
    assert updates and updates[-1]["posisi"] == "tinggi" and updates[-1]["jenis"] == "biasa"
