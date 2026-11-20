from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sahabat import server
from sahabat.config import Config, load
from sahabat.jaringan import COOKIE
from sahabat.kamera import KameraStore
from sahabat.server import create_app
from sahabat.source import VideoSource
from tests.test_jaringan import FakeLan
from tests.test_server import StubEngine

BRAND = {"nama": "A", "jenis": "merek", "merek": "hikvision", "host": "10.0.0.5", "channel": 1,
         "substream": True, "pengguna": "admin", "sandi": "rahasia"}


def test_kamera_api_never_returns_password(tmp_path):
    app = create_app(StubEngine(), kamera=KameraStore(tmp_path / "k.json"))
    with TestClient(app) as c:
        r = c.post("/api/kamera", json=BRAND)
        assert r.status_code == 200 and r.json()["ok"] and "rahasia" not in r.text
        listing = c.get("/api/kamera")
        assert "rahasia" not in listing.text and len(listing.json()["kamera"]) == 1
        assert {m["id"] for m in listing.json()["merek"]} >= {"hikvision", "dahua", "xmeye"}
        bad = c.post("/api/kamera", json={"nama": "", "jenis": "alamat", "alamat": "rtsp://h/x"})
        assert bad.status_code == 400 and bad.json()["ok"] is False and bad.json()["error"]
        assert c.post("/api/kamera", content=b"bukan json",
                      headers={"Content-Type": "application/json"}).status_code == 400


def test_kamera_api_is_laptop_only(tmp_path):
    app = create_app(StubEngine(), lan=FakeLan(), kamera=KameraStore(tmp_path / "k.json"))
    with TestClient(app, client=("192.168.1.50", 5000)) as p:
        p.cookies.set(COOKIE, "ABCD2345")
        for method, path in (("get", "/api/kamera"), ("post", "/api/kamera"), ("post", "/api/kamera/uji"),
                             ("post", "/api/kamera/cari"), ("post", "/api/kamera/onvif"),
                             ("delete", "/api/kamera/k1")):
            assert getattr(p, method)(path).status_code == 403


def test_test_endpoint_uses_stored_password_without_echoing(tmp_path, monkeypatch):
    seen = []

    def fake_test(stream, **kw):
        seen.append(stream.url)
        return {"ok": True, "pesan": "Tersambung", "lebar": 640, "tinggi": 360, "fps": 15.0, "cuplikan": ""}
    monkeypatch.setattr(server, "uji_sambungan", fake_test)
    store = KameraStore(tmp_path / "k.json")
    item = store.save({"nama": "A", "jenis": "alamat", "alamat": "rtsp://u:rahasia@h/x"})
    with TestClient(create_app(StubEngine(), kamera=store)) as c:
        r = c.post("/api/kamera/uji", json={"id": item["id"], "nama": "A", "jenis": "alamat",
                                             "alamat": "rtsp://h/x", "pengguna": "u"})
        assert r.json()["ok"] and "rahasia" not in r.text
        assert c.post("/api/kamera/uji", json={"nama": "A", "jenis": "aneh"}).status_code == 400
    assert seen == ["rtsp://u:rahasia@h/x"]


def test_search_and_onvif_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "onvif_discover", lambda timeout_s=3.0: [{"host": "10.0.0.5", "xaddr": "x",
                                                                         "nama": "Lorong", "merek": "DS"}])
    calls = []
    monkeypatch.setattr(server, "onvif_stream_uri", lambda xaddr, u, p, substream=True, timeout_s=5.0, channel=1:
                        calls.append((xaddr, u, p, substream)) or "rtsp://10.0.0.5:554/sub")
    with TestClient(create_app(StubEngine(), kamera=KameraStore(tmp_path / "k.json"))) as c:
        assert c.post("/api/kamera/cari", json={}).json() == {"ok": True, "ditemukan": [
            {"host": "10.0.0.5", "xaddr": "x", "nama": "Lorong", "merek": "DS"}]}
        r = c.post("/api/kamera/onvif", json={"xaddr": "http://10.0.0.5/onvif/device_service", "pengguna": "admin",
                                               "sandi": "rahasia", "substream": True})
        assert r.json() == {"ok": True, "alamat": "rtsp://10.0.0.5:554/sub"}
        assert c.post("/api/kamera/onvif", json={"xaddr": "file:///x"}).status_code == 400
    assert calls == [("http://10.0.0.5/onvif/device_service", "admin", "rahasia", True)]


def test_onvif_error_is_reported(tmp_path, monkeypatch):
    from sahabat.onvif import MSG_PASSWORD, OnvifError

    def refuse(*a, **k):
        raise OnvifError(MSG_PASSWORD)
    monkeypatch.setattr(server, "onvif_stream_uri", refuse)
    with TestClient(create_app(StubEngine(), kamera=KameraStore(tmp_path / "k.json"))) as c:
        r = c.post("/api/kamera/onvif", json={"xaddr": "http://10.0.0.5/onvif", "pengguna": "a", "sandi": "b"})
    assert r.status_code == 400 and r.json() == {"ok": False, "error": MSG_PASSWORD}


def test_delete_in_use_camera_is_refused(tmp_path):
    store = KameraStore(tmp_path / "k.json")
    item = store.save({"nama": "A", "jenis": "alamat", "alamat": "rtsp://h/x"})
    other = store.save({"nama": "B", "jenis": "alamat", "alamat": "rtsp://h/y"})
    engine = StubEngine()
    engine.source = SimpleNamespace(spec="ipcam:" + item["id"])
    with TestClient(create_app(engine, inputs=None, kamera=store)) as c:
        assert c.delete("/api/kamera/" + item["id"]).status_code == 409
        assert c.delete("/api/kamera/" + other["id"]).json() == {"ok": True}
        assert c.delete("/api/kamera/" + other["id"]).status_code == 404


@pytest.fixture
def cctv_selection(monkeypatch, tmp_path):
    from sahabat import input_control as ic
    monkeypatch.setattr(ic, "list_cameras", lambda: [{"id": "camera:internal", "name": "Internal"}])
    monkeypatch.setattr(ic, "list_microphones", lambda: [])
    store = KameraStore(tmp_path / "k.json")
    item = store.save({"nama": "Lorong A", "jenis": "alamat", "alamat": "rtsp://u:rahasia@h/x"})
    engine = StubEngine()
    engine.cfg, engine.source = Config(), VideoSource(None, resolve=store.resolve)
    engine.select_camera = engine.source.select
    control = ic.InputDevices(engine, None, tmp_path / "settings.json", tmp_path / "video", kamera=store)
    return control, engine, item


def test_device_panel_lists_and_selects_cctv(cctv_selection):
    control, engine, item = cctv_selection
    spec = "ipcam:" + item["id"]
    assert control.scan()["ipcams"] == [{"id": spec, "name": "Lorong A"}]
    status = control.apply({"camera": spec})
    assert engine.source.spec == spec and status["camera_name"] == "Lorong A"
    saved = control.settings_path.read_text(encoding="utf-8")
    assert load(control.settings_path).camera == spec and "rahasia" not in saved
    with pytest.raises(ValueError, match="Tekan Cari ulang"):
        control.apply({"camera": "ipcam:tidakada"})


# ---- perbaikan review akhir ----
class _ReopenSource:
    def __init__(self, spec):
        self.spec, self.reopened = spec, 0

    def reopen(self):
        self.reopened += 1


def test_saving_active_camera_reconnects(tmp_path):                  # Final #3
    store = KameraStore(tmp_path / "k.json")
    item = store.save({"nama": "A", "jenis": "alamat", "alamat": "rtsp://h/x"})
    other = store.save({"nama": "B", "jenis": "alamat", "alamat": "rtsp://h/y"})
    engine = StubEngine()
    engine.source = _ReopenSource("ipcam:" + item["id"])
    with TestClient(create_app(engine, inputs=None, kamera=store)) as c:
        assert c.post("/api/kamera", json={"id": item["id"], "nama": "A", "jenis": "alamat",
                                            "alamat": "rtsp://h/x2"}).status_code == 200
        assert engine.source.reopened == 1
        c.post("/api/kamera", json={"id": other["id"], "nama": "B2", "jenis": "alamat", "alamat": "rtsp://h/y"})
        assert engine.source.reopened == 1                           # kamera lain tidak mengganggu


def test_foreign_host_header_is_refused(tmp_path):                    # Final #6 (DNS rebinding)
    app = create_app(StubEngine(), kamera=KameraStore(tmp_path / "k.json"))
    with TestClient(app) as c:
        assert c.get("/api/kamera", headers={"Host": "evil.example:8765"}).status_code == 403
        assert c.get("/api/kamera", headers={"Host": "127.0.0.1:8765"}).status_code == 200
        assert c.get("/api/kamera", headers={"Host": "localhost:8765"}).status_code == 200


def test_camera_posts_require_json(tmp_path):                         # Final #6 (POST lintas situs)
    app = create_app(StubEngine(), kamera=KameraStore(tmp_path / "k.json"))
    with TestClient(app) as c:
        body = '{"nama": "A", "jenis": "alamat", "alamat": "rtsp://h/x"}'
        r = c.post("/api/kamera", content=body, headers={"Content-Type": "text/plain"})
        assert r.status_code == 415
        assert c.post("/api/kamera/cari", headers={"Content-Type": "text/plain"}).status_code == 415
        assert c.post("/api/kamera", content=body, headers={"Content-Type": "application/json"}).status_code == 200
