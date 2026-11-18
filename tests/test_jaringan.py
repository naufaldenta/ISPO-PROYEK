import json
import time
import urllib.request

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sahabat.jaringan import CODE_ALPHABET, COOKIE, LanServer, is_local, lan_addresses, new_code, qr_png
from sahabat.server import create_app
from tests.test_server import StubEngine

PHONE = ("192.168.1.50", 50000)


class FakeLan:
    def __init__(self, enabled=True, code="ABCD2345"):
        self.enabled, self.code, self.viewers = enabled, code, 0

    def check(self, code):
        return bool(code) and code.strip().upper() == self.code

    def urls(self):
        return ["http://192.168.1.19:8770/"] if self.enabled else []

    def status(self):
        return {"hidup": self.enabled, "alamat": self.urls(), "kode": self.code, "port": 8770, "galat": "",
                "penonton": self.viewers}

    def set_enabled(self, on):
        self.enabled = on
        return self.status()

    def regenerate(self):
        self.code = "ZZZZ9999"
        return self.code


@pytest.fixture
def remembered(monkeypatch):
    from sahabat import server
    saved = []
    monkeypatch.setattr(server, "remember_lan", lambda on, code: saved.append((on, code)))
    return saved


def phone(app):
    return TestClient(app, client=PHONE)


def test_codes_are_readable_and_random():
    codes = {new_code() for _ in range(50)}
    assert len(codes) == 50 and all(len(c) == 8 and set(c) <= set(CODE_ALPHABET) for c in codes)
    assert not set("01OI") & set(CODE_ALPHABET)


def test_only_loopback_is_local():
    assert is_local("127.0.0.1") and is_local("::1") and is_local("localhost")
    assert not is_local("192.168.1.50") and not is_local(None) and not is_local("bukan-ip")


def test_qr_code_holds_the_address():
    png = qr_png("http://192.168.1.19:8770/?kode=ABCD2345")
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    assert text == "http://192.168.1.19:8770/?kode=ABCD2345"


def test_lan_addresses_are_ipv4_strings():
    for ip in lan_addresses():
        assert ip.count(".") == 3 and not ip.startswith("127.")


def test_phone_is_refused_while_lan_is_off():
    app = create_app(StubEngine(), lan=FakeLan(enabled=False))
    assert phone(app).get("/").status_code == 403
    assert create_app(StubEngine()) and phone(create_app(StubEngine())).get("/").status_code == 403


def test_phone_needs_the_code_and_then_gets_a_cookie():
    lan = FakeLan()
    app = create_app(StubEngine(), lan=lan)
    with phone(app) as c:
        first = c.get("/")
        assert first.status_code == 401 and 'name="kode"' in first.text
        assert c.get("/api/riwayat").status_code == 401
        assert c.get("/", params={"kode": "SALAH123"}).status_code == 403
        ok = c.get("/", params={"kode": "abcd2345"}, follow_redirects=False)
        assert ok.status_code == 303 and ok.headers["location"] == "/" and COOKIE in ok.headers["set-cookie"]
        c.cookies.set(COOKIE, lan.code)
        assert c.get("/").status_code == 200 and "Sudah diperiksa" in c.get("/").text
        assert c.post("/api/ack").json() == {"ok": True}


def test_phone_can_watch_but_not_change_settings():
    lan = FakeLan()
    with phone(create_app(StubEngine(), lan=lan)) as c:
        c.cookies.set(COOKIE, lan.code)
        for method, path in (("post", "/api/zona"), ("post", "/api/perangkat"), ("get", "/api/perangkat"),
                             ("post", "/api/video"), ("post", "/api/cermin"), ("post", "/api/kata"),
                             ("post", "/api/riwayat/klip"), ("get", "/api/jaringan"), ("post", "/api/jaringan"),
                             ("post", "/api/jaringan/kode"), ("post", "/api/tanda")):
            assert getattr(c, method)(path).status_code == 403, path


def test_phone_websocket_is_told_it_is_remote_and_counted():
    lan = FakeLan()
    with phone(create_app(StubEngine(), lan=lan)) as c:
        c.cookies.set(COOKIE, lan.code)
        with c.websocket_connect("/ws") as ws:
            m = ws.receive_json()
            assert m["lokal"] is False and "jaringan" not in m
            assert lan.viewers == 1
    assert lan.viewers == 0


def test_phone_websocket_without_code_is_closed():
    from starlette.websockets import WebSocketDisconnect
    with phone(create_app(StubEngine(), lan=FakeLan())) as c:
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws") as ws:
                ws.receive_json()


def test_new_code_locks_out_old_phones(remembered):
    lan = FakeLan()
    app = create_app(StubEngine(), lan=lan)
    with TestClient(app) as laptop, phone(app) as c:
        c.cookies.set(COOKIE, "ABCD2345")
        assert c.get("/").status_code == 200
        assert laptop.post("/api/jaringan/kode").json()["kode"] == "ZZZZ9999"
        assert c.get("/").status_code == 401
    assert remembered == [(True, "ZZZZ9999")]


def test_laptop_controls_lan_and_sees_viewers(remembered):
    lan = FakeLan(enabled=False)
    app = create_app(StubEngine(), lan=lan)
    with TestClient(app) as laptop:
        assert laptop.get("/api/jaringan").json()["hidup"] is False
        assert laptop.get("/api/jaringan/qr.png").status_code == 404
        r = laptop.post("/api/jaringan", json={"hidup": True}).json()
        assert r["ok"] and r["hidup"] and r["alamat"] == ["http://192.168.1.19:8770/"]
        qr = laptop.get("/api/jaringan/qr.png")
        assert qr.status_code == 200 and qr.headers["content-type"] == "image/png"
        assert laptop.post("/api/jaringan", json={"hidup": "ya"}).status_code == 400
        with laptop.websocket_connect("/ws") as ws:
            m = ws.receive_json()
            assert m["lokal"] is True and m["jaringan"] == {"hidup": True, "penonton": 0}
    assert remembered == [(True, "ABCD2345")]


def test_real_lan_server_starts_serves_and_stops():
    lan = LanServer(0, "ABCD2345", host="127.0.0.1", addresses=lambda: ["127.0.0.1"])
    lan.app = create_app(StubEngine(), lan=lan)
    status = lan.set_enabled(True)
    try:
        assert status["hidup"] and status["port"] and status["alamat"] == [f"http://127.0.0.1:{status['port']}/"]
        body = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{status['port']}/api/ping", timeout=1) as r:
                    body = json.loads(r.read())
                break
            except OSError:
                time.sleep(0.1)
        assert body == {"app": "sahabat"}
    finally:
        lan.set_enabled(False)
    assert lan.status()["hidup"] is False and lan.status()["alamat"] == []
