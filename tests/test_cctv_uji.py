from sahabat.cctv_uji import PESAN, rtsp_probe, uji_sambungan
from sahabat.kamera import Stream


def test_probe_ok_after_digest_auth(fake_rtsp_server):
    def responder(req):
        if "Authorization: Digest" in req and 'username="admin"' in req:
            return "RTSP/1.0 200 OK\r\nCSeq: 2\r\n\r\n"
        return 'RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\nWWW-Authenticate: Digest realm="cam", nonce="abc"\r\n\r\n'
    url = fake_rtsp_server(responder)
    assert rtsp_probe(url.replace("rtsp://", "rtsp://admin:benar@")) == "ok"


def test_probe_ok_after_basic_auth(fake_rtsp_server):
    def responder(req):
        if "Authorization: Basic YTpi" in req:
            return "RTSP/1.0 200 OK\r\nCSeq: 2\r\n\r\n"
        return 'RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\nWWW-Authenticate: Basic realm="c"\r\n\r\n'
    assert rtsp_probe(fake_rtsp_server(responder).replace("rtsp://", "rtsp://a:b@")) == "ok"


def test_probe_wrong_password_and_wrong_path(fake_rtsp_server):
    def always_401(req):
        return 'RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\nWWW-Authenticate: Basic realm="c"\r\n\r\n'
    assert rtsp_probe(fake_rtsp_server(always_401).replace("rtsp://", "rtsp://a:salah@")) == "sandi"
    assert rtsp_probe(fake_rtsp_server(always_401)) == "sandi"                 # tanpa login sama sekali
    assert rtsp_probe(fake_rtsp_server(lambda r: "RTSP/1.0 404 Not Found\r\nCSeq: 1\r\n\r\n")) == "jalur"
    assert rtsp_probe(fake_rtsp_server(lambda r: "RTSP/1.0 503 Busy\r\nCSeq: 1\r\n\r\n")) == "lain"


def test_probe_unreachable():
    assert rtsp_probe("rtsp://127.0.0.1:1/x", timeout_s=1) == "tak_terjangkau"


def test_connection_test_reports_size_fps_and_preview(mjpeg_server):
    r = uji_sambungan(Stream(mjpeg_server.url, None), timeout_s=5)
    assert r["ok"] and (r["lebar"], r["tinggi"]) == (640, 360) and r["fps"] > 5 and r["cuplikan"]


def test_connection_test_applies_dvr_tile(mjpeg_server):
    r = uji_sambungan(Stream(mjpeg_server.url, (2, 0)), timeout_s=5)
    assert r["ok"] and (r["lebar"], r["tinggi"]) == (320, 180)


def test_connection_test_message_never_contains_password(monkeypatch):
    monkeypatch.setattr("sahabat.cctv_uji.rtsp_probe", lambda url, timeout_s=3.0: "sandi")
    r = uji_sambungan(Stream("rtsp://admin:rahasia@h/x", None), timeout_s=1)
    assert not r["ok"] and r["pesan"] == PESAN["sandi"] and "rahasia" not in str(r)


def test_connection_test_stream_that_never_opens(mjpeg_server):
    class Closed:
        def isOpened(self):
            return False

        def release(self):
            pass
    url = mjpeg_server.url.replace("http://", "http://admin:rahasia@")      # host terjangkau, stream tidak terbuka
    r = uji_sambungan(Stream(url, None), timeout_s=1, opener=lambda url: Closed())
    assert not r["ok"] and r["pesan"] == PESAN["lain"] and "rahasia" not in str(r)


def test_probe_bad_port_does_not_raise():                           # Final #1
    assert rtsp_probe("rtsp://admin:ab@h:cd@127.0.0.1:x/stream", timeout_s=1) in ("lain", "tak_terjangkau")
    assert rtsp_probe("rtsp://h:abc/x", timeout_s=1) == "lain"
