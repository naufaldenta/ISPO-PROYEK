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


# ---- temuan minor (Bagian 1) ----
def test_digest_with_comma_in_realm_opaque_and_algorithm():                          # minor B1 #11
    import hashlib
    import re as _re
    from sahabat.cctv_uji import _auth_header
    challenge = 'Digest realm="IP Camera, Lobi", nonce="n0nc3", opaque="xyz", algorithm=MD5'
    header = _auth_header([challenge], "admin", "rahasia", "rtsp://10.0.0.1/s")
    fields = dict(_re.findall(r'(\w+)=(?:"([^"]*)"|([^,\s]*))', header[7:]) and
                  [(k, a or b) for k, a, b in _re.findall(r'(\w+)=(?:"([^"]*)"|([^,\s]*))', header[7:])])
    md5 = lambda s: hashlib.md5(s.encode()).hexdigest()                               # noqa: E731
    expected = md5(f"{md5('admin:IP Camera, Lobi:rahasia')}:n0nc3:{md5('DESCRIBE:rtsp://10.0.0.1/s')}")
    assert fields["realm"] == "IP Camera, Lobi" and fields["opaque"] == "xyz" and fields["algorithm"] == "MD5"
    assert fields["response"] == expected


def test_opener_crash_is_a_friendly_failure(monkeypatch):                           # minor B1 #10
    import cv2
    from sahabat import cctv_uji
    from sahabat.cctv_uji import PESAN, uji_sambungan
    from sahabat.kamera import Stream
    monkeypatch.setattr(cctv_uji, "_reachable", lambda url, t: True)       # sampai ke pembuka OpenCV

    def boom(url):
        raise cv2.error("backend tidak tersedia")
    r = uji_sambungan(Stream("http://127.0.0.1:9/video", None), timeout_s=1.0, opener=boom)
    assert not r["ok"] and r["pesan"] in PESAN.values()


def test_unreachable_network_codes_mean_unreachable(monkeypatch):                    # minor B1 #10
    import socket as _socket
    from sahabat import cctv_uji
    for code in (10065, 10051):
        monkeypatch.setattr(_socket, "create_connection",
                            lambda *a, _c=code, **k: (_ for _ in ()).throw(OSError(_c, "tidak terjangkau")))
        assert cctv_uji.rtsp_probe("rtsp://10.9.9.9/x", timeout_s=0.1) == "tak_terjangkau"


def test_capture_card_in_use_has_its_own_message(monkeypatch):                       # minor B1 #13
    from sahabat import cctv_uji
    from sahabat.kamera import Stream

    class Closed:
        def isOpened(self):
            return False

        def release(self):
            pass
    monkeypatch.setattr("sahabat.source.VideoSource._open", lambda self: Closed())
    r = cctv_uji.uji_sambungan(Stream("camera:1", None), timeout_s=1.0)
    assert not r["ok"] and r["pesan"] == cctv_uji.PESAN["dipakai"] and "substream" not in r["pesan"]


def test_digest_prefers_a_challenge_we_can_answer():                                  # review M-6
    from sahabat.cctv_uji import _auth_header
    header = _auth_header(['Digest realm="r", nonce="n1", algorithm=SHA-256',
                           'Digest realm="r", nonce="n2", algorithm=MD5'], "admin", "x", "rtsp://h/s")
    assert "SHA-256" not in header and 'nonce="n2"' in header
    only_sha = _auth_header(['Digest realm="r", nonce="n1", algorithm=SHA-256', 'Basic realm="r"'], "admin", "x",
                            "rtsp://h/s")
    assert only_sha.startswith("Basic ")
