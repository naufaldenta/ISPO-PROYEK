import os
"""Endpoint inspeksi klip: info, frame tunggal, unduh MP4, tindak lanjut guru, dan kamera lain pada saat itu."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sahabat.jaringan import COOKIE
from sahabat.riwayat import Riwayat
from sahabat.server import create_app
from tests.test_jaringan import FakeLan
from tests.test_riwayat import Clock
from tests.test_server import StubEngine


def jpeg(i, w=64, h=48):
    img = np.full((h, w, 3), (i * 7) % 255, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def overlay(i):
    return {"orang": [{"id": 3, "kp": [[10.0 + i, 20.0]] * 17, "conf": [0.9] * 17, "warna": "red", "label": "#3"}],
            "skor": {"jatuh": i == 12, "kekerasan": 0.0}, "ukuran": [64, 48], "zona": None}


@pytest.fixture
def app_with_clips(tmp_path):
    clock = Clock()
    r = Riwayat(tmp_path, background=False, clock=clock, free_bytes=lambda _: 10 ** 12, before_s=1.0, after_s=3.0)
    ids = {}

    def clip(key, t0, kamera, nama, with_overlay=True, finish=True, waktu=0.0):
        clock.now = 1_800_000_000.0 + waktu
        rec = None
        n = 20 if not finish else 12 + 32                  # selesai: frame sampai 3 dtk sesudah kejadian
        for i in range(n):
            t = t0 + i * 0.1
            r.frame(t, jpeg(i), kamera=kamera, overlay=overlay(i) if with_overlay else None)
            if i == 12:
                rec = r.event(t, "jatuh", "full", [3], kamera=kamera, kamera_nama=nama)
            if not finish and i == 13:
                break
        ids[key] = rec.id

    clip("v2", 0.0, "k1", 'Kelas 7B/Lt "2"')
    clip("v1", 10.0, "k9", "Lama", with_overlay=False, waktu=300.0)
    clip("k2_dekat", 20.0, "k2", "Lorong A", waktu=3.5)      # rentang klip 1 + 3 = 4 dtk
    clip("k3_jauh", 30.0, "k3", "Kantin", waktu=60.0)
    clip("menyimpan", 40.0, "k4", "Tangga", finish=False, waktu=900.0)
    engine = StubEngine()
    engine.riwayat = r
    return create_app(engine, lan=FakeLan()), ids


def test_info_and_frames_for_v2_clip(app_with_clips):
    app, ids = app_with_clips
    with TestClient(app) as c:
        info = c.get(f"/api/riwayat/{ids['v2']}/info").json()
        assert info["ok"] and info["versi"] == 2 and info["jumlah"] == len(info["waktu"]) == len(info["overlay"])
        assert info["tulang"] and info["kejadian"]["id"] == ids["v2"] and info["meta"]["orang"] == [3]
        f = c.get(f"/api/riwayat/{ids['v2']}/frame/0")
        assert f.headers["content-type"] == "image/jpeg" and f.content[:2] == b"\xff\xd8"
        assert f.headers["cache-control"] == "no-store"
        assert c.get(f"/api/riwayat/{ids['v2']}/frame/9999").status_code == 404
        assert c.get(f"/api/riwayat/{ids['v2']}/frame/-1").status_code in (404, 422)


def test_old_clip_still_inspectable(app_with_clips):                               # Review Focus 1
    app, ids = app_with_clips
    with TestClient(app) as c:
        info = c.get(f"/api/riwayat/{ids['v1']}/info").json()
        assert info["versi"] == 1 and info["overlay"] is None and info["jumlah"] > 5
        assert c.get(f"/api/riwayat/{ids['v1']}/frame/0").status_code == 200


def test_saving_and_missing_clips_are_friendly(app_with_clips):                   # Review Focus 2 dan 3
    app, ids = app_with_clips
    with TestClient(app) as c:
        r = c.get(f"/api/riwayat/{ids['menyimpan']}/info")
        assert r.status_code == 409 and r.json()["error"] == "Klip sedang disimpan, coba lagi sebentar lagi."
        r = c.get("/api/riwayat/20990101-000000-1/info")
        assert r.status_code == 404 and r.json()["error"] == "Klip tidak ada atau sudah terhapus."
        assert c.get("/api/riwayat/20990101-000000-1/frame/0").status_code == 404
        assert c.get("/api/riwayat/20990101-000000-1/unduh").status_code == 404


def test_other_cameras_at_the_same_time(app_with_clips):
    app, ids = app_with_clips
    with TestClient(app) as c:
        lain = c.get(f"/api/riwayat/{ids['v2']}/info").json()["lain"]
    assert [x["id"] for x in lain] == [ids["k2_dekat"]] and abs(lain[0]["selisih_s"] - 3.5) < 0.01
    assert lain[0]["kamera_nama"] == "Lorong A"


def test_mp4_download_with_safe_name(app_with_clips, tmp_path):                   # Review Focus 5
    app, ids = app_with_clips
    with TestClient(app) as c:
        r = c.get(f"/api/riwayat/{ids['v2']}/unduh")
        old = c.get(f"/api/riwayat/{ids['v1']}/unduh")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4" and old.status_code == 200
    assert f'filename="SAHABAT-Kelas-7B-Lt--2--{ids["v2"]}.mp4"' in r.headers["content-disposition"]
    path = tmp_path / "unduh.mp4"
    path.write_bytes(r.content)
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    assert cap.isOpened() and ok and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 10
    assert frame.shape[1] == 64 and frame.shape[0] > 48            # pita keterangan di bawah gambar
    cap.release()


def test_marks_api_and_summary(app_with_clips):
    app, ids = app_with_clips
    with TestClient(app) as c:
        r = c.post(f"/api/riwayat/{ids['v2']}/tindak", json={"tanda": "benar", "catatan": "Sudah ditangani."})
        assert r.json()["ok"] and r.json()["kejadian"]["tanda"] == "benar"
        bad = c.post(f"/api/riwayat/{ids['v2']}/tindak", json={"tanda": "pelaku"})
        assert bad.status_code == 400 and bad.json()["error"]
        assert c.post("/api/riwayat/20990101-000000-1/tindak", json={"tanda": "benar"}).status_code == 404
        listing = c.get("/api/riwayat").json()
        assert listing["ringkasan"]["benar"] == 1 and listing["ringkasan"]["jatuh"] == 5


def test_phone_can_inspect_but_not_mark_or_download(app_with_clips):
    app, ids = app_with_clips
    with TestClient(app, client=("192.168.1.50", 5000)) as p:
        p.cookies.set(COOKIE, "ABCD2345")
        assert p.get(f"/api/riwayat/{ids['v2']}/info").status_code == 200
        assert p.get(f"/api/riwayat/{ids['v2']}/frame/0").status_code == 200
        assert p.get(f"/api/riwayat/{ids['v2']}/unduh").status_code == 403
        assert p.post(f"/api/riwayat/{ids['v2']}/tindak", json={"tanda": "benar"}).status_code == 403


# ---- perbaikan review akhir ----
def _frames(n=12):
    return [(i * 0.1, jpeg(i)) for i in range(n)]


def test_mp4_decodes_frames_one_at_a_time(monkeypatch):                             # Final #3
    from sahabat import klip_mp4
    log = []
    real_decode = cv2.imdecode

    def decode(buf, flags):
        log.append("decode")
        return real_decode(buf, flags)

    class Writer:
        def __init__(self, *a, **k):
            pass

        def isOpened(self):
            return True

        def write(self, frame):
            log.append("write")

        def release(self):
            pass
    monkeypatch.setattr(klip_mp4.cv2, "imdecode", decode)
    monkeypatch.setattr(klip_mp4.cv2, "VideoWriter", Writer)
    klip_mp4.render(_frames(12), None, None, "uji")
    first_write = log.index("write")
    assert log[:first_write].count("decode") <= 2                    # tidak membuka seluruh klip lebih dulu


def test_mp4_temp_file_is_removed_when_writing_fails(monkeypatch, tmp_path):        # Final #3
    from sahabat import klip_mp4
    monkeypatch.setattr(klip_mp4.tempfile, "mkstemp", lambda suffix="": (os.open(tmp_path / "x.mp4", os.O_CREAT | os.O_RDWR), str(tmp_path / "x.mp4")))
    released = []

    class Writer:
        def __init__(self, *a, **k):
            pass

        def isOpened(self):
            return True

        def write(self, frame):
            raise RuntimeError("disk penuh")

        def release(self):
            released.append(True)
    monkeypatch.setattr(klip_mp4.cv2, "VideoWriter", Writer)
    with pytest.raises(RuntimeError):
        klip_mp4.render(_frames(5), None, None, "uji")
    assert released and not (tmp_path / "x.mp4").exists()


def test_mp4_failure_is_a_clear_503(app_with_clips, monkeypatch):                   # Final #3
    from sahabat import klip_mp4

    def boom(*a, **k):
        raise MemoryError()
    monkeypatch.setattr(klip_mp4, "render", boom)
    app, ids = app_with_clips
    with TestClient(app) as c:
        r = c.get(f"/api/riwayat/{ids['v2']}/unduh")
    assert r.status_code == 503 and r.json()["error"].startswith("MP4 belum bisa dibuat")
