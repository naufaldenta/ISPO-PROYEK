import json

import pytest

from sahabat.kamera import KameraStore, Stream


def store(tmp_path):
    return KameraStore(tmp_path / "kamera.json")


def test_add_brand_camera_hides_password(tmp_path):
    s = store(tmp_path)
    item = s.save({"nama": "Lorong A", "jenis": "merek", "merek": "hikvision", "host": "192.168.1.64",
                   "port": 554, "channel": 1, "substream": True, "pengguna": "admin", "sandi": "rahasia"})
    assert item["ada_sandi"] and "rahasia" not in json.dumps(item)
    assert "rahasia" not in (tmp_path / "kamera.json").read_text(encoding="utf-8")
    assert item["alamat_tampil"] == "rtsp://admin:••••@192.168.1.64:554/Streaming/Channels/102"
    assert s.resolve("ipcam:" + item["id"]) == Stream(
        "rtsp://admin:rahasia@192.168.1.64:554/Streaming/Channels/102", None)


def test_store_survives_reload(tmp_path):
    item = store(tmp_path).save({"nama": "A", "jenis": "alamat", "alamat": "rtsp://u:p@h/x"})
    assert store(tmp_path).resolve("ipcam:" + item["id"]).url == "rtsp://u:p@h/x"


def test_full_address_login_is_split_out(tmp_path):
    s = store(tmp_path)
    item = s.save({"nama": "Gerbang", "jenis": "alamat", "alamat": "rtsp://u:p%40ss@10.0.0.9/live"})
    assert item["pengguna"] == "u" and item["alamat_tampil"] == "rtsp://u:••••@10.0.0.9/live"
    assert s.resolve("ipcam:" + item["id"]).url == "rtsp://u:p%40ss@10.0.0.9/live"


def test_update_keeps_password_unless_given(tmp_path):
    s = store(tmp_path)
    item = s.save({"nama": "A", "jenis": "alamat", "alamat": "rtsp://u:p@h/x"})
    s.save({"id": item["id"], "nama": "B", "jenis": "alamat", "alamat": "rtsp://h/x", "pengguna": "u"})
    assert s.resolve("ipcam:" + item["id"]).url == "rtsp://u:p@h/x" and s.get_public(item["id"])["nama"] == "B"
    s.save({"id": item["id"], "nama": "B", "jenis": "alamat", "alamat": "rtsp://h/x", "pengguna": "u", "sandi": ""})
    assert s.resolve("ipcam:" + item["id"]).url == "rtsp://u@h/x" and not s.get_public(item["id"])["ada_sandi"]


def test_capture_card_tile(tmp_path):
    s = store(tmp_path)
    item = s.save({"nama": "DVR ubin 2", "jenis": "perangkat", "perangkat": "camera:usb#1", "potong": [2, 1]})
    assert s.resolve("ipcam:" + item["id"]) == Stream("camera:usb#1", (2, 1))


@pytest.mark.parametrize("bad", [
    {"nama": "", "jenis": "alamat", "alamat": "rtsp://h/x"},
    {"nama": "x" * 41, "jenis": "alamat", "alamat": "rtsp://h/x"},
    {"nama": "A", "jenis": "alamat", "alamat": "file:///C:/rahasia.txt"},
    {"nama": "A", "jenis": "merek", "merek": "tidakada", "host": "h"},
    {"nama": "A", "jenis": "merek", "merek": "hikvision", "host": "h h", "channel": 1},
    {"nama": "A", "jenis": "perangkat", "perangkat": "camera:x", "potong": [5, 0]},
    {"nama": "A", "jenis": "perangkat", "perangkat": "camera:x", "potong": [2, 4]},
    {"nama": "A", "jenis": "aneh"},
    {"id": "tidakada", "nama": "A", "jenis": "alamat", "alamat": "rtsp://h/x"},
])
def test_invalid_input_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        store(tmp_path).save(bad)


def test_delete_and_unknown(tmp_path):
    s = store(tmp_path)
    item = s.save({"nama": "A", "jenis": "alamat", "alamat": "rtsp://h/x"})
    assert s.delete(item["id"]) and not s.delete(item["id"])
    assert s.resolve("ipcam:" + item["id"]) is None and s.resolve("ipcam:../x") is None


def test_corrupt_file_and_foreign_password(tmp_path):        # Review Focus 5
    (tmp_path / "kamera.json").write_text("{rusak", encoding="utf-8")
    assert store(tmp_path).list_public() == []
    (tmp_path / "kamera.json").write_text(json.dumps({"kamera": [{
        "id": "k1", "nama": "A", "jenis": "alamat", "alamat": "rtsp://h/x", "pengguna": "u",
        "sandi": "dpapi:QUJD"}]}), encoding="utf-8")
    s = store(tmp_path)
    assert s.get_public("k1")["sandi_rusak"] is True
    stream = s.resolve("ipcam:k1")                              # tidak mencoba login tanpa sandi (akun bisa terkunci)
    assert stream.url == "" and "isi ulang" in stream.galat


def test_catalog_for_device_panel(tmp_path):
    s = store(tmp_path)
    item = s.save({"nama": "Lorong A", "jenis": "alamat", "alamat": "rtsp://h/x"})
    assert s.catalog() == [{"id": "ipcam:" + item["id"], "name": "Lorong A"}]


def test_form_stream_without_saving(tmp_path):
    s = store(tmp_path)
    item = s.save({"nama": "A", "jenis": "alamat", "alamat": "rtsp://u:p@h/x"})
    assert s.form_stream({"id": item["id"], "nama": "A", "jenis": "alamat", "alamat": "rtsp://h/x",
                          "pengguna": "u"}).url == "rtsp://u:p@h/x"
    assert s.form_stream({"nama": "B", "jenis": "merek", "merek": "tapo", "host": "10.0.0.2",
                          "pengguna": "a", "sandi": "b"}).url == "rtsp://a:b@10.0.0.2:554/stream2"
    assert len(s.list_public()) == 1


# ---- perbaikan review akhir ----
def test_raw_slash_password_is_split_out(tmp_path):            # Final #1
    s = store(tmp_path)
    item = s.save({"nama": "A", "jenis": "alamat", "alamat": "rtsp://admin:ab/cd@192.168.1.5/stream"})
    assert "ab/cd" not in (tmp_path / "kamera.json").read_text(encoding="utf-8") and item["ada_sandi"]
    assert item["alamat_tampil"] == "rtsp://admin:••••@192.168.1.5/stream"
    assert s.resolve("ipcam:" + item["id"]).url == "rtsp://admin:ab%2Fcd@192.168.1.5/stream"


def test_query_password_moves_to_encrypted_field(tmp_path):    # Final #1 dan #2
    s = store(tmp_path)
    item = s.save({"nama": "A", "jenis": "alamat", "alamat": "http://cam/videostream.cgi?user=admin&pwd=secret"})
    assert "secret" not in (tmp_path / "kamera.json").read_text(encoding="utf-8") and item["ada_sandi"]
    assert "secret" not in str(item) and item["alamat_edit"] == "http://cam/videostream.cgi?user=admin&pwd={SANDI}"
    url = "http://cam/videostream.cgi?user=admin&pwd=secret"
    assert s.resolve("ipcam:" + item["id"]).url == url
    s.save({"id": item["id"], "nama": "B", "jenis": "alamat", "alamat": item["alamat_edit"]})   # ganti nama saja
    assert s.resolve("ipcam:" + item["id"]).url == url


def test_edit_address_has_no_login(tmp_path):                   # Final #2
    item = store(tmp_path).save({"nama": "A", "jenis": "alamat", "alamat": "rtsp://u:p@h:554/x"})
    assert item["alamat_edit"] == "rtsp://h:554/x"


def test_password_without_user_is_kept(tmp_path):
    s = store(tmp_path)
    item = s.save({"nama": "A", "jenis": "alamat", "alamat": "rtsp://:pass@h/x"})
    assert s.resolve("ipcam:" + item["id"]).url == "rtsp://:pass@h/x"


def test_bad_port_rejected(tmp_path):
    with pytest.raises(ValueError):
        store(tmp_path).save({"nama": "A", "jenis": "alamat", "alamat": "rtsp://h:abc/x"})


def test_form_stream_reuses_password_only_for_same_camera(tmp_path):   # Final #6
    s = store(tmp_path)
    item = s.save({"nama": "A", "jenis": "alamat", "alamat": "rtsp://u:p@h/x"})
    evil = s.form_stream({"id": item["id"], "nama": "A", "jenis": "alamat", "alamat": "rtsp://evil.example/x",
                          "pengguna": "u"})
    assert evil.url == "rtsp://u@evil.example/x"


def test_query_password_with_raw_hash_and_slash(tmp_path):     # Final #1 (ditemukan saat uji browser)
    s = store(tmp_path)
    item = s.save({"nama": "A", "jenis": "alamat", "alamat": "http://192.168.1.77/v.cgi?user=admin&pwd=s/e#cret"})
    assert "cret" not in str(item) and "cret" not in (tmp_path / "kamera.json").read_text(encoding="utf-8")
    assert item["alamat_edit"] == "http://192.168.1.77/v.cgi?user=admin&pwd={SANDI}"
    assert s.resolve("ipcam:" + item["id"]).url == "http://192.168.1.77/v.cgi?user=admin&pwd=s%2Fe%23cret"


# ---- Bagian 2: Mode CCTV ----
def test_new_cameras_are_active_and_flags_survive_edits(tmp_path):
    s = store(tmp_path)
    item = s.save({"nama": "A", "jenis": "alamat", "alamat": "rtsp://h/x"})
    assert item["aktif"] is True and item["prioritas"] is False and item["zona"] is None
    s.set_flags(item["id"], aktif=False, prioritas=True)
    s.set_zone(item["id"], [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]])
    edited = s.save({"id": item["id"], "nama": "B", "jenis": "alamat", "alamat": "rtsp://h/x"})
    assert (edited["aktif"], edited["prioritas"], edited["zona"]) == (False, True, [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]])
    assert store(tmp_path).get_public(item["id"])["prioritas"] is True
    assert [c["id"] for c in s.active()] == []
    s.set_flags(item["id"], aktif=True)
    assert [c["id"] for c in s.active()] == [item["id"]]


def test_video_camera_for_simulation(tmp_path):
    videos = tmp_path / "video"
    videos.mkdir()
    (videos / "jatuh.mp4").write_bytes(b"x")
    s = KameraStore(tmp_path / "k.json", video_dir=videos)
    item = s.save({"nama": "Simulasi", "jenis": "video", "video": "jatuh.mp4"})
    assert s.resolve("ipcam:" + item["id"]) == Stream(str(videos / "jatuh.mp4"), None)
    assert item["video"] == "jatuh.mp4"
    (videos / "jatuh.mp4").unlink()
    assert "tidak ada di folder video" in s.resolve("ipcam:" + item["id"]).galat
    for bad in ("../rahasia.mp4", "a.txt", ""):
        with pytest.raises(ValueError):
            s.save({"nama": "X", "jenis": "video", "video": bad})


def test_flags_unknown_camera(tmp_path):
    with pytest.raises(ValueError):
        store(tmp_path).set_flags("tidakada", aktif=True)
    with pytest.raises(ValueError):
        store(tmp_path).set_zone("tidakada", None)
