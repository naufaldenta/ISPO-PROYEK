import pytest

from sahabat.merek import MEREK, alamat_merek, pasang_login, pisah_login, samarkan


def test_hikvision_sub_and_main():
    assert alamat_merek("hikvision", "192.168.1.64", 554, 1, True, "admin", "abc") == \
        "rtsp://admin:abc@192.168.1.64:554/Streaming/Channels/102"
    assert alamat_merek("hikvision", "192.168.1.64", 554, 3, False, "admin", "abc").endswith("/Streaming/Channels/301")


def test_dahua_and_imou_use_realmonitor():
    for m in ("dahua", "imou"):
        assert alamat_merek(m, "10.0.0.5", 554, 2, True, "admin", "x") == \
            "rtsp://admin:x@10.0.0.5:554/cam/realmonitor?channel=2&subtype=1"


def test_other_brands():
    assert alamat_merek("uniview", "h", 554, 1, True, "a", "b") == "rtsp://a:b@h:554/unicast/c1/s1/live"
    assert alamat_merek("tapo", "h", 554, 1, True, "a", "b") == "rtsp://a:b@h:554/stream2"
    assert alamat_merek("ezviz", "h", 554, 1, False, "admin", "b") == "rtsp://admin:b@h:554/h264/ch1/main/av_stream"
    assert alamat_merek("xmeye", "h", 554, 4, True, "admin", "b") == \
        "rtsp://h:554/user=admin&password=b&channel=4&stream=1.sdp?real_stream"
    assert alamat_merek("ipwebcam", "192.168.1.20", 8080, 1, True, "", "") == "http://192.168.1.20:8080/video"


def test_brand_table_shape():
    assert set(MEREK) == {"hikvision", "dahua", "uniview", "tapo", "ezviz", "imou", "xmeye", "ipwebcam"}
    for info in MEREK.values():
        assert isinstance(info["nama"], str) and isinstance(info["port"], int) and isinstance(info["channel"], bool)


def test_special_characters_are_encoded():          # Review Focus 1
    url = alamat_merek("hikvision", "h", 554, 1, True, "ad min", "p@ss:w/rd#1%")
    assert url == "rtsp://ad%20min:p%40ss%3Aw%2Frd%231%25@h:554/Streaming/Channels/102"
    assert pisah_login(url) == ("rtsp://h:554/Streaming/Channels/102", "ad min", "p@ss:w/rd#1%")


def test_login_round_trip_and_empty_login():
    assert pasang_login("rtsp://cam.local/live", "u", "p") == "rtsp://u:p@cam.local/live"
    assert pasang_login("rtsp://cam.local/live", "", "") == "rtsp://cam.local/live"
    assert pisah_login("rtsp://cam.local/live") == ("rtsp://cam.local/live", "", "")


def test_masking_anywhere_in_text():
    msg = "gagal membuka rtsp://admin:rahasia@10.0.0.5:554/x dan rtsp://h/user=admin&password=rahasia&channel=1"
    out = samarkan(msg)
    assert "rahasia" not in out and "admin:••••@10.0.0.5" in out


def test_unknown_brand_raises():
    with pytest.raises(ValueError):
        alamat_merek("tidakada", "h", 554, 1, True, "", "")
