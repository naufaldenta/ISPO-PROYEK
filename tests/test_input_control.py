from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sahabat.config import Config, load
from sahabat.server import create_app
from sahabat.source import VideoSource
from tests.test_server import StubEngine


@pytest.fixture
def selection(monkeypatch, tmp_path):
    from sahabat import input_control as ic
    monkeypatch.setattr(ic, "list_cameras", lambda: [{"id": "camera:internal", "name": "Internal"},
                                                   {"id": "camera:usb", "name": "External"}])
    monkeypatch.setattr(ic, "list_microphones", lambda: [{"id": "mic:usb", "name": "USB mic"}])
    engine = StubEngine()
    engine.cfg, engine.source = Config(), VideoSource(None)
    engine.select_camera = engine.source.select
    class Listener:
        microphone, mic_name = None, ""
        def set_microphone(self, value): self.microphone = value
        def snapshot(self): return SimpleNamespace(status="mati", label="Kata kasar: MATI", notice="", notice_seq=0)
    listener = Listener()
    control = ic.InputDevices(engine, listener, tmp_path / "settings.json")
    return control, engine, listener


def test_list_and_apply_devices_persist_without_touching_detectors(selection):
    control, engine, listener = selection
    with TestClient(create_app(engine, listener, inputs=control)) as client:
        catalog = client.get("/api/perangkat").json()
        assert [c["name"] for c in catalog["cameras"]] == ["Internal", "External"]
        result = client.post("/api/perangkat", json={"camera": "camera:usb", "microphone": "mic:usb"})
        assert result.status_code == 200 and result.json()["ok"]
        assert engine.source.spec == "camera:usb" and listener.microphone == "mic:usb"
        assert load(control.settings_path).camera == "camera:usb"
        assert engine.ack_calls == 0
        assert listener.snapshot().status == "mati"


@pytest.mark.parametrize("body", [[], {"camera": 999}, {"camera": True}, {"camera": "file.mp4"},
                                  {"camera": "camera:missing"}, {"microphone": "mic:missing"}, {"other": 1}])
def test_invalid_selection_changes_nothing(selection, body):
    control, engine, listener = selection
    with TestClient(create_app(engine, listener, inputs=control)) as client:
        response = client.post("/api/perangkat", json=body)
        assert response.status_code == 400 and response.json()["ok"] is False
    assert engine.source.spec is None and listener.microphone is None
    assert not control.settings_path.exists()


def test_scan_failure_in_one_category_does_not_hide_other(selection, monkeypatch):
    from sahabat import input_control as ic
    control, _, _ = selection
    monkeypatch.setattr(ic, "list_microphones", lambda: (_ for _ in ()).throw(OSError("audio unavailable")))
    result = control.scan()
    assert len(result["cameras"]) == 2 and result["microphones"] == [] and result["errors"]


def test_unplugged_selection_can_be_kept_when_only_other_device_changes(selection, monkeypatch):
    from sahabat import input_control as ic
    control, engine, _ = selection
    engine.source.select("camera:usb")
    monkeypatch.setattr(ic, "list_cameras", lambda: [])
    control.apply({"camera": "camera:usb", "microphone": "mic:usb"})
    assert engine.source.spec == "camera:usb"


def test_unwritable_settings_does_not_apply_half_selection(selection):
    control, engine, listener = selection
    control.settings_path.mkdir()
    with pytest.raises(OSError):
        control.apply({"camera": "camera:usb", "microphone": "mic:usb"})
    assert engine.source.spec is None and listener.microphone is None


def test_boolean_is_never_accepted_as_legacy_camera_index(selection):
    control, engine, _ = selection
    engine.source.spec = 0
    with pytest.raises(ValueError):
        control.apply({"camera": False})


@pytest.fixture
def with_videos(selection, tmp_path):
    control, engine, listener = selection
    control.video_dir = tmp_path / "video"
    control.video_dir.mkdir()
    (control.video_dir / "demo jatuh.mp4").write_bytes(b"x")
    (control.video_dir / "catatan.txt").write_text("bukan video")
    return control, engine, listener


def test_videos_are_listed_and_selected_for_this_session_only(with_videos):
    control, engine, listener = with_videos
    with TestClient(create_app(engine, listener, inputs=control)) as client:
        catalog = client.get("/api/perangkat").json()
        assert catalog["videos"] == [{"id": "video:demo jatuh.mp4", "name": "demo jatuh.mp4"}]
        result = client.post("/api/perangkat", json={"camera": "video:demo jatuh.mp4"}).json()
        assert result["ok"] and result["camera"] == "video:demo jatuh.mp4" and result["camera_is_video"] is True
        assert engine.source.spec == str(control.video_dir / "demo jatuh.mp4")
        assert not control.settings_path.exists()            # video tidak diingat setelah dinyalakan ulang
        back = client.post("/api/perangkat", json={"camera": "camera:usb"}).json()
        assert back["camera"] == "camera:usb" and back["camera_is_video"] is False
        assert load(control.settings_path).camera == "camera:usb"


@pytest.mark.parametrize("bad", ["video:hilang.mp4", "video:catatan.txt", "video:../settings.json"])
def test_unknown_or_unsafe_video_is_refused(with_videos, bad):
    control, engine, listener = with_videos
    with TestClient(create_app(engine, listener, inputs=control)) as client:
        assert client.post("/api/perangkat", json={"camera": bad}).status_code == 400
    assert engine.source.spec is None


def test_upload_copies_the_video_and_never_overwrites(with_videos):
    control, engine, listener = with_videos
    with TestClient(create_app(engine, listener, inputs=control)) as client:
        r = client.post("/api/video", params={"nama": "demo jatuh.mp4"}, content=b"isi video")
        assert r.json() == {"ok": True, "id": "video:demo jatuh (2).mp4", "nama": "demo jatuh (2).mp4"}
        assert (control.video_dir / "demo jatuh (2).mp4").read_bytes() == b"isi video"
        assert (control.video_dir / "demo jatuh.mp4").read_bytes() == b"x"
        for bad in ("", "a.exe", "../x.mp4", r"c:\x.mp4", ".mp4", "x" * 130 + ".mp4"):
            assert client.post("/api/video", params={"nama": bad}, content=b"1").status_code == 400
        assert client.post("/api/video", params={"nama": "kosong.mp4"}, content=b"").status_code == 400
    assert not list(control.video_dir.glob("*.part")) and not (control.video_dir / "kosong.mp4").exists()


def test_open_video_folder(with_videos, monkeypatch):
    from sahabat import server
    control, engine, listener = with_videos
    opened = []
    monkeypatch.setattr(server, "open_folder", lambda folder: opened.append(folder))
    with TestClient(create_app(engine, listener, inputs=control)) as client:
        assert client.post("/api/video/folder").json() == {"ok": True}
    assert opened == [control.video_dir]


def test_without_input_control_video_endpoints_say_so():
    from tests.test_server import StubEngine
    with TestClient(create_app(StubEngine())) as client:
        assert client.post("/api/video", params={"nama": "a.mp4"}, content=b"1").status_code == 503
        assert client.post("/api/video/folder").status_code == 503
