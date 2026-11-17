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
