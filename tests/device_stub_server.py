"""UI smoke test tanpa hardware atau perubahan pengaturan pengguna.

runtime/python/python.exe -m uvicorn --factory tests.device_stub_server:app --port 8766
"""
import tempfile
from pathlib import Path

from sahabat.config import Config
from sahabat.source import VideoSource
from sahabat.input_control import InputDevices
from sahabat.server import create_app
from .stub_server import StubEngine, StubListener


def app():
    import sahabat.input_control as ic
    ic.list_cameras = lambda: [{"id": "camera:internal", "name": "USB2.0 HD UVC WebCam"},
                               {"id": "camera:external", "name": "WebCamera"}]
    ic.list_microphones = lambda: [{"id": "mic:internal", "name": "Microphone (Realtek(R) Audio)"},
                                   {"id": "mic:external", "name": "Microphone (WebCamera)"}]
    engine, listener = StubEngine(), StubListener()
    engine.cfg, engine.source = Config(), VideoSource(None)
    engine.select_camera = engine.source.select
    listener.microphone, listener.mic_name = None, ""
    listener.set_microphone = lambda spec: setattr(listener, "microphone", spec)
    folder = tempfile.TemporaryDirectory(prefix="sahabat-ui-")
    inputs = InputDevices(engine, listener, Path(folder.name) / "settings.json")
    application = create_app(engine, listener, inputs)
    application.state.temp_folder = folder
    return application
