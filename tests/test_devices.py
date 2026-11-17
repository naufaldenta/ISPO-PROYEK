import json
from types import SimpleNamespace

import cv2
import pytest

from sahabat import devices


def cam(index, name, path, backend):
    return SimpleNamespace(index=index, name=name, path=path, backend=backend)


def test_camera_catalog_merges_backends_and_keeps_external_beyond_index_three(monkeypatch):
    rows = {cv2.CAP_DSHOW: [cam(0, "Internal", "usb#a#{one}", cv2.CAP_DSHOW),
                           cam(7, "External", "usb#b#{one}", cv2.CAP_DSHOW)],
            cv2.CAP_MSMF: [cam(2, "External", "USB#B#{two}", cv2.CAP_MSMF)]}
    monkeypatch.setattr(devices, "_enumerate_cameras", lambda backend: rows[backend])
    found = devices.list_cameras()
    assert [c["name"] for c in found] == ["Internal", "External"]
    assert found[1]["routes"] == [(7, cv2.CAP_DSHOW), (2, cv2.CAP_MSMF)]
    assert devices.camera_routes(found[1]["id"]) == found[1]["routes"]
    rows[cv2.CAP_DSHOW][1].index = 5
    assert devices.camera_routes(found[1]["id"])[0] == (5, cv2.CAP_DSHOW)
    assert devices.camera_routes("camera:usb#missing") == []


def test_microphone_catalog_excludes_outputs_and_system_aliases():
    rows = [{"name": "Microsoft Sound Mapper - Input", "hostapi": 0, "max_input_channels": 2},
            {"name": "Internal", "hostapi": 0, "max_input_channels": 2},
            {"name": "USB mic", "hostapi": 0, "max_input_channels": 1},
            {"name": "Speaker", "hostapi": 0, "max_input_channels": 0},
            {"name": "USB mic", "hostapi": 1, "max_input_channels": 2}]
    apis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
    found = devices.microphone_catalog(rows, apis)
    assert [d["name"] for d in found] == ["Internal", "USB mic"]
    sd = SimpleNamespace(query_devices=lambda: rows, query_hostapis=lambda: apis)
    assert devices.resolve_microphone(sd, found[1]["id"]) == 2
    rows[1], rows[2] = rows[2], rows[1]
    assert devices.resolve_microphone(sd, found[1]["id"]) == 1
    rows.pop(1)
    with pytest.raises(OSError, match="tidak terhubung"):
        devices.resolve_microphone(sd, found[1]["id"])


def test_microphone_scan_uses_isolated_process_with_timeout(monkeypatch):
    seen = {}
    def run(args, **kw):
        seen.update(kw)
        assert args[-2:] == ["sahabat.devices", "--microphones"]
        return SimpleNamespace(stdout=json.dumps([{"id": "mic:usb", "name": "USB"}]))
    monkeypatch.setattr(devices.subprocess, "run", run)
    assert devices.list_microphones()[0]["name"] == "USB"
    assert seen["timeout"] <= 10 and seen["check"]
