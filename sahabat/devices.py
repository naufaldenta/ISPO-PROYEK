"""Inventaris input tanpa membuka kamera atau merekam suara.

Kamera disimpan dengan device path, mikrofon dengan nama + host API: nomor indeks
dapat berubah saat USB dicabut. Enumerasi audio memakai proses terpisah supaya
penyegaran PortAudio tidak menghentikan stream yang sedang dipakai Listener.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import cv2


def _enumerate_cameras(backend):
    from cv2_enumerate_cameras import enumerate_cameras
    return enumerate_cameras(backend)


def list_cameras() -> list[dict]:
    cameras = {}
    errors = []
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        try:
            rows = _enumerate_cameras(backend)
        except Exception as exc:
            errors.append(exc)
            continue
        for camera in rows:
            # DirectShow/MSMF append different interface-class GUIDs to the same USB path.
            identity = (camera.path or f"name:{camera.name}:{backend}:{camera.index}").lower().split("#{")[0]
            key = "camera:" + identity
            item = cameras.setdefault(key, {"id": key, "name": camera.name, "routes": []})
            item["routes"].append((camera.index, camera.backend))
    if len(errors) == 2:
        raise OSError("Daftar kamera tidak bisa dibaca: " + str(errors[0]))
    return list(cameras.values())


def camera_routes(spec: int | str | None) -> list[tuple[int, int]]:
    if isinstance(spec, int):                 # pengaturan lama tetap berlaku
        if spec >= 100:
            return [(spec % 100, (spec // 100) * 100)]
        return [(spec, cv2.CAP_DSHOW), (spec, cv2.CAP_MSMF)]
    cameras = list_cameras()
    return [route for camera in cameras if spec is None or camera["id"] == spec for route in camera["routes"]]


def microphone_catalog(rows, apis) -> list[dict]:
    inputs = [(i, d) for i, d in enumerate(rows) if d["max_input_channels"] > 0]
    # MME supports resampling to 16 kHz and exposes physical/virtual inputs once.
    # WASAPI/WDM aliases would otherwise show the same hardware several times.
    mme = [(i, d) for i, d in inputs if apis[d["hostapi"]]["name"] == "MME"]
    if mme:
        inputs = mme
    found, counts = [], {}
    for index, info in inputs:
        name, host = info["name"], apis[info["hostapi"]]["name"]
        if name in ("Microsoft Sound Mapper - Input", "Primary Sound Capture Driver"):
            continue
        pair = (host, name)
        ordinal = counts.get(pair, 0)
        counts[pair] = ordinal + 1
        key = "mic:" + json.dumps([host, name, ordinal], ensure_ascii=True, separators=(",", ":"))
        found.append({"id": key, "name": name + (f" ({ordinal + 1})" if ordinal else ""), "index": index})
    return found


def resolve_microphone(sd, spec: int | str | None):
    if not isinstance(spec, str) or not spec.startswith("mic:"):
        return spec
    for device in microphone_catalog(sd.query_devices(), sd.query_hostapis()):
        if device["id"] == spec:
            return device["index"]
    raise OSError("Mikrofon pilihan tidak terhubung")


def list_microphones() -> list[dict]:
    result = subprocess.run(
        [sys.executable, "-m", "sahabat.devices", "--microphones"],
        cwd=os.path.dirname(os.path.dirname(__file__)), capture_output=True,
        text=True, encoding="utf-8", timeout=10, check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return json.loads(result.stdout)


if __name__ == "__main__" and "--microphones" in sys.argv:
    import sounddevice as sd
    print(json.dumps(microphone_catalog(sd.query_devices(), sd.query_hostapis()), ensure_ascii=True))
