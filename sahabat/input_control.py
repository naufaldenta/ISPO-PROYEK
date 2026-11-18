"""Pilihan input layar guru; model dan ambang tidak disentuh."""
from __future__ import annotations

import threading
from pathlib import Path

from .config import SETTINGS_PATH, VIDEO_DIR, update_local
from .devices import VIDEO_PREFIX, list_cameras, list_microphones, list_videos


class InputDevices:
    def __init__(self, engine, listener, settings_path: Path = SETTINGS_PATH, video_dir: Path = VIDEO_DIR):
        self.engine, self.listener = engine, listener
        self.settings_path = Path(settings_path)
        self.video_dir = Path(video_dir)
        self._lock = threading.Lock()

    def _shown(self, spec):
        """Sumber video dari folder video/ ditampilkan sebagai "video:<nama berkas>"."""
        if isinstance(spec, str) and not spec.startswith("camera:"):
            path = Path(spec)
            if path.parent.resolve() == self.video_dir.resolve():
                return VIDEO_PREFIX + path.name
        return spec

    def status(self) -> dict:
        source, listener = self.engine.source, self.listener
        camera = self._shown(source.spec)
        return {"camera": camera,
                "camera_is_video": isinstance(camera, str) and camera.startswith(VIDEO_PREFIX),
                "microphone": listener.microphone if listener is not None else self.engine.cfg.microphone,
                "camera_connected": source.read() is not None,
                "camera_error": source.error,
                "microphone_status": listener.snapshot().label if listener is not None else "Mikrofon tidak tersedia",
                "microphone_name": listener.mic_name if listener is not None else "",
                "microphone_available": listener is not None}

    def scan(self) -> dict:
        result = {"cameras": [], "microphones": [], "errors": [], "videos": list_videos(self.video_dir)}
        for field, discover in (("cameras", list_cameras), ("microphones", list_microphones)):
            try:
                result[field] = [{"id": d["id"], "name": d["name"]} for d in discover()]
            except Exception as exc:
                result["errors"].append(str(exc))
        result.update(self.status())
        return result

    def apply(self, values: dict) -> dict:
        if not isinstance(values, dict) or not values or not set(values) <= {"camera", "microphone"}:
            raise ValueError("Pilih kamera dan/atau mikrofon yang tersedia.")
        if any(isinstance(value, bool) for value in values.values()):
            raise ValueError("Pilihan perangkat tidak valid.")
        with self._lock:
            current = self.status()
            changes = {}
            for name, catalog in (("camera", list_cameras), ("microphone", list_microphones)):
                if name not in values or values[name] == current[name]:
                    continue
                value = values[name]
                if name == "camera" and isinstance(value, str) and value.startswith(VIDEO_PREFIX):
                    if value not in {d["id"] for d in list_videos(self.video_dir)}:
                        raise ValueError("Video pilihan tidak ditemukan di folder video. Tekan Cari ulang.")
                    changes[name] = value
                    continue
                if value is not None and (not isinstance(value, str) or value not in {d["id"] for d in catalog()}):
                    raise ValueError("Perangkat pilihan tidak ditemukan. Tekan Cari ulang.")
                if name == "microphone" and self.listener is None:
                    raise ValueError("Pendengar mikrofon tidak tersedia pada sesi ini.")
                changes[name] = value
            if changes:
                # Video cadangan hanya untuk sesi ini: setelah dinyalakan ulang, SAHABAT kembali ke kamera.
                saved = {k: v for k, v in changes.items()
                         if not (k == "camera" and isinstance(v, str) and v.startswith(VIDEO_PREFIX))}
                if saved:
                    update_local(self.settings_path, **saved)
                if "camera" in changes:
                    camera = changes["camera"]
                    if isinstance(camera, str) and camera.startswith(VIDEO_PREFIX):
                        camera = str(self.video_dir / camera[len(VIDEO_PREFIX):])
                    self.engine.select_camera(camera)
                if "microphone" in changes:
                    self.listener.set_microphone(changes["microphone"])
            return self.status()
