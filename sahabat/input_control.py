"""Pilihan input layar guru; model dan ambang tidak disentuh."""
from __future__ import annotations

import threading
from pathlib import Path

from .config import SETTINGS_PATH, update_local
from .devices import list_cameras, list_microphones


class InputDevices:
    def __init__(self, engine, listener, settings_path: Path = SETTINGS_PATH):
        self.engine, self.listener = engine, listener
        self.settings_path = Path(settings_path)
        self._lock = threading.Lock()

    def status(self) -> dict:
        source, listener = self.engine.source, self.listener
        return {"camera": source.spec,
                "microphone": listener.microphone if listener is not None else self.engine.cfg.microphone,
                "camera_connected": source.read() is not None,
                "camera_error": source.error,
                "microphone_status": listener.snapshot().label if listener is not None else "Mikrofon tidak tersedia",
                "microphone_name": listener.mic_name if listener is not None else "",
                "microphone_available": listener is not None}

    def scan(self) -> dict:
        result = {"cameras": [], "microphones": [], "errors": []}
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
                if value is not None and (not isinstance(value, str) or value not in {d["id"] for d in catalog()}):
                    raise ValueError("Perangkat pilihan tidak ditemukan. Tekan Cari ulang.")
                if name == "microphone" and self.listener is None:
                    raise ValueError("Pendengar mikrofon tidak tersedia pada sesi ini.")
                changes[name] = value
            if changes:
                update_local(self.settings_path, **changes)
                if "camera" in changes:
                    self.engine.select_camera(changes["camera"])
                if "microphone" in changes:
                    self.listener.set_microphone(changes["microphone"])
            return self.status()
