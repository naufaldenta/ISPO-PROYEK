"""Mikrofon lewat sounddevice. Logika pendengar ada di listener.py; di sini hanya membuka, membaca, dan menyadari putus.

Diuji dengan pustaka sounddevice tiruan (tests/test_mic.py). Yang belum pernah dicoba pada perangkat sungguhan:
mencabut dan mencolok ulang mikrofon USB saat aplikasi berjalan.
"""
from __future__ import annotations

import queue
import time
from typing import Protocol

import numpy as np

from .devices import resolve_microphone

READ_WAIT_S = 0.5
STALL_S = 2.0                # tidak ada blok selama ini = perangkat hilang (Windows sering tidak melapor saat USB dicabut)
_opened_before = False


class Mic(Protocol):
    name: str

    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...


class _SoundDeviceMic:
    def __init__(self, device: int | str | None, rate: int) -> None:
        global _opened_before
        import sounddevice as sd
        self._blocks: queue.Queue = queue.Queue(maxsize=50)
        self._closed = False
        self._stream = None
        if _opened_before:                                 # daftar perangkat PortAudio hanya dibaca sekali saat dimuat;
            try:                                           # tanpa ini mikrofon yang dicolok ulang tidak pernah terlihat
                sd._terminate()
                sd._initialize()
            except Exception:
                pass
        _opened_before = True
        try:
            device = resolve_microphone(sd, device)
            info = sd.query_devices(device, "input")
            self.name = str(info["name"])
            self._stream = sd.InputStream(samplerate=rate, channels=1, dtype="int16", blocksize=rate // 10,
                                          device=device, callback=self._on_block)
            self._stream.start()
        except (sd.PortAudioError, ValueError) as e:       # tidak ada perangkat masukan, nama salah, atau gagal dibuka
            self.close()
            raise OSError(str(e)) from e
        self._last_block = time.monotonic()

    def _on_block(self, indata, frames, time_info, status) -> None:
        block = indata[:, 0].copy()
        try:
            self._blocks.put_nowait(block)
        except queue.Full:                                 # pembaca tertinggal: buang yang tertua
            try:
                self._blocks.get_nowait()
                self._blocks.put_nowait(block)
            except (queue.Empty, queue.Full):
                pass

    def read(self) -> np.ndarray | None:
        """Satu blok int16 (±0,1 detik); None bila belum ada; OSError bila perangkatnya hilang atau berhenti mengirim."""
        try:
            block = self._blocks.get(timeout=READ_WAIT_S)
        except queue.Empty:
            stalled = time.monotonic() - self._last_block >= STALL_S
            if self._closed or not self._stream.active or stalled:
                raise OSError("mikrofon berhenti mengirim suara")
            return None
        self._last_block = time.monotonic()
        return block

    def close(self) -> None:
        self._closed = True
        if self._stream is None:
            return
        try:
            self._stream.stop()
        except Exception:                                  # perangkat sudah dicabut: tidak ada yang perlu dihentikan
            pass
        try:
            self._stream.close()
        except Exception:
            pass


def open_mic(device: int | str | None, rate: int = 16000) -> Mic:
    """Membuka mikrofon; None = perangkat masukan bawaan Windows. OSError bila tidak ada, ImportError bila pustakanya tidak ada."""
    return _SoundDeviceMic(device, rate)
