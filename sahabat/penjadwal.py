"""Penjadwal Mode CCTV: kamera mana yang dinilai pose sekarang.

- penuh: ada orang dalam beberapa detik terakhir atau alarmnya belum diperiksa. Dinilai 10x per detik, sama dengan
  laju pelatihan model.
- siaga: tidak ada orang. Pose tidak dijalankan; hanya deteksi gerak murah. Begitu ada gerak, kamera dinilai pose,
  dan bila ada orang langsung menjadi penuh. Pengaman: pose tetap dijalankan sekali tiap beberapa detik.
- lambat: penuh, tetapi laju penilaiannya di bawah batas karena komputer tidak sanggup. Ditampilkan, tidak disembunyikan.
- terputus: tidak dinilai.
Kamera prioritas yang penuh selalu didahulukan; kamera penuh lainnya bergiliran (yang paling lama menunggu dulu).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SLOW_AFTER_S = 2.0          # laju diukur atas jendela 2 dtk; sebelum itu kamera belum bisa disebut lambat
URGENT_WAIT_S = 0.5         # kamera siaga yang bergerak tidak menunggu lebih lama dari ini (spec: naik ≤ 0,5 dtk)
_DUE_SLACK_S = 0.005


@dataclass
class KamInfo:
    id: str
    prioritas: bool
    tersambung: bool
    penuh: bool
    last_step: float
    gerak: bool
    fps: float
    penuh_sejak: float | None


class Penjadwal:
    def __init__(self, rate_hz: float = 10.0, safety_s: float = 5.0, floor_hz: float = 8.0,
                 motion_hz: float = 5.0) -> None:
        self.rate_hz, self.safety_s, self.floor_hz, self.motion_hz = rate_hz, safety_s, floor_hz, motion_hz

    def pilih(self, now: float, kamera: list[KamInfo]) -> list[str]:
        period = 1.0 / self.rate_hz
        priority, urgent, full, idle = [], [], [], []
        for k in kamera:
            if not k.tersambung:
                continue
            waited = now - k.last_step
            if k.penuh:
                if waited >= period - _DUE_SLACK_S:
                    (priority if k.prioritas else full).append(k)
            elif waited >= self.safety_s or (k.gerak and waited >= URGENT_WAIT_S):
                urgent.append(k)            # supaya kamera siaga tidak kelaparan saat kamera penuh menghabiskan CPU
            elif k.gerak:
                idle.append(k)
        order = []
        # Kamera siaga yang mendesak (bergerak ≥ 0,5 dtk menunggu, atau pengaman 5 dtk) didahulukan bahkan dari kamera
        # prioritas: paling sering ±2 kali per detik, dan tanpa ini kamera lain tidak pernah dinilai sama sekali.
        for group in (urgent, priority, full, idle):
            order += [k.id for k in sorted(group, key=lambda k: k.last_step)]
        return order

    def perlu_cek_gerak(self, now: float, last_check: float) -> bool:
        return now - last_check >= 1.0 / self.motion_hz - _DUE_SLACK_S

    def status(self, k: KamInfo, now: float) -> str:
        if not k.tersambung:
            return "terputus"
        if not k.penuh:
            return "siaga"
        if now - k.last_step > 2.5 / self.rate_hz:
            return "lambat"                 # kamera penuh yang tidak dilayani: langsung terlihat, tanpa masa tenggang
        if k.penuh_sejak is not None and now - k.penuh_sejak >= SLOW_AFTER_S and k.fps < self.floor_hz:
            return "lambat"
        return "penuh"


class GerakMurah:
    """Selisih gambar kecil (64x36 abu-abu) dengan gambar sebelumnya; sekitar 1 ms per kamera."""

    SIZE = (64, 36)

    def __init__(self, threshold: float = 2.0) -> None:
        self.threshold = threshold
        self._prev: np.ndarray | None = None

    def berubah(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, self.SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        gray = gray.astype(np.int16)
        prev, self._prev = self._prev, gray
        if prev is None:
            return True
        return float(np.mean(np.abs(gray - prev))) >= self.threshold
