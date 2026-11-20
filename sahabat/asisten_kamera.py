"""Asisten pemasangan kamera: petunjuk dari pose yang terlihat selama ±30 detik terakhir.

- Badan terlihat utuh: persen pengamatan orang dengan mode tampilan "full" (bahu, pinggul, lutut, dan pergelangan
  kaki terlihat), mode yang sama dengan yang dipakai penilai.
- Ukuran orang: median tinggi kotak orang dibagi tinggi gambar.
- Sudut pandang: median (pinggul ke pergelangan kaki) / (bahu ke pinggul) pada orang tegak yang badannya utuh. Dari
  kamera setinggi dada, kaki tampak sekitar 1,3-1,6 kali badan; dari kamera tinggi yang menunduk, kaki memendek.

Murah: hanya beberapa angka per orang per penilaian, tanpa menyalin gambar.
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import deque

import numpy as np

from .history import raw_mode
from .types import ANKLES, HIPS, SHOULDERS, Person

# Kalibrasi 20 Nov 2026 (jejak pose data uji):
# - median keyakinan titik tubuh menurut tinggi orang: 0,10-0,15 -> 0,72; 0,15-0,20 -> 0,82; 0,20-0,30 -> 0,89.
# - median rasio kaki/badan: sesi 16 kamera dada 1,35 (p25 1,30), sesi 16 kamera tinggi 1,23 (p75 1,28), sesi 17 1,44,
#   GMDCSA 1,38, CAUCAFall 1,34, Bus (atap) 1,28, Le2i (dinding tinggi, orang jauh) 1,45. Rasio mengukur curamnya
#   pandangan ke orang, bukan tinggi pemasangan: kamera tinggi yang memandang orang jauh tetap tampak "setinggi dada".
KECIL = 0.15         # tinggi orang < 15% tinggi gambar: keyakinan titik tubuh mulai turun
MENUNDUK = 1.25      # di antara kamera dada (p25 1,30) dan kamera tinggi (median 1,23) di ruangan yang sama
MIN_OBS = 20         # pengamatan minimal sebelum memberi saran (±2 dtk orang terlihat pada 10 Hz)
MIN_RATIO_OBS = 10
MIN_CONF = 0.5
WAIT = "Belum ada orang terlihat. Minta satu orang berjalan di depan kamera."


def _mid(kp: np.ndarray, conf: np.ndarray, idxs) -> np.ndarray | None:
    pts = [kp[i] for i in idxs if conf[i] >= MIN_CONF]
    return np.mean(pts, axis=0) if pts else None


def leg_torso_ratio(person: Person) -> float | None:
    """(pinggul ke pergelangan kaki) / (bahu ke pinggul) bila orangnya tegak dan titiknya terlihat, selain itu None."""
    kp, conf = np.asarray(person.kp, dtype=np.float32), np.asarray(person.conf, dtype=np.float32)
    sh, hip, ank = _mid(kp, conf, SHOULDERS), _mid(kp, conf, HIPS), _mid(kp, conf, ANKLES)
    if sh is None or hip is None or ank is None:
        return None
    torso, leg = hip - sh, ank - hip
    upright = torso[1] > 2 * abs(torso[0]) and leg[1] > 2 * abs(leg[0])     # bahu di atas pinggul di atas kaki
    length = float(np.hypot(*torso))
    if not upright or length < 1.0:
        return None
    return float(np.hypot(*leg)) / length


class Asisten:
    def __init__(self, window_s: float = 30.0, clock=time.monotonic) -> None:
        self.window_s = window_s
        self._clock = clock                 # jam server: saran basi bila tidak ada pengamatan baru selama window_s
        self._last_seen: float | None = None
        self._obs: deque = deque()          # (t, mode, tinggi_orang, rasio | None, id)
        self._last_t: float | None = None
        self._lock = threading.Lock()       # amati() dari utas mesin, ringkas() dari utas server

    def amati(self, t: float, people: list[Person], shape: tuple[int, int], mode: dict[int, str]) -> None:
        height = float(shape[0]) or 1.0
        rows = []
        for p in people:
            m = mode.get(p.track_id) or raw_mode(np.asarray(p.conf), MIN_CONF)
            box_h = float(p.box[3] - p.box[1]) if p.box else 0.0
            rows.append((t, m, box_h / height, leg_torso_ratio(p) if m == "full" else None, p.track_id))
        with self._lock:
            self._last_t = t
            self._last_seen = self._clock()
            self._obs.extend(rows)
            self._prune(t)

    def reset(self) -> None:
        """Sumber berganti (kamera lain): saran kamera sebelumnya tidak berlaku lagi."""
        with self._lock:
            self._obs.clear()
            self._last_t = self._last_seen = None

    def _prune(self, now: float) -> None:
        while self._obs and now - self._obs[0][0] > self.window_s:
            self._obs.popleft()

    def ringkas(self, now: float | None = None) -> dict:
        with self._lock:
            if self._last_seen is not None and self._clock() - self._last_seen > self.window_s:
                self._obs.clear()           # kamera terputus / mesin dijeda: saran lama sudah basi
            now = self._last_t if now is None else now
            if now is not None:
                self._prune(now)
            obs = list(self._obs)
        if len(obs) < MIN_OBS:
            return {"orang": len({o[4] for o in obs}), "utuh_persen": None, "tinggi_orang": None,
                    "rasio_kaki_badan": None, "saran": [WAIT]}
        utuh = round(100 * sum(o[1] == "full" for o in obs) / len(obs))
        tinggi = statistics.median(o[2] for o in obs)
        ratios = [o[3] for o in obs if o[3] is not None]
        rasio = statistics.median(ratios) if len(ratios) >= MIN_RATIO_OBS else None
        saran = [f"Badan terlihat utuh: {utuh}%"]
        if tinggi < KECIL:
            saran.append("Orang terlalu kecil, dekatkan kamera atau pakai lensa lebih sempit")
        if rasio is not None and rasio < MENUNDUK:
            saran.append("Kamera terlalu menunduk")
        if utuh < 50:
            saran.append("Kaki sering terpotong, naikkan atau mundurkan kamera")
        return {"orang": len({o[4] for o in obs}), "utuh_persen": utuh, "tinggi_orang": round(tinggi, 3),
                "rasio_kaki_badan": round(rasio, 3) if rasio is not None else None, "saran": saran}
