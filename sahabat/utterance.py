"""Memotong aliran sampel mikrofon menjadi ucapan: suara yang diapit hening.

Hening dan dengung tetap (kipas, AC) tidak dikeluarkan sama sekali, supaya pengenal ucapan yang mahal
hanya bekerja saat ada orang bicara.

Lantai bising dipelajari terus-menerus dari SEMUA bingkai (persentil rendah dalam 10 detik terakhir), juga saat
ucapan berjalan. Versi pertama membekukan lantai selama ucapan; di sesi uji 15 November 2026 lantai itu terpelajari
dari awal yang senyap, bising ruangan lalu terus dianggap suara, dan semua potongan keluar 6 detik (kata terpotong
di sembarang tempat, pengenal bekerja tanpa henti).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .config import Config

FRAME_S = 0.03               # bingkai penilaian
PREROLL_S = 0.2              # sedikit suara sebelum bingkai bersuara pertama ikut dibawa
SHORT_S = 1.2                # satu kata atau frasa singkat bisa diselesaikan lebih cepat
FLAT_DB = 6.0                # potongan paksa yang kekerasannya serata ini adalah bising, bukan ucapan
MIN_DB = -100.0


class Segmenter:
    def __init__(self, cfg: Config, rate: int = 16000) -> None:
        self.cfg = cfg
        self.rate = rate
        self._frame = int(round(FRAME_S * rate))
        self._silence_frames = int(np.ceil(cfg.utt_silence_s / FRAME_S))
        self._short_silence_frames = min(self._silence_frames, int(np.ceil(cfg.utt_short_silence_s / FRAME_S)))
        self._short_frames = int(round(SHORT_S / FRAME_S))
        self._min_frames = int(np.ceil(cfg.utt_min_s / FRAME_S - 1e-9))
        self._max_frames = int(round(cfg.utt_max_s / FRAME_S))
        self._overlap_frames = int(np.ceil(cfg.utt_overlap_s / FRAME_S))
        self._pending = np.zeros(0, dtype=np.int16)                       # sampel yang belum genap satu bingkai
        self._heard: deque[float] = deque(maxlen=int(round(cfg.utt_floor_s / FRAME_S)))   # kekerasan semua bingkai
        self._preroll: deque[np.ndarray] = deque(maxlen=int(np.ceil(PREROLL_S / FRAME_S)))
        self._frames: list[np.ndarray] | None = None                      # None = di luar ucapan
        self._levels: list[float] = []
        self._speech = 0
        self._quiet = 0

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        """Menerima sampel int16 mono berapa pun banyaknya; mengembalikan ucapan yang selesai pada panggilan ini."""
        data = np.concatenate([self._pending, np.asarray(samples, dtype=np.int16).ravel()])
        whole = len(data) // self._frame * self._frame
        self._pending = data[whole:]
        out: list[np.ndarray] = []
        for start in range(0, whole, self._frame):
            done = self._step(data[start:start + self._frame])
            if done is not None:
                out.append(done)
        return out

    def flush(self) -> list[np.ndarray]:
        """Akhir aliran: ucapan yang sedang berjalan dikeluarkan bila cukup panjang."""
        out = [self._join()] if self._frames is not None and self._speech >= self._min_frames else []
        self._leave()
        self._pending = np.zeros(0, dtype=np.int16)
        return out

    # ---- satu bingkai ---------------------------------------------------------------

    def _step(self, frame: np.ndarray) -> np.ndarray | None:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        level = max(20.0 * np.log10(rms / 32768.0), MIN_DB) if rms > 0 else MIN_DB
        self._heard.append(level)
        floor = float(np.percentile(self._heard, self.cfg.utt_floor_pct))
        voiced = level >= max(floor + self.cfg.utt_gate_db, self.cfg.utt_abs_db)

        if self._frames is None:
            if not voiced:
                self._preroll.append(frame)
                return None
            self._frames, self._levels = list(self._preroll) + [frame], [level]
            self._preroll.clear()
            self._speech, self._quiet = 1, 0
            return None

        self._frames.append(frame)
        self._levels.append(level)
        if voiced:
            self._speech += 1
            self._quiet = 0
        else:
            self._quiet += 1

        active_frames = len(self._levels) - self._quiet
        end_silence = self._short_silence_frames if active_frames <= self._short_frames else self._silence_frames
        if self._quiet >= end_silence:                                     # hening penutup: ucapan selesai
            done = self._join() if self._speech >= self._min_frames else None
            self._leave()
            return done
        if len(self._frames) >= self._max_frames:                         # terlalu panjang: potong paksa
            if np.percentile(self._levels, 95) - np.percentile(self._levels, 5) < FLAT_DB:
                # Bising tetap yang baru muncul (kipas dinyalakan): buang, dan jadikan ia lantai sekarang juga
                # (tanpa ini lantai baru menyusul setelah jendela 10 detiknya terisi bising itu).
                levels = list(self._levels)
                self._leave()
                self._heard.clear()
                self._heard.extend(levels)
                return None
            done = self._join()
            keep = self._overlap_frames
            self._frames, self._levels = self._frames[-keep:], self._levels[-keep:]
            # Tumpang-tindih tidak dihitung sebagai suara baru: tanpa suara baru yang cukup, sambungan ini dibuang,
            # supaya akhir kalimat tidak dikirim (dan diumumkan) dua kali.
            self._speech, self._quiet = 0, 0
            return done
        return None

    def _join(self) -> np.ndarray:
        return np.concatenate(self._frames).astype(np.int16)

    def _leave(self) -> None:
        self._frames, self._levels = None, []
        self._speech = self._quiet = 0
        self._preroll.clear()
