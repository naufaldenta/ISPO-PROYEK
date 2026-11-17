"""Memotong satu klip menjadi jendela fitur 1 detik, persis seperti penilaian langsung.

Dataset hanya memberi label per klip. Jadi:
- klip nonkekerasan: semua jendela berlabel "bukan";
- klip kekerasan: hanya jendela yang geraknya cukup besar yang dipakai; jendela tenang dibuang.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sahabat.config import Config
from sahabat.history import History
from sahabat.pair_features import MOTION_FEATURES as _MOTION
from sahabat.pair_features import WINDOW_FEATURE_NAMES, close_pairs, pair_windows
from sahabat.types import Person



@dataclass
class ClipPairs:
    """Satu klip yang sudah diputar ulang: fitur dua arah untuk tiap pasangan di tiap frame."""
    t: list[float]                                                      # cap waktu tiap frame
    rows: list[tuple[int, int, int, np.ndarray, np.ndarray]]            # (frame, nomor A, nomor B, A->B, B->A)


def load_clip(npz: Path) -> list[tuple[float, list[Person]]]:
    with np.load(npz, allow_pickle=False) as data:
        times = data["t"]
        people = tuple(data[name] for name in ("frame_of", "track_id", "kp", "conf", "box"))
    frames: list[tuple[float, list[Person]]] = [(float(t), []) for t in times]
    for frame, track_id, kp, conf, box in zip(*people, strict=True):
        frames[int(frame)][1].append(Person(int(track_id), kp, conf, tuple(float(v) for v in box)))
    return frames


def clip_pairs(npz: Path, cfg: Config) -> ClipPairs:
    return pairs_from_frames(load_clip(npz), cfg)


def pairs_from_frames(frames: list[tuple[float, list[Person]]], cfg: Config) -> ClipPairs:
    """Memutar ulang klip lewat History yang sama dengan aplikasi."""
    history = History(cfg)
    rows = []
    for i, (t, people) in enumerate(frames):
        history.update(t, people)
        for a, b in close_pairs(history.tracks, cfg.pair_max_dist):
            both = pair_windows(a, b, cfg.violence_window_s)
            if both is not None:
                rows.append((i, a.track_id, b.track_id, both[0], both[1]))
    return ClipPairs([t for t, _ in frames], rows)


def training_rows(clip: ClipPairs, label: int, motion_min: float = 1.0, stride: int = 1) -> np.ndarray:
    """Matriks (jumlah jendela, jumlah fitur): dua arah tiap pasangan, tenang dibuang bila label kekerasan."""
    rows = [f for _, _, _, ab, ba in clip.rows[::stride] for f in (ab, ba)
            if label == 0 or max(f[i] for i in _MOTION) >= motion_min]
    return np.array(rows, dtype=np.float32).reshape(-1, len(WINDOW_FEATURE_NAMES))


def clip_windows(npz: Path, label: int, cfg: Config, motion_min: float = 1.0) -> np.ndarray:
    return training_rows(clip_pairs(npz, cfg), label, motion_min)
