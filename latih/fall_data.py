"""Baris latih penilai jatuh dari tiga sumber: kerangka NTU (HRNet) dan dua set video sungguhan lewat YOLO-pose.

Video sungguhan (UR Fall Detection, GMDCSA-24) dilewatkan YOLO-pose + ByteTrack yang sama dengan aplikasi,
jadi cirinya memuat getar dan titik hilang yang sama dengan yang dilihat aplikasi di webcam. NTU memberi
jumlah (946 jatuh), video sungguhan memberi kemiripan dengan kondisi pakai.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sahabat.config import Config
from sahabat.history import History
from sahabat.person_features import PERSON_FEATURE_NAMES, person_features
from sahabat.types import HIPS

from .putar import PoseTrace

WINDOW_S = 2.0
N_FEAT = len(PERSON_FEATURE_NAMES)
POSITIVE_BEFORE_S = 0.2        # langkah positif: dari 0,2 dtk sebelum orangnya rebah ...
POSITIVE_AFTER_S = 1.0         # ... sampai 1 dtk sesudahnya (sama dengan aturan pada klip NTU)
LANDING_DEFAULT_S = 1.0        # bila saat rebah tak bisa dibaca dari pinggul
LANDING_SEARCH_S = 3.0
_EPS = 1e-6


@dataclass
class ClipRows:
    source: str                # ntu | urfall | gmdcsa
    name: str
    group: int                 # pemeran (NTU), subjek (GMDCSA), nomor klip (UR Fall): tidak boleh terbelah antar lipatan
    is_fall: bool
    kind: str                  # NTU: nomor aksi sebagai teks; video: jatuh | berbaring | biasa
    feats: np.ndarray          # (langkah, ciri); baris NaN = belum bisa dinilai
    labels: np.ndarray         # 1 jatuh, 0 bukan, -1 tidak dipakai melatih
    rule: bool                 # aturan lama memicu (hanya diisi untuk NTU)
    trace: PoseTrace | None    # video: jejak untuk dinilai ulang lewat mesin
    onset: float | None


def clip_features(frames, cfg: Config) -> np.ndarray:
    """(jumlah langkah, jumlah ciri) untuk orang utama di klip; baris NaN bila belum bisa dinilai."""
    history = History(cfg)
    out = np.full((len(frames), N_FEAT), np.nan, dtype=np.float32)
    for i, (t, people) in enumerate(frames):
        history.update(t, people)
        fresh = [tr for tr in history.tracks.values() if tr.samples and tr.samples[-1].t == t and tr.mode == "full"]
        if fresh:
            main = max(fresh, key=lambda tr: tr.ref_size)
            f = person_features(main, WINDOW_S)
            if f is not None:
                out[i] = f
    return out


def time_labels(ts: np.ndarray, onset: float | None, landing: float | None) -> np.ndarray:
    """Label per langkah dari waktu awal jatuh dan saat rebah. Tanpa awal jatuh: semuanya bukan jatuh."""
    ts = np.asarray(ts, dtype=float)
    if onset is None:
        return np.zeros(len(ts), dtype=np.int8)
    landing = onset + LANDING_DEFAULT_S if landing is None else landing
    labels = np.full(len(ts), -1, dtype=np.int8)
    labels[ts < onset - _EPS] = 0
    labels[(ts >= landing - POSITIVE_BEFORE_S - _EPS) & (ts <= landing + POSITIVE_AFTER_S + _EPS)] = 1
    return labels


def landing_time(tr: PoseTrace, onset: float) -> float:
    """Perkiraan saat orangnya rebah: pinggul orang terbesar mencapai 80% turunnya dalam 3 dtk sejak awal jatuh."""
    pts = []
    for t, people in zip(tr.t, tr.people):
        if onset - _EPS <= t <= onset + LANDING_SEARCH_S + _EPS and people:
            p = max(people, key=lambda q: (q.box[2] - q.box[0]) * (q.box[3] - q.box[1]))
            seen = [i for i in HIPS if p.conf[i] >= 0.5]
            if seen:
                pts.append((float(t), float(np.mean([p.kp[i][1] for i in seen])), float(p.box[3] - p.box[1])))
    if len(pts) < 3:
        return onset + LANDING_DEFAULT_S
    y0, height = pts[0][1], pts[0][2]
    drop = max(y for _, y, _ in pts) - y0
    if drop < 0.15 * height:                             # pinggul nyaris tidak turun (jatuh menjauhi kamera, tertutup)
        return onset + LANDING_DEFAULT_S
    return next(t for t, y, _ in pts if y >= y0 + 0.8 * drop)


def video_rows(source: str, name: str, group: int, tr: PoseTrace, cfg: Config, onset: float | None,
               landing: float | None, kind: str) -> ClipRows:
    frames = list(zip((float(t) for t in tr.t), tr.people))
    return ClipRows(source, name, group, onset is not None, kind, clip_features(frames, cfg),
                    time_labels(tr.t, onset, landing), False, tr, onset)


def stack(clips: list[ClipRows], weights: dict[str, float] | None = None):
    """(X, y, bobot) dari baris yang berlabel dan punya ciri. Bobot per sumber; bawaan 1."""
    weights = weights or {}
    X, y, w = [], [], []
    for c in clips:
        usable = (c.labels >= 0) & np.isfinite(c.feats).any(axis=1)
        X.append(c.feats[usable])
        y.append(c.labels[usable])
        w.append(np.full(int(usable.sum()), weights.get(c.source, 1.0)))
    if not X:
        return np.zeros((0, N_FEAT), dtype=np.float32), np.zeros(0, dtype=np.int8), np.zeros(0)
    return np.concatenate(X), np.concatenate(y), np.concatenate(w)


def fold_of(clip: ClipRows, n_folds: int) -> int:
    """Lipatan sebuah klip. Satu grup (pemeran/subjek) selalu utuh di satu lipatan."""
    return int(clip.group) % n_folds
