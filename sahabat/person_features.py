"""Ciri gerak satu orang selama ~2 detik, untuk penilai jatuh terlatih.

Aturan sudut badan saja buta terhadap jatuh ke arah (atau menjauhi) kamera: badan tidak tampak rebah
di gambar, tetapi pinggul dan kepala turun jauh dan badan memendek. Ciri di sini menangkap itu.

Semua jarak dalam ukuran badan (Track.ref_size), kecepatan per detik. Yang tak terukur diisi NaN;
model yang dipakai (gradient boosting) menerima NaN.
"""
from __future__ import annotations

import numpy as np

from .history import EPS, Track
from .types import ANKLES, HIPS, KNEES, NOSE, SHOULDERS

PERSON_FEATURE_NAMES: tuple[str, ...] = (
    "hip_drop",             # seberapa jauh pinggul sekarang di bawah titik tertingginya
    "hip_drop_speed_max",   # laju turun pinggul tertinggi
    "head_drop",
    "head_drop_speed_max",
    "angle_now",            # sudut badan dari tegak, 0 = berdiri, 1 = rebah
    "angle_max",
    "angle_change",         # sudut sekarang dikurangi sudut di awal jendela
    "torso_len_ratio",      # panjang badan sekarang dibanding acuan (memendek = condong ke kamera)
    "head_above_hip",       # tinggi kepala di atas pinggul (berdiri ~1,3; tergeletak ~0)
    "box_aspect_now",       # lebar/tinggi kotak titik tubuh
    "box_height_ratio",     # tinggi kotak sekarang dibanding tertinggi di jendela
    "ankle_below_hip",      # jarak tegak pergelangan kaki di bawah pinggul (berdiri ~1,8)
    "still",                # laju rata-rata tengah badan dalam 0,5 dtk terakhir
    "conf_now",             # rata-rata keyakinan titik tubuh sekarang
    "conf_drop",            # penurunannya dibanding yang terbaik di jendela
    "knee_straight_now",    # 0..1, 1 = kaki lurus
    "since_drop_s",         # berapa detik sejak pinggul turun paling cepat
)

MIN_CONF = 0.5
MIN_SAMPLES = 8
SPEED_LAG = 3               # kecepatan diukur antar sampel berjarak 3 langkah, tahan getar
RECENT_S = 0.5


def _mid(kp: np.ndarray, vis: np.ndarray, idxs) -> np.ndarray:
    """Titik tengah beberapa sendi per frame, (T, 2); NaN bila tak satu pun terlihat."""
    w = vis[:, idxs].astype(float)
    n = w.sum(axis=1)
    total = (kp[:, idxs, :] * w[:, :, None]).sum(axis=1)
    out = np.full((len(kp), 2), np.nan)
    ok = n > 0
    out[ok] = total[ok] / n[ok, None]
    return out


def _last(x: np.ndarray, ts: np.ndarray) -> float:
    """Nilai terakhir yang terukur dalam RECENT_S terakhir; NaN bila tidak ada."""
    ok = np.isfinite(x) & (ts[-1] - ts <= RECENT_S + EPS)
    return float(x[ok][-1]) if ok.any() else float("nan")


def _drop_speed(y: np.ndarray, ts: np.ndarray, ref: float) -> tuple[float, float]:
    """(laju turun tertinggi, cap waktunya); (0, NaN) bila tak terukur."""
    if len(y) <= SPEED_LAG:
        return 0.0, float("nan")
    dt = ts[SPEED_LAG:] - ts[:-SPEED_LAG]
    speed = (y[SPEED_LAG:] - y[:-SPEED_LAG]) / np.where(dt > EPS, dt, np.nan) / ref
    if not np.isfinite(speed).any():
        return 0.0, float("nan")
    i = int(np.nanargmax(speed))
    return max(0.0, float(speed[i])), float(ts[SPEED_LAG + i])


def person_features(track: Track | None, window_s: float) -> np.ndarray | None:
    """Vektor ciri (urutan PERSON_FEATURE_NAMES); None bila riwayat terlalu pendek."""
    if track is None or not track.samples:
        return None
    t_end = track.samples[-1].t
    samples = [s for s in track.samples if t_end - s.t <= window_s + EPS]
    ref = track.ref_size
    if len(samples) < MIN_SAMPLES or ref <= 0:
        return None

    ts = np.array([s.t for s in samples])
    kp = np.stack([s.kp for s in samples]).astype(float)
    conf = np.stack([s.conf for s in samples]).astype(float)
    vis = conf >= MIN_CONF
    sizes = np.array([s.size for s in samples])
    centers = np.stack([s.center for s in samples])

    sh, hip = _mid(kp, vis, SHOULDERS), _mid(kp, vis, HIPS)
    ankle, knee = _mid(kp, vis, ANKLES), _mid(kp, vis, KNEES)
    head = np.where(vis[:, NOSE, None], kp[:, NOSE, :], sh)
    hip_y, head_y = hip[:, 1], head[:, 1]

    hip_now, head_now = _last(hip_y, ts), _last(head_y, ts)
    with np.errstate(all="ignore"):
        hip_drop = (hip_now - np.nanmin(hip_y)) / ref if np.isfinite(hip_y).any() else float("nan")
        head_drop = (head_now - np.nanmin(head_y)) / ref if np.isfinite(head_y).any() else float("nan")
    hip_speed, hip_speed_t = _drop_speed(hip_y, ts, ref)
    head_speed, _ = _drop_speed(head_y, ts, ref)

    d = sh - hip
    angle = np.degrees(np.arctan2(np.abs(d[:, 0]), -d[:, 1])) / 90.0
    angle_now = _last(angle, ts)
    finite_angle = angle[np.isfinite(angle)]
    angle_max = float(finite_angle.max()) if len(finite_angle) else float("nan")
    angle_change = angle_now - float(finite_angle[0]) if len(finite_angle) else float("nan")

    masked = np.where(vis[:, :, None], kp, np.nan)
    with np.errstate(all="ignore"):
        width = np.nanmax(masked[:, :, 0], axis=1) - np.nanmin(masked[:, :, 0], axis=1)
        height = np.nanmax(masked[:, :, 1], axis=1) - np.nanmin(masked[:, :, 1], axis=1)
    h_now, w_now = _last(height, ts), _last(width, ts)
    box_aspect = w_now / h_now if h_now > 0 else float("nan")
    h_max = float(np.nanmax(height)) if np.isfinite(height).any() else float("nan")
    box_height_ratio = h_now / h_max if h_max and h_max > 0 else float("nan")

    recent = ts[-1] - ts[:-1] <= RECENT_S + EPS
    dt = np.diff(ts)
    step_ok = recent & (dt > EPS)
    still = float((np.linalg.norm(np.diff(centers, axis=0), axis=1)[step_ok] / dt[step_ok] / ref).mean()) \
        if step_ok.any() else 0.0

    conf_mean = conf.mean(axis=1)
    knee_straight = float("nan")
    for h_i, k_i, a_i in zip(HIPS, KNEES, ANKLES):
        if vis[-1, h_i] and vis[-1, k_i] and vis[-1, a_i]:
            u, v = kp[-1, h_i] - kp[-1, k_i], kp[-1, a_i] - kp[-1, k_i]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu > 0 and nv > 0:
                value = float(np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1.0, 1.0)))) / 180.0
                knee_straight = value if np.isnan(knee_straight) else max(knee_straight, value)

    out = np.array([
        hip_drop, hip_speed, head_drop, head_speed, angle_now, angle_max, angle_change,
        _last(sizes, ts) / ref, (hip_now - head_now) / ref, box_aspect, box_height_ratio,
        (_last(ankle[:, 1], ts) - hip_now) / ref, still, float(conf_mean[-1]),
        float(conf_mean[-1] - conf_mean.max()), knee_straight,
        t_end - hip_speed_t,
    ], dtype=np.float32)
    out[np.isinf(out)] = np.nan
    return out
