"""Pasangan orang yang berdekatan dan ringkasan gerak berarah "A terhadap B" selama ~1 detik.

Dipakai sama persis oleh penilaian langsung dan oleh pelatihan, supaya keduanya tidak menyimpang.
Semua jarak dalam ukuran badan; semua kecepatan dalam ukuran badan per detik.

Hitungan dilakukan sekaligus untuk seluruh jendela (larik numpy), bukan per sampel: dengan empat
orang ada enam pasangan dan dua arah, dan semuanya harus selesai jauh di bawah 100 ms.
"""
from __future__ import annotations

import numpy as np

from .history import EPS, Track, make_sample, torso_center
from .types import ANKLES, ELBOWS, HIPS, KNEES, SHOULDERS, WRISTS, Sample

FEATURE_NAMES: tuple[str, ...] = (
    "wrist_toward_max",    # laju tertinggi pergelangan tangan A mendekati badan B
    "wrist_dist_min",      # jarak terdekat pergelangan tangan A ke badan B
    "wrist_speed_max",     # laju tertinggi pergelangan tangan A (arah bebas)
    "wrist_speed_std",
    "elbow_straight_max",  # sudut siku terbesar A, 0..1 (1 = lurus)
    "ankle_toward_max",    # seperti wrist_toward_max untuk pergelangan kaki; 0 bila A bukan full
    "ankle_dist_min",      # 9.0 bila A bukan full
    "knee_straight_max",   # 0 bila A bukan full
    "closing_speed_max",   # laju kedua tengah badan saling mendekat
    "b_away_speed_max",    # laju tengah badan B menjauh dari A
    "b_accel_max",         # percepatan puncak tengah badan B
    "torso_dist_mean",
)
MOTION_FEATURES = tuple(FEATURE_NAMES.index(n) for n in
                        ("wrist_speed_max", "ankle_toward_max", "b_away_speed_max"))

# Deret per langkah (10 langkah = 1 detik): bentuk gerak dari waktu ke waktu, bukan hanya nilai puncaknya.
SEQ_TICKS = 10
TICK_S = 0.1
SERIES_NAMES: tuple[str, ...] = (
    "wrist_l", "wrist_r",    # jarak tangan kiri/kanan A ke ruas badan B
    "ankle_l", "ankle_r",    # jarak kaki kiri/kanan A ke ruas badan B
    "torso",                 # jarak kedua tengah badan
    "b_head_shift",          # seberapa jauh kepala B bergeser sejak awal jendela (tersentak)
    "b_torso_shift",         # seberapa jauh badan B bergeser sejak awal jendela (terdorong)
)
WINDOW_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + tuple(
    f"{name}_t{k}" for name in SERIES_NAMES for k in range(SEQ_TICKS))

FAR = 9.0                  # nilai "jauh sekali" bila jarak tak bisa diukur
MIN_CONF = 0.5
MATCH_TOL_S = 0.05
MIN_SHARED = 5
STALE_S = 0.25

_LEG_ZERO = tuple(FEATURE_NAMES.index(n) for n in ("ankle_toward_max", "knee_straight_max"))
_LEG_FAR = FEATURE_NAMES.index("ankle_dist_min")
_LEG_SERIES = tuple(WINDOW_FEATURE_NAMES.index(f"{n}_t{k}") for n in ("ankle_l", "ankle_r") for k in range(SEQ_TICKS))


def tanpa_kaki(rows: np.ndarray) -> np.ndarray:
    """Baris fitur seolah kaki tidak terlihat (kamera dekat), untuk orang yang kakinya terpotong maupun utuh.

    Deret kaki diisi FAR, bukan NaN: kolom yang seluruhnya kosong tidak bisa dilatih gradient boosting.
    """
    out = np.array(rows, dtype=np.float32, copy=True)
    out[..., list(_LEG_ZERO)] = 0.0
    out[..., _LEG_FAR] = FAR
    out[..., list(_LEG_SERIES)] = FAR
    return out


def close_pairs(tracks: dict[int, Track], max_dist: float) -> list[tuple[Track, Track]]:
    """Pasangan bernomor naik yang keduanya bisa dinilai dan tengah badannya berdekatan."""
    usable = [tr for _, tr in sorted(tracks.items()) if tr.mode in ("full", "upper") and tr.samples]
    sizes = [tr.ref_size for tr in usable]
    pairs = []
    for i, a in enumerate(usable):
        for j in range(i + 1, len(usable)):
            b = usable[j]
            sa, sb = a.samples[-1], b.samples[-1]
            if abs(sa.t - sb.t) > STALE_S:
                continue
            scale = (sizes[i] + sizes[j]) / 2.0
            if scale <= 0:
                continue
            if np.linalg.norm(torso_center(sa) - torso_center(sb)) < max_dist * scale:
                pairs.append((a, b))
    return pairs


def _shared(a: Track, b: Track, window_s: float) -> list[tuple[Sample, Sample]]:
    out, j, sb = [], 0, list(b.samples)
    for sa in a.samples:
        while j < len(sb) and sb[j].t < sa.t - MATCH_TOL_S:
            j += 1
        if j < len(sb) and abs(sb[j].t - sa.t) <= MATCH_TOL_S:
            out.append((sa, sb[j]))
    if not out:
        return out
    t_end = out[-1][0].t
    return [(x, y) for x, y in out if t_end - x.t <= window_s + EPS]


def _derived(s: Sample) -> Sample:
    """Sampel yang dibuat tanpa make_sample (misalnya di uji) dilengkapi dulu."""
    return s if s.top is not None else make_sample(s.t, s.kp, s.conf, s.size, MIN_CONF)


def _limb(kp: np.ndarray, conf: np.ndarray, top: np.ndarray, bottom: np.ndarray, dt: np.ndarray,
          limb_idxs, scale: float) -> tuple[float, float, float, float]:
    """(laju mendekat tertinggi, jarak terdekat, laju tertinggi, simpangan baku laju) satu jenis anggota badan.

    kp (T,17,2), conf (T,17): milik A.  top, bottom (T,2): ruas kepala-pinggul B.  dt (T-1,).
    """
    axis = bottom - top
    denom = np.einsum("ij,ij->i", axis, axis)
    safe = np.where(denom > 0, denom, 1.0)
    step_ok = dt > EPS
    toward, dist_min, speeds = 0.0, FAR, []
    for idx in limb_idxs:
        valid = conf[:, idx] >= MIN_CONF
        if not valid.any():
            continue
        p = kp[:, idx, :].astype(float)
        k = np.clip(np.einsum("ij,ij->i", p - top, axis) / safe, 0.0, 1.0)
        k[denom <= 0] = 0.0
        d = np.linalg.norm(p - (top + k[:, None] * axis), axis=1) / scale
        dist_min = min(dist_min, float(d[valid].min()))
        ok = valid[1:] & valid[:-1] & step_ok            # laju hanya antar dua sampel berurutan yang terlihat
        if ok.any():
            toward = max(toward, float(((d[:-1] - d[1:])[ok] / dt[ok]).max()))
            speeds.append(np.linalg.norm(p[1:] - p[:-1], axis=1)[ok] / scale / dt[ok])
    if not speeds:
        return 0.0, dist_min, 0.0, 0.0
    all_speeds = np.concatenate(speeds)
    return toward, dist_min, float(all_speeds.max()), float(all_speeds.std())


def _straightness(kp: np.ndarray, conf: np.ndarray, joints) -> float:
    """Sudut sendi terbesar (0..1, 1 = lurus) dari daftar (pangkal, sendi, ujung); 0 bila tak terukur."""
    best = 0.0
    for a, mid, c in joints:
        u, v = kp[:, a] - kp[:, mid], kp[:, c] - kp[:, mid]
        nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        ok = (conf[:, a] >= MIN_CONF) & (conf[:, mid] >= MIN_CONF) & (conf[:, c] >= MIN_CONF) & (nu > 0) & (nv > 0)
        if ok.any():
            cos = np.clip(np.einsum("ij,ij->i", u[ok], v[ok]) / (nu[ok] * nv[ok]), -1.0, 1.0)
            best = max(best, float(np.degrees(np.arccos(cos)).max()) / 180.0)
    return best


_ARMS = tuple((SHOULDERS[i], ELBOWS[i], WRISTS[i]) for i in (0, 1))
_LEGS = tuple((HIPS[i], KNEES[i], ANKLES[i]) for i in (0, 1))


def _stack(samples: list[Sample]):
    return (np.stack([s.kp for s in samples]).astype(float), np.stack([s.conf for s in samples]),
            np.stack([s.top for s in samples]), np.stack([s.bottom for s in samples]),
            np.stack([s.center for s in samples]))


def _directed(kp, conf, full: bool, top_b, bottom_b, ca, cb, dists, dt, scale: float) -> np.ndarray:
    """Fitur A-terhadap-B dari larik yang sudah disusun."""
    w_toward, w_dist, w_speed, w_std = _limb(kp, conf, top_b, bottom_b, dt, WRISTS, scale)
    elbow = _straightness(kp, conf, _ARMS)
    if full:
        a_toward, a_dist, _, _ = _limb(kp, conf, top_b, bottom_b, dt, ANKLES, scale)
        knee = _straightness(kp, conf, _LEGS)
    else:
        a_toward, a_dist, knee = 0.0, FAR, 0.0

    closing, away, accel = 0.0, 0.0, 0.0
    ok = dt > EPS
    if ok.any():
        closing = max(0.0, float(((dists[:-1] - dists[1:])[ok] / dt[ok]).max()))
        vb = (cb[1:] - cb[:-1])[ok] / scale / dt[ok][:, None]
        gap = (cb[:-1] - ca[:-1])[ok]
        norm = np.linalg.norm(gap, axis=1)
        has_gap = norm > 0
        if has_gap.any():
            away = max(0.0, float((np.einsum("ij,ij->i", vb[has_gap], gap[has_gap]) / norm[has_gap]).max()))
        if len(vb) > 1:
            accel = float((np.linalg.norm(vb[1:] - vb[:-1], axis=1) / dt[ok][1:]).max())

    out = np.array([w_toward, w_dist, w_speed, w_std, elbow, a_toward, a_dist, knee,
                    closing, away, accel, float(dists.mean())], dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=FAR, neginf=0.0)


def both_directions(a: Track, b: Track, window_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    """(fitur A-terhadap-B, fitur B-terhadap-A); None bila sampel bersama kurang.

    Larik jendela disusun sekali lalu dipakai untuk kedua arah.
    """
    shared = _shared(a, b, window_s)
    scale = (a.ref_size + b.ref_size) / 2.0
    if len(shared) < MIN_SHARED or scale <= 0:
        return None
    sa = [_derived(x) for x, _ in shared]
    sb = [_derived(y) for _, y in shared]
    dt = np.diff(np.array([s.t for s in sa]))
    kp_a, conf_a, top_a, bottom_a, ca = _stack(sa)
    kp_b, conf_b, top_b, bottom_b, cb = _stack(sb)
    dists = np.linalg.norm(ca - cb, axis=1) / scale
    return (_directed(kp_a, conf_a, a.mode == "full", top_b, bottom_b, ca, cb, dists, dt, scale),
            _directed(kp_b, conf_b, b.mode == "full", top_a, bottom_a, cb, ca, dists, dt, scale))


def directed_features(a: Track, b: Track, window_s: float) -> np.ndarray | None:
    """Vektor fitur A-terhadap-B (urutan FEATURE_NAMES); None bila sampel bersama kurang."""
    both = both_directions(a, b, window_s)
    return None if both is None else both[0]


_SERIES_LIMBS = (WRISTS[0], WRISTS[1], ANKLES[0], ANKLES[1])
_SERIES_SPAN_S = (SEQ_TICKS - 1) * TICK_S + 0.02


_BRIDGE_S = 0.3            # lubang antar frame sampai sepanjang ini dijembatani; lebih panjang dibiarkan kosong


def _series(kp_a, conf_a, top_b, bottom_b, ca, cb, scale: float, base: int) -> np.ndarray:
    """Deret jarak A-terhadap-B pada tiap sampel, (7, jumlah sampel); sendi tak terlihat berisi NaN."""
    out = np.empty((len(SERIES_NAMES), len(kp_a)), dtype=np.float64)
    axis = bottom_b - top_b
    denom = np.einsum("ij,ij->i", axis, axis)
    safe = np.where(denom > 0, denom, 1.0)
    for row, idx in enumerate(_SERIES_LIMBS):
        p = kp_a[:, idx, :]
        k = np.clip(np.einsum("ij,ij->i", p - top_b, axis) / safe, 0.0, 1.0)
        d = np.linalg.norm(p - (top_b + k[:, None] * axis), axis=1) / scale
        d[conf_a[:, idx] < MIN_CONF] = np.nan
        out[row] = d
    out[4] = np.linalg.norm(ca - cb, axis=1) / scale
    out[5] = np.linalg.norm(top_b - top_b[base], axis=1) / scale     # pergeseran sejak awal deret
    out[6] = np.linalg.norm(cb - cb[base], axis=1) / scale
    return out


def _on_ticks(values: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Menaruh deret (7, sampel) ke SEQ_TICKS langkah tetap berjarak TICK_S yang berakhir di sampel terakhir.

    Kamera tidak selalu 10 gambar per detik (webcam di ruangan redup bisa 7,5). Langkah yang punya sampel
    memakai sampel itu apa adanya; langkah di antara dua sampel diisi dengan garis lurus di antaranya.
    """
    out = np.full((values.shape[0], SEQ_TICKS), np.nan, dtype=np.float32)
    t_end = ts[-1]
    for k in range(SEQ_TICKS):
        tick_t = t_end - (SEQ_TICKS - 1 - k) * TICK_S
        nearest = int(np.argmin(np.abs(ts - tick_t)))
        if abs(ts[nearest] - tick_t) <= TICK_S / 2 + EPS:
            out[:, k] = values[:, nearest]
            continue
        right = int(np.searchsorted(ts, tick_t))
        left = right - 1
        if left < 0 or right >= len(ts) or ts[right] - ts[left] > _BRIDGE_S:
            continue
        w = (tick_t - ts[left]) / (ts[right] - ts[left])
        out[:, k] = values[:, left] * (1.0 - w) + values[:, right] * w
    return out.reshape(-1)


def pair_windows(a: Track, b: Track, window_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    """(vektor A-terhadap-B, vektor B-terhadap-A), masing-masing urutan WINDOW_FEATURE_NAMES.

    Isinya ringkasan (sama dengan both_directions) disambung deret jarak per langkah. None bila sampel
    bersama kurang. Deret memakai cap waktu sebenarnya: langkah terakhir selalu frame terbaru.
    """
    shared = _shared(a, b, window_s)
    scale = (a.ref_size + b.ref_size) / 2.0
    if len(shared) < MIN_SHARED or scale <= 0:
        return None
    sa = [_derived(x) for x, _ in shared]
    sb = [_derived(y) for _, y in shared]
    ts = np.array([s.t for s in sa])
    kp_a, conf_a, top_a, bottom_a, ca = _stack(sa)
    kp_b, conf_b, top_b, bottom_b, cb = _stack(sb)
    dt = np.diff(ts)
    dists = np.linalg.norm(ca - cb, axis=1) / scale
    summary_ab = _directed(kp_a, conf_a, a.mode == "full", top_b, bottom_b, ca, cb, dists, dt, scale)
    summary_ba = _directed(kp_b, conf_b, b.mode == "full", top_a, bottom_a, cb, ca, dists, dt, scale)

    pick = np.flatnonzero(ts[-1] - ts <= _SERIES_SPAN_S)     # deret mencakup SEQ_TICKS langkah terakhir
    if len(pick) < MIN_SHARED:
        return None
    base = int(pick[0])        # sampel sebelum itu hanya dipakai untuk menjembatani langkah pertama
    series_ab = _on_ticks(_series(kp_a, conf_a, top_b, bottom_b, ca, cb, scale, base), ts)
    series_ba = _on_ticks(_series(kp_b, conf_b, top_a, bottom_a, cb, ca, scale, base), ts)
    return np.concatenate([summary_ab, series_ab]), np.concatenate([summary_ba, series_ba])
