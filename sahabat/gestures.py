"""Pengenal gerak per pasangan, memakai titik tubuh yang juga dipakai model."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import cos, hypot, pi
from typing import Literal

from .config import Config
from .history import EPS, Track, _mid
from .pair_features import close_pairs
from .types import ANKLES, ELBOWS, HIPS, KNEES, NOSE, SHOULDERS, WRISTS, Sample

GestureKind = Literal["pukulan", "tendangan", "kuncian"]
_ARMS = tuple(zip(SHOULDERS, ELBOWS, WRISTS))
_LEGS = tuple(zip(HIPS, KNEES, ANKLES))


@dataclass(frozen=True)
class GestureHit:
    kind: GestureKind
    ids: tuple[int, int]
    actor: int


@dataclass
class _Hold:
    start: float
    last: float
    fired: bool = False


def _distance(a, b) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _segment_distance(p, start, end) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denom = dx * dx + dy * dy
    if denom <= EPS:
        return _distance(p, start)
    along = ((p[0] - start[0]) * dx + (p[1] - start[1]) * dy) / denom
    k = min(1.0, max(0.0, along))
    return hypot(p[0] - start[0] - k * dx, p[1] - start[1] - k * dy)


def _straight(a, joint, end, cutoff: float) -> bool:
    ax, ay = joint[0] - a[0], joint[1] - a[1]
    bx, by = end[0] - joint[0], end[1] - joint[1]
    norm = hypot(ax, ay) * hypot(bx, by)
    return norm > EPS and ax * bx + ay * by >= cutoff * norm


def _previous(track: Track, t: float, window: float) -> Sample | None:
    # Jarak waktu cukup untuk melihat ekstensi, bukan perubahan satu titik antargambar.
    for s in reversed(track.samples):
        age = t - s.t
        if age > window + EPS:
            break
        if age >= 0.19:
            return s
    return None


class GestureDetector:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._strike_cos = cos((1.0 - cfg.strike_straight) * pi)
        self._hold_cos = cos((1.0 - cfg.hold_straight) * pi)
        self.reset()

    def reset(self) -> None:
        self._last_tick: float | None = None
        self._holds: dict[tuple[int, int, str], _Hold] = {}
        self._last_strike: dict[tuple[int, int], float] = {}
        self._strike_history: dict[tuple[int, int], deque[float]] = {}

    def tick(self, t: float, tracks: dict[int, Track]) -> list[GestureHit]:
        cfg = self.cfg
        if not cfg.gesture_enabled:
            return []
        interval = t - self._last_tick if self._last_tick is not None else 0.0
        self._last_tick = t
        gap = max(cfg.hold_gap_s, min(0.35, 1.15 * interval))
        self._holds = {k: v for k, v in self._holds.items() if t - v.last <= gap + EPS}
        self._last_strike = {k: v for k, v in self._last_strike.items() if t - v <= cfg.strike_window_s}
        self._strike_history = {k: v for k, v in self._strike_history.items()
                                if t - v[-1] <= cfg.strike_confirm_s}
        fresh = {i: tr for i, tr in tracks.items() if tr.samples and t - tr.samples[-1].t <= EPS}
        if len(fresh) < 2:
            return []
        prev = {i: _previous(tr, t, cfg.strike_window_s) for i, tr in fresh.items()}
        strikes = {}
        hits = []
        for left, right in close_pairs(fresh, cfg.pair_max_dist):
            ids = (left.track_id, right.track_id)
            for a, b in ((left, right), (right, left)):
                sa, sb = a.samples[-1], b.samples[-1]
                scale = (a.ref_size + b.ref_size) / 2
                for zone in self._hold_zones(sa, sb, scale):
                    key = (a.track_id, b.track_id, zone)
                    hold = self._holds.setdefault(key, _Hold(t, t))
                    hold.last = t
                    if not hold.fired and t - hold.start >= cfg.hold_s - EPS:
                        hold.fired = True
                        if not any(hit.ids == ids for hit in hits):
                            hits.append(GestureHit("kuncian", ids, a.track_id))
                pa, pb = prev[a.track_id], prev[b.track_id]
                if pa is None or pb is None or abs(pa.t - pb.t) > 0.05:
                    continue
                for limb, kind, distance in self._strikes(sa, sb, pa, pb, scale, a.mode == "full"):
                    key = (a.track_id, limb)
                    if key not in self._last_strike and (key not in strikes or distance < strikes[key][0]):
                        strikes[key] = (distance, GestureHit(kind, ids, a.track_id))
        for key, (_, hit) in strikes.items():
            self._last_strike[key] = t
            history = self._strike_history.setdefault(hit.ids, deque())
            while history and t - history[0] > cfg.strike_confirm_s:
                history.popleft()
            if not history or t - history[-1] >= 0.25:
                history.append(t)
            if len(history) >= cfg.strike_confirm:
                del self._strike_history[hit.ids]
                hits.append(hit)
        return hits

    def _strikes(self, a, b, prev_a, prev_b, scale, full):
        cfg = self.cfg
        confidence = cfg.kp_min_conf
        dt = a.t - prev_a.t
        groups = ((_ARMS, "pukulan"), (_LEGS, "tendangan")) if full else ((_ARMS, "pukulan"),)
        for limbs, kind in groups:
            for base, joint, tip in limbs:
                if min(a.conf[base], a.conf[joint], a.conf[tip], prev_a.conf[base],
                       prev_a.conf[tip]) < confidence:
                    continue
                p, q = a.kp[tip], prev_a.kp[tip]
                extension = _distance(p, a.kp[base]) - _distance(q, prev_a.kp[base])
                if extension < cfg.strike_reach_gain * scale:
                    continue
                if not _straight(a.kp[base], a.kp[joint], p, self._strike_cos):
                    continue
                d = _segment_distance(p, b.top, b.bottom)
                if d > cfg.strike_reach * scale or p[1] < b.top[1] - 0.05 * scale:
                    continue
                # B yang berjalan mendekat sendiri tidak cukup: ujung A juga harus melesat ke arah B.
                # Kamera lambat bisa mengambil sebagian waktu sebelum atau sesudah
                # ayunan. Dalam selang yang lebih panjang, ekstensi dan kelurusan
                # tetap wajib, tetapi jangan membagi jarak dengan seluruh jeda.
                travel = min(_segment_distance(q, prev_b.top, prev_b.bottom),
                             _segment_distance(q, b.top, b.bottom)) - d
                clear_swing = dt <= 0.36 and travel >= cfg.strike_speed * scale * dt
                # Antargambar yang renggang, beri kelonggaran bila ayunan
                # jelas berhenti sebelum badan. Tumpang tindih titik 2D sering
                # hanya lengan orang mengobrol atau titik berpindah orang.
                clear_air_swing = (0.22 < dt <= 0.36 and d >= 0.32 * scale
                                   and travel >= cfg.strike_speed * scale * 0.22)
                if not (clear_swing or clear_air_swing):
                    continue
                if kind == "tendangan":
                    knees = [b.kp[i, 1] for i in KNEES if b.conf[i] >= confidence]
                    if p[1] >= (min(knees) if knees else b.bottom[1]):
                        continue
                elif any(b.conf[i] >= confidence and _distance(p, b.kp[i]) <= cfg.strike_hand_clear * scale
                         for i in WRISTS):
                    continue
                yield tip, kind, d

    def _hold_zones(self, a: Sample, b: Sample, scale: float):
        cfg = self.cfg
        confidence = cfg.kp_min_conf
        sh = _mid(b.kp, b.conf, SHOULDERS, confidence)
        if sh is None:
            return
        neck = (b.kp[NOSE] + sh) / 2 if b.conf[NOSE] >= confidence else (sh[0], sh[1] - 0.15 * scale)
        arms = [(base, joint, tip) for base, joint, tip in _ARMS
                if min(a.conf[base], a.conf[joint], a.conf[tip]) >= confidence]
        # Tangan di badan orang yang berimpit dengan kedua lengan menekuk menyerupai pelukan.
        straight = {_arm[2] for _arm in arms if _straight(*(a.kp[i] for i in _arm), self._hold_cos)}
        if not straight and _distance(a.center, b.center) < 0.65 * scale:
            return
        for _, _, tip in arms:
            if _distance(a.kp[tip], neck) <= cfg.hold_neck_dist * scale:
                yield "leher"
                break
        for tip in straight:
            p = a.kp[tip]
            if (0.18 * scale <= p[1] - sh[1] and p[1] <= b.bottom[1]
                    and _segment_distance(p, sh, b.bottom) <= cfg.hold_center_frac * scale):
                yield "badan"
                break
