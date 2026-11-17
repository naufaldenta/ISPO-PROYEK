"""Riwayat gerak per orang: sampel bercap waktu, mode pandang, dan ukuran badan."""
from __future__ import annotations

import dataclasses
from collections import deque

import numpy as np

from .config import Config
from .types import ANKLES, ELBOWS, HIPS, KNEES, NOSE, SHOULDERS, WRISTS, Person, Sample, ViewMode

EPS = 1e-6
MIN_SIZE_PX = 8.0
GHOST_MIN_CONF = 0.3         # titik yang dibandingkan saat mencari deteksi dobel
GHOST_MIN_POINTS = 4
SHOULDER_TO_TORSO = 1.4      # perkiraan panjang badan dari lebar bahu bila pinggul tak terlihat


def _any(conf: np.ndarray, idxs, min_conf: float) -> bool:
    return any(conf[i] >= min_conf for i in idxs)


def _mid(kp: np.ndarray, conf: np.ndarray, idxs, min_conf: float) -> np.ndarray | None:
    pts = [kp[i] for i in idxs if conf[i] >= min_conf]
    return np.mean(pts, axis=0) if pts else None


def raw_mode(conf: np.ndarray, min_conf: float) -> ViewMode:
    """Mode pandang dari satu frame, sebelum histeresis."""
    if not _any(conf, SHOULDERS, min_conf):
        return "none"
    if _any(conf, HIPS, min_conf) and _any(conf, KNEES, min_conf) and _any(conf, ANKLES, min_conf):
        return "full"
    if _any(conf, ELBOWS, min_conf) and _any(conf, WRISTS, min_conf):
        return "upper"
    return "none"


def body_size(kp: np.ndarray, conf: np.ndarray, min_conf: float) -> float | None:
    """Panjang badan (tengah bahu ke tengah pinggul) dalam piksel; None bila tak bisa diukur."""
    sh = _mid(kp, conf, SHOULDERS, min_conf)
    if sh is None:
        return None
    hip = _mid(kp, conf, HIPS, min_conf)
    if hip is not None:
        size = float(np.linalg.norm(sh - hip))
    elif all(conf[i] >= min_conf for i in SHOULDERS):
        size = float(np.linalg.norm(kp[SHOULDERS[0]] - kp[SHOULDERS[1]])) * SHOULDER_TO_TORSO
    else:
        return None
    if not np.isfinite(size) or size < MIN_SIZE_PX:
        return None
    return size


def make_sample(t: float, kp: np.ndarray, conf: np.ndarray, size: float, min_conf: float) -> Sample:
    """Sampel beserta tengah badan dan ruas kepala-pinggul, dihitung sekali di sini."""
    sh = _mid(kp, conf, SHOULDERS, min_conf)
    hip = _mid(kp, conf, HIPS, min_conf)
    if sh is None:
        sh = np.mean(kp, axis=0)
    center = (sh + hip) / 2.0 if hip is not None else sh
    top = kp[NOSE] if conf[NOSE] >= min_conf else sh
    bottom = hip if hip is not None else sh + np.array([0.0, size])
    return Sample(t, kp, conf, size, np.asarray(center, dtype=float), np.asarray(top, dtype=float),
                  np.asarray(bottom, dtype=float))


def torso_center(s: Sample, min_conf: float = 0.5) -> np.ndarray:
    """Tengah badan: antara bahu dan pinggul; tengah bahu bila pinggul tak terlihat."""
    if s.center is not None:
        return s.center
    sh = _mid(s.kp, s.conf, SHOULDERS, min_conf)
    hip = _mid(s.kp, s.conf, HIPS, min_conf)
    if sh is not None and hip is not None:
        return (sh + hip) / 2.0
    if sh is not None:
        return sh
    return np.mean(s.kp, axis=0)


class Track:
    """Riwayat satu orang selama nomornya bertahan."""

    def __init__(self, track_id: int, t: float) -> None:
        self.track_id = track_id
        self.samples: deque[Sample] = deque()
        self.mode: ViewMode = "none"
        self.last_seen = t
        self._candidate: ViewMode = "none"
        self._candidate_since = t
        self._ref_size: float | None = None
        self._ref_key: tuple[int, float] | None = None

    @property
    def ref_size(self) -> float:
        """Ukuran badan acuan: persentil 80 dari riwayat, supaya tahan saat badan memendek di gambar.

        Dipanggil berkali-kali per putaran, jadi hasilnya disimpan sampai riwayat berubah.
        """
        if not self.samples:
            return 0.0
        key = (len(self.samples), self.samples[-1].t)
        if key != self._ref_key:
            self._ref_size = float(np.percentile([s.size for s in self.samples], 80))
            self._ref_key = key
        return self._ref_size

    def _observe_mode(self, t: float, raw: ViewMode, hysteresis_s: float) -> None:
        if raw != self._candidate:
            self._candidate, self._candidate_since = raw, t
        if self._candidate != self.mode and t - self._candidate_since >= hysteresis_s - EPS:
            self.mode = self._candidate


class History:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.tracks: dict[int, Track] = {}
        self.overflow_ids: list[int] = []
        self._alias: dict[int, int] = {}            # nomor pelacak baru -> nomor riwayat yang diteruskannya
        self._alias_seen: dict[int, float] = {}

    def update(self, t: float, people: list[Person]) -> list[Person]:
        """Mencatat satu frame. Mengembalikan orang-orangnya dengan nomor riwayat, tanpa deteksi dobel."""
        cfg = self.cfg
        people = self._stitch(t, self._drop_ghosts(self._apply_aliases(t, people)))
        ranked = sorted(people, key=lambda p: (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]), reverse=True)
        assessed, extra = ranked[:cfg.max_people], ranked[cfg.max_people:]
        self.overflow_ids = [p.track_id for p in extra]
        for track_id in self.overflow_ids:
            self.tracks.pop(track_id, None)

        for p in assessed:
            track = self.tracks.get(p.track_id)
            if track is None:
                track = self.tracks[p.track_id] = Track(p.track_id, t)
            size = body_size(p.kp, p.conf, cfg.kp_min_conf)
            raw = raw_mode(p.conf, cfg.kp_min_conf) if size is not None else "none"
            track._observe_mode(t, raw, cfg.mode_hysteresis_s)
            if size is not None:
                track.samples.append(make_sample(t, p.kp, p.conf, size, cfg.kp_min_conf))
            track.last_seen = t

        for track_id in list(self.tracks):
            track = self.tracks[track_id]
            if t - track.last_seen > cfg.track_ttl_s + EPS:
                del self.tracks[track_id]
                continue
            while track.samples and t - track.samples[0].t > cfg.history_s + EPS:
                track.samples.popleft()
        for new_id in [i for i, seen in self._alias_seen.items()
                       if t - seen > cfg.track_ttl_s + EPS or self._alias[i] not in self.tracks]:
            del self._alias[new_id], self._alias_seen[new_id]
        return people

    # ---- nomor pelacak -> nomor riwayat ------------------------------------------

    def _apply_aliases(self, t: float, people: list[Person]) -> list[Person]:
        out, used = [], {p.track_id for p in people if p.track_id not in self._alias}
        for p in people:
            target = self._alias.get(p.track_id)
            if target is not None and target not in used:
                used.add(target)
                self._alias_seen[p.track_id] = t
                p = dataclasses.replace(p, track_id=target)
            elif target is not None:                 # nomor lamanya muncul lagi bersamaan: mereka dua orang
                del self._alias[p.track_id], self._alias_seen[p.track_id]
            out.append(p)
        return out

    def _drop_ghosts(self, people: list[Person]) -> list[Person]:
        """Membuang deteksi dobel: dua kerangka yang nyaris berimpit adalah satu orang."""
        dropped: set[int] = set()
        for i, a in enumerate(people):
            for j in range(i + 1, len(people)):
                b = people[j]
                if i in dropped or j in dropped or not self._same_body(a, b):
                    continue
                dropped.add(j if self._keep_rank(a) >= self._keep_rank(b) else i)
        return [p for i, p in enumerate(people) if i not in dropped]

    def _keep_rank(self, p: Person) -> tuple[int, float]:
        track = self.tracks.get(p.track_id)
        return (len(track.samples) if track is not None else -1, float(np.mean(p.conf)))

    def _same_body(self, a: Person, b: Person) -> bool:
        both = (a.conf >= GHOST_MIN_CONF) & (b.conf >= GHOST_MIN_CONF)
        if int(both.sum()) < GHOST_MIN_POINTS:
            return False
        scale = max(a.box[2] - a.box[0], a.box[3] - a.box[1], b.box[2] - b.box[0], b.box[3] - b.box[1])
        if scale <= 0:
            return False
        gap = float(np.median(np.linalg.norm(a.kp[both] - b.kp[both], axis=1)))
        return gap / scale < self.cfg.ghost_dist

    def _stitch(self, t: float, people: list[Person]) -> list[Person]:
        """Nomor yang belum dikenal meneruskan riwayat orang terdekat yang baru saja hilang."""
        cfg = self.cfg
        present = {p.track_id for p in people}
        lost = {i: tr for i, tr in self.tracks.items()
                if i not in present and tr.samples and t - tr.last_seen <= cfg.stitch_s + EPS}
        out = []
        for p in people:
            if p.track_id not in self.tracks and lost:
                center = self._center(p)
                best = min(lost, key=lambda i: float(np.linalg.norm(lost[i].samples[-1].center - center)))
                track = lost[best]
                reach = cfg.stitch_dist * track.ref_size
                if float(np.linalg.norm(track.samples[-1].center - center)) <= reach:
                    self._alias[p.track_id], self._alias_seen[p.track_id] = best, t
                    del lost[best]
                    p = dataclasses.replace(p, track_id=best)
            out.append(p)
        return out

    def _center(self, p: Person) -> np.ndarray:
        sh = _mid(p.kp, p.conf, SHOULDERS, self.cfg.kp_min_conf)
        hip = _mid(p.kp, p.conf, HIPS, self.cfg.kp_min_conf)
        if sh is not None and hip is not None:
            return (sh + hip) / 2.0
        if sh is not None or hip is not None:
            return sh if sh is not None else hip
        return np.array([(p.box[0] + p.box[2]) / 2.0, (p.box[1] + p.box[3]) / 2.0])
