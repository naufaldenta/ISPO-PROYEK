"""Penilai jatuh berbasis aturan. Tidak perlu pelatihan.

Seluruh badan: tegak -> rebah dengan pinggul turun cepat -> tetap rendah sebentar.
Setengah badan: bahu turun cepat di bagian bawah gambar lalu orangnya hilang.
Turun ke bawah: kamera tinggi, bahu melesat ke seperlima bawah gambar lalu tidak naik lagi (hilang dari bawah gambar
atau hanya bahu yang tersisa).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .history import EPS, Track, _mid, raw_mode
from .person_features import PERSON_FEATURE_NAMES, person_features
from .types import ANKLES, HIPS, NOSE, SHOULDERS, WRISTS, Sample

SPEED_SPAN_S = 0.25      # kecepatan diukur pada rentang ini supaya tahan getar titik tubuh


@dataclass
class _State:
    name: str            # upright | tilted | dropped | fired | down_unknown
    since: float = 0.0


def _torso_angle(s: Sample, min_conf: float) -> float | None:
    """Sudut badan dari tegak, derajat: 0 berdiri, 90 rebah. None bila pinggul tak terlihat."""
    sh = _mid(s.kp, s.conf, SHOULDERS, min_conf)
    hip = _mid(s.kp, s.conf, HIPS, min_conf)
    if sh is None or hip is None:
        return None
    dx, dy = sh - hip
    return float(np.degrees(np.arctan2(abs(dx), -dy)))


def _peak_down_speed(track: Track, idxs, t: float, window_s: float, min_conf: float) -> float:
    """Kecepatan turun tertinggi titik tengah `idxs` dalam `window_s` terakhir, ukuran badan per detik."""
    pts = []
    for s in track.samples:
        if t - s.t > window_s + EPS:
            continue
        mid = _mid(s.kp, s.conf, idxs, min_conf)
        if mid is not None:
            pts.append((s.t, float(mid[1])))
    size = track.ref_size
    if size <= 0:
        return 0.0
    best = 0.0
    for i, (t1, y1) in enumerate(pts):
        for t0, y0 in pts[:i]:
            dt = t1 - t0
            if SPEED_SPAN_S - EPS <= dt <= 2 * SPEED_SPAN_S + EPS:
                best = max(best, (y1 - y0) / dt / size)
    return best


_HIP_DROP = PERSON_FEATURE_NAMES.index("hip_drop")
_SINCE_DROP = PERSON_FEATURE_NAMES.index("since_drop_s")
REFIRE_S = 5.0            # satu kejadian jatuh hanya memicu sekali, walau aturan dan model sama-sama menangkapnya
LOST_GRACE_S = 1.0        # orang yang hilang tepat setelah tanda jatuh dianggap masih jatuh selama ini
# Turun ke bawah gambar (sesi webcam 16 November 2026, kamera sekitar 2 m): saat jatuh, lengan langsung keluar dari
# gambar sehingga aturan setengah badan tidak pernah aktif. Jongkok di depan kamera turun sama cepatnya tetapi bangun
# lagi dalam kurang dari 1 detik; jatuh tetap di bawah atau hilang 5-10 detik.
LOW_DROP_SPEED = 2.0      # ukuran badan per detik, bahu ke bawah
LOW_STAY_S = 3.0          # bahu tidak naik lagi selama ini
LOW_RISE_FRAC = 0.72      # bahu kembali di atas garis ini (bagian tinggi gambar) = orangnya bangun
# Bahu jatuh sungguhan di rekaman paling cepat sekitar 4 ukuran badan per detik antargambar. Lompatan yang jauh lebih
# cepat adalah pelacak yang menyambung ke orang lain di tepi bawah, bukan jatuh.
LOW_JUMP_SPEED = 8.0
LOW_FROM_S = 1.5          # bahu harus berasal dari atas LOW_RISE_FRAC dalam selang ini (dari berdiri, bukan duduk)
LOW_SPEED_WINDOW_S = 1.0  # bahu sampai di tepi bawah bisa beberapa gambar setelah turunnya yang paling cepat


class FallDetector:
    """Aturan gerak dan penilai terlatih untuk alarm jatuh."""

    def __init__(self, cfg: Config, model_path: Path | None = None) -> None:
        self.cfg = cfg
        self._states: dict[int, _State] = {}
        self._armed: dict[int, float] = {}      # setengah badan: nomor -> cap waktu sampel terakhir yang memenuhi
        self._model = None
        if model_path is not None and Path(model_path).is_file():
            import joblib
            self._model = joblib.load(model_path)
            if hasattr(self._model, "n_jobs"):
                self._model.n_jobs = 1
            expected = getattr(self._model, "n_features_in_", len(PERSON_FEATURE_NAMES))
            if expected != len(PERSON_FEATURE_NAMES):
                print(f"Peringatan: {Path(model_path).name} dilatih dengan {expected} ciri, aplikasi memakai "
                      f"{len(PERSON_FEATURE_NAMES)}. Model diabaikan; latih ulang dengan latih.fall_train.")
                self._model = None
        self._support_threshold = (cfg.fall_rule_model_threshold
                                   if cfg.fall_rule_model_threshold is not None else
                                   getattr(self._model, "fall_rule_support_threshold_", 0.0))
        self._flags: dict[int, deque[bool]] = {}        # model: tanda jatuh per putaran, per orang
        self._rule_support: dict[int, float] = {}       # peluang dari fitur lengkap pada putaran ini
        self._flag_t: dict[int, float] = {}             # kapan orang itu terakhir dinilai model
        # Yang sudah memicu: nomor -> tinggi pinggul (piksel) yang harus dicapai lagi sebelum boleh memicu ulang.
        self._model_fired: dict[int, float] = {}
        self._last_fire: dict[int, float] = {}
        self._last_hip: dict[int, float] = {}
        self._upright: dict[int, float] = {}
        self._edge: dict[int, float] = {}
        self._low: dict[int, float] = {}        # turun ke bawah: nomor -> waktu bahu sampai di tepi bawah
        self._low_done: set[int] = set()        # sudah memicu; menunggu bahunya naik lagi
        self.sources: dict[int, str] = {}

    @property
    def uses_model(self) -> bool:
        return self._model is not None

    def tick(self, t: float, tracks: dict[int, Track], frame_h: int) -> list[int]:
        """Nomor orang yang jatuhnya baru terkonfirmasi pada panggilan ini."""
        self.sources = {}
        # Pakai bukti tegak dari frame sebelum gerakan, bukan pose pada frame pemicu.
        self._upright = {i: when for i, when in self._upright.items()
                         if t - when <= self.cfg.fall_upright_before_s + self.cfg.fall_window_s + EPS}
        full_rule = self._model is None or self.cfg.fall_rule_with_model
        candidates = self._rule_tick(t, tracks, frame_h, full_rule)
        if self._model is not None:
            learned = self._model_tick(t, tracks)
            # Nilai 1 mempertahankan keputusan aturan ketika fitur lengkap belum tersedia.
            candidates = {i: source for i, source in candidates.items()
                          if source != "aturan"
                          or self._rule_support.get(i, 1.0) >= self._support_threshold}
            for i in learned:
                candidates.setdefault(i, "model")
        for i, track in tracks.items():
            if track.samples and t - track.samples[-1].t <= EPS and self._is_upright(track):
                self._upright[i] = t
        fired = []
        for track_id, source in candidates.items():
            if t - self._last_fire.get(track_id, float("-inf")) >= REFIRE_S:
                self._last_fire[track_id] = t
                fired.append(track_id)
                self.sources[track_id] = source
        for track_id in [i for i, when in self._last_fire.items() if t - when > 4 * REFIRE_S]:
            del self._last_fire[track_id]
        return fired

    def _can_fall(self, track_id: int, t: float, since_drop: float = 0.0) -> bool:
        if not self.cfg.fall_need_upright:
            return True
        elapsed = since_drop if np.isfinite(since_drop) else 0.0
        return t - elapsed - self._upright.get(track_id, float("-inf")) <= self.cfg.fall_upright_before_s + EPS

    def _is_upright(self, track: Track) -> bool:
        s = track.samples[-1]
        conf = self.cfg.kp_min_conf
        angle = _torso_angle(s, conf)
        if angle is not None:
            if s.size < 0.55 * track.ref_size:
                return False             # satu lonjakan titik pada badan rebah tidak cukup
            if angle < 45:
                return True
            if angle <= 70:
                hip = _mid(s.kp, s.conf, HIPS, conf)
                feet = _mid(s.kp, s.conf, ANKLES, conf)
                shoulder = _mid(s.kp, s.conf, SHOULDERS, conf)
                return (feet is not None and feet[1] - hip[1] >= 0.8 * track.ref_size
                        and hip[1] - shoulder[1] >= 0.30 * track.ref_size)
            return False
        if raw_mode(s.conf, conf) != "upper" or s.conf[NOSE] < conf:
            return False
        sh = _mid(s.kp, s.conf, SHOULDERS, conf)
        dx, dy = sh - s.kp[NOSE]
        return dy > max(abs(dx) * 0.6, 0.08 * s.size)

    def _supported_on_hands(self, track: Track) -> bool:
        """Postur bertumpu untuk mulai push-up: telapak di bawah bahu, setinggi kaki."""
        if not self.cfg.fall_need_upright or not track.samples:
            return False
        now = track.samples[-1]
        conf = self.cfg.kp_min_conf
        scale = track.ref_size
        kept_since = now.t
        for s in reversed(track.samples):
            if now.t - s.t > 0.55:
                break
            hand = _mid(s.kp, s.conf, WRISTS, conf)
            feet = _mid(s.kp, s.conf, ANKLES, conf)
            sh = _mid(s.kp, s.conf, SHOULDERS, conf)
            if hand is None or feet is None or sh is None:
                continue
            if (abs(hand[0] - sh[0]) > 0.45 * scale or hand[1] - sh[1] < 0.35 * scale
                    or abs(hand[1] - feet[1]) > 0.35 * scale):
                break
            kept_since = s.t
            if now.t - kept_since >= 0.28:
                return True
        return False

    def _model_tick(self, t: float, tracks: dict[int, Track]) -> list[int]:
        cfg = self.cfg
        self._rule_support = {}
        ids, feats = [], []
        for track_id, track in tracks.items():
            if track.mode == "full" and track.samples and t - track.samples[-1].t <= EPS:
                f = person_features(track, cfg.fall_window_s)
                if f is not None:
                    ids.append(track_id)
                    feats.append(f)
        if ids:                                          # satu panggilan untuk semua orang
            probs = self._model.predict_proba(np.array(feats))[:, 1]
            self._rule_support = dict(zip(ids, map(float, probs), strict=True))
            for track_id, p, features in zip(ids, probs, feats):
                positive = p >= cfg.fall_model_threshold and self._can_fall(track_id, t, features[_SINCE_DROP])
                self._flags.setdefault(track_id, deque(maxlen=cfg.fall_confirm_n)).append(bool(positive))
                self._flag_t[track_id] = t

        scored = dict(zip(ids, feats))
        fired = []
        for track_id in list(self._flags):
            flags = self._flags[track_id]
            track = tracks.get(track_id)
            if track_id not in scored:
                detected_now = track is not None and t - track.last_seen <= EPS
                lost_for = t - self._flag_t[track_id]
                if detected_now:
                    flags.append(False)                  # terlihat tetapi tak terukur: bukan tanda jatuh
                elif flags and flags[-1] and lost_for <= LOST_GRACE_S + EPS:
                    flags.append(True)                   # tergeletak lalu sama sekali tak terdeteksi: masih dihitung jatuh
                elif lost_for > LOST_GRACE_S + EPS and track is None:
                    del self._flags[track_id], self._flag_t[track_id]
                    self._model_fired.pop(track_id, None)
                    continue
            hip_y = self._hip_y(track) if track_id in scored else None
            if track_id in self._model_fired:
                # Boleh memicu lagi hanya setelah orangnya bangun: pinggul naik kembali setidaknya separuh jatuhnya.
                if hip_y is not None and hip_y <= self._model_fired[track_id]:
                    del self._model_fired[track_id]
                    flags.clear()
            elif sum(flags) >= cfg.fall_confirm_k:
                drop_px = 0.0
                if track_id in scored and track is not None:
                    drop = float(scored[track_id][_HIP_DROP])
                    drop_px = (drop if np.isfinite(drop) else 0.0) * track.ref_size
                base = hip_y if hip_y is not None else self._last_hip.get(track_id, float("inf"))
                self._model_fired[track_id] = base - 0.5 * max(drop_px, 0.0) if np.isfinite(base) else float("-inf")
                fired.append(track_id)
            if hip_y is not None:
                self._last_hip[track_id] = hip_y
        for track_id in [i for i in self._last_hip if i not in self._flags]:
            del self._last_hip[track_id]
        return fired

    @staticmethod
    def _hip_y(track: Track | None) -> float | None:
        if track is None or not track.samples:
            return None
        s = track.samples[-1]
        hip = _mid(s.kp, s.conf, HIPS, 0.5)
        return float(hip[1]) if hip is not None else None

    def _rule_tick(self, t: float, tracks: dict[int, Track], frame_h: int,
                   full_rule: bool = True) -> dict[int, str]:
        """full_rule False: aturan seluruh-badan tetap mengikuti keadaan orangnya tetapi tidak memicu."""
        fired: dict[int, str] = {}
        for track_id, track in tracks.items():
            if not track.samples:
                continue
            fresh = t - track.samples[-1].t <= EPS
            if fresh:
                self._full_body_step(t, track)
                self._upper_body_step(track, frame_h)
                self._edge_step(t, track, frame_h)
                self._low_step(t, track, frame_h)

        for track_id, st in list(self._states.items()):
            if st.name == "dropped" and t - st.since >= self.cfg.fall_stay_s - EPS:
                track = tracks.get(track_id)
                supported = track is not None and t - track.last_seen <= EPS and self._supported_on_hands(track)
                st.name = "down_unknown" if supported else "fired"
                if full_rule and not supported:
                    fired[track_id] = "aturan"
            elif st.name == "tilted" and t - st.since > self.cfg.fall_drop_window_s + EPS:
                st.name = "down_unknown"
            if track_id not in tracks and st.name != "dropped":
                del self._states[track_id]

        for track_id, armed_t in list(self._armed.items()):
            if t - armed_t >= self.cfg.upper_vanish_s - EPS:
                del self._armed[track_id]
                fired.setdefault(track_id, "setengah_badan")
        for track_id, when in list(self._low.items()):
            if t - when >= LOW_STAY_S - EPS:
                del self._low[track_id]
                self._low_done.add(track_id)
                fired.setdefault(track_id, "turun_bawah")
        for track_id, when in list(self._edge.items()):
            if t - when < self.cfg.fall_edge_vanish_s - EPS:
                continue
            del self._edge[track_id]
            track = tracks.get(track_id)
            if track is None or track.last_seen < t - EPS:
                fired.setdefault(track_id, "keluar_gambar")
            elif track.samples:
                s = track.samples[-1]
                sh = _mid(s.kp, s.conf, SHOULDERS, self.cfg.kp_min_conf)
                if (raw_mode(s.conf, self.cfg.kp_min_conf) != "full" and sh is not None
                        and sh[1] > self.cfg.upper_bottom_frac * frame_h):
                    fired.setdefault(track_id, "keluar_gambar")
        return fired

    def _edge_step(self, t: float, track: Track, frame_h: int) -> None:
        cfg = self.cfg
        i = track.track_id
        s = track.samples[-1]
        mode = raw_mode(s.conf, cfg.kp_min_conf)
        sh = _mid(s.kp, s.conf, SHOULDERS, cfg.kp_min_conf)
        if mode != "full":
            if sh is not None and sh[1] <= cfg.upper_bottom_frac * frame_h:
                self._edge.pop(i, None)
            return
        visible = s.kp[s.conf >= cfg.kp_min_conf]
        hip = _mid(s.kp, s.conf, HIPS, cfg.kp_min_conf)
        near_bottom = (max(visible[:, 1]) >= cfg.fall_edge_frac * frame_h
                       and hip[1] >= cfg.upper_bottom_frac * frame_h)
        if near_bottom and self._can_fall(i, t):
            # Kedua titik tengah bergerak turun. Kaki saja di tepi bukan jatuh.
            span = 2 * SPEED_SPAN_S
            if all(_peak_down_speed(track, idxs, t, span, cfg.kp_min_conf) >= cfg.fall_hip_speed
                   for idxs in (HIPS, SHOULDERS)):
                self._edge[i] = t
                return
        self._edge.pop(i, None)

    def _low_step(self, t: float, track: Track, frame_h: int) -> None:
        cfg = self.cfg
        i = track.track_id
        s = track.samples[-1]
        sh = _mid(s.kp, s.conf, SHOULDERS, cfg.kp_min_conf)
        if sh is None:
            return                       # hilang atau hanya kepala: calon tetap berjalan
        if i in self._low_done or i in self._model_fired:
            if sh[1] <= LOW_RISE_FRAC * frame_h:
                self._low_done.discard(i)    # sudah bangun; jatuh berikutnya boleh memicu lagi
            return                       # masih tergeletak: bergeser turun lagi bukan jatuh baru
        if i in self._low:
            if sh[1] <= LOW_RISE_FRAC * frame_h:
                del self._low[i]         # bahu naik lagi: jongkok atau membungkuk, bukan jatuh
            return
        # Hanya orang yang kakinya tidak terlihat (dekat kamera tinggi). Orang seluruh badan yang duduk atau berbaring
        # di lantai juga bisa berakhir dengan bahu di bawah gambar (UR Fall adl-31..40); mereka dinilai aturan
        # seluruh badan dan model.
        if (sh[1] > cfg.upper_bottom_frac * frame_h and not self._legs_seen(track, t) and self._can_fall(i, t)
                and _peak_down_speed(track, SHOULDERS, t, LOW_SPEED_WINDOW_S, cfg.kp_min_conf) >= LOW_DROP_SPEED
                and self._came_from_above(track, t, frame_h) and not self._jumped(track, t)):
            self._low[i] = t

    def _came_from_above(self, track: Track, t: float, frame_h: int) -> bool:
        conf = self.cfg.kp_min_conf
        for s in track.samples:
            if t - s.t <= LOW_FROM_S + EPS:
                sh = _mid(s.kp, s.conf, SHOULDERS, conf)
                if sh is not None and sh[1] <= LOW_RISE_FRAC * frame_h:
                    return True
        return False

    def _legs_seen(self, track: Track, t: float) -> bool:
        """Seluruh badan sempat terlihat sesaat sebelum turun (kaki baru tertutup karena duduk, bukan di luar gambar)."""
        window = self.cfg.fall_upright_before_s + LOW_SPEED_WINDOW_S
        return any(raw_mode(s.conf, self.cfg.kp_min_conf) == "full" for s in track.samples if t - s.t <= window + EPS)

    def _jumped(self, track: Track, t: float) -> bool:
        """Bahu berpindah antargambar lebih cepat dari jatuh mana pun dalam jendela turun."""
        conf, size = self.cfg.kp_min_conf, track.ref_size
        pts = [(s.t, _mid(s.kp, s.conf, SHOULDERS, conf)) for s in track.samples if t - s.t <= LOW_SPEED_WINDOW_S + EPS]
        pts = [(ts, p) for ts, p in pts if p is not None]
        return size > 0 and any(np.hypot(*(p1 - p0)) / max(t1 - t0, 0.1) / size > LOW_JUMP_SPEED
                                for (t0, p0), (t1, p1) in zip(pts, pts[1:]))

    def _full_body_step(self, t: float, track: Track) -> None:
        cfg = self.cfg
        angle = _torso_angle(track.samples[-1], cfg.kp_min_conf)
        if angle is None:
            return
        st = self._states.get(track.track_id)
        if st is None:
            # Pertama kali terlihat: yang sudah rebah tidak boleh memicu sampai ia tegak.
            name = "upright" if angle < cfg.fall_upright_deg else "down_unknown"
            self._states[track.track_id] = _State(name, t)
            return
        if angle < cfg.fall_upright_deg:
            st.name, st.since = "upright", t
            return
        if st.name == "upright" and angle > cfg.fall_lying_deg:
            st.name, st.since = "tilted", t
        if st.name == "tilted":
            speed = _peak_down_speed(track, HIPS, t, cfg.fall_drop_window_s, cfg.kp_min_conf)
            if speed >= cfg.fall_hip_speed and self._can_fall(track.track_id, t):
                st.name = "dropped"          # `since` tetap saat badan mulai rebah

    def _upper_body_step(self, track: Track, frame_h: int) -> None:
        cfg = self.cfg
        s = track.samples[-1]
        sh = _mid(s.kp, s.conf, SHOULDERS, cfg.kp_min_conf)
        low = sh is not None and sh[1] > cfg.upper_bottom_frac * frame_h
        current_mode = raw_mode(s.conf, cfg.kp_min_conf)
        # Sisa bahu di tepi bawah bukan tanda bangun. Pertahankan waktu calon jatuh,
        # termasuk setelah histeresis mode berubah menjadi none, tanpa memulai calon baru.
        if low and current_mode == "none" and track.track_id in self._armed:
            return
        if track.mode != "upper" or current_mode != "upper":
            self._armed.pop(track.track_id, None)
            return
        fast = _peak_down_speed(track, SHOULDERS, s.t, 2 * SPEED_SPAN_S, cfg.kp_min_conf) >= cfg.upper_drop_speed
        if low and fast and self._can_fall(track.track_id, s.t):
            self._armed[track.track_id] = s.t
        else:
            self._armed.pop(track.track_id, None)   # muncul lagi tanpa memenuhi syarat: batal
