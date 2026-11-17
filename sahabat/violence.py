"""Penilai indikasi kekerasan per pasangan orang.

Bila berkas model Random Forest ada, model itu yang dipakai. Bila belum ada, dipakai aturan
sederhana sebagai pengganti sementara supaya seluruh alur bisa diuji.

Model pendamping `violence_model_setengah.joblib` (di folder yang sama) menilai pasangan yang kaki keduanya tidak
terlihat (kamera dekat). Model utama sebagian besar belajar dari seluruh badan; pendamping dilatih tambahan dengan sesi
webcam setengah badan (latih.latih_sesi_webcam). Ambang pendamping disimpan di modelnya (`own_threshold_`) dan
peluangnya diskalakan ke ambang aplikasi, supaya konfirmasi alarm tetap satu aturan.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .history import Track
from .pair_features import FEATURE_NAMES, MOTION_FEATURES, WINDOW_FEATURE_NAMES, close_pairs, pair_windows

_F = {name: i for i, name in enumerate(FEATURE_NAMES)}
COMPANION_NAME = "violence_model_setengah.joblib"


def _load_model(path: Path):
    import joblib
    model = joblib.load(path)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1          # beberapa baris saja per putaran: banyak utas justru memperlambat
    expected = getattr(model, "n_features_in_", len(WINDOW_FEATURE_NAMES))
    if expected != len(WINDOW_FEATURE_NAMES):
        print(f"Peringatan: {Path(path).name} dilatih dengan {expected} fitur, aplikasi memakai "
              f"{len(WINDOW_FEATURE_NAMES)}. Model diabaikan; latih ulang dengan latih.violence_train.")
        return None
    return model


@dataclass(frozen=True)
class PairScore:
    ids: tuple[int, int]        # terurut naik
    prob: float                 # 0..1


class ViolenceScorer:
    def __init__(self, cfg: Config, model_path: Path | None = None) -> None:
        self.cfg = cfg
        self._model = None
        self._half = None
        self._half_scale = 1.0
        if model_path is not None and Path(model_path).is_file():
            self._model = _load_model(model_path)
            companion = Path(model_path).with_name(COMPANION_NAME)
            if self._model is not None and companion.is_file():
                self._half = _load_model(companion)
                own = float(getattr(self._half, "own_threshold_", cfg.violence_threshold))
                self._half_scale = cfg.violence_threshold / own

    @property
    def uses_model(self) -> bool:
        return self._model is not None

    def score(self, tracks: dict[int, Track]) -> list[PairScore]:
        ids, feats = [], []
        for a, b in close_pairs(tracks, self.cfg.pair_max_dist):
            both = pair_windows(a, b, self.cfg.violence_window_s)
            if both is None:
                continue
            ids.append((a.track_id, b.track_id))
            feats += [both[0], both[1]]
        if not ids:
            return []
        if self._model is not None:              # satu panggilan untuk semua pasangan dan kedua arah
            rows = np.array(feats)
            probs = self._model.predict_proba(rows)[:, 1]
            if self._half is not None:
                half = np.repeat([tracks[i].mode != "full" and tracks[j].mode != "full" for i, j in ids], 2)
                if half.any():
                    probs[half] = np.minimum(self._half.predict_proba(rows[half])[:, 1] * self._half_scale, 1.0)
            movement = rows[:, MOTION_FEATURES].reshape(len(ids), 2, -1).max(axis=(1, 2))
            # Pasangan diam kadang diberi peluang tinggi oleh model. Abaikan peluang itu
            # bila tidak ada gerak terukur pada kedua arah; kuncian dinilai terpisah.
            probs = np.where(np.repeat(movement >= self.cfg.violence_min_motion, 2), probs, 0.0)
        else:
            probs = [self._prob(f) for f in feats]
        return [PairScore(pair, float(max(probs[2 * i], probs[2 * i + 1]))) for i, pair in enumerate(ids)]

    def _prob(self, f: np.ndarray) -> float:
        """Aturan sementara (dipakai hanya bila belum ada model); hanya melihat bagian ringkasan."""
        cfg = self.cfg
        hand_near = f[_F["wrist_dist_min"]] <= cfg.rule_limb_dist
        punch = f[_F["wrist_toward_max"]] >= cfg.rule_limb_speed and hand_near
        kick = (f[_F["ankle_toward_max"]] >= cfg.rule_limb_speed
                and f[_F["ankle_dist_min"]] <= cfg.rule_limb_dist)
        push = f[_F["b_away_speed_max"]] >= cfg.rule_push_speed and hand_near
        return 1.0 if (punch or kick or push) else 0.0
