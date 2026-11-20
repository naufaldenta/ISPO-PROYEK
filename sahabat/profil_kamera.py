"""Profil posisi kamera: setinggi dada/kepala (model sekarang) atau tinggi di pojok/langit-langit (model kamera tinggi).

Model sekarang tidak diubah. Model kamera tinggi adalah berkas terpisah dan baru dipasang bila lolos uji pada data kamera
tinggi yang tidak ikut dilatih (latih.kamera_tinggi). Selama berkasnya belum ada, rusak, atau tidak bisa dimuat, profil
tinggi memakai model sekarang dan menampilkan peringatan. Bawaannya selalu "dada", jadi hasil kamera lama tidak berubah.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import MODELS_DIR

POSISI = ("dada", "tinggi")
LABEL = {"dada": "Setinggi dada/kepala (sekitar 1,2–1,8 m)",
         "tinggi": "Tinggi, di pojok atau langit-langit (sekitar 2,2–3 m)"}
PERINGATAN = "Akurasi untuk kamera tinggi belum teruji."
FALL_NAME, VIOLENCE_NAME = "fall_model.joblib", "violence_model.joblib"
FALL_TINGGI_NAME, VIOLENCE_TINGGI_NAME = "fall_model_tinggi.joblib", "violence_model_tinggi.joblib"
FALL_TINGGI, VIOLENCE_TINGGI = MODELS_DIR / FALL_TINGGI_NAME, MODELS_DIR / VIOLENCE_TINGGI_NAME


@dataclass(frozen=True)
class ModelKamera:
    violence: Path
    fall: Path
    peringatan: str


def posisi_bersih(value) -> str:
    return value if value in POSISI else "dada"


def _loadable(path: Path, n_features: int) -> bool:
    """Model bisa dipakai penilai: termuat, jumlah cirinya sama dengan aplikasi, dan benar-benar bisa menilai.

    Hanya memuat tidak cukup: model dengan jumlah ciri lain akan diabaikan penilai (diam-diam memakai aturan saja), dan
    model dari versi scikit-learn lain bisa termuat tetapi gagal tiap kali menilai (kamera "tidak dapat dinilai").
    """
    if not path.is_file():
        return False
    try:
        import joblib
        import numpy as np
        model = joblib.load(path)
        if getattr(model, "n_features_in_", n_features) != n_features:
            return False
        proba = np.asarray(model.predict_proba(np.zeros((1, n_features), dtype=np.float32)))
        return proba.shape == (1, 2)
    except Exception:                       # berkas rusak atau dari versi pustaka lain: pakai model sekarang
        return False


def model_untuk(posisi: str, models_dir: Path = MODELS_DIR) -> ModelKamera:
    models_dir = Path(models_dir)
    violence, fall = models_dir / VIOLENCE_NAME, models_dir / FALL_NAME
    if posisi_bersih(posisi) != "tinggi":
        return ModelKamera(violence, fall, "")
    high_v, high_f = models_dir / VIOLENCE_TINGGI_NAME, models_dir / FALL_TINGGI_NAME
    from .pair_features import WINDOW_FEATURE_NAMES
    from .person_features import PERSON_FEATURE_NAMES
    use_v = _loadable(high_v, len(WINDOW_FEATURE_NAMES))
    use_f = _loadable(high_f, len(PERSON_FEATURE_NAMES))
    return ModelKamera(high_v if use_v else violence, high_f if use_f else fall,
                       "" if use_v and use_f else PERINGATAN)
