"""Profil posisi kamera (setinggi dada / tinggi) dan pertukaran model tanpa menghentikan pemantauan."""
import json
from pathlib import Path

import joblib

from sahabat.config import LOCAL_FIELDS, MODELS_DIR, Config
from sahabat.engine import Engine
from sahabat.kamera import KameraStore
from sahabat.profil_kamera import LABEL, PERINGATAN, POSISI, model_untuk, posisi_bersih

FALL, VIOLENCE = MODELS_DIR / "fall_model.joblib", MODELS_DIR / "violence_model.joblib"


class _Pose:
    def detect(self, frame):
        return []


def _models(folder: Path) -> None:
    for name in ("fall_model.joblib", "violence_model.joblib"):
        joblib.dump({"model": name}, folder / name)


def test_default_and_bad_values_are_dada():                                # Review Focus 3
    assert POSISI == ("dada", "tinggi")
    assert posisi_bersih(None) == "dada" and posisi_bersih("atap") == "dada" and posisi_bersih(3) == "dada"
    assert posisi_bersih("tinggi") == "tinggi"
    assert LABEL == {"dada": "Setinggi dada/kepala (sekitar 1,2–1,8 m)",
                     "tinggi": "Tinggi, di pojok atau langit-langit (sekitar 2,2–3 m)"}


class _Usable:
    """Model tiruan yang bisa menilai: jumlah ciri sesuai dan predict_proba bekerja."""

    def __init__(self, n):
        self.n_features_in_ = n

    def predict_proba(self, x):
        return [[0.9, 0.1]] * len(x)


def test_high_profile_uses_high_models_only_when_loadable(tmp_path):       # Review Focus 2
    _models(tmp_path)
    m = model_untuk("tinggi", tmp_path)
    assert (m.fall.name, m.violence.name, m.peringatan) == ("fall_model.joblib", "violence_model.joblib", PERINGATAN)
    from sahabat.pair_features import WINDOW_FEATURE_NAMES
    from sahabat.person_features import PERSON_FEATURE_NAMES
    joblib.dump(_Usable(len(WINDOW_FEATURE_NAMES)), tmp_path / "violence_model_tinggi.joblib")
    (tmp_path / "fall_model_tinggi.joblib").write_bytes(b"rusak")
    m = model_untuk("tinggi", tmp_path)
    assert m.violence.name == "violence_model_tinggi.joblib" and m.fall.name == "fall_model.joblib"
    assert m.peringatan == PERINGATAN == "Akurasi untuk kamera tinggi belum teruji."
    joblib.dump(_Usable(len(PERSON_FEATURE_NAMES)), tmp_path / "fall_model_tinggi.joblib")
    assert model_untuk("tinggi", tmp_path).peringatan == ""
    dada = model_untuk("dada", tmp_path)
    assert (dada.fall, dada.violence, dada.peringatan) == (tmp_path / "fall_model.joblib",
                                                           tmp_path / "violence_model.joblib", "")


def test_swap_models_keeps_alarm_history_and_tracks():                     # Review Focus 1
    engine = Engine(Config(), None, _Pose(), violence_model=VIOLENCE, fall_model=FALL)
    alarm, history, gestures = engine.alarm, engine.history, engine.gestures
    old_violence, old_fall = engine.violence, engine.fall
    engine.ganti_model(VIOLENCE, FALL)
    assert engine.alarm is alarm and engine.history is history and engine.gestures is gestures
    assert engine.violence is not old_violence and engine.fall is not old_fall
    assert (engine.violence_model_path, engine.fall_model_path) == (VIOLENCE, FALL)


def test_config_position_is_local():
    assert Config().posisi_kamera == "dada" and "posisi_kamera" in LOCAL_FIELDS


def _store(tmp_path, extra=None):
    item = {"id": "k1", "nama": "Lorong", "jenis": "alamat", "alamat": "rtsp://10.0.0.1/x", "pengguna": "",
            "sandi": "", "aktif": True, "prioritas": False, "zona": None, "potong": None, **(extra or {})}
    (tmp_path / "kamera.json").write_text(json.dumps({"kamera": [item]}), encoding="utf-8")
    return KameraStore(tmp_path / "kamera.json")


def test_old_store_without_position_reads_dada(tmp_path):                 # Review Focus 3
    assert _store(tmp_path).get_public("k1")["posisi"] == "dada"
    assert _store(tmp_path, {"posisi": "atap"}).get_public("k1")["posisi"] == "dada"


def test_position_is_saved_without_reconnecting(tmp_path):
    store = _store(tmp_path)
    before = store.get_public("k1")
    form = {"id": "k1", "nama": "Lorong", "jenis": "alamat", "alamat": "rtsp://10.0.0.1/x", "pengguna": ""}
    after = store.save({**form, "posisi": "tinggi"})
    assert after["posisi"] == "tinggi" and after["rev"] == before["rev"]       # hanya posisi: sumber tidak dibuka ulang
    assert store.save({**form, "posisi": "tinggi", "nama": "Lorong A"})["rev"] == before["rev"] + 1
    created = store.save({"nama": "Baru", "jenis": "alamat", "alamat": "rtsp://10.0.0.2/x", "posisi": "tinggi"})
    assert created["posisi"] == "tinggi"


class _WrongWidth:
    n_features_in_ = 3

    def predict_proba(self, x):
        return [[0.5, 0.5]] * len(x)


class _Broken:
    """Seolah dimuat dari versi scikit-learn lain: bisa dimuat, tetapi gagal saat menilai."""

    def __init__(self, n):
        self.n_features_in_ = n

    def predict_proba(self, x):
        raise AttributeError("'HistGradientBoostingClassifier' object has no attribute '_preprocessor'")


def test_high_model_that_cannot_score_falls_back_with_warning(tmp_path):     # Final #3
    from sahabat.person_features import PERSON_FEATURE_NAMES
    from sahabat.pair_features import WINDOW_FEATURE_NAMES
    _models(tmp_path)
    joblib.dump(_WrongWidth(), tmp_path / "violence_model_tinggi.joblib")
    joblib.dump(_Broken(len(PERSON_FEATURE_NAMES)), tmp_path / "fall_model_tinggi.joblib")
    m = model_untuk("tinggi", tmp_path)
    assert (m.violence.name, m.fall.name, m.peringatan) == ("violence_model.joblib", "fall_model.joblib", PERINGATAN)
    joblib.dump(_Broken(len(WINDOW_FEATURE_NAMES)), tmp_path / "violence_model_tinggi.joblib")
    assert model_untuk("tinggi", tmp_path).violence.name == "violence_model.joblib"
