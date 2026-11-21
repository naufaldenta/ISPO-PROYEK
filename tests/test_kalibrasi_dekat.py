"""Kalibrasi model kamera lomba: label, gerbang, pemindahan rekaman, pasang/lepas, dan penilai tanpa kaki."""
import json

import joblib
import numpy as np
import pytest

from latih import kalibrasi_dekat as kd
from sahabat.config import Config
from sahabat.pair_features import FEATURE_NAMES, WINDOW_FEATURE_NAMES, tanpa_kaki
from sahabat.profil_kamera import PERINGATAN_DEKAT, PERINGATAN_KALIBRASI, VIOLENCE_DEKAT_NAME, model_untuk


def _m(tp=30, pos=40, fp=2, neg=20, per_jam=10.0, menit=10.0):
    return {"tertangkap": tp, "positif": pos, "keliru_klip": fp, "negatif": neg, "keliru_per_jam": per_jam,
            "menit_biasa": menit}


NOW = _m(tp=10)


def test_repeated_presses_are_one_take():
    marks = [(10.0, "kekerasan"), (11.0, "kekerasan"), (13.5, "kekerasan"), (20.0, "kekerasan"), (21.0, "biasa")]
    assert kd.gabung_tanda(marks) == [(10.0, "kekerasan"), (20.0, "kekerasan"), (21.0, "biasa")]


def test_gate_passes_good_candidate():
    assert kd.alasan_gagal(NOW, _m()) == []


@pytest.mark.parametrize("bad, word", [
    (_m(pos=19, tp=15), "ulangan kekerasan"), (_m(neg=9, fp=0), "ulangan biasa"), (_m(menit=4.0), "kegiatan biasa"),
    (_m(tp=10), "tidak lebih baik"), (_m(tp=27), "minimal 70%"), (_m(fp=5), "maksimal 20%"), (_m(per_jam=35.0), "per jam"),
])
def test_gate_rejects(bad, word):
    assert any(word in reason for reason in kd.alasan_gagal(NOW, bad))


def test_threshold_choice_prefers_catches_then_fewer_false_alarms():
    per = {0.8: _m(tp=36, per_jam=40.0), 0.85: _m(tp=33, per_jam=20.0), 0.9: _m(tp=33, per_jam=12.0), 0.95: _m(tp=20)}
    assert kd.pilih_ambang(NOW, per) == 0.9
    assert kd.pilih_ambang(NOW, {0.8: _m(tp=5)}) is None


def _rec(root, name, posisi="dekat"):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({"jenis": "adegan", "posisi": posisi}), encoding="utf-8")
    (folder / "0001.avi").write_bytes(b"video")
    (folder / "0001.txt").write_text("1.0\n1.1\n1.2\n", encoding="utf-8")
    (folder / "tanda.jsonl").write_text(json.dumps({"t": 1.2, "label": "kekerasan"}) + "\n", encoding="utf-8")
    return folder


def test_only_close_recordings_are_collected(tmp_path):
    _rec(tmp_path, "a")
    _rec(tmp_path, "b", "dada")
    assert [p.name for p in kd.rekaman_dekat(tmp_path)] == ["a"]
    assert kd.rekaman_dekat(tmp_path / "tidak-ada") == []


def test_move_verifies_then_removes_original(tmp_path):
    source = _rec(tmp_path / "c", "sesi")
    target = kd.pindahkan(source, tmp_path / "d")
    assert not source.exists() and (target / "0001.avi").read_bytes() == b"video"
    again = _rec(tmp_path / "c", "sesi")
    with pytest.raises(FileExistsError):
        kd.pindahkan(again, tmp_path / "d")
    assert again.exists()


def test_move_keeps_original_when_copy_differs(tmp_path, monkeypatch):
    source = _rec(tmp_path / "c", "sesi")
    real = kd._sha
    monkeypatch.setattr(kd, "_sha", lambda p: "beda" if ".part" in str(p) else real(p))
    with pytest.raises(OSError, match="tidak sama"):
        kd.pindahkan(source, tmp_path / "d")
    assert (source / "0001.avi").is_file() and not list((tmp_path / "d").iterdir())


def test_recording_summary(tmp_path):
    r = kd.ringkas_rekaman(_rec(tmp_path, "sesi"))
    assert (r["kekerasan"], r["biasa"], r["jenis"]) == (1, 0, "adegan") and r["laju"] == pytest.approx(10.0)


def test_delete_video_keeps_pose_and_skips_untraced(tmp_path):
    done = _rec(tmp_path, "selesai")
    (done / "jejak_416.npz").write_bytes(b"x")
    (done / "suara").mkdir()
    pending = _rec(tmp_path, "belum")
    assert kd.hapus_video(tmp_path) == 1
    assert not (done / "0001.avi").exists() and not (done / "suara").exists() and (done / "jejak_416.npz").is_file()
    assert (pending / "0001.avi").is_file()


class _Close:
    """Model tiruan: peluang = isi kolom jarak kaki terkecil dibagi 10, supaya terlihat bila kaki sudah disamarkan."""
    n_features_in_ = len(WINDOW_FEATURE_NAMES)
    upper_only_ = True
    own_threshold_ = 0.9

    def predict_proba(self, x):
        p = np.asarray(x)[:, FEATURE_NAMES.index("ankle_dist_min")] / 10.0
        return np.column_stack([1 - p, p])


def _candidate(folder, lolos=True):
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(_Close(), folder / VIOLENCE_DEKAT_NAME)
    (folder / "hasil.json").write_text(json.dumps({"lolos": lolos, "sha256": kd._sha(folder / VIOLENCE_DEKAT_NAME)}),
                                       encoding="utf-8")


def test_install_and_remove(tmp_path):
    cand, models = tmp_path / "k", tmp_path / "m"
    models.mkdir()
    _candidate(cand)
    assert kd.pasang(cand, models) and (models / VIOLENCE_DEKAT_NAME).is_file()
    assert kd.lepas(cand, models) and not (models / VIOLENCE_DEKAT_NAME).exists()
    assert not kd.lepas(cand, models)


def test_install_refuses_failed_or_changed_candidate(tmp_path):
    cand, models = tmp_path / "k", tmp_path / "m"
    models.mkdir()
    _candidate(cand, lolos=False)
    assert not kd.pasang(cand, models)
    _candidate(cand)
    (cand / VIOLENCE_DEKAT_NAME).write_bytes(b"berubah")
    assert not kd.pasang(cand, models)
    assert not (models / VIOLENCE_DEKAT_NAME).exists()


def _base_models(folder):
    for name in ("fall_model.joblib", "violence_model.joblib"):
        joblib.dump({"model": name}, folder / name)


def test_profile_uses_calibrated_model_only_when_valid(tmp_path):
    _base_models(tmp_path)
    assert model_untuk("dekat", tmp_path).violence.name == "violence_model.joblib"
    (tmp_path / VIOLENCE_DEKAT_NAME).write_bytes(b"rusak")
    m = model_untuk("dekat", tmp_path)
    assert m.violence.name == "violence_model.joblib" and m.peringatan == PERINGATAN_DEKAT
    joblib.dump(_Close(), tmp_path / VIOLENCE_DEKAT_NAME)
    m = model_untuk("dekat", tmp_path)
    assert m.violence.name == VIOLENCE_DEKAT_NAME and m.peringatan == PERINGATAN_KALIBRASI
    assert model_untuk("dada", tmp_path).violence.name == "violence_model.joblib"


class _Plain(_Close):
    upper_only_ = False


def test_model_without_close_flag_is_not_used(tmp_path):
    _base_models(tmp_path)
    joblib.dump(_Plain(), tmp_path / VIOLENCE_DEKAT_NAME)
    assert model_untuk("dekat", tmp_path).violence.name == "violence_model.joblib"


def test_leg_masking_matches_a_half_body_row():
    row = np.ones(len(WINDOW_FEATURE_NAMES), dtype=np.float32)
    out = tanpa_kaki(row[None])[0]
    assert out[FEATURE_NAMES.index("ankle_toward_max")] == 0 and out[FEATURE_NAMES.index("ankle_dist_min")] == 9.0
    assert out[WINDOW_FEATURE_NAMES.index("ankle_r_t9")] == 9.0 and out[WINDOW_FEATURE_NAMES.index("wrist_l_t9")] == 1.0
    assert row[FEATURE_NAMES.index("ankle_dist_min")] == 1.0          # masukan tidak diubah


def test_scorer_masks_legs_and_scales_to_app_threshold(tmp_path, monkeypatch):
    from sahabat import violence as v
    path = tmp_path / VIOLENCE_DEKAT_NAME
    joblib.dump(_Close(), path)
    row = np.full(len(WINDOW_FEATURE_NAMES), 2.0, dtype=np.float32)    # bergerak; jarak kaki asli 2,0
    monkeypatch.setattr(v, "close_pairs", lambda tracks, dist: [(tracks[1], tracks[2])])
    monkeypatch.setattr(v, "pair_windows", lambda a, b, w: (row, row))

    class T:
        def __init__(self, i):
            self.track_id, self.mode = i, "full"
    cfg = Config()
    got = v.ViolenceScorer(cfg, path).score({1: T(1), 2: T(2)})
    assert got[0].prob == pytest.approx(0.9 * cfg.violence_threshold / 0.9)   # kolom kaki disamarkan menjadi 9,0
