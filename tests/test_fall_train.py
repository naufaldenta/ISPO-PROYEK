import numpy as np

from latih.fall_train import clip_features, confirmed, tick_labels, wanted
from sahabat.config import Config
from sahabat.person_features import PERSON_FEATURE_NAMES

from .helpers import ticks
from .test_fall import fall_script

F = {n: i for i, n in enumerate(PERSON_FEATURE_NAMES)}


def test_fall_clip_labels_only_the_end_as_positive():
    labels = tick_labels(n_real=30, n_total=45, is_fall=True)
    assert (labels[:28] == -1).all() and (labels[28:41] == 1).all() and (labels[41:] == -1).all()


def test_other_clips_are_all_negative():
    assert (tick_labels(30, 45, False) == 0).all()


def test_short_fall_clip_does_not_overflow():
    labels = tick_labels(n_real=1, n_total=5, is_fall=True)
    assert len(labels) == 5 and (labels == 1).all()


def test_confirmed_needs_k_within_n_consecutive_ticks():
    assert confirmed(np.array([1, 1, 0, 0, 0], dtype=bool), 2, 3) is True
    assert confirmed(np.array([1, 0, 0, 1, 0, 0, 1], dtype=bool), 2, 3) is False
    assert confirmed(np.array([0, 1, 0, 1, 1, 0], dtype=bool), 3, 5) is True
    assert confirmed(np.array([], dtype=bool), 2, 3) is False


def test_wanted_keeps_fall_and_lookalikes_and_samples_the_rest():
    assert wanted("S001C001P001R001A043", 43) and wanted("S001C001P001R001A008", 8)
    assert not wanted("S001C001P001R001A050", 50)                       # aksi dua orang: bukan urusan penilai jatuh
    names = [f"S001C001P{p:03d}R001A001" for p in range(1, 201)]
    kept = [n for n in names if wanted(n, 1)]
    assert 5 <= len(kept) <= 40 and kept == [n for n in names if wanted(n, 1)]      # ~1 dari 10, selalu sama


def test_clip_features_follow_the_person_through_a_fall():
    frames = [(t, fall_script(t)) for t in ticks(0.0, 6.0)]
    X = clip_features(frames, Config())
    assert X.shape == (60, len(PERSON_FEATURE_NAMES))
    assert np.isnan(X[:5]).all()                                         # belum cukup riwayat
    assert abs(X[19, F["hip_drop"]]) < 0.05                              # masih berdiri
    assert X[30, F["hip_drop"]] > 0.7 and X[30, F["angle_now"]] > 0.9    # sudah tergeletak


def test_clip_features_with_nobody_is_all_nan():
    X = clip_features([(t, []) for t in ticks(0.0, 2.0)], Config())
    assert X.shape == (20, len(PERSON_FEATURE_NAMES)) and np.isnan(X).all()
