import numpy as np
import pytest

from latih.fall_data import ClipRows, fold_of, landing_time, stack, time_labels, video_rows
from latih.putar import PoseTrace
from sahabat.config import Config
from sahabat.person_features import PERSON_FEATURE_NAMES

from .helpers import person, ticks
from .test_fall import fall_script


def make_trace(script, seconds):
    ts = ticks(0.0, seconds)
    return PoseTrace(np.array(ts), 480, 640, [script(t) for t in ts])


def test_everyday_clip_is_all_negative():
    assert list(time_labels(np.array([0.0, 0.1, 0.2]), None, None)) == [0, 0, 0]


def test_fall_clip_labels_before_during_and_after():
    ts = np.arange(0, 60) / 10.0                                     # 0,0 .. 5,9 dtk
    labels = time_labels(ts, onset=2.0, landing=2.6)
    assert set(labels[ts < 2.0 - 1e-9]) == {0}                       # sebelum jatuh: bukan jatuh
    assert set(labels[(ts >= 2.0) & (ts < 2.4 - 1e-9)]) == {-1}      # sedang jatuh: tidak dipakai melatih
    assert set(labels[(ts >= 2.4 - 1e-9) & (ts <= 3.6 + 1e-9)]) == {1}   # 0,2 dtk sebelum rebah s.d. 1 dtk sesudahnya
    assert set(labels[ts > 3.6 + 1e-9]) == {-1}                      # tergeletak lama: tidak dipakai


def test_landing_is_found_from_the_hips():
    tr = make_trace(fall_script, 5.0)                                # jatuh mulai 2,0 dtk, rebah dalam 0,4 dtk
    assert landing_time(tr, onset=2.0) == pytest.approx(2.3, abs=0.11)


def test_landing_falls_back_when_hips_are_not_seen():
    tr = make_trace(lambda t: [person(1, cx=300, hip_y=300, visible="none")], 5.0)
    assert landing_time(tr, onset=2.0) == pytest.approx(3.0)         # tanpa pinggul: anggap 1 detik
    assert landing_time(make_trace(lambda t: [], 5.0), onset=2.0) == pytest.approx(3.0)


def test_landing_falls_back_when_the_hips_barely_move():
    tr = make_trace(lambda t: [person(1, cx=300, hip_y=300)], 5.0)
    assert landing_time(tr, onset=2.0) == pytest.approx(3.0)


def test_video_rows_have_one_feature_row_per_tick():
    tr = make_trace(fall_script, 5.0)
    rows = video_rows("gmdcsa", "s1-fall-01", 1, tr, Config(), onset=2.0, landing=2.4, kind="jatuh")
    assert rows.feats.shape == (50, len(PERSON_FEATURE_NAMES)) and rows.labels.shape == (50,)
    assert rows.is_fall and rows.source == "gmdcsa" and rows.trace is tr
    assert np.isnan(rows.feats[0]).all() and np.isfinite(rows.feats[30]).any()      # awal: riwayat belum cukup
    assert (rows.labels == 1).sum() == 13


def rows(source, group, is_fall, n=4, value=0.0):
    labels = np.array([0, -1, 1, 1] if is_fall else [0, 0, 0, 0], dtype=np.int8)[:n]
    feats = np.full((n, len(PERSON_FEATURE_NAMES)), value, dtype=np.float32)
    feats[0] = np.nan                                                # baris tanpa ciri tidak ikut dilatih
    return ClipRows(source, f"{source}-{group}", group, is_fall, "x", feats, labels, False, None, None)


def test_stack_keeps_only_labelled_rows_with_features_and_weights_by_source():
    X, y, w = stack([rows("ntu", 1, True), rows("urfall", 2, False, value=5.0)], {"urfall": 10.0})
    assert list(y) == [1, 1, 0, 0, 0]                                # -1 dan baris NaN dibuang
    assert list(w) == [1.0, 1.0, 10.0, 10.0, 10.0]
    assert X.shape == (5, len(PERSON_FEATURE_NAMES)) and X[-1, 0] == 5.0


def test_folds_keep_a_group_together_and_spread_groups():
    clips = [rows("ntu", g, False) for g in (1, 1, 2, 3, 4, 5, 6)] + [rows("gmdcsa", g, False) for g in (1, 2, 3, 4)]
    folds = [fold_of(c, 5) for c in clips]
    assert folds[0] == folds[1]                                      # pemeran yang sama, lipatan yang sama
    assert len({fold_of(c, 5) for c in clips if c.source == "gmdcsa"}) == 4
    assert all(0 <= f < 5 for f in folds)
