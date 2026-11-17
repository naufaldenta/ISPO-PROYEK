import numpy as np

from latih.ntu_build import build_clip
from latih.violence_train import action_name, alarm_rates, fit
from latih.windows import training_rows
from sahabat.config import Config
from sahabat.pair_features import WINDOW_FEATURE_NAMES

from .scenes import punch, two
from .test_ntu import as_ntu


def test_action_names_cover_both_groups():
    assert action_name(50) == "pukul/tampar" and action_name(58) == "salaman" and action_name(999) == "aksi 999"


def test_alarm_rates_count_per_action():
    hit = np.array([True, False, True, True, False])
    actions = np.array([50, 50, 58, 58, 58])
    assert alarm_rates(hit, actions) == {50: (1, 2), 58: (2, 3)}


def test_training_rows_can_skip_overlapping_windows():
    clip = build_clip(*as_ntu(two, seconds=4.0), Config())
    every = training_rows(clip, 0, 1.0)
    half = training_rows(clip, 0, 1.0, 2)
    assert every.shape[1] == half.shape[1] == len(WINDOW_FEATURE_NAMES)
    assert abs(len(half) - len(every) / 2) <= 2 and len(half) % 2 == 0          # tetap dua arah per jendela
    assert np.array_equal(half[:2], every[:2], equal_nan=True)


def test_model_learns_to_separate_punches_and_accepts_missing_values():
    quiet = training_rows(build_clip(*as_ntu(two, seconds=8.0), Config()), 0, 1.0)
    hits = training_rows(build_clip(*as_ntu(lambda t: punch(t, t0=1.0 + 1.5 * int(t / 1.5)), seconds=8.0), Config()), 1, 1.0)
    X = np.concatenate([quiet, hits] * 4)
    y = np.concatenate([np.zeros(len(quiet)), np.ones(len(hits))] * 4)
    assert np.isnan(X).any() or True                         # deret boleh berisi NaN; model harus menerimanya
    X[0, 20] = np.nan
    model = fit(X, y)
    assert model.predict_proba(hits[:5])[:, 1].min() > 0.5 and model.predict_proba(quiet[:5])[:, 1].max() < 0.5
    assert model.n_features_in_ == len(WINDOW_FEATURE_NAMES)


# ---- video sungguhan ikut melatih ----

def synthetic_item(source, violent, group, script, seconds=6.0):
    from latih.violence_train import Item
    return Item(source, "pukul" if violent else "salaman", violent, group, build_clip(*as_ntu(script, seconds=seconds), Config()))


def test_trace_pairs_gives_the_same_rows_as_the_live_path():
    from latih.putar import PoseTrace
    from latih.violence_train import trace_pairs
    from latih.ntu import to_frames
    frames = to_frames(*as_ntu(two, seconds=4.0))
    tr = PoseTrace(np.array([t for t, _ in frames]), 480, 640, [p for _, p in frames])
    a, b = trace_pairs(tr, Config()), build_clip(*as_ntu(two, seconds=4.0), Config())
    assert len(a.rows) == len(b.rows) > 0 and np.array_equal(a.rows[0][3], b.rows[0][3], equal_nan=True)


def test_stack_weights_video_rows_and_thins_only_ntu():
    from latih.violence_train import stack
    ntu = synthetic_item("ntu", False, 1, two)
    ut = synthetic_item("ut", False, 1101, two)
    Xn, yn, wn = stack([ntu], 10.0)
    Xu, yu, wu = stack([ut], 10.0)
    assert set(wn) == {1.0} and set(wu) == {10.0} and set(yn) == set(yu) == {0}
    assert abs(len(Xu) - 2 * len(Xn)) <= 4                    # NTU diambil selang-seling, video semuanya


def test_fit_accepts_sample_weights():
    quiet = training_rows(build_clip(*as_ntu(two, seconds=8.0), Config()), 0, 1.0)
    hits = training_rows(build_clip(*as_ntu(lambda t: punch(t, t0=1.0 + 1.5 * int(t / 1.5)), seconds=8.0), Config()), 1, 1.0)
    X = np.concatenate([quiet, hits] * 4)
    y = np.concatenate([np.zeros(len(quiet)), np.ones(len(hits))] * 4)
    model = fit(X, y, np.where(y == 1, 10.0, 1.0))
    assert model.predict_proba(hits[:5])[:, 1].min() > 0.5


def test_alarms_use_the_app_confirmation():
    from latih.violence_train import alarms
    item = synthetic_item("ut", True, 1101, lambda t: punch(t, t0=1.0 + 1.5 * int(t / 1.5)), seconds=8.0)
    always = [np.ones(len(item.clip.rows))]
    never = [np.zeros(len(item.clip.rows))]
    assert alarms([item], always, Config()).tolist() == [True] and alarms([item], never, Config()).tolist() == [False]


def test_peak_rows_keep_only_the_highest_scored_moving_windows():
    from latih.violence_train import peak_rows
    item = synthetic_item("ut", True, 1101, lambda t: punch(t, t0=1.0 + 1.5 * int(t / 1.5)), seconds=8.0)
    n = len(item.clip.rows)
    first = np.zeros(n)
    first[[3, 7]] = [0.9, 0.8]
    rows = peak_rows(item.clip, first, k=2)
    wanted = [f for j in (3, 7) for f in (item.clip.rows[j][3], item.clip.rows[j][4])]
    assert 1 <= len(rows) <= 4 and all(any(np.array_equal(r, w, equal_nan=True) for w in wanted) for r in rows)
    assert len(peak_rows(item.clip, first, k=1000)) <= 2 * n


def test_stack_with_first_round_uses_peaks_only_for_violent_clips():
    from latih.violence_train import PEAK_ROWS, stack
    hit = synthetic_item("ut", True, 1101, lambda t: punch(t, t0=1.0 + 1.5 * int(t / 1.5)), seconds=8.0)
    calm = synthetic_item("ut", False, 1102, two)
    first = [np.linspace(0, 1, len(hit.clip.rows)), np.zeros(len(calm.clip.rows))]
    X1, y1, _ = stack([hit, calm], 10.0)
    X2, y2, _ = stack([hit, calm], 10.0, first)
    assert (y2 == 0).sum() == (y1 == 0).sum()                  # klip ramah tidak berubah
    assert 0 < (y2 == 1).sum() <= 2 * PEAK_ROWS < (y1 == 1).sum()


def test_time_labeled_webcam_actions_keep_moving_rows_even_if_first_model_misses_them():
    from latih.violence_train import PEAK_ROWS, stack
    hit = synthetic_item("webcam", True, 2000, lambda t: punch(t, t0=1.0 + 1.5 * int(t / 1.5)), seconds=8.0)
    X1, y1, w1 = stack([hit], 10.0)
    X2, y2, w2 = stack([hit], 10.0, [np.zeros(len(hit.clip.rows))])
    assert len(X2) > 2 * PEAK_ROWS
    assert np.array_equal(X1, X2, equal_nan=True)
    assert np.array_equal(y1, y2) and np.array_equal(w1, w2)


# ---- rekaman webcam sendiri sebagai data latih (sesi uji pemilik proyek, 15 November 2026) ----

def test_session_ranges_become_one_item_each_with_only_their_own_rows():
    from latih.ntu import to_frames
    from latih.putar import PoseTrace
    from latih.violence_train import WEBCAM_GROUP_BASE, session_items
    frames = to_frames(*as_ntu(lambda t: punch(t, t0=1.0 + 1.5 * int(t / 1.5)), seconds=12.0))
    start = 5000.0                                             # jam mesin, bukan nol
    tr = PoseTrace(np.array([start + t for t, _ in frames]), 480, 640, [p for _, p in frames])
    ranges = [{"mulai": 2.0, "selesai": 5.0, "kekerasan": True, "grup": 0},
              {"mulai": 7.0, "selesai": 11.0, "kekerasan": False, "grup": 1},
              {"mulai": 50.0, "selesai": 60.0, "kekerasan": False, "grup": 1}]      # di luar rekaman: tidak ada baris
    items = session_items(tr, ranges, Config())
    assert [(i.source, i.name, i.violent, i.group) for i in items] == [
        ("webcam", "kekerasan", True, WEBCAM_GROUP_BASE), ("webcam", "biasa", False, WEBCAM_GROUP_BASE + 1),
        ("webcam", "biasa", False, WEBCAM_GROUP_BASE + 1)]
    first, second, empty = (i.clip for i in items)
    assert len(first.t) == 31 and len(second.t) == 41 and empty.rows == [] and empty.t == []
    assert first.rows and second.rows
    assert all(0 <= r[0] < len(first.t) for r in first.rows) and all(0 <= r[0] < len(second.t) for r in second.rows)
    whole = build_clip(*as_ntu(lambda t: punch(t, t0=1.0 + 1.5 * int(t / 1.5)), seconds=12.0), Config())
    at_3s = next(r for r in whole.rows if r[0] == 30)          # riwayat tidak diputus di awal rentang
    assert np.array_equal(first.rows[[r[0] for r in first.rows].index(10)][3], at_3s[3], equal_nan=True)


def test_session_rows_get_the_video_weight():
    from latih.violence_train import Item, stack
    cam = synthetic_item("webcam", False, 2001, two)
    _, _, w = stack([cam], 10.0)
    assert set(w) == {10.0}
