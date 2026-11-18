import numpy as np

from latih.eksperimen_abc import FALL_A, VIOLENCE_A, Columns, _prf, fall_alarms
from sahabat.pair_features import WINDOW_FEATURE_NAMES
from sahabat.person_features import PERSON_FEATURE_NAMES


def ticks_for(track_ids_per_tick, hz=10):
    return [(i / hz, ids) for i, ids in enumerate(track_ids_per_tick)]


def test_single_prediction_alarms_immediately_without_confirmation():
    ticks = ticks_for([[1]] * 5)
    probs = np.array([0.1, 0.95, 0.1, 0.1, 0.1])
    assert fall_alarms(ticks, probs, 0.9, 1, 1) == [0.1]


def test_confirmation_needs_k_of_last_n():
    ticks = ticks_for([[1]] * 6)
    probs = np.array([0.95, 0.1, 0.95, 0.1, 0.95, 0.1])
    assert fall_alarms(ticks, probs, 0.9, 3, 5) == [0.4]
    assert fall_alarms(ticks, np.array([0.95, 0.1, 0.1, 0.1, 0.1, 0.95]), 0.9, 3, 5) == []


def test_each_person_has_its_own_votes_and_refire_pause():
    ticks = ticks_for([[1, 2]] * 3 + [[1]] * 60)
    probs = np.array([0.95, 0.1] * 3 + [0.95] * 60)
    got = fall_alarms(ticks, probs, 0.9, 1, 1)
    assert got[0] == 0.0 and all(b - a >= 5.0 - 1e-9 for a, b in zip(got, got[1:]))


def test_single_frame_columns_are_instantaneous_features():
    names = [PERSON_FEATURE_NAMES[i] for i in FALL_A]
    assert not any(("speed" in n or "drop" in n or "change" in n or "max" in n or n == "still") for n in names)
    v = [WINDOW_FEATURE_NAMES[i] for i in VIOLENCE_A]
    assert all(n.endswith("_t9") for n in v) and not any("shift" in n for n in v)


def test_columns_model_reads_only_its_columns():
    class Echo:
        def predict_proba(self, x):
            return np.stack([1 - x[:, 0], x[:, 0]], axis=1)
    m = Columns(Echo(), [2])
    assert m.predict_proba(np.array([[0.0, 0.0, 0.7]]))[0, 1] == 0.7


def test_precision_recall_f1():
    assert _prf(8, 2, 2) == {"tp": 8, "fn": 2, "fp": 2, "precision": 0.8, "recall": 0.8, "f1": 0.8}
    assert _prf(0, 0, 0)["f1"] == 0.0
