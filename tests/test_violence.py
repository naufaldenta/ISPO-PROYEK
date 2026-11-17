from sahabat.config import Config
from sahabat.history import History
from sahabat.violence import PairScore, ViolenceScorer

from .helpers import feed, person, with_hidden_legs
from .scenes import handshake, kick, mirrored, punch, push, two


def score_after(script, seconds=1.5):
    h = History(Config())
    feed(h, script, seconds=seconds)
    return ViolenceScorer(Config()).score(h.tracks)


def test_without_model_uses_rules():
    assert ViolenceScorer(Config()).uses_model is False


def test_missing_model_file_falls_back_to_rules(tmp_path):
    assert ViolenceScorer(Config(), tmp_path / "tidak-ada.joblib").uses_model is False


def test_rule_scores_punch_from_either_side():
    assert score_after(punch) == [PairScore((1, 2), 1.0)]
    assert score_after(mirrored(punch)) == [PairScore((1, 2), 1.0)]   # B memukul A


def test_rule_scores_kick_in_full_mode():
    assert score_after(kick)[0].prob == 1.0


def test_rule_scores_push():
    assert score_after(push)[0].prob == 1.0


def test_standing_and_slow_handshake_score_zero():
    assert score_after(two) == [PairScore((1, 2), 0.0)]
    assert score_after(handshake)[0].prob == 0.0                # tangan mendekat 0,5 badan/dtk


def test_single_person_has_no_pairs():
    assert score_after(lambda t: [person(1, cx=250, hip_y=300)]) == []


def test_upper_mode_ignores_legs():
    upper_kick = lambda t: [with_hidden_legs(p) if p.track_id == 1 else p for p in kick(t)]
    assert score_after(upper_kick)[0].prob == 0.0               # tendangan tidak dinilai


def test_degenerate_person_is_skipped_without_error():
    assert score_after(lambda t: two(t, b_kw=dict(size=3))) == []


def test_pair_with_too_little_history_is_skipped():
    assert score_after(two, seconds=0.2) == []


class CountingModel:
    """Model tiruan: peluang = 1 bila tangan mendekat cepat, dan menghitung berapa kali dipanggil."""
    n_jobs = 8

    def __init__(self):
        self.calls, self.rows = 0, 0

    def predict_proba(self, X):
        import numpy as np
        self.calls += 1
        self.rows += len(X)
        p = (X[:, 0] >= 2.0).astype(float)
        return np.column_stack([1 - p, p])


def test_model_is_called_once_per_tick_for_all_pairs(tmp_path):
    import joblib
    path = tmp_path / "model.joblib"
    joblib.dump(CountingModel(), path)
    scorer = ViolenceScorer(Config(), path)
    assert scorer.uses_model and scorer._model.n_jobs == 1        # satu utas: baris sedikit, utas hanya menambah jeda
    h = History(Config())
    three = lambda t: punch(t) + [person(3, cx=480, hip_y=300)]        # ketiganya saling berdekatan
    feed(h, three, seconds=1.5)
    scores = scorer.score(h.tracks)
    assert [s.ids for s in scores] == [(1, 2), (1, 3), (2, 3)]
    assert (scorer._model.calls, scorer._model.rows) == (1, 6)     # 3 pasangan x 2 arah, satu panggilan
    assert scores[0].prob == 1.0 and scores[2].prob == 0.0


def test_no_pairs_means_no_model_call(tmp_path):
    import joblib
    path = tmp_path / "model.joblib"
    joblib.dump(CountingModel(), path)
    scorer = ViolenceScorer(Config(), path)
    h = History(Config())
    feed(h, lambda t: [person(1, cx=250, hip_y=300)], seconds=1.5)
    assert scorer.score(h.tracks) == [] and scorer._model.calls == 0


class OldTwelveFeatureModel(CountingModel):
    n_features_in_ = 12


def test_model_trained_on_other_features_is_ignored_not_crashed(tmp_path):
    import joblib
    path = tmp_path / "lama.joblib"
    joblib.dump(OldTwelveFeatureModel(), path)
    scorer = ViolenceScorer(Config(), path)
    assert scorer.uses_model is False                     # kembali ke aturan; tidak memakai model yang tak cocok
    assert score_after(punch) == [PairScore((1, 2), 1.0)]


def test_model_receives_summary_and_series(tmp_path):
    import joblib
    from sahabat.pair_features import WINDOW_FEATURE_NAMES

    class ShapeModel(CountingModel):
        def predict_proba(self, X):
            assert X.shape[1] == len(WINDOW_FEATURE_NAMES) == 82
            return super().predict_proba(X)
    path = tmp_path / "m.joblib"
    import tests.test_violence as here
    here.ShapeModel = ShapeModel
    ShapeModel.__qualname__ = "ShapeModel"
    joblib.dump(ShapeModel(), path)
    h = History(Config())
    feed(h, punch, seconds=1.5)
    assert ViolenceScorer(Config(), path).score(h.tracks)[0].prob == 1.0


def test_model_cannot_raise_the_score_for_a_motionless_pair():
    import numpy as np

    class Overconfident:
        def predict_proba(self, X):
            return np.column_stack((np.zeros(len(X)), np.ones(len(X))))

    h = History(Config())
    feed(h, two, seconds=2.0)
    scorer = ViolenceScorer(Config())
    scorer._model = Overconfident()
    assert scorer.score(h.tracks) == [PairScore((1, 2), 0.0)]


# ---- posisi duduk tidak boleh membuat kekerasan diabaikan ----

def seated(p):
    """Orang yang sama tetapi duduk bersila: lutut dan pergelangan kaki setinggi pinggul."""
    import dataclasses

    from sahabat.types import L_ANK, L_HIP, L_KNEE, R_ANK, R_HIP, R_KNEE
    kp = p.kp.copy()
    hip_y = float(kp[[L_HIP, R_HIP], 1].mean())
    kp[[L_KNEE, R_KNEE], 1] = hip_y + 10.0
    kp[[L_ANK, R_ANK], 1] = hip_y + 15.0
    return dataclasses.replace(p, kp=kp)


def test_people_sitting_on_the_floor_are_still_scored_for_violence():
    # Gerak pukulan tetap dinilai, termasuk bila korbannya sedang duduk.
    sitting = lambda t: [seated(p) for p in punch(t)]
    assert score_after(sitting) == [PairScore((1, 2), 1.0)]
    one_sits = lambda t: [punch(t)[0], seated(punch(t)[1])]
    assert score_after(one_sits) == [PairScore((1, 2), 1.0)]


def test_standing_people_are_still_scored():
    assert score_after(punch) == [PairScore((1, 2), 1.0)]


def test_half_body_view_is_still_scored_because_legs_cannot_be_judged():
    upper = lambda t: [with_hidden_legs(p) for p in punch(t)]
    assert score_after(upper, seconds=2.0)[0].prob == 1.0


# ---- model pendamping untuk pasangan setengah badan (kaki keduanya tidak terlihat) ----

class ConstantModel:
    n_jobs = 4

    def __init__(self, value, **meta):
        self.value, self.calls = value, 0
        self.__dict__.update(meta)

    def predict_proba(self, X):
        import numpy as np
        self.calls += 1
        return np.column_stack((np.full(len(X), 1 - self.value), np.full(len(X), self.value)))


def scorer_with_companion(tmp_path, main=0.0, half=0.9, own=0.9):
    import joblib
    joblib.dump(ConstantModel(main), tmp_path / "violence_model.joblib")
    joblib.dump(ConstantModel(half, own_threshold_=own), tmp_path / "violence_model_setengah.joblib")
    return ViolenceScorer(Config(), tmp_path / "violence_model.joblib")


def tracks_of(script, seconds=1.5):
    h = History(Config())
    feed(h, script, seconds=seconds)
    return h.tracks


def test_half_body_pair_is_scored_by_the_companion_on_its_own_threshold(tmp_path):
    scorer = scorer_with_companion(tmp_path, half=0.9, own=0.9)
    assert scorer._half.n_jobs == 1
    upper = lambda t: [with_hidden_legs(p) for p in punch(t)]
    [score] = scorer.score(tracks_of(upper, seconds=2.0))
    # 0,9 pada ambang pendamping 0,9 setara 0,8 pada ambang aplikasi
    assert abs(score.prob - Config().violence_threshold) < 1e-9


def test_whole_body_pair_keeps_the_main_model(tmp_path):
    scorer = scorer_with_companion(tmp_path, main=0.3, half=1.0)
    assert scorer.score(tracks_of(punch)) == [PairScore((1, 2), 0.3)]
    assert scorer._half.calls == 0


def test_pair_with_one_whole_body_keeps_the_main_model(tmp_path):
    scorer = scorer_with_companion(tmp_path, main=0.3, half=1.0)
    one_upper = lambda t: [punch(t)[0], with_hidden_legs(punch(t)[1])]
    assert scorer.score(tracks_of(one_upper, seconds=2.0)) == [PairScore((1, 2), 0.3)]


def test_companion_cannot_raise_a_motionless_pair(tmp_path):
    scorer = scorer_with_companion(tmp_path, half=1.0, own=0.5)
    still = lambda t: [with_hidden_legs(p) for p in two(t)]
    assert scorer.score(tracks_of(still, seconds=2.0)) == [PairScore((1, 2), 0.0)]


def test_without_companion_file_half_body_pairs_use_the_main_model(tmp_path):
    import joblib
    joblib.dump(ConstantModel(0.3), tmp_path / "violence_model.joblib")
    scorer = ViolenceScorer(Config(), tmp_path / "violence_model.joblib")
    upper = lambda t: [with_hidden_legs(p) for p in punch(t)]
    assert scorer._half is None
    assert scorer.score(tracks_of(upper, seconds=2.0)) == [PairScore((1, 2), 0.3)]


def test_companion_with_other_feature_count_is_ignored(tmp_path):
    import joblib
    joblib.dump(ConstantModel(0.3), tmp_path / "violence_model.joblib")
    joblib.dump(ConstantModel(1.0, n_features_in_=12), tmp_path / "violence_model_setengah.joblib")
    assert ViolenceScorer(Config(), tmp_path / "violence_model.joblib")._half is None
