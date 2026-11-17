from latih.evaluasi_sesi import EventRecorder
from sahabat.config import Config
from sahabat.violence import PairScore


class BrokenFallModel:
    n_features_in_ = 17

    def predict_proba(self, X):
        raise RuntimeError("simulasi model rusak saat prediksi")


class QuietViolenceModel:
    n_features_in_ = 82

    def predict_proba(self, X):
        import numpy as np
        return np.column_stack((np.ones(len(X)), np.zeros(len(X))))


def test_recorder_keeps_simultaneous_events_and_pending_alarm_does_not_hide_new_ones():
    import dataclasses
    cfg = dataclasses.replace(Config(), confirm_k=1, confirm_n=1)
    recorder = EventRecorder(cfg, 100.0)
    recorder.tick(101.0, [1], [PairScore((2, 3), 1.0)], {1: "upper", 2: "full", 3: "full"})
    recorder.tick(102.0, [4], [], {4: "full"})
    assert [(e["t"], e["kind"], e["ids"]) for e in recorder.events] == [
        (1.0, "fall", [1]), (1.0, "violence", [2, 3]), (2.0, "fall", [4])]
    assert recorder.active.ids == (1,)


def test_replay_fails_if_prediction_fails_instead_of_reporting_a_clean_session(tmp_path):
    import joblib
    import numpy as np
    import pytest
    from latih.evaluasi_sesi import replay
    from latih.putar import PoseTrace
    from .helpers import person
    fall, violence = tmp_path / "fall.joblib", tmp_path / "violence.joblib"
    joblib.dump(BrokenFallModel(), fall)
    joblib.dump(QuietViolenceModel(), violence)
    tr = PoseTrace(np.arange(30) / 10, 480, 640,
                   [[person(1, cx=300, hip_y=300)] for _ in range(30)])
    with pytest.raises(RuntimeError, match="Replay gagal"):
        replay(tr, violence, fall)
