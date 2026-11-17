from dataclasses import replace

import numpy as np
import pytest

from sahabat.config import Config
from sahabat.fall import FallDetector
from sahabat.history import History

from .helpers import person, ticks
from .test_fall import fall_script


class ConstantModel:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, features):
        p = np.full(len(features), self.probability)
        return np.column_stack((1 - p, p))


def replay(script, probability, threshold=0.5):
    cfg = replace(Config(), fall_rule_model_threshold=threshold)
    history = History(cfg)
    detector = FallDetector(cfg)
    if probability is not None:
        detector._model = ConstantModel(probability)
    events = []
    for t in ticks(0, 6):
        history.update(t, script(t))
        events.extend((t, detector.sources[i]) for i in detector.tick(t, history.tracks, 480))
    return events


@pytest.mark.parametrize(("probability", "fired"), [(0.0, False), (0.6, True), (None, True)])
def test_whole_body_rule_uses_available_model_support(probability, fired):
    assert bool(replay(fall_script, probability)) == fired


def test_zero_threshold_preserves_rule_without_model_support():
    assert replay(fall_script, 0.0, threshold=0)


@pytest.mark.parametrize(("metadata", "fired"), [(None, True), (0.5, False)])
def test_support_threshold_follows_the_installed_model(tmp_path, metadata, fired):
    import joblib

    model = ConstantModel(0.0)
    if metadata is not None:
        model.fall_rule_support_threshold_ = metadata
    path = tmp_path / "fall.joblib"
    joblib.dump(model, path)
    cfg = Config()
    history = History(cfg)
    detector = FallDetector(cfg, path)
    events = []
    for t in ticks(0, 6):
        history.update(t, fall_script(t))
        events.extend(detector.tick(t, history.tracks, 480))
    assert bool(events) == fired


def test_upper_body_fall_with_no_full_body_features_keeps_its_alarm():
    def drop(t):
        if t < 2:
            return [person(1, cx=300, hip_y=400, visible="upper")]
        if t < 2.4:
            return [person(1, cx=300, hip_y=400 + 400 * (t - 2), visible="upper")]
        return []

    assert any(source == "setengah_badan" for _, source in replay(drop, 0.0))


@pytest.mark.parametrize(("raw", "value"), [("0.5", 0.5), ("none", None)])
def test_replay_can_override_optional_support_threshold(raw, value):
    from latih.putar import _overrides

    assert _overrides([f"fall_rule_model_threshold={raw}"]) == {"fall_rule_model_threshold": value}


def test_training_snapshot_accepts_fields_from_other_project_revisions(tmp_path):
    from latih.putar import load_config_snapshot

    path = tmp_path / "config.json"
    path.write_text('{"violence_threshold": 0.7, "future_audio_option": 9}', encoding="utf-8")
    assert load_config_snapshot(path).violence_threshold == 0.7
