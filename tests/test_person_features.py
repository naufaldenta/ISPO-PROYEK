import numpy as np

from sahabat.config import Config
from sahabat.history import History
from sahabat.person_features import PERSON_FEATURE_NAMES, person_features

from .helpers import feed, person
from .test_fall import fall_script


def after(script, seconds):
    h = History(Config())
    feed(h, script, seconds=seconds)
    f = person_features(h.tracks[1], 2.0)
    return None if f is None else dict(zip(PERSON_FEATURE_NAMES, f))


def standing(t):
    return [person(1, cx=300, hip_y=300)]


def sit_down(t):                    # pinggul turun 0,5 badan dalam 0,5 dtk, badan tetap tegak
    k = min(max((t - 2.0) / 0.5, 0.0), 1.0)
    return [person(1, cx=300, hip_y=300 + 50 * k)]


def fall_toward_camera(t):          # pinggul turun 1,2 badan, badan memendek di gambar, tetap "tegak" di gambar
    k = min(max((t - 2.0) / 0.4, 0.0), 1.0)
    return [person(1, cx=300, hip_y=300 + 120 * k, size=100 - 45 * k)]


def test_shape_and_names():
    f = person_features(History(Config()).tracks.get(1), 2.0) if False else None
    h = History(Config())
    feed(h, standing, seconds=2.0)
    v = person_features(h.tracks[1], 2.0)
    assert v.shape == (len(PERSON_FEATURE_NAMES),) and v.dtype == np.float32
    assert len(set(PERSON_FEATURE_NAMES)) == len(PERSON_FEATURE_NAMES)


def test_too_little_history_is_none():
    assert after(standing, seconds=0.5) is None


def test_standing_is_quiet():
    f = after(standing, seconds=3.0)
    assert abs(f["hip_drop"]) < 0.05 and f["hip_drop_speed_max"] < 0.1 and f["angle_now"] < 0.1
    assert 0.9 < f["torso_len_ratio"] < 1.1 and f["head_above_hip"] > 1.0 and f["still"] < 0.05


def test_sideways_fall_shows_drop_and_lying_posture():
    f = after(fall_script, seconds=3.2)                 # jatuh di t=2,0..2,4 lalu diam
    assert f["hip_drop"] > 0.7 and f["hip_drop_speed_max"] > 1.5
    assert f["angle_now"] > 0.9 and f["angle_change"] > 0.8
    assert f["head_above_hip"] < 0.3 and f["box_height_ratio"] < 0.7
    assert 0.5 < f["since_drop_s"] < 1.5 and f["still"] < 0.1


def test_fall_toward_camera_is_visible_without_any_tilt():
    f = after(fall_toward_camera, seconds=3.2)
    assert f["angle_now"] < 0.1                          # aturan sudut buta di sini
    assert f["hip_drop"] > 1.0 and f["torso_len_ratio"] < 0.7 and f["head_drop"] > 0.5


def test_sitting_down_drops_less_and_keeps_posture():
    f = after(sit_down, seconds=3.2)
    assert 0.3 < f["hip_drop"] < 0.7 and f["angle_now"] < 0.1 and f["torso_len_ratio"] > 0.9


def test_missing_legs_give_nan_not_a_crash():
    f = after(lambda t: [person(1, cx=300, hip_y=300, visible="upper")], seconds=3.0)
    assert np.isnan(f["ankle_below_hip"]) and np.isnan(f["knee_straight_now"])
    assert np.isfinite(f["head_drop"]) and np.isfinite(f["still"])
