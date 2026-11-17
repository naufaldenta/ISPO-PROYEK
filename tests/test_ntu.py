import pickle

import numpy as np
import pytest

from latih.ntu import parse_name, safe_load, to_frames
from latih.ntu_build import build_clip, load_cache, save_cache
from sahabat.config import Config
from sahabat.pair_features import WINDOW_FEATURE_NAMES as FEATURE_NAMES

from .helpers import ticks
from .scenes import punch, two


def as_ntu(script, seconds, fps=30.0):
    """Adegan sintetis dalam bentuk larik NTU: keypoint (M, T, 17, 2), score (M, T, 17)."""
    times = ticks(0.0, seconds, hz=fps)
    kp = np.zeros((2, len(times), 17, 2), dtype=np.float16)
    score = np.zeros((2, len(times), 17), dtype=np.float16)
    for i, t in enumerate(times):
        for p in script(t):
            kp[p.track_id - 1, i] = p.kp
            score[p.track_id - 1, i] = p.conf
    return kp, score


def test_parse_name():
    assert parse_name("S001C002P003R002A050") == dict(setup=1, camera=2, performer=3, replication=2, action=50)
    with pytest.raises(ValueError):
        parse_name("bukan-nama-ntu")


def test_safe_load_reads_arrays(tmp_path):
    path = tmp_path / "aman.pkl"
    path.write_bytes(pickle.dumps({"annotations": [{"label": 3, "keypoint": np.ones((1, 2, 17, 2), dtype=np.float16)}]}))
    assert safe_load(path)["annotations"][0]["keypoint"].shape == (1, 2, 17, 2)


def test_safe_load_refuses_to_run_code(tmp_path):
    class Evil:
        def __reduce__(self):
            import os
            return (os.system, ("echo dijebol",))
    path = tmp_path / "jahat.pkl"
    path.write_bytes(pickle.dumps({"x": Evil()}))
    with pytest.raises(pickle.UnpicklingError):
        safe_load(path)


def test_to_frames_resamples_to_10_fps_and_keeps_both_people():
    kp, score = as_ntu(two, seconds=2.0)
    frames = to_frames(kp, score)
    assert len(frames) == 20 and [t for t, _ in frames][:3] == [0.0, 0.1, 0.2]
    assert [p.track_id for p in frames[5][1]] == [1, 2] and frames[5][1][0].kp.shape == (17, 2)


def test_to_frames_skips_absent_person_and_can_hold_the_last_frame():
    kp, score = as_ntu(two, seconds=1.0)
    score[1] = 0.0                                             # orang kedua tidak pernah ada
    frames = to_frames(kp, score, hold_last_s=1.5)
    assert all(len(people) == 1 for _, people in frames)
    assert len(frames) == 10 + 15 and frames[-1][0] == pytest.approx(2.4)


def test_build_clip_gives_pair_features():
    kp, score = as_ntu(punch, seconds=3.0)
    clip = build_clip(kp, score, Config())
    assert len(clip.t) == 30 and len(clip.rows) > 10
    assert clip.rows[0][3].shape == (len(FEATURE_NAMES),)


def test_cache_roundtrip(tmp_path):
    kp, score = as_ntu(punch, seconds=3.0)
    clip = build_clip(kp, score, Config())
    meta = [dict(action=50, performer=7, setup=1, camera=2), dict(action=58, performer=8, setup=1, camera=1)]
    empty = build_clip(*as_ntu(lambda t: [], seconds=1.0), Config())
    save_cache(tmp_path / "c.npz", [clip, empty], meta)
    clips, got = load_cache(tmp_path / "c.npz")
    assert got == meta and len(clips) == 2
    assert clips[0].t == clip.t and len(clips[0].rows) == len(clip.rows) and clips[1].rows == []
    assert clips[0].rows[3][:3] == clip.rows[3][:3] and np.array_equal(clips[0].rows[3][4], clip.rows[3][4], equal_nan=True)
    assert len(clips[1].t) == 10


def test_fall_fired_replays_a_clip_through_the_app_detector():
    from latih.ntu_fall import fall_fired
    from .test_fall import fall_script
    falling = [(t, fall_script(t)) for t in ticks(0.0, 6.0)]
    standing = [(t, fall_script(t, t_fall=99.0)) for t in ticks(0.0, 6.0)]
    assert fall_fired(falling, Config(), 480) is True
    assert fall_fired(standing, Config(), 480) is False
