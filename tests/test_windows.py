import dataclasses

import numpy as np

from latih.extract_pose import save_clip
from latih.train_rf import (clip_alarm, folds_by_clip, group_of, keep_clip, label_of, load_actions,
                            training_rows)
from latih.windows import clip_pairs, clip_windows, load_clip
from sahabat.config import Config
from sahabat.pair_features import WINDOW_FEATURE_NAMES as FEATURE_NAMES

from .helpers import ticks
from .scenes import punch_then_still, single_person, two

CFG = dataclasses.replace(Config(), violence_threshold=0.6, confirm_k=4, confirm_n=6)


def make_npz(tmp_path, script, seconds):
    path = tmp_path / "klip.npz"
    save_clip(path, [(t, script(t)) for t in ticks(0.0, seconds)], (480, 640))
    return path


def test_clip_roundtrip(tmp_path):
    frames = load_clip(make_npz(tmp_path, two, seconds=1.0))
    assert len(frames) == 10 and [p.track_id for p in frames[3][1]] == [1, 2]
    assert frames[3][0] == 0.3 and frames[3][1][0].kp.shape == (17, 2)


def test_roundtrip_keeps_empty_frames(tmp_path):
    frames = load_clip(make_npz(tmp_path, lambda t: [] if t < 0.5 else single_person(t), seconds=1.0))
    assert [len(people) for _, people in frames] == [0] * 5 + [1] * 5


def test_clip_pairs_gives_both_directions_per_frame(tmp_path):
    clip = clip_pairs(make_npz(tmp_path, two, seconds=2.0), Config())
    assert clip.t == ticks(0.0, 2.0) and len(clip.rows) >= 10
    frame, a, b, fab, fba = clip.rows[0]
    assert (a, b) == (1, 2) and fab.shape == fba.shape == (len(FEATURE_NAMES),)
    assert [r[0] for r in clip.rows] == sorted(r[0] for r in clip.rows)


def test_nonviolent_clip_keeps_all_windows(tmp_path):
    X = clip_windows(make_npz(tmp_path, two, seconds=4.0), 0, Config())
    assert X.shape[1] == len(FEATURE_NAMES) and len(X) >= 40          # 2 arah x ±30 langkah


def test_violent_clip_drops_quiet_windows(tmp_path):
    X = clip_windows(make_npz(tmp_path, punch_then_still, seconds=6.0), 1, Config())
    assert 0 < len(X) < 40                                             # hanya sekitar pukulan


def test_clip_without_pairs_gives_empty(tmp_path):
    X = clip_windows(make_npz(tmp_path, single_person, seconds=3.0), 1, Config())
    assert X.shape == (0, len(FEATURE_NAMES))


def test_training_rows_match_clip_windows(tmp_path):
    path = make_npz(tmp_path, punch_then_still, seconds=6.0)
    clip = clip_pairs(path, Config())
    for label in (0, 1):
        assert np.array_equal(training_rows(clip, label, 1.0), clip_windows(path, label, Config()), equal_nan=True)


def test_labels_come_from_folder_names():
    assert label_of("violence-detection-dataset/violent/cam1/12.npz") == 1
    assert label_of("violence-detection-dataset/non-violent/cam2/12.npz") == 0
    assert label_of("lain/12.npz") is None


def test_two_cameras_of_one_scene_share_a_group():
    a = group_of("violence-detection-dataset/violent/cam1/12.npz")
    b = group_of("violence-detection-dataset/violent/cam2/12.npz")
    c = group_of("violence-detection-dataset/non-violent/cam1/12.npz")
    assert a == b and a != c


def test_folds_never_share_a_scene():
    groups = np.repeat(np.arange(10), 3)
    folds = folds_by_clip(groups, 5)
    assert len(folds) == 5
    for train, test in folds:
        assert set(groups[train]).isdisjoint(groups[test]) and len(set(groups[test])) == 2
    assert sorted(i for _, test in folds for i in test) == list(range(30))     # tiap klip diuji tepat sekali


def test_only_target_violent_actions_are_kept(tmp_path):
    csv = tmp_path / "violent-action-classes.csv"
    csv.write_text("FILE; ACTION CLASSES\n1.mp4;slap\n4.mp4;club\n6.mp4;walk,gunshot\n3.mp4;slap,kick\n", encoding="utf-8")
    actions = load_actions(csv)
    assert actions == {"1": {"slap"}, "4": {"club"}, "6": {"walk", "gunshot"}, "3": {"slap", "kick"}}
    assert keep_clip(1, "1", actions) and keep_clip(1, "3", actions)
    assert not keep_clip(1, "4", actions) and not keep_clip(1, "6", actions)     # tongkat, tembakan: di luar cakupan
    assert not keep_clip(1, "99", actions)                                        # tidak ada di daftar
    assert keep_clip(0, "4", actions)                                             # klip nonkekerasan selalu dipakai
    assert keep_clip(1, "4", None)                                                # tanpa daftar: semua dipakai


def test_clip_alarm_uses_the_same_confirmation_as_the_app(tmp_path):
    clip = clip_pairs(make_npz(tmp_path, two, seconds=3.0), Config())
    n = len(clip.rows)
    assert clip_alarm(clip, np.full(n, 0.9), CFG) is True
    assert clip_alarm(clip, np.full(n, 0.5), CFG) is False
    spike = np.zeros(n)
    spike[5] = 0.99                                                    # satu lonjakan saja: bukan alarm
    assert clip_alarm(clip, spike, CFG) is False
    strict = dataclasses.replace(CFG, violence_threshold=0.95)
    assert clip_alarm(clip, np.full(n, 0.9), strict) is False
