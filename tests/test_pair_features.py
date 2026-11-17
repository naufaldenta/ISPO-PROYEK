from sahabat.config import Config
from sahabat.history import History
from sahabat.pair_features import (FEATURE_NAMES, SEQ_TICKS, SERIES_NAMES, WINDOW_FEATURE_NAMES,
                                   both_directions, close_pairs, directed_features, pair_windows)

from .helpers import feed, person, with_hidden_legs
from .scenes import handshake, kick, punch, two


def tracks_of(positions):
    h = History(Config())
    feed(h, lambda t: [person(i + 1, cx=x, hip_y=y) for i, (x, y) in enumerate(positions)], seconds=1.0)
    return h.tracks


def features_after(script, seconds):
    h = History(Config())
    feed(h, script, seconds=seconds)
    return directed_features(h.tracks[1], h.tracks[2], 1.0)


def named(script, seconds=1.5):
    return dict(zip(FEATURE_NAMES, features_after(script, seconds)))


def test_close_pairs_distance_and_count():
    assert [(a.track_id, b.track_id) for a, b in close_pairs(tracks_of([(250, 300), (400, 300)]), 2.5)] == [(1, 2)]
    assert close_pairs(tracks_of([(100, 300), (500, 300)]), 2.5) == []
    assert len(close_pairs(tracks_of([(200, 300), (300, 300), (400, 300), (500, 300)]), 4.0)) == 6


def test_feature_vector_shape():
    f = features_after(two, seconds=1.5)
    assert f.shape == (len(FEATURE_NAMES),) and len(FEATURE_NAMES) == 12


def test_punch_features():
    f = named(punch)
    assert f["wrist_toward_max"] >= 2.0 and f["wrist_dist_min"] <= 0.6
    assert f["elbow_straight_max"] > 0.5


def test_kick_features():
    f = named(kick)
    assert f["ankle_toward_max"] >= 2.0 and f["ankle_dist_min"] <= 0.6


def test_standing_still_features_are_small():
    f = named(two)
    assert f["wrist_toward_max"] < 0.2 and f["closing_speed_max"] < 0.2
    assert f["b_away_speed_max"] < 0.2 and f["b_accel_max"] < 0.2
    assert 1.4 < f["torso_dist_mean"] < 1.6


def test_handshake_is_slow():
    assert named(handshake)["wrist_toward_max"] < 1.0


def test_too_few_shared_samples_is_none():
    assert features_after(two, seconds=0.2) is None


def test_upper_mode_actor_has_neutral_leg_features():
    upper_kick = lambda t: [with_hidden_legs(p) if p.track_id == 1 else p for p in kick(t)]
    f = named(upper_kick)
    assert (f["ankle_toward_max"], f["ankle_dist_min"], f["knee_straight_max"]) == (0.0, 9.0, 0.0)


def test_both_directions_equals_two_separate_calls():
    import numpy as np
    h = History(Config())
    feed(h, kick, seconds=1.5)
    a, b = h.tracks[1], h.tracks[2]
    ab, ba = both_directions(a, b, 1.0)
    assert np.array_equal(ab, directed_features(a, b, 1.0))
    assert np.array_equal(ba, directed_features(b, a, 1.0))
    assert not np.array_equal(ab, ba)
    h2 = History(Config())
    feed(h2, two, seconds=0.2)
    assert both_directions(h2.tracks[1], h2.tracks[2], 1.0) is None


def test_reference_size_follows_history_changes():
    h = History(Config())
    feed(h, lambda t: [person(1, cx=300, hip_y=300, size=100)], seconds=1.0)
    assert abs(h.tracks[1].ref_size - 100) < 1
    feed(h, lambda t: [person(1, cx=300, hip_y=300, size=200)], t0=1.0, seconds=3.0)
    assert abs(h.tracks[1].ref_size - 200) < 1            # riwayat lama sudah terbuang; nilai simpanan ikut berubah


# ---- jendela lengkap: ringkasan + deret jarak per langkah ----

def windows_after(script, seconds):
    h = History(Config())
    feed(h, script, seconds=seconds)
    return h.tracks[1], h.tracks[2], pair_windows(h.tracks[1], h.tracks[2], 1.0)


def series(vec, name):
    i = len(FEATURE_NAMES) + SERIES_NAMES.index(name) * SEQ_TICKS
    return vec[i:i + SEQ_TICKS]


def test_window_vector_is_summary_plus_series():
    import numpy as np
    a, b, (ab, ba) = windows_after(kick, 1.5)
    assert len(WINDOW_FEATURE_NAMES) == 12 + 7 * 10 == ab.shape[0] == ba.shape[0] and ab.dtype == np.float32
    assert len(set(WINDOW_FEATURE_NAMES)) == len(WINDOW_FEATURE_NAMES)
    summary_ab, summary_ba = both_directions(a, b, 1.0)
    assert np.array_equal(ab[:12], summary_ab) and np.array_equal(ba[:12], summary_ba)


def test_series_shows_the_punch_arriving_over_time():
    import numpy as np
    _, _, (ab, ba) = windows_after(punch, 1.4)            # pukulan t=1,0..1,3; jendela berakhir di t=1,3
    wrist = series(ab, "wrist_r")
    assert wrist[-1] < 0.4 and wrist[0] > 1.0 and np.all(np.diff(wrist[-4:]) <= 1e-3)      # makin dekat
    assert np.allclose(series(ab, "torso"), 1.5, atol=0.05)
    assert np.allclose(series(ab, "b_head_shift"), 0.0, atol=0.02)
    assert np.allclose(series(ba, "wrist_r"), series(ba, "wrist_r")[0], atol=0.02)         # B diam


def test_series_shows_the_victim_being_pushed():
    from .scenes import push
    _, _, (ab, _) = windows_after(push, 1.4)
    assert series(ab, "b_torso_shift")[-1] > 0.4 and series(ab, "b_head_shift")[-1] > 0.4


def test_hidden_legs_give_nan_series_not_a_crash():
    import numpy as np
    upper = lambda t: [with_hidden_legs(p) if p.track_id == 1 else p for p in kick(t)]
    _, _, (ab, ba) = windows_after(upper, 1.5)
    assert np.isnan(series(ab, "ankle_l")).all() and np.isnan(series(ab, "ankle_r")).all()
    assert np.isfinite(series(ab, "wrist_r")).all() and np.isfinite(series(ba, "ankle_r")).all()


def test_too_little_history_gives_no_window():
    _, _, both = windows_after(two, 0.2)
    assert both is None


def test_uneven_frame_times_still_fill_the_latest_slot():
    import numpy as np
    h = History(Config())
    t = 0.0
    for i in range(40):                                     # jarak antar frame 0,07..0,13 dtk
        t += 0.1 + 0.03 * (1 if i % 3 == 0 else -1 if i % 3 == 1 else 0)
        h.update(t, two(t))
    ab, ba = pair_windows(h.tracks[1], h.tracks[2], 1.0)
    torso = series(ab, "torso")
    assert np.isfinite(torso[-1]) and np.isfinite(torso).sum() >= 7 and np.nanmax(np.abs(torso - 1.5)) < 0.05


def test_slow_camera_still_fills_every_series_slot():
    import numpy as np
    from .scenes import handshake
    h = History(Config())
    t = 0.0
    for _ in range(30):                                     # 7,5 gambar per detik, seperti webcam di ruangan redup
        t += 1 / 7.5
        h.update(t, handshake(t))
    ab, _ = pair_windows(h.tracks[1], h.tracks[2], 1.0)
    torso, wrist = series(ab, "torso"), series(ab, "wrist_r")
    assert np.isfinite(torso).all() and np.allclose(torso, 1.5, atol=0.05)
    assert np.isfinite(wrist).all() and np.all(np.diff(wrist) <= 1e-3)          # tangan terus mendekat, tanpa lubang


def test_long_gap_is_not_bridged():
    import numpy as np
    h = History(Config())
    for t in (0.0, 0.1, 0.2, 0.3, 0.9, 1.0, 1.1, 1.2):      # 0,6 dtk tanpa data di tengah
        h.update(t, two(t))
    ab, _ = pair_windows(h.tracks[1], h.tracks[2], 1.0)
    torso = series(ab, "torso")
    assert np.isfinite(torso[-4:]).all() and np.isnan(torso[3:6]).all()
