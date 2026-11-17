import numpy as np
import pytest

from sahabat.config import Config
from sahabat.history import History, body_size, raw_mode, torso_center
from sahabat.types import L_SH

from .helpers import feed, person, skeleton


def test_raw_mode():
    assert raw_mode(skeleton(0, 0)[1], 0.5) == "full"
    assert raw_mode(skeleton(0, 0, visible="upper")[1], 0.5) == "upper"
    assert raw_mode(skeleton(0, 0, visible="none")[1], 0.5) == "none"


def test_body_size_full_and_upper():
    kp, conf = skeleton(300, 300, size=100)
    assert body_size(kp, conf, 0.5) == pytest.approx(100, abs=1)
    kp, conf = skeleton(300, 300, size=100, visible="upper")
    assert body_size(kp, conf, 0.5) == pytest.approx(98, abs=2)      # 70 * 1.4


def test_body_size_degenerate_is_none():
    kp, conf = skeleton(300, 300, size=3)                              # < 8 piksel
    assert body_size(kp, conf, 0.5) is None
    kp, conf = skeleton(300, 300)
    kp[L_SH] = np.nan
    assert body_size(kp, conf, 0.5) is None
    kp, conf = skeleton(300, 300, visible="none")
    assert body_size(kp, conf, 0.5) is None


def test_new_track_is_none_then_full_after_hysteresis():
    h = History(Config())
    feed(h, lambda t: [person(1, cx=300, hip_y=300)], seconds=0.3)
    assert h.tracks[1].mode == "none"
    feed(h, lambda t: [person(1, cx=300, hip_y=300)], t0=0.4, seconds=0.3)
    assert h.tracks[1].mode == "full"


def test_mode_does_not_flicker():
    h = History(Config())
    t = feed(h, lambda t: [person(1, cx=300, hip_y=300)], seconds=1.0)
    h.update(t + 0.1, [person(1, cx=300, hip_y=300, visible="upper")])
    assert h.tracks[1].mode == "full"                                 # satu frame tidak mengubah mode
    feed(h, lambda t: [person(1, cx=300, hip_y=300, visible="upper")], t0=t + 0.2, seconds=0.6)
    assert h.tracks[1].mode == "upper"


def test_only_four_largest_are_assessed():
    h = History(Config())
    ppl = [person(i, cx=100 * i, hip_y=300, size=40 + 10 * i) for i in range(1, 7)]
    h.update(0.0, ppl)
    assert sorted(h.tracks) == [3, 4, 5, 6] and sorted(h.overflow_ids) == [1, 2]


def test_old_tracks_and_samples_are_pruned():
    h = History(Config())
    for i in range(200):                                              # nomor berganti tiap frame
        h.update(i * 0.1, [person(i, cx=300, hip_y=300)])
    assert len(h.tracks) <= 11                                        # ttl 1,0 dtk pada 10 Hz
    h2 = History(Config())
    feed(h2, lambda t: [person(1, cx=300, hip_y=300)], seconds=10.0)
    assert len(h2.tracks[1].samples) <= 21                            # history_s 2,0 dtk


def test_degenerate_person_gets_track_without_samples():
    h = History(Config())
    feed(h, lambda t: [person(1, cx=300, hip_y=300, size=3)], seconds=1.0)
    assert h.tracks[1].mode == "none" and len(h.tracks[1].samples) == 0


def test_torso_center_full_and_upper():
    h = History(Config())
    feed(h, lambda t: [person(1, cx=300, hip_y=300, size=100),
                       person(2, cx=600, hip_y=300, size=100, visible="upper")], seconds=1.0)
    assert torso_center(h.tracks[1].samples[-1]) == pytest.approx([300, 250], abs=1)
    assert torso_center(h.tracks[2].samples[-1]) == pytest.approx([600, 200], abs=1)


# ---- menyambung nomor baru ke orang yang baru saja hilang, dan membuang deteksi dobel ----

def lost_then_new_id(gap_s, new_cx=310, new_hip_y=380):
    """Orang #1 berdiri 1 dtk, hilang gap_s detik, lalu muncul lagi sebagai #7 (misalnya sudah tergeletak)."""
    h = History(Config())
    feed(h, lambda t: [person(1, cx=300, hip_y=300)], seconds=1.0)
    t = feed(h, lambda t: [], t0=1.0, seconds=gap_s)
    back = h.update(t + 0.1, [person(7, cx=new_cx, hip_y=new_hip_y, torso_deg=80)])
    return h, back


def test_new_number_near_a_just_lost_person_continues_that_history():
    h, back = lost_then_new_id(gap_s=0.4)
    assert sorted(h.tracks) == [1] and [p.track_id for p in back] == [1]
    assert len(h.tracks[1].samples) == 11                              # 10 sampel lama + 1 baru
    more = h.update(1.6, [person(7, cx=310, hip_y=380, torso_deg=85)])
    assert [p.track_id for p in more] == [1] and len(h.tracks[1].samples) == 12     # sambungan bertahan


def test_new_number_far_away_is_a_new_person():
    h, back = lost_then_new_id(gap_s=0.4, new_cx=600)
    assert [p.track_id for p in back] == [7] and 7 in h.tracks


def test_new_number_after_a_long_gap_is_a_new_person():
    h, back = lost_then_new_id(gap_s=0.95)                             # hilang lebih dari stitch_s
    assert [p.track_id for p in back] == [7]


def test_new_number_while_the_old_one_is_still_seen_is_a_new_person():
    h = History(Config())
    feed(h, lambda t: [person(1, cx=300, hip_y=300)], seconds=1.0)
    back = h.update(1.0, [person(1, cx=300, hip_y=300), person(7, cx=420, hip_y=300)])
    assert sorted(p.track_id for p in back) == [1, 7] and sorted(h.tracks) == [1, 7]


def test_duplicate_detection_of_one_person_is_dropped():
    h = History(Config())
    feed(h, lambda t: [person(1, cx=300, hip_y=300)], seconds=1.0)
    back = h.update(1.0, [person(5, cx=304, hip_y=303), person(1, cx=300, hip_y=300)])
    assert [p.track_id for p in back] == [1] and sorted(h.tracks) == [1]      # yang sudah punya riwayat dipertahankan


def test_two_people_side_by_side_are_both_kept():
    h = History(Config())
    back = h.update(0.0, [person(1, cx=300, hip_y=300), person(2, cx=370, hip_y=300)])     # bahu bersentuhan
    assert sorted(p.track_id for p in back) == [1, 2]


def test_stitched_numbers_do_not_pile_up():
    h = History(Config())
    for i in range(300):                                               # nomor berganti tiap frame, orangnya sama
        back = h.update(i * 0.1, [person(i + 1, cx=300, hip_y=300)])
    assert sorted(h.tracks) == [1] and [p.track_id for p in back] == [1]
    assert len(h._alias) <= 12
