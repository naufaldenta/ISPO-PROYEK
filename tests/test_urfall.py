import numpy as np
import pytest

from latih.putar import Alarm, PoseTrace
from latih.urfall import ClipResult, fall_onsets, hold, summarize

from .helpers import person

CSV = """fall-01,1,-1,3.1,2.9
fall-01,2,-1,3.3,2.9
fall-01,61,0,3.1,3.0
fall-01,62,0,3.1,3.0
fall-01,91,1,3.1,3.0
fall-02,10,-1,1,1
fall-02,31,1,1,1
adl-05,1,-1,1,1
"""


def test_fall_onset_is_the_first_falling_frame(tmp_path):
    path = tmp_path / "label.csv"
    path.write_text(CSV)
    onsets = fall_onsets(path, fps=30.0)
    assert onsets["fall-01"] == pytest.approx(2.0)          # frame 61 pada 30 gambar per detik
    assert onsets["fall-02"] == pytest.approx(1.0)          # tidak ada label "sedang jatuh": pakai "tergeletak"
    assert "adl-05" not in onsets


def test_hold_repeats_the_last_pose():
    tr = PoseTrace(np.array([0.0, 0.1, 0.2]), 480, 640, [[], [], [person(1, cx=300, hip_y=300)]])
    held = hold(tr, seconds=0.5, rate_hz=10.0)
    assert list(held.t) == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert [len(ps) for ps in held.people] == [0, 0, 1, 1, 1, 1, 1, 1]
    assert len(tr.t) == 3                                     # jejak asli tidak diubah


def test_hold_zero_or_empty_changes_nothing():
    tr = PoseTrace(np.array([0.0, 0.1]), 480, 640, [[], []])
    assert len(hold(tr, 0.0, 10.0).t) == 2
    assert len(hold(PoseTrace(np.array([]), 480, 640, []), 1.5, 10.0).t) == 0


def test_summary_counts_clips_and_delays():
    results = [
        ClipResult("fall-01", True, 0, [Alarm(3.0, "fall")], 2.0),
        ClipResult("fall-01", True, 1, [], 2.0),
        ClipResult("fall-02", True, 0, [Alarm(0.5, "fall"), Alarm(4.0, "fall")], 1.0),   # 0,5 dtk: sebelum jatuh
        ClipResult("adl-01", False, 0, [], None),
        ClipResult("adl-02", False, 0, [Alarm(1.0, "fall")], None),
        ClipResult("adl-03", False, 0, [Alarm(1.0, "violence")], None),
    ]
    s = summarize(results)
    assert (s.falls, s.falls_caught) == (3, 2)
    assert s.delays == pytest.approx([1.0, 3.0])               # alarm pertama setelah awal jatuh
    assert s.early == ["fall-02"]                              # alarm sebelum orangnya jatuh
    assert (s.adls, s.adl_false) == (3, 1)
    assert s.false_names == ["adl-02"] and s.missed_names == ["fall-01"]
    assert s.violence_names == ["adl-03"]


def test_fall_times_give_onset_and_lying_start(tmp_path):
    from latih.urfall import fall_times
    path = tmp_path / "label.csv"
    path.write_text(CSV)
    times = fall_times(path, fps=30.0)
    assert times["fall-01"] == pytest.approx((2.0, 3.0))       # jatuh di frame 61, tergeletak di frame 91
    assert times["fall-02"] == pytest.approx((1.0, 1.0)) and "adl-05" not in times
