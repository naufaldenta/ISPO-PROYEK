import numpy as np
import pytest

from latih.laju_rendah import subsample
from latih.putar import PoseTrace
from sahabat.types import Person


def make_trace(n=30, hz=10.0):
    people = [[Person(1, np.zeros((17, 2), np.float32), np.ones(17, np.float32), (0.0, 0.0, 1.0, 1.0))]
              for _ in range(n)]
    return PoseTrace(np.arange(n, dtype=np.float64) / hz, 480, 640, people)


def test_subsample_keeps_every_nth_tick_and_times():
    tr = make_trace(n=30, hz=10.0)
    half = subsample(tr, 2)
    assert len(half.t) == 15 and len(half.people) == 15
    assert half.t[1] - half.t[0] == pytest.approx(0.2) and (half.frame_h, half.frame_w) == (480, 640)
    assert list(subsample(tr, 1).t) == list(tr.t)
    third = subsample(tr, 3)
    assert len(third.t) == 10 and third.t[1] == pytest.approx(0.3)
