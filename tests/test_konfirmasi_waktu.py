"""Konfirmasi berbasis waktu (profil kamera dekat): pukulan singkat tetap terkonfirmasi walau penilaian hanya 5 per detik."""
import numpy as np

from sahabat.alarm import AlarmManager
from sahabat.asisten_kamera import Asisten
from sahabat.config import Config
from sahabat.engine import Engine
from sahabat.types import Person
from sahabat.violence import PairScore

PAIR = (1, 2)


def _run(manager, scores, hz):
    fired = []
    for i, p in enumerate(scores):
        fired += [e.t for e in manager.tick(i / hz, [], [PairScore(PAIR, p)], {1: "upper", 2: "upper"})]
    return fired


def _timed():
    manager = AlarmManager(Config())
    manager.atur_konfirmasi(1.0)
    return manager


def test_default_rule_is_unchanged_and_misses_short_burst_at_5hz():
    assert Config().confirm_window_s is None
    assert _run(AlarmManager(Config()), [0.1, 0.9, 0.9, 0.9, 0.1, 0.1], 5) == []


def test_default_rule_still_fires_on_six_of_ten():
    assert _run(AlarmManager(Config()), [0.9] * 6, 10) != []


def test_time_rule_confirms_short_burst_at_5hz():
    assert _run(_timed(), [0.1, 0.9, 0.9, 0.9, 0.1, 0.1], 5) != []


def test_time_rule_ignores_single_spikes():
    assert _run(_timed(), [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9], 5) == []


def test_time_rule_needs_majority_of_the_last_second_at_10hz():
    scores = [0.1] * 8 + [0.9] * 3 + [0.1] * 8
    assert _run(_timed(), scores, 10) == []
    assert _run(_timed(), [0.1] * 8 + [0.9] * 7, 10) != []


def test_switching_profile_drops_old_votes():
    manager = _timed()
    _run(manager, [0.9, 0.9], 5)
    manager.atur_konfirmasi(None)
    assert not manager._timed and not manager._windows


def test_close_assistant_does_not_ask_for_legs():
    assistant = Asisten(clock=lambda: 0.0)
    assistant.dekat = True
    conf = np.zeros(17)
    conf[[5, 6, 7, 8, 9, 10, 11, 12]] = 0.9
    person = Person(1, np.zeros((17, 2)), conf, (0.0, 0.0, 100.0, 300.0))
    for i in range(40):
        assistant.amati(i / 10, [person], (480, 640), {1: "upper"})
    assert assistant.ringkas()["saran"] == ["Badan bisa dinilai: 100%"]


class _Pose:
    def detect(self, frame):
        return []


def _slow_text(engine, hz):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    snap = None
    for i in range(int(hz * 6)):
        snap = engine.step(i / hz, frame)
    return snap.slow_text


def test_close_profile_warns_when_assessment_is_slow():
    engine = Engine(Config(), None, _Pose())
    engine.atur_profil(1.0, True)
    assert engine.alarm.confirm_window_s == 1.0 and engine.asisten.dekat
    assert "5 kali per detik" in _slow_text(engine, 5)


def test_close_profile_is_quiet_at_full_rate():
    engine = Engine(Config(), None, _Pose())
    engine.atur_profil(1.0, True)
    assert _slow_text(engine, 10) == ""


def test_default_profile_has_no_rate_warning():
    assert _slow_text(Engine(Config(), None, _Pose()), 5) == ""
