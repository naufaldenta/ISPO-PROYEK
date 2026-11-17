from sahabat.alarm import AlarmManager, Event
from sahabat.config import Config
from sahabat.violence import PairScore

HIGH = [PairScore((1, 2), 0.9)]
M = {1: "full", 2: "full"}
CFG = Config(violence_threshold=0.6, confirm_k=4, confirm_n=6)     # mekanisme diuji lepas dari nilai bawaan


def test_violence_needs_k_of_n():
    a = AlarmManager(CFG)
    assert [a.tick(0.1 * i, [], HIGH, M) for i in range(3)] == [[], [], []]
    ev = a.tick(0.3, [], HIGH, M)
    assert [(e.kind, e.ids) for e in ev] == [("violence", (1, 2))]
    assert a.active.kind == "violence" and a.red_ids == {1, 2}


def test_single_spike_is_ignored():
    a = AlarmManager(CFG)
    a.tick(0.0, [], HIGH, M)
    assert all(a.tick(0.1 * i, [], [], M) == [] for i in range(1, 20)) and a.active is None


def test_low_probability_never_alarms():
    a = AlarmManager(CFG)
    low = [PairScore((1, 2), 0.5)]
    assert all(a.tick(0.1 * i, [], low, M) == [] for i in range(50))


def test_fall_is_immediate_and_latched():
    a = AlarmManager(CFG)
    assert a.tick(1.0, [7], [], {7: "upper"})[0] == Event("fall", 1.0, (7,), "upper")
    for i in range(50):
        a.tick(1.1 + 0.1 * i, [], [], {})
    assert a.active.kind == "fall" and a.red_ids == {7}


def test_violence_event_reports_weakest_mode():
    a = AlarmManager(CFG)
    ev = [e for i in range(6) for e in a.tick(0.1 * i, [], HIGH, {1: "full", 2: "upper"})]
    assert ev[0].mode == "upper"


def test_same_pair_does_not_duplicate_while_latched():
    a = AlarmManager(CFG)
    events = [e for i in range(40) for e in a.tick(0.1 * i, [], HIGH, M)]
    assert len(events) == 1


def test_oldest_unacknowledged_event_is_active():
    a = AlarmManager(CFG)
    a.tick(1.0, [7], [], {7: "full"})
    for i in range(6):
        a.tick(1.1 + 0.1 * i, [], HIGH, M)
    assert a.active.kind == "fall" and a.red_ids == {7, 1, 2}


def test_acknowledge_clears_everything_and_reports():
    a = AlarmManager(CFG)
    a.tick(1.0, [7], [], {7: "full"})
    a.tick(1.1, [8], [], {8: "full"})
    assert a.acknowledge(2.0) is True and a.active is None and a.red_ids == set()
    assert a.acknowledge(2.1) is False


def test_pair_cooldown_after_acknowledge():
    a = AlarmManager(CFG)
    for i in range(6):
        a.tick(0.1 * i, [], HIGH, M)
    a.acknowledge(1.0)
    assert all(a.tick(1.0 + 0.1 * i, [], HIGH, M) == [] for i in range(1, 50))   # sampai 5,9 dtk
    assert any(a.tick(6.1 + 0.1 * i, [], HIGH, M) for i in range(10))


def test_state_for_vanished_pairs_is_forgotten():
    a = AlarmManager(CFG)
    for i in range(500):
        a.tick(0.1 * i, [], [PairScore((i, i + 1), 0.1)], {})
    assert a.tracked_pairs() <= 60


def test_default_confirmation_matches_the_chosen_operating_point():
    a = AlarmManager(Config())
    assert all(a.tick(0.1 * i, [], HIGH, M) == [] for i in range(5))
    assert len(a.tick(0.5, [], HIGH, M)) == 1                      # 6 dari 10 pada ambang 0,8
    b = AlarmManager(Config())
    weak = [PairScore((1, 2), 0.75)]
    assert all(b.tick(0.1 * i, [], weak, M) == [] for i in range(50))


def test_gesture_and_model_share_one_latched_alarm_and_cooldown():
    from sahabat.gestures import GestureHit
    a = AlarmManager(Config())
    held = [GestureHit("kuncian", (1, 2), 1)]
    assert [e.source for e in a.tick(1, [], HIGH, M, hits=held)] == ["kuncian"]
    assert a.tick(1.1, [], HIGH, M, hits=held) == []
    assert a.acknowledge(1.2)
    assert a.tick(2, [], HIGH, M, hits=held) == []


def test_victim_loss_requires_recent_votes_and_one_exit():
    a = AlarmManager(Config())
    for i in range(3):
        assert a.tick(i / 10, [], HIGH, M, seen={1, 2}, exiting=set()) == []
    assert a.tick(0.3, [], [], M, seen={1, 2}, exiting=set()) == []
    assert a.tick(0.4, [], [], M, seen={1}, exiting=set()) == []
    assert a.tick(0.5, [], [], M, seen=set(), exiting={1, 2}) == []
    assert [e.source for e in a.tick(0.6, [], [], M, seen={1}, exiting={2})] == ["keluar_gambar"]


def test_exit_votes_discard_observations_outside_the_confirmation_window():
    alarm = AlarmManager(Config(confirm_n=6, confirm_k=5))
    low = [PairScore((1, 2), 0.1)]
    for i in range(7):
        alarm.tick(i / 10, [], HIGH if i % 3 == 0 else low, M, seen={1, 2})
    assert alarm.tick(0.7, [], [], M, seen={1}, exiting={2}) == []
