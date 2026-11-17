import pytest

from latih.putar import Alarm
from latih.utinteraction import CLASSES, PairResult, load_index, per_class


@pytest.fixture
def root(tmp_path):
    for folder, names in (("segmented_set1", ["0_1_4.avi", "17_3_0.avi", "13_3_2.avi", "catatan.txt"]),
                          ("segmented_set2", ["16_13_5.avi", "15_13_3.avi", "aneh.avi"])):
        (tmp_path / folder).mkdir()
        for n in names:
            (tmp_path / folder / n).write_bytes(b"")
    return tmp_path


def test_class_numbers_mean_what_the_pictures_show():
    assert CLASSES == {0: ("salaman", False), 1: ("pelukan", False), 2: ("tendang", True),
                       3: ("menunjuk", False), 4: ("pukul", True), 5: ("dorong", True)}


def test_index_reads_set_sequence_and_class_from_the_file_name(root):
    clips = load_index(root)
    assert [c.name for c in clips] == ["set1-00-seq01-pukul", "set1-13-seq03-tendang", "set1-17-seq03-salaman",
                                       "set2-15-seq13-menunjuk", "set2-16-seq13-dorong"]      # berkas lain dilewati
    assert [c.violent for c in clips] == [True, True, False, False, True]
    assert clips[0].sequence == 1 and clips[4].sequence == 13 and clips[4].path.name == "16_13_5.avi"


def test_per_class_counts_violence_and_fall_alarms():
    results = [
        PairResult("a", "pukul", True, [Alarm(2.0, "violence")], 2),
        PairResult("b", "pukul", True, [], 2),
        PairResult("c", "salaman", False, [Alarm(1.0, "violence"), Alarm(1.5, "fall")], 2),
        PairResult("d", "salaman", False, [Alarm(1.0, "fall")], 2),
        PairResult("e", "menunjuk", False, [], 1),
    ]
    rows = per_class(results)
    assert rows["pukul"] == {"n": 2, "violence": 1, "fall": 0, "pair_seen": 2}
    assert rows["salaman"] == {"n": 2, "violence": 1, "fall": 2, "pair_seen": 2}
    assert rows["menunjuk"] == {"n": 1, "violence": 0, "fall": 0, "pair_seen": 0}
