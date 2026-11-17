import pytest

from latih.gmdcsa import load_index, parse_classes

FALL_CSV = """File Name,Length (seconds),Time of Recording,Attire,Description, Classes
01.mp4,06, Day (Light On),Full T-shirt & Trousers, Falling from sitting on the chair. Left side Fall, Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]
02.mp4,05, Day,T-shirt, Standing to falling. Full body (both ankles) is not visible, Falling (BW) [ 0.3 to 5]; Standing [0 to 0.2]
03.mp4,03,Day,Full T-shirt & Trouser,falling forward, Fall (FW)[0 to 3]
04.mp4,10,Day,Full T-shirt & Trouser,Falling slowly from standing position to the floor,
"""
ADL_CSV = """File Name,Length (seconds),Time of Recording,Attire,Description, Classes
01.mp4,08, Day (Light On), Full T-shirt, Sitting on the bed to sleeping right side on the bed, Sitting[0 to 1]; Sleeping[2 to 8]
02.mp4,10,Day (Lights Off),Full Shirt & Pant,Talking on the mobile phone while walking, Walking[0 to 10]
"""


def test_classes_are_parsed_with_their_times():
    assert parse_classes(" Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]") == [("falling (sw)", 3.4, 6.0), ("sitting", 0.0, 3.4)]
    assert parse_classes("Falling (BW) [ 0.3 to 5]:; Standing [0 to 0.2] ;") == [("falling (bw)", 0.3, 5.0),
                                                                              ("standing", 0.0, 0.2)]
    assert parse_classes("") == [] and parse_classes("tanpa waktu") == []


@pytest.fixture
def root(tmp_path):
    for subject, files in (("Subject 1", {"Fall": 4, "ADL": 2}), ("Subject 2", {"ADL": 1})):
        for kind, n in files.items():
            (tmp_path / subject / kind).mkdir(parents=True)
            for i in range(1, n + 1):
                (tmp_path / subject / kind / f"{i:02d}.mp4").write_bytes(b"")
    (tmp_path / "Subject 1" / "Fall.csv").write_text(FALL_CSV, encoding="utf-8")
    (tmp_path / "Subject 1" / "ADL.csv").write_text(ADL_CSV, encoding="utf-8")
    (tmp_path / "Subject 2" / "ADL.csv").write_text(ADL_CSV, encoding="utf-8")
    return tmp_path


def test_index_lists_every_labelled_video_that_exists(root):
    clips = load_index(root)
    assert [c.name for c in clips] == ["s1-adl-01", "s1-adl-02", "s1-fall-01", "s1-fall-02", "s1-fall-03",
                                       "s1-fall-04", "s2-adl-01"]              # s2-adl-02 tidak ada berkasnya
    assert {c.subject for c in clips} == {1, 2} and clips[2].path.name == "01.mp4"


def test_fall_onset_is_the_start_of_the_fall_class(root):
    by = {c.name: c for c in load_index(root)}
    assert by["s1-fall-01"].onset == pytest.approx(3.4) and by["s1-fall-02"].onset == pytest.approx(0.3)
    assert by["s1-fall-03"].onset == 0.0
    assert by["s1-fall-04"].onset is None                    # tidak diberi waktu oleh penyusunnya
    assert by["s1-adl-01"].onset is None and by["s1-adl-01"].is_fall is False


def test_lying_and_partly_hidden_clips_are_flagged(root):
    by = {c.name: c for c in load_index(root)}
    assert by["s1-adl-01"].lying is True and by["s1-adl-02"].lying is False
    assert by["s1-fall-02"].partly_hidden is True and by["s1-fall-01"].partly_hidden is False
