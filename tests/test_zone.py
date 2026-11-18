import numpy as np
import pytest

from sahabat import zone as Z

from .helpers import person

SQUARE = ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75))


def test_no_zone_means_the_whole_picture():
    assert Z.normalize(None) is None
    assert Z.contains(None, 0.0, 0.99)
    people = [person(1, cx=10, hip_y=300)]
    assert Z.split(people, None, (480, 640)) == (people, [])


def test_normalize_accepts_a_polygon_and_rounds_it():
    got = Z.normalize([[0.1, 0.1], [0.9, 0.1], [0.5, 0.912345]])
    assert got == ((0.1, 0.1), (0.9, 0.1), (0.5, 0.9123))


@pytest.mark.parametrize("bad", [
    [], [[0, 0], [1, 1]], [[0, 0]] * 25, "abc", [[0, 0], [1, 0], [1]], [[0, 0], [1, 0], [1.2, 1]],
    [[0, 0], [1, 0], ["a", 1]], [[0, 0], [1, 0], [True, 1]], [[0.5, 0.5], [0.5001, 0.5], [0.5, 0.5001]],
])
def test_normalize_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        Z.normalize(bad)


def test_load_turns_a_broken_saved_zone_into_the_whole_picture():
    assert Z.load([[0, 0]]) is None
    assert Z.load("rusak") is None
    assert Z.load([list(p) for p in SQUARE]) == SQUARE


def test_contains_inside_outside_and_concave():
    assert Z.contains(SQUARE, 0.5, 0.5)
    assert not Z.contains(SQUARE, 0.1, 0.5)
    assert not Z.contains(SQUARE, 0.5, 0.9)
    letter_l = ((0.0, 0.0), (0.4, 0.0), (0.4, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0))
    assert Z.contains(letter_l, 0.2, 0.2) and Z.contains(letter_l, 0.8, 0.8)
    assert not Z.contains(letter_l, 0.8, 0.2)


def test_split_uses_the_middle_of_the_body():
    inside = person(1, cx=320, hip_y=260)
    outside = person(2, cx=40, hip_y=260)
    a, b = Z.split([inside, outside], SQUARE, (480, 640))
    assert [p.track_id for p in a] == [1] and [p.track_id for p in b] == [2]


def test_draw_darkens_outside_and_keeps_inside():
    canvas = np.full((100, 100, 3), 200, dtype=np.uint8)
    Z.draw(canvas, SQUARE)
    assert canvas[50, 50].tolist() == [200, 200, 200]
    assert canvas[5, 5, 0] < 200
    untouched = np.full((10, 10, 3), 200, dtype=np.uint8)
    Z.draw(untouched, None)
    assert (untouched == 200).all()
