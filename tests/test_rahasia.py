import sys

import pytest

from sahabat.rahasia import buka, simpan


def test_round_trip_and_not_plain():
    nilai = simpan("rahasia@:/#% 1")
    assert "rahasia" not in nilai and buka(nilai) == "rahasia@:/#% 1"


def test_windows_uses_dpapi():
    if sys.platform == "win32":
        assert simpan("x").startswith("dpapi:")


@pytest.mark.parametrize("bad", ["dpapi:bukan-base64!", "dpapi:QUJD", "aneh:xx", ""])
def test_garbage_raises_value_error(bad):
    with pytest.raises(ValueError):
        buka(bad)
