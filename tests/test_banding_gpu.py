"""Gerbang kesetaraan GPU: GPU hanya boleh dipakai otomatis bila tidak lebih buruk di angka mana pun."""
import json
from pathlib import Path

from latih import banding_gpu as bg
from latih.banding_gpu import lolos

CPU = {"jatuh": {"f1": 0.833}, "kekerasan_sesi17": {"f1": 0.753, "tp": 32, "alarm_keliru_per_jam": 34.12}}


def copy():
    return json.loads(json.dumps(CPU))


def test_gate_requires_not_worse_on_every_number():
    assert lolos(CPU, copy())
    worse = copy()
    worse["kekerasan_sesi17"]["alarm_keliru_per_jam"] = 35.0
    assert not lolos(CPU, worse)
    better = copy()
    better["jatuh"]["f1"] = 0.85
    assert lolos(CPU, better)
    for key, value in (("f1", 0.75), ("tp", 31)):
        worse = copy()
        worse["kekerasan_sesi17"][key] = value
        assert not lolos(CPU, worse), key
    worse = copy()
    worse["jatuh"]["f1"] = 0.80
    assert not lolos(CPU, worse)


def test_gpu_path_mirrors_original_trace_layout():
    original = bg.DATA / "putaran3" / "cauca_416" / "FallS9.npz"
    assert bg.gpu_path(original) == bg.OUT_TRACES / "putaran3" / "cauca_416" / "FallS9.npz"
    assert bg.gpu_path(Path("C:/lain/x.npz")) is None


def test_redirect_loads_gpu_trace_and_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(bg, "DATA", tmp_path / "data")
    monkeypatch.setattr(bg, "OUT_TRACES", tmp_path / "gpu")
    original = tmp_path / "data" / "putaran3" / "le2i_416" / "a.npz"
    mirror = tmp_path / "gpu" / "putaran3" / "le2i_416" / "a.npz"
    mirror.parent.mkdir(parents=True)
    mirror.write_bytes(b"")
    seen, used = [], set()
    load = bg.redirect(lambda path: seen.append(Path(path)) or "jejak", used)
    assert load(original) == "jejak" and seen == [mirror] and used == {mirror}
    missing = tmp_path / "data" / "putaran3" / "le2i_416" / "b.npz"
    try:
        load(missing)
    except FileNotFoundError as exc:
        assert "ekstrak" in str(exc)
    else:
        raise AssertionError("jejak GPU yang belum ada harus menghentikan pengukuran")


def test_summary_text_names_each_number():
    gpu = copy()
    gpu["kekerasan_sesi17"]["alarm_keliru_per_jam"] = 36.0
    text = bg.ringkas(CPU, gpu)
    assert "F1 jatuh 0.833 → 0.833" in text and "alarm keliru/jam 34.12 → 36.0" in text
