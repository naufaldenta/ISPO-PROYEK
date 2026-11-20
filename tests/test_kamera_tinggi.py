"""Model kamera tinggi: data, gerbang, dan alat rekaman sekolah (latih.kamera_tinggi)."""
from zipfile import ZipFile

import pytest

from latih import kamera_tinggi as kt


def _bus_zip(path, names):
    with ZipFile(path, "w") as z:
        z.writestr("train.txt", "\n".join(names))
        z.writestr("test.txt", "")
        for n in names:
            folder = "Violence" if n.startswith("VIOLENCE_") else "NoViolence"
            z.writestr(f"{folder}/{n}", b"mp4")


def test_bus_train_names_and_labels(tmp_path):
    _bus_zip(tmp_path / "bus.zip", ["VIOLENCE_946.mp4", "NONVIOLENCE_3195.mp4"])
    assert kt.bus_train_videos(tmp_path / "bus.zip") == [("VIOLENCE_946.mp4", True), ("NONVIOLENCE_3195.mp4", False)]


def test_bus_train_rejects_paths(tmp_path):
    _bus_zip(tmp_path / "bus.zip", ["../x.mp4"])
    with pytest.raises(ValueError):
        kt.bus_train_videos(tmp_path / "bus.zip")


def test_bus_train_extracts_into_folders(tmp_path):
    _bus_zip(tmp_path / "bus.zip", ["VIOLENCE_1.mp4", "NONVIOLENCE_2.mp4"])
    paths = kt.ekstrak_bus_train(tmp_path / "bus.zip", tmp_path / "out")
    assert [p.relative_to(tmp_path / "out").as_posix() for p in paths] == ["Violence/VIOLENCE_1.mp4",
                                                                            "NoViolence/NONVIOLENCE_2.mp4"]
    assert all(p.read_bytes() == b"mp4" for p in paths)


def test_dev_split_is_stable_and_about_a_fifth():
    names = [f"VIOLENCE_{i}.mp4" for i in range(500)]
    dev = kt.split_dev(names)
    assert dev == kt.split_dev(list(reversed(names)))
    assert 0.15 <= len(dev) / len(names) <= 0.25 and dev <= set(names)


def test_scfd_videos(tmp_path):
    for folder, name in (("fight", "fi001.mp4"), ("noFight", "nofi001.mp4"), ("noFight", "catatan.txt")):
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / name).write_bytes(b"")
    got = kt.scfd_videos(tmp_path)
    assert [(p.name, v) for p, v in got] == [("fi001.mp4", True), ("nofi001.mp4", False)]


def test_trace_targets_mirror_sources(tmp_path):
    assert kt.trace_target(kt.ROOT / "bus_train" / "Violence" / "VIOLENCE_1.mp4") == \
        kt.ROOT / "bus_train_416" / "VIOLENCE_1.npz"
    assert kt.trace_target(kt.ROOT / "scfd" / "noFight" / "nofi001.mp4") == kt.ROOT / "scfd_416" / "noFight" / "nofi001.npz"


# ---- kandidat kekerasan ----
from types import SimpleNamespace  # noqa: E402


def _replay(name, group, split, violent=True, source="bus"):
    return SimpleNamespace(item=SimpleNamespace(name=name, group=group, violent=violent, source=source), split=split)


def test_high_gate():
    now = {"tertangkap": 36, "keliru_klip": 12, "keliru_per_jam": 34.1}
    assert kt.lolos_tinggi(now, {**now, "tertangkap": 40})
    assert not kt.lolos_tinggi(now, now)                                  # sama saja = tidak lolos
    assert not kt.lolos_tinggi(now, {**now, "tertangkap": 40, "keliru_klip": 13})
    assert not kt.lolos_tinggi(now, {**now, "tertangkap": 40, "keliru_per_jam": 35.0})


def test_test_set_never_overlaps_training():                                 # Review Focus 5
    bus_train = [_replay("VIOLENCE_1.mp4", 7000, "train"), _replay("VIOLENCE_2.mp4", 7001, "dev")]
    airtlab = [_replay("a/1.npz", 3001, "train"), _replay("a/2.npz", 3002, "test"),
               _replay("a/3.npz", 3003, "train", violent=False), _replay("a/4.npz", 3004, "dev")]
    tr, dev, te = kt.susun_kekerasan(bus_train, airtlab, [_replay("VIOLENCE_9.mp4", 5000, "test")],
                                     [_replay("fi001.mp4", 8000, "test", source="scfd")])
    assert [r.item.name for r in tr] == ["VIOLENCE_1.mp4", "a/1.npz"]       # AIRTLab latih: positif saja
    assert [r.item.name for r in dev] == ["VIOLENCE_2.mp4"]
    assert [r.item.name for r in te] == ["VIOLENCE_9.mp4", "a/2.npz", "fi001.mp4"]
    leak = [_replay("VIOLENCE_9.mp4", 7002, "train")]
    with pytest.raises(ValueError):
        kt.susun_kekerasan(leak, [], [_replay("VIOLENCE_9.mp4", 5000, "test")], [])
    with pytest.raises(ValueError):
        kt.susun_kekerasan([_replay("x.mp4", 5000, "train")], [], [_replay("y.mp4", 5000, "test")], [])


def test_install_refuses_when_gate_fails(tmp_path):
    candidate = tmp_path / "kandidat.joblib"
    candidate.write_bytes(b"model")
    target = tmp_path / "models" / "violence_model_tinggi.joblib"
    assert not kt.pasang(candidate, target, lolos=False) and not target.exists()
    assert kt.pasang(candidate, target, lolos=True) and target.read_bytes() == b"model"


# ---- kandidat jatuh ----
def _clip(name, source, subject):
    return (SimpleNamespace(name=name, source=source, group=subject), None)


def test_fall_test_set_never_overlaps_training():                            # Review Focus 5
    train = [_clip("FallS1", "cauca", 1), _clip("WalkS6", "cauca", 6)]
    dev = [_clip("FallS7", "cauca", 7)]
    test = [_clip("FallS9", "cauca", 9), _clip("Coffee_room_01/video_43", "le2i", 0)]
    assert kt.susun_jatuh(train, dev, test) == (train, dev, test)
    with pytest.raises(ValueError):                                           # subjek uji ikut latih
        kt.susun_jatuh(train + [_clip("SitS9", "cauca", 9)], dev, test)
    with pytest.raises(ValueError):                                           # klip uji ikut dev
        kt.susun_jatuh(train, dev + [_clip("Coffee_room_01/video_43", "le2i", 0)], test)


# ---- rekaman sekolah ----
def test_labels_from_marks():
    rows = kt.label_dari_tanda([(1100.0, "kekerasan"), (1200.0, "biasa"), (1300.0, "jatuh")], "adegan",
                               t0=1000.0, durasi=400.0)
    assert rows[0] == {"mulai": 92.0, "selesai": 100.5, "kekerasan": True, "jatuh": False, "grup": 0}
    assert rows[1] == {"mulai": 188.0, "selesai": 200.0, "kekerasan": False, "jatuh": False, "grup": 1}
    assert rows[2] == {"mulai": 292.0, "selesai": 300.5, "kekerasan": False, "jatuh": True, "grup": 2}
    assert kt.label_dari_tanda([], "biasa", t0=0.0, durasi=150.0) == [
        {"mulai": 0.0, "selesai": 60.0, "kekerasan": False, "jatuh": False, "grup": 0},
        {"mulai": 60.0, "selesai": 120.0, "kekerasan": False, "jatuh": False, "grup": 1},
        {"mulai": 120.0, "selesai": 150.0, "kekerasan": False, "jatuh": False, "grup": 2}]
    assert kt.label_dari_tanda([(5.0, "kekerasan")], "adegan", t0=0.0, durasi=50.0)[0]["mulai"] == 0.0


def test_school_command_on_synthetic_session(tmp_path):
    import json
    import numpy as np
    from latih.putar import PoseTrace
    folder = tmp_path / "sesi"
    folder.mkdir()
    PoseTrace(np.arange(0.0, 30.0, 0.1), 480, 640, [[] for _ in range(300)]).save(folder / "jejak_416.npz")
    (folder / "tanda.jsonl").write_text(json.dumps({"t": 12.0, "label": "kekerasan"}) + "\n" +
                                        json.dumps({"t": 25.0, "label": "biasa"}) + "\n", encoding="utf-8")
    (folder / "meta.json").write_text(json.dumps({"jenis": "adegan", "kamera": "Uji", "posisi": "tinggi"}),
                                      encoding="utf-8")
    report = tmp_path / "laporan.md"
    assert kt.sekolah([folder], laporan=report) == 0
    text = report.read_text(encoding="utf-8")
    assert "sekarang" in text and "Uji" in text and "kekerasan 0/1" in text


# ---- perbaikan review akhir ----
def test_school_folds_never_score_trained_takes():                          # Final #1
    from latih.violence_train import WEBCAM_GROUP_BASE
    items = [SimpleNamespace(group=WEBCAM_GROUP_BASE + g, name=f"t{g}") for g in range(9)]   # seperti session_items
    replays = [(_replay(f"sesi/{g}", g, "test"), False) for g in range(9)]                 # seperti kb.prepare
    seen = set()
    for train_items, held in kt.lipatan_sekolah(items, replays):
        trained = {it.group - WEBCAM_GROUP_BASE for it in train_items}
        scored = {rep.item.group for rep, _ in held}
        assert not trained & scored
        seen |= scored
    assert seen == set(range(9))                                            # tiap ulangan dinilai tepat sekali


def test_overlapping_marks_do_not_share_frames():                          # Final #7
    rows = kt.label_dari_tanda([(100.0, "kekerasan"), (106.0, "biasa"), (110.0, "jatuh")], "adegan",
                               t0=0.0, durasi=200.0)
    assert [(r["mulai"], r["selesai"]) for r in rows] == [(92.0, 100.5), (100.5, 106.0), (106.0, 110.5)]


def test_install_needs_ordinary_activity_recording():                      # Final #2
    ok = {"tertangkap": 5, "keliru_klip": 0, "keliru_per_jam": 0.0}
    assert not kt.boleh_pasang_sekolah(lolos=True, jam_biasa=0.0)
    assert kt.boleh_pasang_sekolah(lolos=True, jam_biasa=0.2) and not kt.boleh_pasang_sekolah(lolos=False, jam_biasa=1.0)


def test_fall_counts_use_every_clip_that_can_false_alarm():                          # minor B5 #10
    stats = {"cauca": {"tp": 8, "pos": 10, "fp": 3, "neg": 10, "early": 1}}
    tp, pos, fp, denom = kt._fall_counts(stats)
    assert (tp, pos, fp) == (8, 10, 4) and denom == 20         # alarm sebelum jatuh bisa terjadi di klip jatuh juga


def test_fall_result_names_its_denominator():                                        # review M-10
    import inspect
    assert '"bisa_keliru"' in inspect.getsource(kt.jatuh) and '"negatif": neg' not in inspect.getsource(kt.jatuh)
