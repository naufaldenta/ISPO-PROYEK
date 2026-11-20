import numpy as np

from sahabat.recorder import Recorder, read_marks, read_session


def frame(value, size=(120, 160)):
    return np.full((size[0], size[1], 3), value, dtype=np.uint8)


def record(tmp_path, times, **kw):
    r = Recorder(tmp_path, name="sesi", **kw)
    for i, t in enumerate(times):
        r.write(t, frame(40 + 20 * (i % 8)))
    r.close()
    return r


def test_recorded_frames_come_back_with_their_timestamps(tmp_path):
    times = [100.0, 100.1, 100.23, 100.3, 100.41]
    r = record(tmp_path, times)
    got = list(read_session(r.folder))
    assert [t for t, _ in got] == times
    assert all(f.shape == (120, 160, 3) for _, f in got)
    # JPEG mengubah nilai sedikit; tingkat terang tiap frame harus tetap bisa dibedakan
    assert [round(float(f.mean()) / 20) for _, f in got] == [2, 3, 4, 5, 6]


def test_long_recording_is_split_into_segments(tmp_path):
    r = record(tmp_path, [i * 0.1 for i in range(35)], segment_s=1.0)
    videos = sorted(p.name for p in r.folder.glob("*.avi"))
    assert videos == ["0001.avi", "0002.avi", "0003.avi", "0004.avi"]
    assert len(list(read_session(r.folder))) == 35


def test_changed_frame_size_starts_a_new_segment(tmp_path):
    r = Recorder(tmp_path, name="sesi")
    r.write(0.0, frame(50))
    r.write(0.1, frame(50, size=(240, 320)))
    r.close()
    assert [f.shape[:2] for _, f in read_session(r.folder)] == [(120, 160), (240, 320)]


def test_stops_when_disk_is_nearly_full(tmp_path, capsys):
    free = iter([10_000, 10_000, 500])                  # segmen ketiga: ruang habis
    r = Recorder(tmp_path, name="sesi", segment_s=1.0, min_free=1_000, free_bytes=lambda folder: next(free, 500))
    for i in range(40):
        r.write(i * 0.1, frame(60))
    r.close()
    assert r.active is False
    assert len(list(read_session(r.folder))) == 20      # dua segmen penuh, lalu berhenti
    assert "penuh" in capsys.readouterr().out


def test_unwritable_folder_disables_recording_without_crashing(tmp_path, capsys):
    blocker = tmp_path / "berkas"
    blocker.write_text("x")
    r = Recorder(blocker)                                # induknya berkas, bukan folder
    r.write(0.0, frame(50))
    assert r.mark(0.0, "jatuh") is False
    r.close()
    assert r.active is False
    assert "tidak bisa" in capsys.readouterr().out


def test_marks_are_saved_in_order(tmp_path):
    r = Recorder(tmp_path, name="sesi")
    r.write(5.0, frame(50))
    assert r.mark(5.2, "jatuh") is True
    assert r.mark(9.0, "biasa") is True
    r.close()
    assert read_marks(r.folder) == [(5.2, "jatuh"), (9.0, "biasa")]


def test_write_after_close_is_ignored(tmp_path):
    r = record(tmp_path, [0.0, 0.1])
    r.write(0.2, frame(50))
    assert r.active is False and len(list(read_session(r.folder))) == 2


def test_full_queue_drops_frames_instead_of_blocking(tmp_path):
    r = Recorder(tmp_path, name="sesi", queue_size=2, start=False)   # penulis belum jalan: antrean cepat penuh
    for i in range(5):
        r.write(i * 0.1, frame(50))
    assert r.dropped == 3
    r.start()
    r.close()
    assert len(list(read_session(r.folder))) == 2


def test_recorder_writes_session_meta(tmp_path):                                   # Bagian 5
    rec = Recorder(tmp_path, name="sesi", start=False, meta={"jenis": "biasa", "kamera": "Lorong", "posisi": "tinggi"})
    import json
    assert json.loads((tmp_path / "sesi" / "meta.json").read_text(encoding="utf-8")) == \
        {"jenis": "biasa", "kamera": "Lorong", "posisi": "tinggi"}
    rec.close()
    assert not (tmp_path / "tanpa").exists()
    Recorder(tmp_path, name="tanpa", start=False).close()
    assert not (tmp_path / "tanpa" / "meta.json").exists()


def test_session_meta_names_the_cctv_camera(tmp_path, monkeypatch):                # Bagian 5
    from sahabat import server
    from sahabat.config import Config
    store = type("S", (), {"get_public": lambda self, cam_id: {"nama": "Kelas 7B"} if cam_id == "k1" else None})()
    monkeypatch.setattr(server, "default_store", lambda: store)
    cfg = Config(camera="ipcam:k1", posisi_kamera="tinggi")
    assert server.session_meta(cfg, biasa=True) == {"jenis": "biasa", "kamera": "Kelas 7B", "posisi": "tinggi"}
    assert server.session_meta(Config(camera=0), biasa=False) == {"jenis": "adegan", "kamera": "0", "posisi": "dada"}


def test_session_meta_can_be_updated_while_recording(tmp_path):                      # minor B5 #6
    import json
    rec = Recorder(tmp_path, name="sesi", start=False, meta={"jenis": "biasa", "kamera": "0", "posisi": "dada"})
    rec.perbarui_meta(kamera="Kelas 7B", posisi="tinggi")
    assert json.loads((tmp_path / "sesi" / "meta.json").read_text(encoding="utf-8")) == \
        {"jenis": "biasa", "kamera": "Kelas 7B", "posisi": "tinggi"}
    rec.close()
