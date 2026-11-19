import pytest
import json
import zipfile

from sahabat.riwayat import CLIP_DIR, LOG_FILE, Riwayat


class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def make(tmp_path, **kw):
    kw.setdefault("background", False)
    kw.setdefault("clock", Clock())
    kw.setdefault("free_bytes", lambda _: 10 ** 12)
    return Riwayat(tmp_path, **kw)


def feed(r, t0, t1, hz=10):
    n = int(round((t1 - t0) * hz))
    for i in range(n + 1):
        t = t0 + i / hz
        r.frame(t, f"jpg{t:.1f}".encode())


def test_event_is_listed_newest_first(tmp_path):
    r = make(tmp_path, keep_clips=False)
    a = r.event(1.0, "jatuh", "full", [3])
    b = r.event(2.0, "kekerasan", "upper", (1, 2), "keluar_gambar")
    items = r.list()
    assert [i["id"] for i in items] == [b.id, a.id]
    assert items[0]["jenis"] == "kekerasan" and items[0]["orang"] == [1, 2] and items[0]["sumber"] == "keluar_gambar"
    assert items[1]["mode"] == "full"
    assert r.seq == 2


def test_ids_are_unique_within_the_same_second(tmp_path):
    r = make(tmp_path, keep_clips=False)
    ids = {r.event(float(i), "jatuh", "full", [1]).id for i in range(5)}
    assert len(ids) == 5


def test_clip_holds_frames_from_before_to_after(tmp_path):
    r = make(tmp_path, before_s=2.0, after_s=1.0)
    feed(r, 0.0, 5.0)
    rec = r.event(5.0, "jatuh", "full", [1])
    assert r.list()[0]["klip"] is False and r.list()[0]["menyimpan"] is True     # belum lengkap
    feed(r, 5.1, 7.0)
    frames = r.clip(rec.id)
    assert frames is not None
    assert frames[0][1] == b"jpg3.0" and frames[-1][1] == b"jpg6.0"
    assert frames[0][0] == 0.0 and abs(frames[-1][0] - 3.0) < 1e-6
    assert r.list()[0]["klip"] is True


def test_no_frames_kept_and_no_clip_when_clips_are_off(tmp_path):
    r = make(tmp_path, keep_clips=False)
    feed(r, 0.0, 3.0)
    rec = r.event(3.0, "kekerasan", "full", [1, 2])
    feed(r, 3.1, 8.0)
    assert r.clip(rec.id) is None
    assert r.list()[0]["klip"] is False
    assert not list((tmp_path / CLIP_DIR).glob("*.zip"))


def test_turning_clips_off_drops_the_buffer(tmp_path):
    r = make(tmp_path, before_s=2.0, after_s=0.5)
    feed(r, 0.0, 2.0)
    r.set_keep_clips(False)
    r.set_keep_clips(True)
    rec = r.event(2.0, "jatuh", "full", [1])
    feed(r, 2.1, 3.0)
    frames = r.clip(rec.id)
    assert frames[0][1] == b"jpg2.1"                  # tidak ada frame dari sebelum dimatikan


def test_event_during_another_clip_gets_its_own_clip(tmp_path):
    r = make(tmp_path, before_s=1.0, after_s=1.0)
    feed(r, 0.0, 2.0)
    a = r.event(2.0, "kekerasan", "full", [1, 2])
    feed(r, 2.1, 2.5)
    b = r.event(2.5, "jatuh", "full", [2])
    feed(r, 2.6, 4.0)
    assert r.clip(a.id)[-1][1] == b"jpg3.0"
    assert r.clip(b.id)[0][1] == b"jpg1.5" and r.clip(b.id)[-1][1] == b"jpg3.5"


def test_close_saves_an_unfinished_clip(tmp_path):
    r = make(tmp_path, before_s=1.0, after_s=3.0)
    feed(r, 0.0, 1.0)
    rec = r.event(1.0, "jatuh", "full", [1])
    feed(r, 1.1, 1.5)
    r.close()
    frames = r.clip(rec.id)
    assert frames[-1][1] == b"jpg1.5"


def test_no_clip_when_the_disk_is_nearly_full(tmp_path):
    r = make(tmp_path, free_bytes=lambda _: 10, min_free=1000)
    feed(r, 0.0, 2.0)
    rec = r.event(2.0, "jatuh", "full", [1])
    feed(r, 2.1, 6.0)
    assert r.clip(rec.id) is None
    assert r.list()[0]["id"] == rec.id                 # catatannya tetap ada


def test_old_records_and_clips_are_removed(tmp_path):
    clock = Clock()
    r = make(tmp_path, clock=clock, before_s=0.5, after_s=0.5, keep_days=7)
    feed(r, 0.0, 1.0)
    old = r.event(1.0, "jatuh", "full", [1])
    feed(r, 1.1, 2.0)
    clock.now += 8 * 86400
    feed(r, 10.0, 11.0)
    new = r.event(11.0, "jatuh", "full", [1])
    feed(r, 11.1, 12.0)
    r.prune()
    assert [i["id"] for i in r.list()] == [new.id]
    assert r.clip(old.id) is None and r.clip(new.id) is not None
    # dibuka ulang (aplikasi dinyalakan lagi): pembersihan juga jalan saat mulai
    clock.now += 8 * 86400
    r2 = make(tmp_path, clock=clock)
    assert r2.list() == [] and not list((tmp_path / CLIP_DIR).glob("*.zip"))


def test_history_survives_a_restart_and_counts_from_it(tmp_path):
    clock = Clock()
    r = make(tmp_path, clock=clock, keep_clips=False)
    rec = r.event(1.0, "jatuh", "full", [1])
    r2 = make(tmp_path, clock=clock, keep_clips=False)
    assert r2.seq == 1 and r2.list()[0]["id"] == rec.id
    assert r2.event(2.0, "jatuh", "full", [1]).id != rec.id


def test_broken_lines_are_skipped(tmp_path):
    r = make(tmp_path, keep_clips=False)
    r.event(1.0, "jatuh", "full", [1])
    with open(tmp_path / LOG_FILE, "a", encoding="utf-8") as f:
        f.write("bukan json\n" + json.dumps({"id": "x"}) + "\n")
    assert len(r.list()) == 1


def test_clip_rejects_ids_that_point_elsewhere(tmp_path):
    r = make(tmp_path)
    for bad in ("../pengaturan", "..\\x", "", "a" * 50, "20261118-101010-1/../../x"):
        assert r.clip(bad) is None


def test_clip_file_is_a_plain_zip_of_jpegs(tmp_path):
    r = make(tmp_path, before_s=0.2, after_s=0.2)
    feed(r, 0.0, 0.5)
    rec = r.event(0.5, "jatuh", "full", [1])
    feed(r, 0.6, 1.0)
    with zipfile.ZipFile(tmp_path / CLIP_DIR / f"{rec.id}.zip") as z:
        names = z.namelist()
    assert "waktu.txt" in names and all(n.endswith(".jpg") for n in names if n != "waktu.txt")


def test_read_only_folder_turns_history_off(tmp_path):
    blocker = tmp_path / "berkas"
    blocker.write_text("x")
    r = make(blocker / "riwayat")                     # induknya berkas biasa: folder tidak bisa dibuat
    assert r.disabled
    assert r.event(1.0, "jatuh", "full", [1]) is None
    r.frame(1.0, b"x")
    assert r.list() == [] and r.clip("20261118-101010-1") is None


# ---- Bagian 2: riwayat per kamera ----
def test_clips_never_mix_cameras(tmp_path):
    r = make(tmp_path, before_s=1.0, after_s=0.5)
    rec = None
    for i in range(20):
        t = i * 0.1
        r.frame(t, b"A%d" % i, kamera="k1")
        r.frame(t, b"B%d" % i, kamera="k2")
        if i == 12:
            rec = r.event(t, "jatuh", "full", [3], kamera="k2", kamera_nama="Lorong")
    frames = r.clip(rec.id)
    assert frames and all(j.startswith(b"B") for _, j in frames)
    item = r.list()[0]
    assert (item["kamera"], item["kamera_nama"]) == ("k2", "Lorong")
    assert r.list(kamera="k1") == [] and len(r.list(kamera="k2")) == 1


def test_old_records_without_camera_still_load(tmp_path):
    clock = Clock()
    (tmp_path / LOG_FILE).write_text(json.dumps({"id": "20261118-100000-1", "waktu": clock.now, "jenis": "jatuh",
                                                 "mode": "full", "orang": [1], "sumber": "model", "klip": False})
                                     + "\n", encoding="utf-8")
    assert make(tmp_path, clock=clock).list()[0]["kamera"] == ""


def test_turning_clips_off_clears_every_camera(tmp_path):
    r = make(tmp_path)
    r.frame(0.0, b"A", kamera="k1")
    r.frame(0.0, b"B", kamera="k2")
    r.set_keep_clips(False)
    r.set_keep_clips(True)
    rec = r.event(0.1, "jatuh", "full", [1], kamera="k1")
    r.frame(3.1, b"A2", kamera="k1")                 # akhir klip: 0,1 + 3 dtk
    assert [j for _, j in r.clip(rec.id)] == [b"A2"]


# ---- Bagian 3: klip versi 2 ----
def test_v2_clip_keeps_raw_frames_and_overlay(tmp_path):
    r = make(tmp_path, before_s=1.0, after_s=0.5)
    ov = {"orang": [{"id": 3, "kp": [[1.0, 2.0]] * 17, "conf": [0.9] * 17, "warna": "red", "label": "#3"}],
          "skor": {"jatuh": False, "kekerasan": 0.91}, "ukuran": [640, 480], "zona": [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]}
    rec = None
    for i in range(20):
        r.frame(i * 0.1, b"raw%d" % i, overlay=ov)
        if i == 12:
            rec = r.event(i * 0.1, "kekerasan", "full", [3, 5])
    info = r.clip_info(rec.id)
    assert info["versi"] == 2 and len(info["overlay"]) == len(info["waktu"])
    assert info["overlay"][0]["skor"]["kekerasan"] == 0.91
    assert info["meta"]["orang"] == [3, 5] and info["meta"]["lebar"] == 640 and info["meta"]["tinggi"] == 480
    assert info["meta"]["zona"] == [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
    assert abs(info["meta"]["kejadian_t"] - 1.0) < 1e-6        # 1,2 dtk - mulai klip 0,2 dtk
    assert r.clip(rec.id)[0][1] == b"raw2"                     # klip() lama tetap memberi (t, jpeg)


def test_v1_clip_info_without_overlay(tmp_path):
    r = make(tmp_path, before_s=0.2, after_s=0.2)
    feed(r, 0.0, 0.5)
    rec = r.event(0.5, "jatuh", "full", [1])
    feed(r, 0.6, 1.0)
    info = r.clip_info(rec.id)
    assert info["versi"] == 1 and info["overlay"] is None and info["meta"] is None and len(info["waktu"]) > 3
    assert r.clip_info("../x") is None and r.clip_info("20990101-000000-1") is None


# ---- Bagian 3: tanda tindak lanjut guru ----
def test_teacher_marks_and_notes_survive_restart(tmp_path):
    clock = Clock()
    r = make(tmp_path, clock=clock)
    a = r.event(0.0, "jatuh", "full", [1])
    b = r.event(0.0, "kekerasan", "full", [2, 3])
    item = r.set_tindak(a.id, "keliru", "Siswa duduk di lantai, bukan jatuh. <script>x</script>")
    assert item["tanda"] == "keliru" and item["catatan"].endswith("<script>x</script>")   # disimpan apa adanya
    r.set_tindak(b.id, "tindak", None)
    again = make(tmp_path, clock=clock)
    assert {i["id"]: i["tanda"] for i in again.list()} == {a.id: "keliru", b.id: "tindak"}
    assert again.ringkasan() == {"benar": 0, "keliru": 1, "tindak": 1, "belum": 0, "jatuh": 1, "kekerasan": 1}
    again.set_tindak(a.id, "", None)
    assert again.ringkasan()["belum"] == 1
    assert [i for i in again.list() if i["id"] == a.id][0]["catatan"].startswith("Siswa")


def test_bad_marks_are_rejected(tmp_path):
    r = make(tmp_path)
    a = r.event(0.0, "jatuh", "full", [1])
    for bad in (("pelaku", None), ("benar", "x" * 301), (1, None), (None, 5)):
        with pytest.raises(ValueError):
            r.set_tindak(a.id, *bad)
    assert r.set_tindak("20990101-000000-1", "benar", None) is None
    assert r.set_tindak("../x", "benar", None) is None


def test_summary_per_camera(tmp_path):
    r = make(tmp_path)
    a = r.event(0.0, "jatuh", "full", [1], kamera="k1")
    r.event(0.0, "jatuh", "full", [1], kamera="k2")
    r.set_tindak(a.id, "benar", None)
    assert r.ringkasan(kamera="k1") == {"benar": 1, "keliru": 0, "tindak": 0, "belum": 0, "jatuh": 1, "kekerasan": 0}
