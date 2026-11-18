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
