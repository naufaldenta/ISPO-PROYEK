import json

import numpy as np

from sahabat.config import Config
from sahabat.engine import BAR_TEXT, Engine
from sahabat.predlog import PredLog

from .helpers import person, ticks
from .scenes import repeated_punch
from .test_fall import fall_script

BLACK = np.zeros((480, 640, 3), dtype=np.uint8)


class FakePose:
    def __init__(self, script):
        self.script, self.t = script, 0.0

    def detect(self, frame):
        return self.script(self.t)


def make_engine(script, log=None, slow=False):
    return Engine(Config(), None, FakePose(script), log, slow)


def run(e, seconds, t0=0.0):
    snap = None
    for t in ticks(t0, seconds):
        e.pose.t = t
        snap = e.step(t, BLACK)
    return snap


def run_engine(script, seconds):
    return run(make_engine(script), seconds)


def test_bar_text_is_exact():
    assert BAR_TEXT == {"aman": "AMAN", "jatuh": "JATUH TERDETEKSI",
                        "kekerasan": "INDIKASI KEKERASAN TERDETEKSI",
                        "tidak_dinilai": "TIDAK DAPAT DINILAI", "kamera_terputus": "KAMERA TERPUTUS"}


def test_empty_scene_is_aman():
    s = run_engine(lambda t: [], seconds=1.0)
    assert (s.bar, s.text, s.people, s.alarm_seq) == ("aman", "AMAN", 0, 0)


def test_camera_switch_resets_motion_but_keeps_unacknowledged_alarm():
    from sahabat.source import VideoSource
    e = make_engine(fall_script)
    run(e, seconds=6.0)
    assert e.alarm.active is not None and e.history.tracks
    e.source = VideoSource(0)
    e.source.select(1)
    e.step(7.0, None, input_version=1)
    assert not e.history.tracks
    assert e.alarm.active is not None and e.snapshot().bar == "jatuh"
    assert not e.alarm.red_ids         # ID kamera lama tidak menandai orang pada kamera baru
    assert e.acknowledge()


def test_frame_from_old_camera_is_discarded_if_selection_changes_during_pose():
    from sahabat.source import VideoSource
    e = make_engine(lambda t: [person(1, cx=300, hip_y=300)])
    e.source = VideoSource(0)
    def detect(frame):
        e.source.select(1)
        return [person(1, cx=300, hip_y=300)]
    e.pose.detect = detect
    e.step(1.0, BLACK, input_version=0)
    assert not e.history.tracks and e.snapshot().people == 0


def test_select_camera_waits_for_current_scoring_to_finish():
    import threading
    from sahabat.source import VideoSource
    e = make_engine(lambda t: [])
    e.source = VideoSource(0)
    started, selected = threading.Event(), threading.Event()
    def select():
        started.set()
        e.select_camera(1)
        selected.set()
    with e._lock:
        thread = threading.Thread(target=select)
        thread.start()
        assert started.wait(1)
        assert not selected.wait(0.05)
        assert e.source.spec == 0
    thread.join(1)
    assert selected.is_set() and e.source.spec == 1


def test_no_frame_is_kamera_terputus():
    e = make_engine(lambda t: [])
    s = e.step(0.0, None)
    assert (s.bar, s.text, s.overlays) == ("kamera_terputus", "KAMERA TERPUTUS", ())


def test_only_unassessable_people_is_tidak_dinilai():
    s = run_engine(lambda t: [person(1, cx=300, hip_y=300, visible="none")], seconds=1.0)
    assert s.bar == "tidak_dinilai" and s.overlays[0].color == "gray"
    assert "tidak dapat dinilai" in s.overlays[0].label


def test_standing_person_is_green():
    s = run_engine(lambda t: [person(1, cx=300, hip_y=300)], seconds=1.5)
    assert s.bar == "aman" and [(o.color, o.label) for o in s.overlays] == [("green", "#1")]


def test_fall_turns_bar_red_then_ack_clears():
    e = make_engine(fall_script)
    s = run(e, seconds=6.0)
    assert (s.bar, s.text, s.alarm_seq) == ("jatuh", "JATUH TERDETEKSI", 1)
    assert [o.color for o in s.overlays] == ["red"]
    assert e.acknowledge() is True
    e.pose.t = 6.1
    assert e.step(6.1, BLACK).bar == "aman"
    assert e.acknowledge() is False


def test_alarm_stays_red_while_camera_is_briefly_lost():
    e = make_engine(fall_script)
    run(e, seconds=6.0)
    s = e.step(6.1, None)                                   # kabel kamera tersenggol saat alarm menyala
    assert s.bar == "jatuh" and "kamera terputus" in s.slow_text and s.alarm_seq == 1
    e.pose.t = 6.2
    assert e.step(6.2, BLACK).bar == "jatuh"
    assert e.acknowledge() is True
    assert e.step(6.3, None).bar == "kamera_terputus"


def test_punch_gives_kekerasan():
    s = run_engine(repeated_punch, seconds=4.0)
    assert (s.bar, s.text) == ("kekerasan", "INDIKASI KEKERASAN TERDETEKSI")
    assert sorted(o.color for o in s.overlays) == ["red", "red"]


def test_six_people_do_not_crash_and_extras_are_gray():
    s = run_engine(lambda t: [person(i, cx=90 * i, hip_y=300, size=40 + 10 * i) for i in range(1, 7)], seconds=1.5)
    assert s.people == 6 and sum(o.color == "gray" for o in s.overlays) == 2


def test_upper_mode_label():
    s = run_engine(lambda t: [person(1, cx=300, hip_y=300, visible="upper")], seconds=1.5)
    assert "setengah badan" in s.overlays[0].label and s.overlays[0].color == "green"


def test_fps_reflects_step_rate():
    assert 9.0 <= run_engine(lambda t: [], seconds=3.0).fps <= 11.0


def test_log_has_one_line_per_step(tmp_path):
    e = make_engine(fall_script, log=PredLog(tmp_path))
    run(e, seconds=1.0)
    lines = next(tmp_path.glob("prediksi-*.jsonl")).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10 and set(json.loads(lines[0])) == {
        "t", "people", "fall_ids", "pairs", "alarm", "sumber", "kejadian"}
    assert json.loads(lines[-1])["people"] == [{"id": 1, "mode": "full"}]


def test_air_punch_uses_gesture_source_in_engine(tmp_path):
    from .scenes import air_strike
    e = make_engine(air_strike, log=PredLog(tmp_path))
    assert run(e, 2.0).bar == "kekerasan"
    rows = [json.loads(line) for line in next(tmp_path.glob("prediksi-*.jsonl")).read_text().splitlines()]
    assert any(row["alarm"] == "violence" and row["sumber"] == ["pukulan"] for row in rows)


def test_stitched_track_is_not_a_victim_exit():
    from .scenes import two
    from sahabat.violence import PairScore

    def scene(t):
        # Pelacak memberi nomor 3 kepada B tanpa mengubah posenya.
        a, b = two(t)
        return [a, b] if t < 1.0 else [a, person(3, cx=400, hip_y=300)]

    e = make_engine(scene)
    e.violence.score = lambda tracks: [PairScore((1, 2), 0.9)] if e.pose.t < 1.0 else []
    run(e, 0.3, t0=0.7)
    run(e, 0.5, t0=1.0)
    assert e.alarm.active is None
    assert set(e.history.tracks) == {1, 2}


def test_slow_flag_passes_through():
    assert make_engine(lambda t: [], slow=True).step(0.0, BLACK).slow is True


def test_latest_jpeg_without_source_is_a_valid_jpeg():
    e = make_engine(lambda t: [person(1, cx=300, hip_y=300)])
    run(e, seconds=1.0)
    assert e.latest_jpeg()[:2] == b"\xff\xd8"


def test_pose_error_does_not_kill_the_step():
    def boom(t):
        raise RuntimeError("model tersedak")
    s = make_engine(boom).step(0.0, BLACK)
    assert (s.bar, s.people) == ("tidak_dinilai", 0)        # model gagal: jangan mengaku AMAN


class FakeSource:
    def __init__(self, fps):
        self.fps = fps

    def read(self):
        return None


def test_dim_room_camera_gets_its_own_warning():
    from sahabat.engine import CAMERA_SLOW_TEXT, SLOW_TEXT
    e = Engine(Config(), FakeSource(7.5), FakePose(lambda t: []))
    s = e.step(0.0, BLACK)
    assert s.slow is True and s.slow_text == CAMERA_SLOW_TEXT.format(8) and "kamera" in s.slow_text
    assert s.slow_text != SLOW_TEXT


def test_fast_camera_and_fast_laptop_have_no_warning():
    s = Engine(Config(), FakeSource(29.0), FakePose(lambda t: [])).step(0.0, BLACK)
    assert (s.slow, s.slow_text) == (False, "")
    assert make_engine(lambda t: []).step(0.0, BLACK).slow_text == ""


def test_slow_laptop_warning_text():
    from sahabat.engine import SLOW_TEXT
    assert make_engine(lambda t: [], slow=True).step(0.0, BLACK).slow_text == SLOW_TEXT


def test_camera_not_yet_measured_is_not_a_warning():
    s = Engine(Config(), FakeSource(0.0), FakePose(lambda t: [])).step(0.0, BLACK)
    assert (s.slow, s.slow_text) == (False, "")


def test_snapshot_and_acknowledge_do_not_wait_for_inference():
    import threading
    import time

    class SlowPose:
        def detect(self, frame):
            time.sleep(0.6)
            return []
    e = Engine(Config(), None, SlowPose())
    worker = threading.Thread(target=e.step, args=(0.0, BLACK))
    worker.start()
    time.sleep(0.15)                                        # mesin sedang di dalam inferensi
    started = time.perf_counter()
    e.snapshot()
    e.acknowledge()
    e.latest_jpeg()
    waited = time.perf_counter() - started
    worker.join()
    assert waited < 0.2


def test_scoring_error_shows_tidak_dinilai_instead_of_a_frozen_bar():
    e = make_engine(lambda t: [person(1, cx=300, hip_y=300)])
    assert run(e, seconds=1.5).bar == "aman"

    def broken(tracks):
        raise ValueError("model tidak cocok")
    e.violence.score = broken
    e.pose.t = 1.6
    s = e.step(1.6, BLACK)
    assert s.bar == "tidak_dinilai" and e.snapshot() is s and s.people == 1


class FakeRecorder:
    def __init__(self, active=True):
        self.active, self.frames, self.marks, self.closed = active, [], [], False

    def write(self, t, frame):
        self.frames.append((t, frame))

    def mark(self, t, label):
        self.marks.append((t, label))
        return self.active

    def close(self):
        self.closed = True


def test_recorder_gets_every_scored_frame_with_its_time():
    rec = FakeRecorder()
    e = Engine(Config(), None, FakePose(lambda t: []), recorder=rec)
    s = run(e, seconds=0.5)
    assert [t for t, _ in rec.frames] == ticks(0.0, 0.5) and rec.frames[0][1] is BLACK
    assert s.recording is True
    assert e.step(0.6, None).recording is True          # kamera terputus: tanda merekam tetap tampil
    assert len(rec.frames) == 5                          # tetapi tidak ada frame yang ditulis


def test_no_recorder_means_not_recording():
    e = make_engine(lambda t: [])
    assert run(e, seconds=0.3).recording is False and e.mark("jatuh") is False


def test_recorder_that_stopped_itself_is_shown_as_not_recording():
    e = Engine(Config(), None, FakePose(lambda t: []), recorder=FakeRecorder(active=False))
    assert run(e, seconds=0.3).recording is False


def test_mark_uses_the_time_of_the_last_step():
    rec = FakeRecorder()
    e = Engine(Config(), None, FakePose(lambda t: []), recorder=rec)
    run(e, seconds=0.5)
    assert e.mark("jatuh") is True and rec.marks == [(0.4, "jatuh")]


def test_stop_closes_the_recorder():
    rec = FakeRecorder()
    Engine(Config(), None, FakePose(lambda t: []), recorder=rec).stop()
    assert rec.closed is True


def test_recording_sign_shows_before_the_first_frame():
    e = Engine(Config(), None, FakePose(lambda t: []), recorder=FakeRecorder())
    assert e.snapshot().recording is True                # model masih dimuat, kamera sudah direkam sebentar lagi


class FakeRiwayat:
    def __init__(self, wants=True, fail=False):
        self.wants_frames = wants
        self.fail = fail
        self.frames, self.events, self.closed = [], [], False

    def frame(self, t, jpeg):
        if self.fail:
            raise OSError("disk penuh")
        self.frames.append((t, jpeg))

    def event(self, t, jenis, mode, ids, sumber="model"):
        self.events.append((t, jenis, mode, tuple(ids), sumber))

    def close(self):
        self.closed = True


def test_riwayat_gets_screen_frames_and_each_new_event():
    r = FakeRiwayat()
    e = Engine(Config(), None, FakePose(fall_script), riwayat=r)
    s = run(e, seconds=6.0)
    assert s.alarm_seq == 1
    assert len(r.events) == 1
    t, jenis, mode, ids, _ = r.events[0]
    assert (jenis, mode, ids) == ("jatuh", "full", (1,))
    assert len(r.frames) == len(list(ticks(0.0, 6.0)))
    assert all(j[:2] == b"\xff\xd8" for _, j in r.frames)          # JPEG
    assert any(ft == t for ft, _ in r.frames)                    # frame saat kejadian ikut


def test_riwayat_without_clips_gets_events_but_no_frames():
    r = FakeRiwayat(wants=False)
    run(Engine(Config(), None, FakePose(repeated_punch), riwayat=r), seconds=4.0)
    assert r.frames == [] and r.events and r.events[0][1] == "kekerasan"


def test_riwayat_failure_does_not_stop_the_alarm():
    r = FakeRiwayat(fail=True)
    s = run(Engine(Config(), None, FakePose(fall_script), riwayat=r), seconds=6.0)
    assert s.bar == "jatuh"
    assert [ev[1] for ev in r.events] == ["jatuh"]               # klip gagal, catatan tetap masuk


def test_stop_closes_the_riwayat():
    r = FakeRiwayat()
    Engine(Config(), None, FakePose(lambda t: []), riwayat=r).stop()
    assert r.closed


LEFT_HALF = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]


def shifted(script, dx):
    """Adegan yang sama, digeser ke kanan sejauh dx piksel."""
    from sahabat.types import Person

    def moved(t):
        return [Person(p.track_id, p.kp + np.array([dx, 0], dtype=np.float32), p.conf,
                       (p.box[0] + dx, p.box[1], p.box[2] + dx, p.box[3])) for p in script(t)]
    return moved


def test_fall_outside_the_zone_is_not_assessed_and_drawn_faint():
    from dataclasses import replace
    script = shifted(fall_script, 400)                   # jatuh di separuh kanan gambar
    e = Engine(replace(Config(), zone=LEFT_HALF), None, FakePose(script))
    s = run(e, seconds=6.0)
    assert (s.bar, s.alarm_seq, s.people) == ("aman", 0, 0)
    assert [o.color for o in s.overlays] == ["out"] and s.overlays[0].label == ""


def test_same_fall_inside_the_zone_still_alarms():
    from dataclasses import replace
    e = Engine(replace(Config(), zone=LEFT_HALF), None, FakePose(fall_script))
    assert run(e, seconds=6.0).bar == "jatuh"


def test_set_zone_validates_and_none_means_whole_picture():
    import pytest
    e = make_engine(lambda t: [])
    assert e.zone is None
    assert e.set_zone(LEFT_HALF) == tuple(tuple(p) for p in [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        e.set_zone([[0, 0], [1, 1]])
    assert e.zone is not None                             # zona lama tetap berlaku bila yang baru ditolak
    assert e.set_zone(None) is None and e.zone is None


def test_broken_saved_zone_falls_back_to_whole_picture():
    from dataclasses import replace
    assert Engine(replace(Config(), zone=[[0, 0]]), None, FakePose(lambda t: [])).zone is None


def test_screen_picture_shows_the_zone():
    from dataclasses import replace
    e = Engine(replace(Config(), zone=LEFT_HALF), None, FakePose(lambda t: []))
    e.step(0.0, np.full((480, 640, 3), 200, dtype=np.uint8))
    import cv2
    img = cv2.imdecode(np.frombuffer(e.latest_jpeg(), np.uint8), cv2.IMREAD_COLOR)
    assert img[240, 600].mean() < 150 and img[240, 100].mean() > 180     # kanan (di luar zona) digelapkan


class CameraRiwayat(FakeRiwayat):
    def frame(self, t, jpeg, kamera=""):
        self.frames.append((t, kamera))

    def event(self, t, jenis, mode, ids, sumber="model", kamera="", kamera_nama=""):
        self.events.append((jenis, kamera, kamera_nama))


def test_engine_tells_riwayat_its_camera():
    r = CameraRiwayat()
    run(Engine(Config(), None, FakePose(fall_script), riwayat=r, kamera=("k1", "Lorong")), seconds=6.0)
    assert r.events == [("jatuh", "k1", "Lorong")] and {k for _, k in r.frames} == {"k1"}


def test_latest_jpeg_can_be_small_for_grid_tiles():
    import cv2

    class Src:
        version, fps = 0, 0.0

        def read(self):
            return 1.0, np.full((480, 640, 3), 60, np.uint8)
    e = Engine(Config(), Src(), FakePose(lambda t: []))
    e._input_version = 0
    small = cv2.imdecode(np.frombuffer(e.latest_jpeg(width=480, quality=60), np.uint8), cv2.IMREAD_COLOR)
    full = cv2.imdecode(np.frombuffer(e.latest_jpeg(), np.uint8), cv2.IMREAD_COLOR)
    assert small.shape[:2] == (360, 480) and full.shape[:2] == (480, 640)


def test_pause_stops_steps_without_closing_and_resume_continues():
    import time
    r = FakeRiwayat()

    class Src:
        version, fps = 0, 0.0

        def packet(self):
            return 0, (time.monotonic(), np.zeros((48, 64, 3), np.uint8))

        def read(self):
            return None

    class CountingPose:
        calls = 0

        def detect(self, frame):
            CountingPose.calls += 1
            return []
    e = Engine(Config(), Src(), CountingPose(), riwayat=r)
    e.start()
    time.sleep(0.3)
    e.pause()
    paused_at = CountingPose.calls
    time.sleep(0.3)
    assert paused_at > 0 and CountingPose.calls == paused_at and e._thread is None and not r.closed
    e.resume()
    time.sleep(0.3)
    assert CountingPose.calls > paused_at
    e.stop()
    assert r.closed


class OverlayRiwayat(FakeRiwayat):
    accepts_overlay = True

    def frame(self, t, jpeg, kamera="", overlay=None):
        self.frames.append((t, jpeg, overlay))


def test_engine_sends_raw_frames_with_overlay_data():
    import cv2
    r = OverlayRiwayat()
    run(Engine(Config(), None, FakePose(fall_script), riwayat=r), seconds=6.0)
    assert r.frames and all(set(ov) >= {"orang", "skor", "ukuran"} for _, _, ov in r.frames)
    assert any(ov["skor"]["jatuh"] for _, _, ov in r.frames)
    assert any(p["warna"] == "red" for _, _, ov in r.frames for p in ov["orang"])
    _, jpeg, ov = r.frames[-1]
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert ov["ukuran"] == [img.shape[1], img.shape[0]]
    green = (img[:, :, 1] > 150) & (img[:, :, 0] < 100) & (img[:, :, 2] < 100)
    assert not green.any()                                       # frame asli: tidak ada garis kerangka
