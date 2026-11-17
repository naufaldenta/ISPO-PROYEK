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
