import dataclasses
import zipfile

import cv2
import numpy as np
import pytest

from latih.putar import (Alarm, PoseTrace, match_marks, replay, session_ticks, tick_indices, trace, video_ticks,
                         zip_ticks)
from sahabat.config import Config
from sahabat.recorder import Recorder

from .helpers import person, ticks
from .test_fall import fall_script


def test_engine_sees_every_third_frame_of_a_30fps_video():
    got = tick_indices(10, fps=30.0, rate_hz=10.0)
    assert [i for _, i in got] == [0, 3, 6, 9]
    assert [t for t, _ in got] == pytest.approx([0.0, 0.1, 0.2, 0.3])


def test_phase_shifts_which_frames_are_seen():
    assert [i for _, i in tick_indices(10, fps=30.0, rate_hz=10.0, phase=2)] == [2, 5, 8]


def test_slow_camera_gives_each_frame_once():
    got = tick_indices(4, fps=7.5, rate_hz=10.0)
    assert [i for _, i in got] == [0, 1, 2, 3]
    assert [t for t, _ in got] == pytest.approx([0.0, 1 / 7.5, 2 / 7.5, 3 / 7.5])


def test_25fps_video_is_sampled_near_10hz_without_repeats():
    idx = [i for _, i in tick_indices(26, fps=25.0, rate_hz=10.0)]
    assert idx == sorted(set(idx)) and idx[:5] == [0, 2, 5, 7, 10]


def make_trace(script, seconds):
    ts = ticks(0.0, seconds)
    return PoseTrace(np.array(ts), 480, 640, [script(t) for t in ts])


def test_trace_survives_saving_and_loading(tmp_path):
    tr = make_trace(lambda t: [person(1, cx=200, hip_y=300), person(7, cx=400, hip_y=300)] if t > 0.15 else [], 0.5)
    tr.save(tmp_path / "jejak.npz")
    back = PoseTrace.load(tmp_path / "jejak.npz")
    assert (back.frame_h, back.frame_w) == (480, 640) and np.allclose(back.t, tr.t)
    assert [[p.track_id for p in ps] for ps in back.people] == [[], [], [1, 7], [1, 7], [1, 7]]
    a, b = tr.people[3][1], back.people[3][1]
    assert np.array_equal(a.kp, b.kp) and np.array_equal(a.conf, b.conf) and a.box == pytest.approx(b.box)


def test_trace_runs_pose_once_per_frame():
    class CountingPose:
        calls = 0

        def detect(self, frame):
            self.calls += 1
            return [person(1, cx=200, hip_y=300)] if frame.mean() > 100 else []

    pose = CountingPose()
    frames = [(0.0, np.zeros((240, 320, 3), np.uint8)), (0.1, np.full((240, 320, 3), 200, np.uint8))]
    tr = trace(frames, pose)
    assert pose.calls == 2 and (tr.frame_h, tr.frame_w) == (240, 320)
    assert [len(ps) for ps in tr.people] == [0, 1] and list(tr.t) == [0.0, 0.1]


def test_replay_of_a_fall_gives_one_fall_alarm():
    alarms = replay(Config(), make_trace(fall_script, 6.0))
    assert [a.kind for a in alarms] == ["fall"] and 1.0 < alarms[0].t < 5.0


def test_replay_of_standing_gives_no_alarm():
    assert replay(Config(), make_trace(lambda t: [person(1, cx=300, hip_y=300)], 4.0)) == []


def test_replay_uses_the_given_thresholds():
    strict = dataclasses.replace(Config(), fall_stay_s=60.0)        # harus tergeletak semenit: tidak mungkin di klip ini
    assert replay(strict, make_trace(fall_script, 6.0)) == []


def test_replay_of_empty_trace_is_empty():
    assert replay(Config(), PoseTrace(np.array([]), 480, 640, [])) == []


def test_marks_are_matched_to_alarms_before_them():
    alarms = [Alarm(12.0, "fall"), Alarm(40.0, "violence"), Alarm(71.0, "fall")]
    marks = [(14.0, "jatuh"), (30.0, "jatuh"), (42.0, "kekerasan"), (60.0, "biasa"), (73.0, "biasa")]
    rows = match_marks(alarms, marks)
    assert [(r.label, r.ok) for r in rows] == [("jatuh", True), ("jatuh", False), ("kekerasan", True),
                                                 ("biasa", True), ("biasa", False)]
    assert rows[0].delay_s == pytest.approx(-2.0) and rows[1].delay_s is None


def test_wrong_kind_of_alarm_does_not_count_as_a_hit():
    rows = match_marks([Alarm(12.0, "violence")], [(14.0, "jatuh")])
    assert rows[0].ok is False and rows[0].other == ["violence"]


def test_video_ticks_reads_a_file_at_engine_rate(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120))
    for i in range(12):
        w.write(np.full((120, 160, 3), i * 20, np.uint8))
    w.release()
    got = list(video_ticks(path, rate_hz=10.0))
    assert [t for t, _ in got] == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert [round(float(f.mean()) / 20) for _, f in got] == [0, 3, 6, 9]


def test_zip_ticks_reads_images_in_name_order(tmp_path):
    path = tmp_path / "klip.zip"
    with zipfile.ZipFile(path, "w") as z:
        for i in (3, 1, 2, 4, 6, 5, 7):                  # urutan di arsip sengaja acak
            ok, buf = cv2.imencode(".png", np.full((60, 80, 3), i * 30, np.uint8))
            z.writestr(f"klip/klip-{i:03d}.png", buf.tobytes())
        z.writestr("klip/catatan.txt", "bukan gambar")
    got = list(zip_ticks(path, fps=30.0, rate_hz=10.0))
    assert [int(f[0, 0, 0]) // 30 for _, f in got] == [1, 4, 7]
    assert [t for t, _ in got] == pytest.approx([0.0, 0.1, 0.2])


def test_session_ticks_reads_a_recording(tmp_path):
    r = Recorder(tmp_path, name="sesi")
    for i in range(3):
        r.write(50.0 + i * 0.1, np.full((120, 160, 3), 90, np.uint8))
    r.close()
    assert [t for t, _ in session_ticks(r.folder)] == [50.0, 50.1, 50.2]


def test_video_ticks_agrees_with_tick_indices_for_every_phase(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (160, 120))
    for i in range(23):
        w.write(np.full((120, 160, 3), i * 10, np.uint8))
    w.release()
    for phase in (0, 1, 2):
        got = list(video_ticks(path, rate_hz=10.0, phase=phase))
        want = tick_indices(23, 25.0, 10.0, phase)
        assert [t for t, _ in got] == pytest.approx([t for t, _ in want])
        assert [round(float(f.mean()) / 10) for _, f in got] == [i for _, i in want]
