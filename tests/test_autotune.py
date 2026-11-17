import platform

import numpy as np

from sahabat.autotune import choose_imgsz, ensure_tuned, measure, pipeline_fps
from sahabat.config import Config

FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


class FakePose:
    made: list[int] = []

    def __init__(self, imgsz):
        FakePose.made.append(imgsz)

    def detect(self, frame):
        return []


def test_picks_largest_size_meeting_target():
    assert choose_imgsz({480: 7.0, 416: 10.5, 320: 14.0}, 10.0, 8.0) == (416, False)


def test_falls_back_to_floor():
    assert choose_imgsz({480: 5.0, 416: 8.4, 320: 9.0}, 10.0, 8.0) == (416, False)


def test_flags_slow_laptop():
    assert choose_imgsz({480: 3.0, 416: 4.0, 320: 6.0}, 10.0, 8.0) == (320, True)


def test_measure_reports_fps_per_size():
    fps = measure(FakePose, (480, 320), FRAME, seconds=0.05)
    assert set(fps) == {480, 320} and all(v > 100 for v in fps.values())


def test_ensure_tuned_measures_once_per_machine():
    FakePose.made = []
    cfg = ensure_tuned(Config(), FRAME, make_pose=FakePose, seconds=0.05)
    assert (cfg.imgsz, cfg.slow, cfg.tuned_for) == (480, False, platform.node())
    FakePose.made = []
    assert ensure_tuned(cfg, FRAME, make_pose=FakePose, seconds=0.05) == cfg and FakePose.made == []


def test_ensure_tuned_remeasures_on_another_laptop():
    FakePose.made = []
    moved = Config(imgsz=320, tuned_for="LAPTOP-LAIN", slow=True)
    cfg = ensure_tuned(moved, FRAME, make_pose=FakePose, seconds=0.05)
    assert (cfg.imgsz, cfg.slow) == (480, False) and FakePose.made != []


def test_pipeline_fps_counts_scoring_time_not_just_inference():
    # terukur di laptop pengembang: inferensi 12,9 FPS (480) dan 16,6 FPS (416); penilaian ~30 ms per frame
    assert pipeline_fps(12.9) < 10.0 < pipeline_fps(16.6)
    assert pipeline_fps(1000.0) < 40.0                       # secepat apa pun modelnya, penilaian tetap makan waktu
    assert pipeline_fps(0.0) == 0.0


def test_ensure_tuned_uses_pipeline_fps(monkeypatch):
    import sahabat.autotune as autotune
    monkeypatch.setattr(autotune, "measure", lambda make, sizes, frame, seconds=3.0: {480: 12.9, 416: 16.6, 320: 30.5})
    cfg = ensure_tuned(Config(), FRAME, make_pose=FakePose)
    assert (cfg.imgsz, cfg.slow) == (416, False)             # 480 hanya cukup untuk inferensi, tidak untuk seluruh putaran
