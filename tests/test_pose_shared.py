"""Satu model pose bersama untuk banyak kamera; setiap kamera punya pelacak sendiri."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from sahabat.pose import CameraPose, SharedPose, YoloPose, model_dir

FRAME = np.zeros((48, 64, 3), np.uint8)
FALLS = Path("D:/sahabat-data/gmdcsa24/ekstrak/Subject 1/Fall")
VIDEO_1, VIDEO_2 = FALLS / "11.mp4", FALLS / "01.mp4"


class FakeYoloPose:
    """Meniru YoloPose: predictor dibuat pada panggilan pertama; daftar pelacak kosong mendapat pelacak baru
    (seperti on_predict_start dengan persist=True)."""

    def __init__(self):
        self._model = SimpleNamespace(predictor=None)
        self.used_tracker_ids = []
        self.imgsz = 320

    def detect(self, frame):
        if self._model.predictor is None:
            self._model.predictor = SimpleNamespace(trackers=[], vid_path=[None])
        pred = self._model.predictor
        if not pred.trackers:
            pred.trackers = [object()]
        self.used_tracker_ids.append(id(pred.trackers[0]))
        return []


def test_trackers_are_swapped_per_camera():
    fake = FakeYoloPose()
    shared = SharedPose(fake)
    a, b = CameraPose(shared, "k1"), CameraPose(shared, "k2")
    a.detect(FRAME)
    b.detect(FRAME)
    a.detect(FRAME)
    used = list(fake.used_tracker_ids)
    assert used[0] == used[2] and used[0] != used[1]
    a.reset()
    a.detect(FRAME)
    assert fake.used_tracker_ids[-1] not in used          # pelacak baru untuk k1
    b.detect(FRAME)
    assert fake.used_tracker_ids[-1] == used[1]           # pelacak k2 tidak tersentuh
    shared.forget("k2")
    b.detect(FRAME)
    assert fake.used_tracker_ids[-1] != used[1]


def _frames(path, n):
    cap = cv2.VideoCapture(str(path))
    out = []
    while len(out) < n:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return out


@pytest.mark.skipif(not (model_dir(320).exists() and VIDEO_1.exists() and VIDEO_2.exists()),
                    reason="butuh model pose 320 dan video GMDCSA-24")
def test_interleaved_cameras_match_single_camera_exactly():
    """Kesetaraan dengan Mode Pameran: video 1 yang diselingi video 2 memberi nomor orang, kotak, dan titik tubuh
    yang identik dengan YoloPose sendirian."""
    f1, f2 = _frames(VIDEO_1, 40), _frames(VIDEO_2, 40)
    alone = YoloPose(320)
    expected = [alone.detect(f) for f in f1]
    shared = SharedPose(YoloPose(320))
    cam1, cam2 = CameraPose(shared, "k1"), CameraPose(shared, "k2")
    got, other = [], []
    for a, b in zip(f1, f2):
        got.append(cam1.detect(a))
        other.append(cam2.detect(b))
    assert sum(len(ps) for ps in expected) > 20                      # memang ada orang yang dilacak
    assert sum(len(ps) for ps in other) > 20
    assert [[p.track_id for p in ps] for ps in got] == [[p.track_id for p in ps] for ps in expected]
    for ps, qs in zip(got, expected):
        for p, q in zip(ps, qs):
            np.testing.assert_allclose(p.kp, q.kp, atol=1e-4)
            np.testing.assert_allclose(p.box, q.box, atol=1e-3)
