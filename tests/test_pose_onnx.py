"""Model pose ONNX dan backend OnnxPose (jalur GPU Mode CCTV)."""
from pathlib import Path

import numpy as np
import pytest

from sahabat.config import MODELS_DIR

ONNX_MODEL = MODELS_DIR / "yolo11n-pose_416.onnx"


@pytest.mark.skipif(not ONNX_MODEL.exists(), reason="model ONNX belum diekspor")
def test_exported_model_shape_and_dynamic_batch():
    import onnxruntime as ort
    s = ort.InferenceSession(str(ONNX_MODEL), providers=["CPUExecutionProvider"])
    out = s.run(None, {"images": np.zeros((3, 3, 416, 416), np.float32)})[0]
    assert out.shape == (3, 56, 3549)


# ---- backend OnnxPose ----
import cv2  # noqa: E402

from sahabat.pose import YoloPose, model_dir  # noqa: E402

FALLS = Path("D:/sahabat-data/gmdcsa24/ekstrak/Subject 1/Fall")
VIDEO_1 = FALLS / "11.mp4"


class FakeSession:
    """Satu deteksi tetap per gambar (ruang 416), dengan titik tubuh di dalam kotak."""

    def __init__(self, box_in_416):
        x1, y1, x2, y2 = box_in_416
        self.box = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        self.batches = []

    def run(self, names, feeds):
        x = feeds["images"]
        self.batches.append(x.shape[0])
        out = np.zeros((x.shape[0], 56, 3549), np.float32)
        cx, cy, w, h = self.box
        out[:, 0:4, 10] = np.array([cx, cy, w, h], np.float32)[None, :]
        out[:, 4, 10] = 0.9
        for k in range(17):
            out[:, 5 + 3 * k, 10] = cx
            out[:, 6 + 3 * k, 10] = cy
            out[:, 7 + 3 * k, 10] = 0.9
        return [out]


def test_batch_letterboxes_each_frame_and_maps_points_back():                    # Review Focus 4
    from sahabat.pose_onnx import OnnxPose
    pose = OnnxPose(session=FakeSession(box_in_416=(100, 100, 200, 300)))
    small, big = np.zeros((360, 640, 3), np.uint8), np.zeros((720, 1280, 3), np.uint8)
    out = pose.detect_many({"k1": small, "k2": big})
    assert out["k1"][0].box[2] == pytest.approx(200 * 640 / 416, abs=1.0)
    assert out["k2"][0].box[2] == pytest.approx(200 * 1280 / 416, abs=2.0)
    assert out["k1"][0].kp[0][0] == pytest.approx(150 * 640 / 416, abs=1.0)
    # DirectML melambat bila bentuk masukan berganti-ganti (campur 1-4 kamera: 77 ms/gambar, tetap 1: 43 ms), jadi
    # model selalu dijalankan dengan bentuk tetap 1x3x416x416, satu gambar per run.
    assert pose.session.batches == [1, 1]


def test_trackers_are_per_camera():
    from sahabat.pose_onnx import OnnxPose
    pose = OnnxPose(session=FakeSession(box_in_416=(100, 100, 200, 300)))
    frame = np.zeros((360, 640, 3), np.uint8)
    for _ in range(3):
        a = pose.detect("k1", frame)
        b = pose.detect("k2", frame)
    assert a and b and a[0].track_id == b[0].track_id == 1                      # nomor per pelacak
    pose.reset("k1")
    assert [p.track_id for p in pose.detect("k1", frame)] == [1]
    assert pose.nama == "CPU (ONNX)"


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


@pytest.mark.skipif(not (ONNX_MODEL.exists() and model_dir(416).exists() and VIDEO_1.exists()), reason="model/video")
def test_onnx_cpu_matches_openvino_path():
    """Kesetaraan backend: 40 frame GMDCSA, jalur ONNX (CPU) vs YoloPose OpenVINO (Mode Pameran)."""
    from sahabat.pose_onnx import OnnxPose
    frames = _frames(VIDEO_1, 40)
    ref = YoloPose(416)
    onnx = OnnxPose()
    expected = [ref.detect(f) for f in frames]
    got = [onnx.detect("k1", f) for f in frames]
    assert sum(len(ps) for ps in expected) > 20
    assert [[p.track_id for p in ps] for ps in got] == [[p.track_id for p in ps] for ps in expected]
    for ps, qs in zip(got, expected):
        for p, q in zip(ps, qs):
            np.testing.assert_allclose(p.kp, q.kp, atol=1.5)
            np.testing.assert_allclose(p.box, q.box, atol=2.0)


@pytest.mark.skipif(not ONNX_MODEL.exists(), reason="model ONNX belum diekspor")
def test_real_model_runs_at_other_resolutions():
    """Model diekspor dengan ukuran dinamis: resolusi autotune (480/320) dipakai juga di GPU, tidak diturunkan."""
    from sahabat.pose_onnx import OnnxPose
    frame = np.full((360, 640, 3), 90, np.uint8)
    for size in (320, 480):
        pose = OnnxPose(imgsz=size)
        assert pose.imgsz == size and pose.detect("k1", frame) == []
