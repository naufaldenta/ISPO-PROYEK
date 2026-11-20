"""Jalur GPU Mode CCTV: YOLO11n-pose lewat ONNX Runtime (DirectML untuk GPU NVIDIA/AMD/Intel, atau CPU).

Antarmukanya sama dengan pose.SharedPose (detect per kamera, reset, forget) ditambah detect_many untuk beberapa kamera
dalam satu panggilan (model tetap dijalankan satu gambar per run, dengan bentuk tetap). Langkahnya meniru YOLO.track()
Ultralytics yang dipakai Mode Pameran, dengan objek Ultralytics yang sama (LetterBox, non_max_suppression, Results,
BYTETracker per kamera), supaya hasilnya setara:
- letterbox 416x416 tanpa "auto" (sama dengan model OpenVINO berukuran tetap), pad 114 di tengah;
- NMS dengan nilai yang dipakai track(): keyakinan 0,1, IoU 0,5, maks. 300, satu kelas;
- titik dan kotak dikembalikan ke ukuran asli tiap frame, lalu pelacak kamera itu memberi nomor orang.
Backend ONNX bawaan Ultralytics tidak dipakai karena ia memasang paket sendiri lewat internet saat model dimuat.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .config import MODELS_DIR
from .pose import people_from_result
from .types import Person

ONNX_MODEL = MODELS_DIR / "yolo11n-pose_416.onnx"
DML = "DmlExecutionProvider"
TRACK_CONF = 0.1            # model.track(): keyakinan bawaan supaya pelacak menerima deteksi lemah
TRACK_IOU = 0.5             # IoU NMS bawaan predictor
MAX_DET = 300


class OnnxPose:
    def __init__(self, model_path: Path = ONNX_MODEL, providers=("CPUExecutionProvider",), provider_options=None,
                 imgsz: int = 416, session=None, nama: str | None = None) -> None:
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml
        if session is None:
            import onnxruntime as ort
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model {Path(model_path).name} belum ada (scripts/ekspor_onnx.py).")
            session = ort.InferenceSession(str(model_path), providers=list(providers),
                                           provider_options=provider_options)
        self.session = session
        self.imgsz = imgsz
        self.nama = nama or ("GPU (DirectML)" if DML in providers else "CPU (ONNX)")
        self._letterbox = LetterBox((imgsz, imgsz), auto=False, stride=32)
        self._tracker_cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        self._trackers: dict = {}
        self._lock = threading.Lock()

    # ---- antarmuka SharedPose ----
    def detect(self, cam_id: str, frame: np.ndarray) -> list[Person]:
        return self.detect_many({cam_id: frame})[cam_id]

    def reset(self, cam_id: str) -> None:
        with self._lock:
            self._trackers.pop(cam_id, None)        # pelacak baru (nomor mulai 1) pada penilaian berikutnya

    def forget(self, cam_id: str) -> None:
        self.reset(cam_id)

    # ---- inti ----
    def _tracker(self, cam_id: str):
        from ultralytics.trackers.byte_tracker import BYTETracker
        tracker = self._trackers.get(cam_id)
        if tracker is None:
            tracker = self._trackers[cam_id] = BYTETracker(args=self._tracker_cfg)
        return tracker

    def detect_many(self, frames: dict[str, np.ndarray]) -> dict[str, list[Person]]:
        """Satu session.run untuk semua frame (bisa beda ukuran); pelacak tiap kamera diperbarui sendiri-sendiri."""
        import torch
        from ultralytics.engine.results import Results
        from ultralytics.utils import nms, ops
        if not frames:
            return {}
        ids = list(frames)
        with self._lock:
            boxed = [self._letterbox(image=frames[i]) for i in ids]
            x = np.ascontiguousarray(np.stack(boxed)[..., ::-1].transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
            # Bentuk masukan tetap 1x3x416x416: DirectML menyusun ulang rencana bila ukuran batch berganti-ganti
            # (Vega 8: campur 1-4 kamera 77 ms/gambar, satu per satu 43 ms/gambar).
            out = np.concatenate([self.session.run(None, {"images": x[i:i + 1]})[0] for i in range(len(x))])
            preds = nms.non_max_suppression(torch.from_numpy(np.asarray(out, dtype=np.float32)), TRACK_CONF,
                                            TRACK_IOU, None, False, max_det=MAX_DET, nc=1)
            shape = (self.imgsz, self.imgsz)
            result: dict[str, list[Person]] = {}
            for cam_id, pred in zip(ids, preds):
                orig = frames[cam_id]
                pred[:, :4] = ops.scale_boxes(shape, pred[:, :4], orig.shape)
                kpts = ops.scale_coords(shape, pred[:, 6:].view(pred.shape[0], 17, 3), orig.shape)
                res = Results(orig, path="", names={0: "person"}, boxes=pred[:, :6], keypoints=kpts)
                tracks = self._tracker(cam_id).update(res.boxes.cpu().numpy(), orig)
                if len(tracks) == 0:
                    result[cam_id] = []
                    continue
                res = res[tracks[:, -1].astype(int)]
                res.update(boxes=torch.as_tensor(tracks[:, :-1]))
                result[cam_id] = people_from_result(res)
            return result
