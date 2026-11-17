"""Pembungkus YOLO-pose (Ultralytics) + pelacak ByteTrack, dijalankan lewat OpenVINO di CPU.

    python -m sahabat.pose --ekspor   mengekspor model ke OpenVINO untuk tiap resolusi (butuh internet sekali)
    python -m sahabat.pose --cek      mengukur FPS tiap resolusi di laptop ini
"""
from __future__ import annotations

import os
import shutil
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import DATA_DIR, MODELS_DIR, Config
from .types import Person

# Ultralytics menulis berkas pengaturannya sendiri; arahkan ke dalam folder aplikasi.
os.environ.setdefault("YOLO_CONFIG_DIR", str(DATA_DIR / "ultralytics"))

MODEL_NAME = "yolo11n-pose"
WEIGHTS = MODELS_DIR / f"{MODEL_NAME}.pt"
_SEM_FAILCRITICALERRORS = 0x0001


@contextmanager
def _windows_loader_errors():
    """Kembalikan galat DLL ke pemanggil selama OpenVINO memuat plugin."""
    if sys.platform != "win32":
        yield
        return

    from ctypes import WinDLL, WinError, byref, get_last_error
    from ctypes.wintypes import BOOL, DWORD, LPDWORD

    kernel = WinDLL("kernel32", use_last_error=True)
    kernel.GetThreadErrorMode.restype = DWORD
    kernel.SetThreadErrorMode.argtypes = (DWORD, LPDWORD)
    kernel.SetThreadErrorMode.restype = BOOL
    previous = DWORD()
    # Ultralytics memeriksa plugin GPU opsional bahkan saat CPU diminta.
    # DLL OpenCL lama dapat memunculkan dialog sebelum OpenVINO menerima galat.
    mode = kernel.GetThreadErrorMode() | _SEM_FAILCRITICALERRORS
    if not kernel.SetThreadErrorMode(mode, byref(previous)):
        raise WinError(get_last_error())
    try:
        yield
    finally:
        kernel.SetThreadErrorMode(previous, None)


class PoseProvider(Protocol):
    def detect(self, frame: np.ndarray) -> list[Person]: ...


def model_dir(imgsz: int) -> Path:
    return MODELS_DIR / f"{MODEL_NAME}_{imgsz}_openvino_model"


def export_all(sizes: tuple[int, ...]) -> None:
    from ultralytics import YOLO
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        target = model_dir(size)
        if target.exists():
            continue
        exported = Path(YOLO(str(WEIGHTS)).export(format="openvino", imgsz=size))
        shutil.move(str(exported), str(target))


class YoloPose:
    def __init__(self, imgsz: int) -> None:
        from ultralytics import YOLO
        path = model_dir(imgsz)
        if not path.exists():
            raise FileNotFoundError(f"Model {path.name} belum ada. Jalankan: python -m sahabat.pose --ekspor")
        self.imgsz = imgsz
        self._model = YOLO(str(path), task="pose")

    def reset(self) -> None:
        """Melupakan semua nomor pelacak (dipakai saat berganti video)."""
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    def detect(self, frame: np.ndarray) -> list[Person]:
        # Predictor Ultralytics memuat model pada panggilan pertama.
        loader = _windows_loader_errors() if self._model.predictor is None else nullcontext()
        with loader:
            result = self._model.track(frame, persist=True, tracker="bytetrack.yaml",
                                       imgsz=self.imgsz, device="cpu", verbose=False)[0]
        boxes, keypoints = result.boxes, result.keypoints
        if boxes is None or boxes.id is None or keypoints is None or keypoints.conf is None:
            return []
        ids = boxes.id.int().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        xy = keypoints.xy.cpu().numpy().astype(np.float32)
        conf = keypoints.conf.cpu().numpy().astype(np.float32)
        return [Person(int(ids[i]), xy[i], conf[i], tuple(float(v) for v in xyxy[i])) for i in range(len(ids))]


def _check() -> int:
    import cv2

    from .autotune import choose_imgsz, measure, pipeline_fps
    from .source import find_camera

    cfg = Config()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    index = find_camera()
    if index is not None:
        cap = cv2.VideoCapture(index)   # find_camera menyertakan ID backend pada indeks
        for _ in range(10):
            ok, got = cap.read()
            if ok:
                frame = got
        cap.release()
    print(f"kamera: {'indeks ' + str(index) if index is not None else 'tidak ada (pakai frame hitam)'}")
    fps = measure(YoloPose, cfg.imgsz_candidates, frame)
    for size, value in fps.items():
        print(f"  resolusi {size}: {value:5.1f} FPS inferensi saja, ~{pipeline_fps(value):4.1f} FPS seluruh putaran")
    size, slow = choose_imgsz({s: pipeline_fps(v) for s, v in fps.items()}, cfg.fps_target, cfg.fps_floor)
    print(f"dipilih: {size}{'  (LAMBAT: di bawah ' + str(cfg.fps_floor) + ' FPS)' if slow else ''}")
    people = YoloPose(size).detect(frame)
    print(f"orang terdeteksi pada frame uji: {len(people)}")
    return 1 if slow else 0


if __name__ == "__main__":
    if "--ekspor" in sys.argv:
        export_all(Config().imgsz_candidates)
        print("ekspor selesai:", ", ".join(model_dir(s).name for s in Config().imgsz_candidates))
    elif "--cek" in sys.argv:
        sys.exit(_check())
    else:
        print(__doc__)
