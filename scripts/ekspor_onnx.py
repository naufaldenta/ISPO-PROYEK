"""Alat pengembang: ekspor YOLO11n-pose ke ONNX (416 px, batch dinamis) untuk jalur GPU Mode CCTV.

    runtime\python\python.exe scripts\ekspor_onnx.py

Butuh paket `onnx` (hanya di laptop pengembang). Hasilnya models/yolo11n-pose_416.onnx; pengguna akhir tidak perlu
mengekspor ulang.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sahabat.config import MODELS_DIR  # noqa: E402

TARGET = MODELS_DIR / "yolo11n-pose_416.onnx"


def main() -> int:
    from ultralytics import YOLO
    exported = Path(YOLO(str(MODELS_DIR / "yolo11n-pose.pt")).export(format="onnx", imgsz=416, dynamic=True,
                                                                         simplify=False))
    shutil.move(str(exported), str(TARGET))
    print("tersimpan:", TARGET, round(TARGET.stat().st_size / 1e6, 1), "MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
