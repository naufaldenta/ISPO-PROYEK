"""Ekstrak jejak pose CAUCAFall atau split uji resmi Bus Violence."""
from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
from zipfile import ZipFile

from .putar import trace, video_ticks
from sahabat.config import Config
from sahabat.pose import YoloPose


def batch_trace(video: Path, rate: float, model, batch: int):
    import numpy as np
    from sahabat.pose import _windows_loader_errors
    from sahabat.types import Person
    from .putar import PoseTrace

    frames = iter(video_ticks(video, rate))
    times, people = [], []
    height, width = 0, 0
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", ()):
        tracker.reset()
    while chunk := list(islice(frames, batch)):
        with _windows_loader_errors():
            results = model.track([frame for _, frame in chunk], persist=True, tracker="bytetrack.yaml",
                                  imgsz=416, device="cpu", verbose=False)
        for (t, frame), result in zip(chunk, results, strict=True):
            times.append(t)
            height, width = frame.shape[:2]
            current = []
            if result.boxes.id is not None and result.keypoints.conf is not None:
                for track_id, kp, conf, box in zip(result.boxes.id.int().tolist(),
                                                 result.keypoints.xy.cpu().numpy(),
                                                 result.keypoints.conf.cpu().numpy(),
                                                 result.boxes.xyxy.cpu().numpy(), strict=True):
                    current.append(Person(track_id, kp.astype(np.float32), conf.astype(np.float32),
                                          tuple(map(float, box))))
            people.append(current)
    return PoseTrace(np.array(times), height, width, people)


def batch_model(root: Path, batch: int):
    import openvino as ov
    from ultralytics import YOLO
    from ultralytics.utils import YAML
    from sahabat.pose import model_dir

    target = root / f"b{batch}_openvino_model"
    original = model_dir(416)
    xml = next(original.glob("*.xml"))
    expected = (target / xml.name, target / xml.with_suffix(".bin").name, target / "metadata.yaml")
    if not all(path.is_file() for path in expected):
        target.mkdir(exist_ok=True)
        graph = ov.Core().read_model(xml)
        graph.reshape([batch, 3, 416, 416])
        ov.save_model(graph, target / xml.name, compress_to_fp16=False)
        meta = YAML.load(original / "metadata.yaml")
        meta["batch"] = batch
        meta["args"]["batch"] = batch
        YAML.save(target / "metadata.yaml", meta)
    return YOLO(str(target), task="pose")


def bus_videos(root: Path) -> list[Path]:
    with ZipFile(root / "bus-violence.zip") as archive:
        names = archive.read("test.txt").decode().splitlines()
        videos = []
        for name in names:
            if Path(name).name != name or not name.endswith(".mp4"):
                raise ValueError(f"Nama video Bus Violence tidak valid: {name}")
            folder = "Violence" if name.startswith("VIOLENCE_") else "NoViolence"
            source = f"{folder}/{name}"
            target = root / "bus" / folder / name
            if not target.exists() or target.stat().st_size != archive.getinfo(source).file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".part")
                temporary.write_bytes(archive.read(source))
                temporary.replace(target)
            videos.append(target)
    return videos


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("cauca", "bus"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--part", type=int, choices=(0, 1))
    parser.add_argument("--batch", type=int, choices=(1, 8), default=1)
    args = parser.parse_args(argv)
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    if args.dataset == "cauca":
        videos = sorted((args.root / "cauca").rglob("*.avi"))
        if len(videos) != 100:
            raise ValueError(f"CAUCAFall: {len(videos)} video, seharusnya 100")
    else:
        videos = bus_videos(args.root)
    if args.part is not None:
        videos = videos[args.part::2]
    out = args.root / f"{args.dataset}_416"
    out.mkdir(exist_ok=True)
    pose = YoloPose(416) if args.batch == 1 else batch_model(args.root, args.batch)
    rate = Config().rate_hz
    for i, video in enumerate(videos, 1):
        target = out / f"{video.stem}.npz"
        if not target.exists():
            if args.batch == 1:
                pose.reset()
                poses = trace(video_ticks(video, rate), pose)
            else:
                poses = batch_trace(video, rate, pose, args.batch)
            if not len(poses.t):
                raise ValueError(f"Tidak ada frame yang berhasil dibaca dari {video}")
            temporary = target.with_suffix(".part.npz")
            poses.save(temporary)
            temporary.replace(target)
        if i % 5 == 0 or i == len(videos):
            print(f"{args.dataset} {i}/{len(videos)} {video.name}", flush=True)


if __name__ == "__main__":
    main()
