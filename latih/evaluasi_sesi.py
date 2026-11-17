"""Replay jejak pose melalui Engine, merekam SEMUA kejadian baru (bukan hanya alarm aktif).

python -m latih.evaluasi_sesi JEJAK --violence MODEL --fall MODEL --output HASIL.json

Konfirmasi pemeriksaan disimulasikan setiap 3 detik alarm aktif. Ini evaluasi offline,
bukan pengukuran perilaku guru atau kecepatan YOLO. Tidak membaca video/audio mentah.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from sahabat.alarm import AlarmManager
from sahabat.config import Config
from sahabat.engine import Engine
from .putar import PoseTrace, _TracePose


class _ReplayEngine(Engine):
    def _report(self, where: str) -> None:
        # Mesin live bertahan saat inferensi gagal. Evaluasi harus gagal jelas,
        # bukan menghitung frame tersebut sebagai tidak ada alarm.
        raise RuntimeError(f"Replay gagal di {where}")


class EventRecorder(AlarmManager):
    def __init__(self, cfg: Config, origin: float):
        super().__init__(cfg)
        self.origin = origin
        self.events: list[dict] = []

    def tick(self, *args, **kwargs):
        new = super().tick(*args, **kwargs)
        self.events.extend({"t": round(e.t - self.origin, 3), "kind": e.kind,
                            "ids": list(e.ids), "mode": e.mode, "source": e.source} for e in new)
        return new


def replay(tr: PoseTrace, violence: Path, fall: Path, cfg: Config | None = None) -> dict:
    if not len(tr.t):
        raise ValueError("Jejak pose kosong")
    # Gagal jelas jika salah path; jangan diam-diam mengevaluasi aturan pengganti.
    models = {kind: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
              for kind, p in (("violence", violence), ("fall", fall))}
    cfg = cfg or Config()
    engine = _ReplayEngine(cfg, None, _TracePose(tr.people), violence_model=violence, fall_model=fall)
    if not engine.violence.uses_model or not engine.fall.uses_model:
        raise ValueError("Model tidak kompatibel dengan fitur aplikasi")
    recorder = EventRecorder(cfg, float(tr.t[0]))
    engine.alarm = recorder
    blank = np.zeros((tr.frame_h, tr.frame_w, 3), dtype=np.uint8)
    since = None
    for t in tr.t:
        state = engine.step(float(t), blank)
        if state.bar in ("jatuh", "kekerasan"):
            since = float(t) if since is None else since
            if t - since >= 3.0:
                engine.acknowledge()
                since = None
        else:
            since = None
    return {"models": models, "config": dataclasses.asdict(cfg), "ack_after_s": 3.0,
            "frames": len(tr.t), "duration_s": float(tr.t[-1] - tr.t[0]), "events": recorder.events}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--violence", type=Path, required=True)
    parser.add_argument("--fall", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result = replay(PoseTrace.load(args.trace), args.violence, args.fall)
    result["trace"] = {"path": str(args.trace), "sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result["events"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
