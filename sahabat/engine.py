"""Mesin: merangkai sumber video -> pose -> riwayat -> penilai -> alarm, 10 kali per detik."""
from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .alarm import AlarmManager
from .config import Config
from .fall import FallDetector
from .gestures import GestureDetector
from .history import History
from .pose import PoseProvider
from .predlog import PredLog
from .source import VideoSource
from .types import SKELETON_EDGES
from .violence import ViolenceScorer
from . import zone as zones

BarState = Literal["aman", "jatuh", "kekerasan", "tidak_dinilai", "kamera_terputus"]
BAR_TEXT: dict[str, str] = {
    "aman": "AMAN",
    "jatuh": "JATUH TERDETEKSI",
    "kekerasan": "INDIKASI KEKERASAN TERDETEKSI",
    "tidak_dinilai": "TIDAK DAPAT DINILAI",
    "kamera_terputus": "KAMERA TERPUTUS",
}
SLOW_TEXT = "laptop ini terlalu lambat, deteksi bisa meleset"
CAMERA_SLOW_TEXT = "kamera hanya mengirim {} gambar per detik (ruangan kurang terang?), deteksi bisa meleset"
CAMERA_LOST_NOTE = "kamera terputus; alarm sebelumnya belum diperiksa"

_BGR = {"green": (89, 199, 52), "red": (48, 59, 255), "gray": (160, 160, 160), "out": (125, 125, 125)}
_DRAW_CONF = 0.5
_FPS_WINDOW_S = 2.0


@dataclass(frozen=True, eq=False)
class Overlay:
    track_id: int
    kp: np.ndarray
    conf: np.ndarray
    color: Literal["green", "red", "gray", "out"]   # out = di luar zona pantau, tidak dinilai
    label: str


@dataclass(frozen=True, eq=False)
class Snapshot:
    bar: str
    text: str
    fps: float
    people: int
    slow: bool
    alarm_seq: int                 # naik 1 tiap kejadian baru; klien memakainya untuk memulai bunyi
    overlays: tuple[Overlay, ...]
    slow_text: str = ""            # peringatan untuk layar; kosong bila tidak ada
    recording: bool = False        # perekam uji sedang menyimpan video


def _label(track_id: int, color: str, mode: str) -> str:
    if color == "gray":
        return f"#{track_id} tidak dapat dinilai"
    if mode == "upper":
        return f"#{track_id} setengah badan (tendangan tidak dinilai)"
    return f"#{track_id}"


class Engine:
    def __init__(self, cfg: Config, source: VideoSource | None, pose: PoseProvider,
                 log: PredLog | None = None, slow: bool = False,
                 violence_model: Path | None = None, fall_model: Path | None = None,
                 recorder=None, riwayat=None, *, kamera: tuple[str, str] = ("", "")) -> None:
        self.cfg = cfg
        self.source = source           # None hanya untuk uji lewat step()
        self.pose = pose
        self.log = log
        self.slow = slow
        self.recorder = recorder       # sahabat.recorder.Recorder; None = tidak merekam
        self.riwayat = riwayat         # sahabat.riwayat.Riwayat; None = tanpa riwayat kejadian
        self.kamera = kamera           # Mode CCTV: (ID, nama) kamera ini; ("", "") = Mode Pameran
        self.zone = zones.load(cfg.zone)   # None = seluruh gambar
        self.history = History(cfg)
        self.fall = FallDetector(cfg, fall_model)
        self.gestures = GestureDetector(cfg)
        self._fall_model_path = fall_model
        self._input_version = 0
        self.violence = ViolenceScorer(cfg, violence_model)
        self.alarm = AlarmManager(cfg)
        self._alarm_seq = 0
        self._step_times: deque[float] = deque()
        self._last_t = 0.0
        self._last_report: dict[str, float] = {}
        self._lock = threading.Lock()
        self._snapshot = Snapshot("kamera_terputus", BAR_TEXT["kamera_terputus"], 0.0, 0, slow, 0, (),
                                  recording=self._recording())
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- satu putaran penilaian -------------------------------------------------

    def step(self, t: float, frame: np.ndarray | None, input_version: int | None = None) -> Snapshot:
        """Satu putaran. frame None berarti kamera terputus.

        Inferensi pose (bagian terlama) berjalan di luar kunci, supaya layar dan tombol tidak ikut menunggu.
        """
        if input_version is not None:
            if self.source is not None and input_version != self.source.version:
                return self._snapshot
            if input_version != self._input_version:
                if hasattr(self.pose, "reset"):
                    self.pose.reset()
                with self._lock:
                    self.history = History(self.cfg)
                    self.fall = FallDetector(self.cfg, self._fall_model_path)
                    self.gestures.reset()
                    self.alarm.reset_input()
                    self._step_times.clear()
                    self._frame = None
                    self._input_version = input_version
                    self._snapshot = self._camera_lost_snapshot()
        if frame is None:
            with self._lock:
                self._last_t = t
                snap = self._camera_lost_snapshot()
                self._snapshot = snap
            return snap

        pose_ok = True
        try:
            people = self.pose.detect(frame)
        except Exception:                          # model tersedak pada satu frame: jangan matikan sistem
            self._report("pose")
            people, pose_ok = [], False

        with self._lock:
            if input_version is not None and self.source is not None and input_version != self.source.version:
                return self._snapshot
            self._last_t = t
            scoring_ok = True
            fall_ids, pairs, events, modes = [], [], [], {}
            people, outside = zones.split(people, self.zone, frame.shape)
            try:
                if pose_ok:
                    people = self.history.update(t, people)
                    tracks = self.history.tracks
                    modes = {track_id: tr.mode for track_id, tr in tracks.items()}
                    fall_ids = self.fall.tick(t, tracks, frame.shape[0])
                    pairs = self.violence.score(tracks)
                    seen = {p.track_id for p in people}
                    hits = self.gestures.tick(t, tracks)
                    exiting = self._exiting(t, tracks, seen, frame.shape[:2]) if seen else set()
                    events = self.alarm.tick(t, fall_ids, pairs, modes, hits=hits,
                                             seen=seen if self.cfg.gesture_enabled else None,
                                             fall_sources=self.fall.sources, exiting=exiting)
            except Exception:                      # penilai gagal: katakan "tidak dapat dinilai", jangan membeku di AMAN
                self._report("penilai")
                scoring_ok = False
            self._alarm_seq += len(events)

            red, overflow = self.alarm.red_ids, set(self.history.overflow_ids)
            overlays, assessable = [], 0
            for p in people:
                mode = "none" if p.track_id in overflow else modes.get(p.track_id, "none")
                if mode != "none":
                    assessable += 1
                color = "red" if p.track_id in red else ("gray" if mode == "none" else "green")
                overlays.append(Overlay(p.track_id, p.kp, p.conf, color, _label(p.track_id, color, mode)))
            overlays.extend(Overlay(p.track_id, p.kp, p.conf, "out", "") for p in outside)

            active = self.alarm.active
            if active is not None:
                bar = "jatuh" if active.kind == "fall" else "kekerasan"
            elif not pose_ok or not scoring_ok or (people and assessable == 0):
                bar = "tidak_dinilai"
            else:
                bar = "aman"

            self._step_times.append(t)
            while self._step_times and t - self._step_times[0] > _FPS_WINDOW_S:
                self._step_times.popleft()
            span = self._step_times[-1] - self._step_times[0]
            fps = (len(self._step_times) - 1) / span if span > 0 else 0.0

            camera_fps = getattr(self.source, "fps", 0.0)
            if 0.0 < camera_fps < self.cfg.fps_floor:        # webcam melambat sendiri di ruangan redup
                slow, slow_text = True, CAMERA_SLOW_TEXT.format(round(camera_fps))
            else:
                slow, slow_text = self.slow, (SLOW_TEXT if self.slow else "")
            snap = Snapshot(bar, BAR_TEXT[bar], round(fps, 1), len(people), slow,
                            self._alarm_seq, tuple(overlays), slow_text, self._recording())
            self._snapshot = snap
            self._frame = frame

        if self.recorder is not None:
            self.recorder.write(t, frame)
        if self.riwayat is not None:
            self._feed_riwayat(t, frame, snap.overlays, events)
        if self.log is not None:
            self.log.write({
                "t": round(t, 3),
                "people": [{"id": p.track_id, "mode": modes.get(p.track_id, "none")} for p in people],
                "fall_ids": [int(i) for i in fall_ids],
                "pairs": [{"ids": list(ps.ids), "prob": round(float(ps.prob), 3)} for ps in pairs],
                "alarm": events[0].kind if events else None,
                "sumber": [e.source for e in events],
                "kejadian": [{"jenis": e.kind, "ids": list(e.ids), "sumber": e.source} for e in events],
            })
        return snap

    @staticmethod
    def _exiting(t, tracks, seen, shape) -> set[int]:
        height, width = shape
        out = set()
        for i, tr in tracks.items():
            if i in seen or not tr.samples or t - tr.last_seen < 0.15:
                continue
            recent = tr.samples[-1]
            x, y = recent.center
            older = next((s for s in reversed(tr.samples) if recent.t - s.t >= 0.20), None)
            moving = (older is not None and recent.t - older.t <= 0.5
                      and np.linalg.norm(recent.center - older.center)
                      >= 1.4 * tr.ref_size * (recent.t - older.t))
            if x <= 0.12 * width or x >= 0.88 * width or y >= 0.84 * height or moving:
                out.add(i)
        return out

    def _feed_riwayat(self, t, frame, overlays, events) -> None:
        """Frame layar (berkerangka) untuk klip bukti, lalu kejadian baru. Gagal di sini tidak menghentikan alarm."""
        try:
            if self.riwayat.wants_frames:
                jpeg = annotated_jpeg(frame, overlays, quality=70, zone=self.zone)
                if jpeg is not None:
                    if self.kamera[0]:
                        self.riwayat.frame(t, jpeg, kamera=self.kamera[0])
                    else:                                      # Mode Pameran: panggilan lama
                        self.riwayat.frame(t, jpeg)
        except Exception:
            self._report("klip bukti")
        try:
            for e in events:
                jenis = "jatuh" if e.kind == "fall" else "kekerasan"
                if self.kamera[0]:
                    self.riwayat.event(t, jenis, e.mode, e.ids, e.source, kamera=self.kamera[0],
                                       kamera_nama=self.kamera[1])
                else:
                    self.riwayat.event(t, jenis, e.mode, e.ids, e.source)
        except Exception:
            self._report("riwayat")

    def _camera_lost_snapshot(self) -> Snapshot:
        """Kamera terputus. Alarm yang belum diperiksa tetap merah dan tetap bisa diperiksa."""
        active = self.alarm.active
        if active is None:
            return Snapshot("kamera_terputus", BAR_TEXT["kamera_terputus"], 0.0, 0, self.slow, self._alarm_seq, (),
                            SLOW_TEXT if self.slow else "", self._recording())
        bar = "jatuh" if active.kind == "fall" else "kekerasan"
        return Snapshot(bar, BAR_TEXT[bar], 0.0, 0, True, self._alarm_seq, (), CAMERA_LOST_NOTE, self._recording())

    def _recording(self) -> bool:
        return self.recorder is not None and bool(self.recorder.active)

    def _report(self, where: str) -> None:
        """Mencetak galat paling sering sekali per 5 detik per tempat, supaya jendela hitam tidak banjir."""
        now = time.monotonic()
        if now - self._last_report.get(where, float("-inf")) >= 5.0:
            self._last_report[where] = now
            print(f"Galat di {where}:")
            traceback.print_exc()

    def snapshot(self) -> Snapshot:
        return self._snapshot                      # rujukan diganti utuh oleh step(); membaca tanpa kunci aman

    def select_camera(self, spec: int | str | None) -> None:
        # Pergantian menunggu penilaian yang sudah berjalan; inferensi pose lama
        # yang masih di luar kunci akan dibuang lewat pemeriksaan versi di step().
        with self._lock:
            if self.source is not None:
                self.source.select(spec)

    def acknowledge(self) -> bool:
        with self._lock:
            return self.alarm.acknowledge(self._last_t)

    def set_zone(self, points) -> zones.Zone | None:
        """Zona pantau baru (titik ternormalisasi) atau None untuk seluruh gambar. ValueError bila tidak sah."""
        zone = zones.normalize(points)
        with self._lock:
            self.zone = zone
        return zone

    def mark(self, label: str) -> bool:
        """Tanda adegan dari pencatat, dicatat pada waktu penilaian terakhir. False bila tidak merekam."""
        if self.recorder is None:
            return False
        return bool(self.recorder.mark(self._last_t, label))

    # ---- utas latar -------------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="mesin", daemon=True)
            self._thread.start()

    def pause(self) -> None:
        """Mode CCTV menyala: utas berhenti, tetapi log, Riwayat, dan perekam tetap terbuka untuk resume()."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def resume(self) -> None:
        self.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self.log is not None:
            self.log.close()
        if self.recorder is not None:
            self.recorder.close()
        if self.riwayat is not None:
            self.riwayat.close()

    def _run(self) -> None:
        period = 1.0 / self.cfg.rate_hz
        last_frame_t = None
        next_tick = time.monotonic()
        while not self._stop.is_set():
            version, got = self.source.packet() if self.source is not None else (None, None)
            try:
                if got is None:
                    self.step(time.monotonic(), None, input_version=version)
                elif got[0] != last_frame_t:             # hanya frame baru yang dinilai
                    last_frame_t = got[0]
                    self.step(got[0], got[1], input_version=version)
            except Exception:
                traceback.print_exc()
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay < -period:                          # tertinggal jauh: jangan mengejar
                next_tick = time.monotonic()
            elif delay > 0:
                self._stop.wait(delay)

    # ---- gambar untuk layar -----------------------------------------------------

    def latest_jpeg(self, width: int | None = None, quality: int = 75) -> bytes | None:
        """Frame kamera terbaru dengan kerangka dari penilaian terakhir, sebagai JPEG.
        width = lebar hasil (ubin grid Mode CCTV); kerangka digambar dulu pada ukuran asli."""
        got = self.source.read() if self.source is not None else None
        snap = self._snapshot
        frame = got[1] if got is not None else self._frame
        if frame is None or (self.source is not None and got is None):
            return annotated_jpeg(np.full((480, 640, 3), 40, dtype=np.uint8), (), quality, width=width)
        overlays = snap.overlays if self.source is None or self.source.version == self._input_version else ()
        return annotated_jpeg(frame, overlays, quality, zone=self.zone, width=width)


def annotated_jpeg(frame: np.ndarray, overlays, quality: int = 75, zone=None, width: int | None = None) -> bytes | None:
    """Frame dengan zona pantau, kerangka, dan label tiap orang, sebagai JPEG. Frame asli tidak diubah."""
    canvas = frame.copy()
    zones.draw(canvas, zone)
    for ov in overlays:
        _draw(canvas, ov)
    if width and canvas.shape[1] > width:
        canvas = cv2.resize(canvas, (width, round(canvas.shape[0] * width / canvas.shape[1])),
                            interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def _draw(canvas: np.ndarray, ov: Overlay) -> None:
    color = _BGR[ov.color]
    thick = 3 if ov.color == "red" else (1 if ov.color == "out" else 2)
    pts = ov.kp
    for a, b in SKELETON_EDGES:
        if ov.conf[a] >= _DRAW_CONF and ov.conf[b] >= _DRAW_CONF:
            cv2.line(canvas, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), color, thick,
                     cv2.LINE_AA)
    seen = [pts[i] for i in range(len(pts)) if ov.conf[i] >= _DRAW_CONF]
    for x, y in seen:
        cv2.circle(canvas, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
    if seen and ov.label:
        top = min(seen, key=lambda p: p[1])
        x = max(4, int(top[0]) - 20)
        y = max(18, int(top[1]) - 12)
        cv2.putText(canvas, ov.label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, ov.label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
