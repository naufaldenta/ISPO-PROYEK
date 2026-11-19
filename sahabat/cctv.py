"""Mode CCTV: menilai banyak kamera sekaligus dengan satu model pose bersama.

Setiap kamera punya VideoSource, pelacak (CameraPose), dan Engine sendiri. Engine itu sama persis dengan Mode Pameran
(penilai jatuh/kekerasan, alarm, zona, Riwayat), tetapi tanpa utas sendiri: satu utas pekerja memanggil step() sesuai
keputusan Penjadwal. Komputer yang tidak sanggup menilai semua kamera penuh tidak menurunkan resolusi diam-diam;
kameranya diberi status "lambat" dan layar mengatakannya, sementara alarm tetap dibunyikan.
"""
from __future__ import annotations

import statistics
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .autotune import pipeline_fps
from .engine import BAR_TEXT
from .penjadwal import GerakMurah, KamInfo, Penjadwal

RED = ("jatuh", "kekerasan")
SLOW_SUFFIX = " (dinilai lambat)"
LOST_STEP_S = 0.5           # kamera terputus: mesin diberi tahu sesering ini (pita "kamera terputus")
MEASURE_ID = "__ukur__"
SMALL_WIDTH, SMALL_QUALITY = 480, 60


class CctvError(Exception):
    """Mode CCTV tidak bisa dinyalakan; pesannya ditampilkan ke guru."""


@dataclass
class _Cam:
    id: str
    nama: str
    prioritas: bool
    zona: list | None
    signature: tuple
    source: object
    engine: object
    gerak: GerakMurah = field(default_factory=GerakMurah)
    last_step: float = float("-inf")
    last_lost: float = float("-inf")
    last_motion: float = float("-inf")
    motion: bool = False
    person_t: float = float("-inf")
    people_last: bool = False                           # penilaian terakhir kamera ini melihat orang
    penuh_sejak: float | None = None
    alarm_mono: float | None = None
    alarm_wall: float | None = None
    last_frame_t: float | None = None
    connected: bool = False
    step_times: deque = field(default_factory=deque)    # waktu penilaian 2 dtk terakhir (laju diukur runner)


def _signature(cam: dict) -> tuple:
    """Isian yang menentukan stream; bila berubah, sumber dibuka ulang."""
    return (cam.get("jenis"), cam.get("alamat_tampil"), cam.get("pengguna"), tuple(cam.get("potong") or ()),
            cam.get("ada_sandi"), cam.get("rev", 0))    # rev naik tiap simpan, termasuk bila hanya sandi berubah


class CctvRunner:
    def __init__(self, cfg, store, shared_pose, riwayat, *, engine_factory: Callable | None = None,
                 source_factory: Callable | None = None, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time, connect_timeout_s: float = 10.0,
                 sleep: Callable[[float], None] = time.sleep, start_thread: bool = True) -> None:
        self.cfg, self.store, self.shared_pose, self.riwayat = cfg, store, shared_pose, riwayat
        self._engine_factory = engine_factory or self._default_engine
        self._source_factory = source_factory or self._default_source
        self._clock, self._wall, self._sleep = clock, wall, sleep
        self.connect_timeout_s = connect_timeout_s
        self._start_thread = start_thread
        self.penjadwal = Penjadwal(cfg.rate_hz, cfg.cctv_safety_s, cfg.fps_floor, cfg.cctv_motion_hz)
        self._cams: dict[str, _Cam] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_report: dict[str, float] = {}
        self.active = False
        self.kapasitas: int | None = None

    # ---- pembuat bawaan ---------------------------------------------------------------
    def _default_source(self, cam: dict):
        from .kamera import IPCAM_PREFIX
        from .source import VideoSource
        return VideoSource(IPCAM_PREFIX + cam["id"], resolve=self.store.resolve)

    def _default_engine(self, cam: dict, source, pose):
        from .config import DATA_DIR
        from .engine import Engine
        from .predlog import PredLog
        from .server import FALL_MODEL, VIOLENCE_MODEL
        engine = Engine(self.cfg, source, pose, PredLog(DATA_DIR / "log", prefix="cctv-" + cam["id"]), False,
                        VIOLENCE_MODEL, FALL_MODEL, None, self.riwayat, kamera=(cam["id"], cam["nama"]))
        engine.set_zone(cam.get("zona"))           # selalu: jangan mewarisi zona Mode Pameran dari cfg.zone
        return engine

    def _build(self, cam: dict) -> _Cam:
        from .pose import CameraPose
        source = self._source_factory(cam)
        source.start()
        engine = self._engine_factory(cam, source, CameraPose(self.shared_pose, cam["id"]))
        return _Cam(cam["id"], cam["nama"], bool(cam.get("prioritas")), cam.get("zona"), _signature(cam), source,
                    engine)

    def _release(self, cam: _Cam) -> None:
        try:
            cam.source.stop()
        finally:
            log = getattr(cam.engine, "log", None)
            if log is not None:
                log.close()                     # Riwayat bersama tidak ditutup di sini
            self.shared_pose.forget(cam.id)

    # ---- menyalakan / mematikan ---------------------------------------------------------
    def start(self) -> dict:
        cams = self.store.active()
        if not cams:
            raise CctvError("Tidak ada kamera CCTV aktif. Tambahkan atau aktifkan kamera di panel Kamera.")
        with self._lock:
            self._cams = {c["id"]: self._build(c) for c in cams}
        deadline = self._clock() + self.connect_timeout_s
        frame = None
        while frame is None:
            for cam in self._cams.values():
                got = cam.source.read()
                if got is not None:
                    frame = got[1]
                    break
            if frame is None:
                if self._clock() >= deadline:
                    self._release_all()
                    raise CctvError("Belum ada kamera CCTV yang tersambung. Periksa panel Kamera.")
                self._sleep(0.1)
        self.kapasitas = self._measure(frame)
        self.active = True
        if self._start_thread:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="mode-cctv", daemon=True)
            self._thread.start()
        return self.snapshot()

    def _measure(self, frame) -> int:
        """Berapa kamera bisa dinilai penuh 10x per detik: 5 kali pose pada gambar kamera sungguhan."""
        times = []
        try:
            for _ in range(5):
                t0 = self._clock()
                self.shared_pose.detect(MEASURE_ID, frame)
                times.append(self._clock() - t0)
        finally:
            self.shared_pose.forget(MEASURE_ID)
        median = statistics.median(times)
        return int(pipeline_fps(1.0 / median) // self.cfg.rate_hz) if median > 0 else 0

    def _release_all(self) -> None:
        with self._lock:
            cams, self._cams = list(self._cams.values()), {}
        for cam in cams:
            self._release(cam)

    def stop(self) -> None:
        self.active = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        self._release_all()

    def reload(self) -> None:
        """Daftar kamera berubah di panel Kamera: tambah, buang, perbarui, atau buka ulang sumber."""
        wanted = {c["id"]: c for c in self.store.active()}
        with self._lock:
            removed = [cam for cid, cam in self._cams.items() if cid not in wanted]
            for cam in removed:
                del self._cams[cam.id]
            for cid, info in wanted.items():
                cam = self._cams.get(cid)
                if cam is None:
                    self._cams[cid] = self._build(info)
                    continue
                cam.nama, cam.prioritas = info["nama"], bool(info.get("prioritas"))
                if hasattr(cam.engine, "kamera"):
                    cam.engine.kamera = (cid, info["nama"])
                if info.get("zona") != cam.zona:
                    cam.zona = info.get("zona")
                    cam.engine.set_zone(cam.zona)
                signature = _signature(info)
                if signature != cam.signature:
                    cam.signature = signature
                    cam.source.reopen()
            # urutan layar mengikuti urutan daftar kamera
            self._cams = {cid: self._cams[cid] for cid in wanted}
        for cam in removed:
            self._release(cam)

    # ---- satu putaran pekerja -----------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                wait = self._cycle(self._clock())
            except Exception:
                self._report("mode CCTV")
                wait = 0.05
            self._stop.wait(wait)

    def _is_full(self, cam: _Cam, now: float) -> bool:
        # Orang yang terlihat pada penilaian terakhir tetap dianggap ada sampai penilaian berikutnya berkata lain,
        # supaya kamera yang kelaparan tidak turun diam-diam menjadi "siaga".
        return (cam.people_last or now - cam.person_t <= self.cfg.cctv_hold_s
                or cam.engine.snapshot().bar in RED)

    def _cycle(self, now: float) -> float:
        with self._lock:
            cams = list(self._cams.values())
        packets, infos = {}, []
        for cam in cams:
            version, got = cam.source.packet()
            if got is None:
                cam.connected = False
                if now - cam.last_lost >= LOST_STEP_S:
                    cam.last_lost = now
                    self._step(cam, now, None, version)
                infos.append(self._info(cam, now))
                continue
            cam.connected = True
            packets[cam.id] = (version, got)
            if not self._is_full(cam, now) and self.penjadwal.perlu_cek_gerak(now, cam.last_motion):
                cam.last_motion = now
                cam.motion = cam.motion or cam.gerak.berubah(got[1])
            infos.append(self._info(cam, now))
        # Satu kamera per putaran, lalu dijadwalkan ulang: kamera prioritas yang jatuh tempo lagi tidak perlu menunggu
        # semua kamera lain selesai dinilai.
        for cid in self.penjadwal.pilih(now, infos):
            cam = self._cams.get(cid)
            if cam is None or cid not in packets:
                continue
            version, (frame_t, frame) = packets[cid]
            if frame_t == cam.last_frame_t:
                continue                                    # belum ada gambar baru dari kamera ini
            cam.last_frame_t, cam.last_step, cam.motion = frame_t, now, False
            cam.step_times.append(now)
            self._step(cam, frame_t, frame, version)
            cam.people_last = cam.engine.snapshot().people > 0
            if cam.people_last:
                cam.person_t = now
            self._track_alarm(cam, self._clock())
            return 0.0
        for cam in cams:
            self._track_alarm(cam, now)
        waits = [cam.last_step + 1.0 / self.cfg.rate_hz - self._clock() for cam in cams
                 if cam.connected and self._is_full(cam, now)]
        return min(0.05, max(0.005, min(waits))) if waits else 0.05

    def _track_alarm(self, cam: _Cam, now: float) -> bool:
        """Mencatat kapan alarm kamera ini mulai (untuk urutan "terlama dulu" dan jam di pita)."""
        red = cam.engine.snapshot().bar in RED
        if red and cam.alarm_mono is None:
            cam.alarm_mono, cam.alarm_wall = now, self._wall()
        elif not red:
            cam.alarm_mono = cam.alarm_wall = None
        return red

    def _step(self, cam: _Cam, t: float, frame, version) -> None:
        try:
            cam.engine.step(t, frame, input_version=version)
        except Exception:
            self._report("kamera " + cam.nama)

    @staticmethod
    def _rate(cam: _Cam, now: float) -> float:
        """Laju penilaian kamera ini dalam 2 dtk terakhir, diukur runner sendiri. FPS dari snapshot mesin membeku
        bila kamera tidak dinilai lagi, sehingga kamera yang kelaparan akan tampak "penuh"."""
        while cam.step_times and now - cam.step_times[0] > 2.0:
            cam.step_times.popleft()
        return len(cam.step_times) / 2.0

    def _info(self, cam: _Cam, now: float) -> KamInfo:
        full = cam.connected and self._is_full(cam, now)
        if full and cam.penuh_sejak is None:
            cam.penuh_sejak = now
        elif not full:
            cam.penuh_sejak = None
        return KamInfo(cam.id, cam.prioritas, cam.connected, full, cam.last_step, cam.motion,
                       self._rate(cam, now), cam.penuh_sejak)

    def _report(self, where: str) -> None:
        now = time.monotonic()
        if now - self._last_report.get(where, float("-inf")) >= 5.0:
            self._last_report[where] = now
            print(f"Galat di {where}:")
            traceback.print_exc()

    # ---- dibaca layar ---------------------------------------------------------------------
    def snapshot(self) -> dict:
        now = self._clock()
        with self._lock:
            cams = list(self._cams.values())
        rows, alarms, seq, people, fps = [], [], 0, 0, []
        for cam in cams:
            snap = cam.engine.snapshot()
            status = self.penjadwal.status(self._info(cam, now), now)
            rate = self._rate(cam, now) if cam.connected else 0.0
            if self._track_alarm(cam, now):
                alarms.append(cam)
            seq += snap.alarm_seq
            people += snap.people
            if status in ("penuh", "lambat"):
                fps.append(rate)
            rows.append({"id": cam.id, "nama": cam.nama, "bar": snap.bar, "text": snap.text, "fps": round(rate, 1),
                         "people": snap.people, "status": status, "prioritas": cam.prioritas, "zona": cam.zona,
                         "galat": getattr(cam.source, "error", "")})
        alarms.sort(key=lambda c: c.alarm_mono)
        status_of = {r["id"]: r["status"] for r in rows}
        if alarms:
            first = alarms[0]
            bar = first.engine.snapshot().bar
            stamp = time.strftime("%H.%M.%S", time.localtime(first.alarm_wall))
            text = f"{BAR_TEXT[bar]} — {first.nama} — {stamp}"
            if status_of.get(first.id) == "lambat":
                text += SLOW_SUFFIX
        elif not rows or all(r["status"] == "terputus" for r in rows):      # tanpa kamera bukan "aman"
            bar, text = "kamera_terputus", BAR_TEXT["kamera_terputus"]
        else:
            bar, text = "aman", BAR_TEXT["aman"]
        return {"aktif": self.active, "bar": bar, "text": text, "fps": round(statistics.mean(fps), 1) if fps else 0.0,
                "people": people, "alarm_seq": seq, "alarm_kamera": alarms[0].id if alarms else None,
                "alarm_list": [c.id for c in alarms], "kapasitas": self.kapasitas,
                "lambat": any(r["status"] == "lambat" for r in rows), "kamera": rows}

    def has_unacked(self, cam_id: str | None = None) -> bool:
        with self._lock:
            cams = [self._cams[cam_id]] if cam_id in self._cams else ([] if cam_id else list(self._cams.values()))
        return any(c.engine.snapshot().bar in RED for c in cams)

    def acknowledge(self, cam_id: str | None = None) -> bool:
        if cam_id is None:
            cam_id = self.snapshot()["alarm_kamera"]
        cam = self._cams.get(cam_id) if cam_id else None
        return bool(cam is not None and cam.engine.acknowledge())

    def acknowledge_all(self) -> int:
        with self._lock:
            cams = list(self._cams.values())
        return sum(1 for cam in cams if cam.engine.acknowledge())

    def set_zone(self, cam_id: str, points):
        cam = self._cams.get(cam_id)
        if cam is None:
            raise ValueError("Kamera tidak ada di Mode CCTV.")
        zone = cam.engine.set_zone(points)
        cam.zona = [list(p) for p in zone] if zone else None
        return zone

    def latest_jpeg(self, cam_id: str, small: bool = False) -> bytes | None:
        cam = self._cams.get(cam_id)
        if cam is None:
            return None
        if small:
            return cam.engine.latest_jpeg(width=SMALL_WIDTH, quality=SMALL_QUALITY)
        return cam.engine.latest_jpeg()


ALARM_FIRST = "Periksa alarm yang menyala dulu, lalu ganti mode."


class ModeControl:
    """Berganti antara Mode Pameran (satu kamera, mesin yang sekarang) dan Mode CCTV (CctvRunner).

    Mode Pameran tidak dimatikan permanen: utas mesinnya dihentikan sementara dan kameranya dilepas, supaya webcam
    bisa dipakai Mode CCTV; saat Mode CCTV dimatikan, keduanya dinyalakan lagi dengan pengaturan yang sama.
    """

    def __init__(self, engine, runner_factory: Callable[[], "CctvRunner"], remember: Callable[[bool], None]) -> None:
        self.engine, self.runner_factory, self.remember = engine, runner_factory, remember
        self.cctv = None
        self._lock = threading.Lock()

    @property
    def hidup(self) -> bool:
        return self.cctv is not None and bool(self.cctv.active)

    def status(self) -> dict:
        if not self.hidup:
            return {"hidup": False, "pesan": "Mode Pameran: satu kamera."}
        snap = self.cctv.snapshot()
        full = sum(1 for c in snap["kamera"] if c["status"] == "penuh")
        pesan = f"Mode CCTV aktif: {len(snap['kamera'])} kamera, {full} dinilai penuh"
        if snap.get("kapasitas") is not None and snap["kapasitas"] < 1:
            pesan += ". Komputer ini belum sanggup menilai satu kamera penuh 10 kali per detik"
        elif snap.get("kapasitas") is not None:
            pesan += f". Komputer ini sanggup menilai {snap['kapasitas']} kamera penuh sekaligus"
        return {**snap, "hidup": True, "pesan": pesan}

    def _pameran_alarm(self) -> bool:
        alarm = getattr(self.engine, "alarm", None)
        return alarm is not None and getattr(alarm, "active", None) is not None

    def set_cctv(self, on: bool) -> dict:
        with self._lock:
            if on == self.hidup:
                return self.status()
            if on:
                if self._pameran_alarm():
                    raise ValueError(ALARM_FIRST)
                self.engine.pause()
                source = getattr(self.engine, "source", None)
                if source is not None:
                    source.stop()
                runner = None
                try:
                    runner = self.runner_factory()
                    runner.start()
                except Exception as exc:                # galat apa pun: Mode Pameran harus kembali memantau
                    if runner is not None:
                        try:
                            runner.stop()
                        except Exception:
                            pass
                    if source is not None:
                        source.start()
                    self.engine.resume()
                    message = str(exc) if isinstance(exc, CctvError) else f"Mode CCTV gagal dinyalakan: {exc}"
                    raise ValueError(message) from None
                self.cctv = runner
            else:
                if self.cctv.has_unacked():
                    raise ValueError(ALARM_FIRST)
                self.cctv.stop()
                self.cctv = None
                source = getattr(self.engine, "source", None)
                if source is not None:
                    source.start()
                self.engine.resume()
            try:
                self.remember(on)
            except OSError:
                pass                                    # tetap berlaku untuk sesi ini
            return self.status()
