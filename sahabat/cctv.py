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
OVERDUE_WAIT_S = 0.02      # kamera jatuh tempo tanpa gambar baru: periksa lagi 50x per detik, bukan 200x
FALLBACK_RETRY_S = 2.0      # CPU cadangan gagal dimuat: coba lagi sesudah selang ini (bukan berputar terus)
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
    pose: object = None                                 # CameraPose kamera ini (menerima pose hitungan runner)
    seen_version: int | None = None                     # versi sumber terakhir yang sudah dinilai
    posisi: str = "dada"                                # profil_kamera: dada / tinggi
    peringatan: str = ""                                # profil tinggi tanpa model tinggi yang lolos uji
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
                 sleep: Callable[[float], None] = time.sleep, start_thread: bool = True,
                 cpu_fallback: Callable[[], object] | None = None, perangkat: str = "CPU",
                 pesan_perangkat: str = "") -> None:
        self.cfg, self.store, self.shared_pose, self.riwayat = cfg, store, shared_pose, riwayat
        self._cpu_fallback = cpu_fallback
        self.perangkat, self.pesan_perangkat, self.galat_gpu = perangkat, pesan_perangkat, ""
        self._engine_factory = engine_factory or self._default_engine
        self._source_factory = source_factory or self._default_source
        self._clock, self._wall, self._sleep = clock, wall, sleep
        self.connect_timeout_s = connect_timeout_s
        self._start_thread = start_thread
        self.penjadwal = Penjadwal(cfg.rate_hz, cfg.cctv_safety_s, cfg.fps_floor, cfg.cctv_motion_hz)
        self._cams: dict[str, _Cam] = {}
        self._lock = threading.RLock()
        self._step_lock = threading.Lock()          # dipegang pekerja selama satu kamera dinilai (pose + mesin)
        self._reload_lock = threading.Lock()        # reload() berurutan; layar tetap memakai _lock yang singkat
        self._stopped = False
        self._alarm_lock = threading.Lock()         # jam mulai alarm ditulis pekerja dan dibaca layar
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
        from .profil_kamera import model_untuk, posisi_bersih
        models = model_untuk(cam.get("posisi"))
        engine = Engine(self.cfg, source, pose, PredLog(DATA_DIR / "log", prefix="cctv-" + cam["id"]), False,
                        models.violence, models.fall, None, self.riwayat, kamera=(cam["id"], cam["nama"]))
        engine.atur_profil(models.confirm_window_s, posisi_bersih(cam.get("posisi")) == "dekat")
        engine.set_zone(cam.get("zona"))           # selalu: jangan mewarisi zona Mode Pameran dari cfg.zone
        return engine

    def _build(self, cam: dict) -> _Cam:
        from .pose import CameraPose
        source = self._source_factory(cam)
        source.start()
        pose = CameraPose(self.shared_pose, cam["id"])
        engine = self._engine_factory(cam, source, pose)
        from .profil_kamera import model_untuk, posisi_bersih
        posisi = posisi_bersih(cam.get("posisi"))
        return _Cam(cam["id"], cam["nama"], bool(cam.get("prioritas")), cam.get("zona"), _signature(cam), source,
                    engine, pose, posisi=posisi, peringatan=model_untuk(posisi).peringatan)

    def _release(self, cam: _Cam, forget: bool = True) -> None:
        """Melepas kamera yang dibuang. Menunggu penilaian yang sedang berjalan selesai (paling lama satu kamera),
        supaya sumber tidak dihentikan di tengah penilaian dan pelacaknya tidak dibuat ulang sesudah dilupakan.
        forget=False: kamera duplikat yang tidak pernah dipakai; pelacak dan Riwayat ber-ID sama milik kamera hidup."""
        with self._step_lock:
            try:
                cam.source.stop()
            finally:
                if forget and self._cams.get(cam.id) is None:
                    self._forget(cam)
                else:
                    log = getattr(cam.engine, "log", None)
                    if log is not None:
                        log.close()

    def _forget(self, cam: _Cam) -> None:
        log = getattr(cam.engine, "log", None)
        if log is not None:
            log.close()                         # Riwayat bersama tidak ditutup di sini
        if self.shared_pose is not None:                # None: perangkat hitung belum dipilih
            self.shared_pose.forget(cam.id)
        if self.riwayat is not None and hasattr(self.riwayat, "lupakan_kamera"):
            self.riwayat.lupakan_kamera(cam.id)

    # ---- menyalakan / mematikan ---------------------------------------------------------
    def start(self) -> dict:
        self._stopped = False
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
        try:
            self.kapasitas = self._measure(frame)
        except Exception as exc:
            if self._cpu_fallback is None:
                self._release_all()
                raise
            self._fall_back(exc, frame)
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
        from .perangkat_hitung import kapasitas
        return kapasitas(statistics.median(times), self.cfg.rate_hz)

    def _release_all(self) -> None:
        """Semua kamera dilepas bersamaan: menghentikan stream yang macet bisa makan beberapa detik per kamera."""
        with self._lock:
            cams, self._cams = list(self._cams.values()), {}

        def stop(cam):
            try:
                cam.source.stop()
            except Exception:
                self._report("melepas kamera " + cam.nama)
        threads = [threading.Thread(target=stop, args=(cam,), name="lepas-kamera", daemon=True) for cam in cams]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        for cam in cams:
            self._forget(cam)

    def stop(self) -> None:
        self.active = False
        self._stopped = True                    # reload() yang sedang membangun tidak menghidupkan kamera lagi
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        self._release_all()

    def reload(self) -> None:
        """Daftar kamera berubah di panel Kamera: tambah, buang, perbarui, atau buka ulang sumber.

        Kamera baru dan model posisi dimuat di luar kunci runner (memuat model makan waktu, dan kunci itu dipakai
        layar tiap 0,2 dtk); kunci hanya dipegang saat menukar isi daftar."""
        with self._reload_lock:
            self._reload()

    def _reload(self) -> None:
        from .profil_kamera import model_untuk, posisi_bersih
        wanted = {c["id"]: c for c in self.store.active()}
        with self._lock:
            current = dict(self._cams)
        built = {cid: self._build(info) for cid, info in wanted.items() if cid not in current}
        for cid, info in wanted.items():
            cam = current.get(cid)
            posisi = posisi_bersih(info.get("posisi"))
            if cam is not None and posisi != cam.posisi:
                models = model_untuk(posisi)
                swap = getattr(cam.engine, "ganti_model", None)
                if swap is not None:
                    swap(models.violence, models.fall)
                profile = getattr(cam.engine, "atur_profil", None)
                if profile is not None:
                    profile(models.confirm_window_s, posisi == "dekat")
                cam.posisi, cam.peringatan = posisi, models.peringatan
        extra = []
        with self._lock:
            if self._stopped:
                # Mode CCTV dimatikan selagi kamera baru dibangun: jangan hidupkan lagi daftar yang sudah dilepas.
                extra, built, wanted = list(built.values()), {}, {}
            removed = [cam for cid, cam in self._cams.items() if cid not in wanted]
            for cid, cam in built.items():
                if cid in self._cams:
                    extra.append(cam)           # sudah ada (tidak terjadi lagi berkat _reload_lock, tetapi aman)
                else:
                    self._cams[cid] = cam
            for cid, info in wanted.items():
                cam = self._cams.get(cid)
                if cam is None or (cid in built and built[cid] is cam):
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
            self._cams = {cid: self._cams[cid] for cid in wanted if cid in self._cams}
        for cam in removed:
            self._release(cam)
        for cam in extra:
            self._release(cam, forget=False)

    @staticmethod
    def _short(exc: Exception) -> str:
        return (str(exc).strip().splitlines() or [type(exc).__name__])[0][:120]

    def _fall_back(self, exc: Exception, frame) -> bool:
        """Pose GPU melempar galat (driver, GPU dicabut, memori habis): Mode CCTV pindah ke CPU dan tetap memantau.
        Tidak ada percobaan GPU lagi sampai Mode CCTV dinyalakan ulang. False bila CPU cadangan tidak ada atau belum
        bisa dimuat (dicoba lagi pada galat berikutnya)."""
        reason = self._short(exc)
        make = self._cpu_fallback
        if make is None:
            self.galat_gpu = f"GPU bermasalah ({reason}); tidak ada cadangan CPU."
            print(self.galat_gpu)
            return False
        try:
            new = make()
        except Exception as load_exc:                   # misalnya memori habis, sama dengan penyebab galat GPU
            self.galat_gpu = (f"GPU bermasalah ({reason}); CPU belum bisa dimuat ({self._short(load_exc)}), "
                              "dicoba lagi.")
            print(self.galat_gpu)
            return False
        self._cpu_fallback = None
        self.galat_gpu = f"GPU bermasalah ({reason}), Mode CCTV pindah ke CPU."
        print(self.galat_gpu)
        old = self.shared_pose
        with self._lock:
            cams = list(self._cams.values())
            self.shared_pose, self.pesan_perangkat = new, ""
            self.perangkat = getattr(new, "nama", None) or "CPU (OpenVINO)"
            for cam in cams:
                if cam.pose is not None:
                    cam.pose.shared = new
        for cam in cams:
            try:
                old.forget(cam.id)
            except Exception:
                pass
            forget_people = getattr(cam.engine, "lupakan_orang", None)
            if forget_people is not None:
                forget_people()                     # nomor orang dari pelacak baru mulai lagi dari 1
        try:
            self.kapasitas = self._measure(frame)
        except Exception:
            self._report("ukur ulang CPU")
        return True

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
        # semua kamera lain selesai dinilai. Berlaku juga untuk GPU (batch tidak mempercepat DirectML, lihat pose_onnx).
        for cid in self.penjadwal.pilih(now, infos):
            cam = self._cams.get(cid)
            if cam is None or cid not in packets:
                continue
            version, (frame_t, frame) = packets[cid]
            if frame_t == cam.last_frame_t:
                continue                                    # belum ada gambar baru dari kamera ini
            with self._step_lock:
                if self._cams.get(cam.id) is not cam:
                    continue                                # dibuang (atau diganti) reload() tepat sebelum dinilai
                # Pose GPU dihitung di sini, bukan di dalam mesin, supaya galatnya memindahkan Mode CCTV ke CPU (mesin
                # menelan galat pose). Frame pertama sesudah sumber dibuka ulang dibiarkan ke mesin: mesin me-reset
                # pelacak dulu, baru menilai.
                if hasattr(self.shared_pose, "detect_many") and cam.pose is not None and version == cam.seen_version:
                    try:
                        people = self.shared_pose.detect(cam.id, frame)
                    except Exception as exc:
                        failed = exc
                    else:
                        failed = None
                        cam.pose.prefetch(frame, people)
                    if failed is not None:
                        return 0.0 if self._fall_back(failed, frame) else FALLBACK_RETRY_S
                cam.seen_version = version
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
        if not waits:
            return 0.05
        soonest = min(waits)
        # Sudah jatuh tempo tetapi belum ada gambar baru (kamera macet atau lebih lambat dari 10 fps): tunggu sebentar,
        # jangan berputar ratusan kali per detik.
        return OVERDUE_WAIT_S if soonest <= 0 else min(0.05, max(0.005, soonest))

    def _track_alarm(self, cam: _Cam, now: float, bar: str | None = None) -> tuple[float, float] | None:
        """Mencatat kapan alarm kamera ini mulai (untuk urutan "terlama dulu" dan jam di pita). Mengembalikan
        (mulai monotonic, mulai jam dinding) yang dibaca sekaligus, atau None bila tidak ada alarm. `bar`: pita dari
        snapshot yang sudah dibaca pemanggil, supaya pita dan status alarm berasal dari saat yang sama."""
        red = (cam.engine.snapshot().bar if bar is None else bar) in RED
        with self._alarm_lock:
            if red and cam.alarm_mono is None:
                cam.alarm_mono, cam.alarm_wall = now, self._wall()
            elif not red:
                cam.alarm_mono = cam.alarm_wall = None
            return (cam.alarm_mono, cam.alarm_wall) if red and cam.alarm_mono is not None else None

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
            started = self._track_alarm(cam, now, snap.bar)
            if started is not None:
                alarms.append((started, cam, snap))
            seq += snap.alarm_seq
            people += snap.people
            if status in ("penuh", "lambat"):
                fps.append(rate)
            rows.append({"id": cam.id, "nama": cam.nama, "bar": snap.bar, "text": snap.text, "fps": round(rate, 1),
                         "people": snap.people, "status": status, "prioritas": cam.prioritas, "zona": cam.zona,
                         "posisi": cam.posisi, "peringatan": cam.peringatan,
                         "galat": getattr(cam.source, "error", "")})
        alarms.sort(key=lambda a: a[0][0])         # jam yang dibaca saat itu, bukan atribut yang bisa berubah
        status_of = {r["id"]: r["status"] for r in rows}
        if alarms:
            (_, first_wall), first, first_snap = alarms[0]
            bar = first_snap.bar
            stamp = time.strftime("%H.%M.%S", time.localtime(first_wall))
            text = f"{BAR_TEXT[bar]} — {first.nama} — {stamp}"
            if status_of.get(first.id) == "lambat":
                text += SLOW_SUFFIX
        elif not rows or all(r["status"] == "terputus" for r in rows):      # tanpa kamera bukan "aman"
            bar, text = "kamera_terputus", BAR_TEXT["kamera_terputus"]
        else:
            bar, text = "aman", BAR_TEXT["aman"]
        return {"aktif": self.active, "bar": bar, "text": text, "fps": round(statistics.mean(fps), 1) if fps else 0.0,
                "people": people, "alarm_seq": seq, "alarm_kamera": alarms[0][1].id if alarms else None,
                "alarm_list": [c.id for _, c, _ in alarms], "kapasitas": self.kapasitas, "perangkat": self.perangkat,
                "galat_gpu": self.galat_gpu, "pesan_perangkat": self.pesan_perangkat,
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

    def asisten(self, cam_id: str) -> dict | None:
        """Petunjuk pemasangan kamera ini (asisten_kamera), atau None bila kamera tidak ada di Mode CCTV."""
        cam = self._cams.get(cam_id)
        helper = getattr(cam.engine, "asisten", None) if cam is not None else None
        return helper.ringkas() if helper is not None else None

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
        perangkat = snap.get("perangkat") or "CPU"
        if perangkat.startswith("GPU (") and perangkat.endswith(")"):
            perangkat = "GPU: " + perangkat[5:-1] + ","        # "GPU (DirectML #1)" -> "GPU: DirectML #1,"
        else:
            perangkat += ":"
        if snap.get("kapasitas") is not None and snap["kapasitas"] < 1:
            pesan += ". Komputer ini belum sanggup menilai satu kamera penuh 10 kali per detik"
        elif snap.get("kapasitas") is not None:
            pesan += f". {perangkat} sanggup menilai {snap['kapasitas']} kamera penuh sekaligus"
        extras = [e.rstrip(".") for e in (snap.get("pesan_perangkat"), snap.get("galat_gpu")) if e]
        if extras:
            pesan += ". " + ". ".join(extras) + "."
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
