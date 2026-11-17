"""Pendengar kata kasar: mikrofon -> pemotong ucapan -> pengenal -> pencocok -> peringatan untuk layar.

Hidup di samping mesin kamera dan tidak menyentuhnya. Apa pun yang gagal di sini (mikrofon, model, pustaka,
galat pengenal) hanya mengubah tulisan tombol di layar; bagian kamera tetap jalan.

Dua utas: "mikrofon" (satu-satunya yang membuka dan menutup mikrofon) dan "pekerja" (satu-satunya yang memanggil
pengenal ucapan). Paling banyak satu ucapan menunggu; yang lebih lama dibuang supaya jeda tidak menumpuk.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import Config
from .mic import Mic
from .predlog import PredLog
from .speech import Recognizer, Verdict, judge
from .utterance import Segmenter
from .words import build_prompt, censor, load_words

STATUS_LABEL: dict[str, str] = {
    "hidup": "Kata kasar: HIDUP",
    "mati": "Kata kasar: MATI",
    "memuat": "Kata kasar: memuat…",
    "tanpa_mikrofon": "Kata kasar: mikrofon tidak ada",
    "tidak_tersedia": "Kata kasar: tidak tersedia",
    "daftar_kosong": "Kata kasar: daftar kata kosong",
}
NOTICE_TEXT = "KATA KASAR TERDENGAR: {}"
RATE = 16000
IDLE_S = 0.05
REPORT_EVERY_S = 5.0


@dataclass(frozen=True)
class ListenState:
    status: str            # kunci STATUS_LABEL
    label: str             # tulisan tombol
    notice: str            # tulisan pita kuning; kosong bila tidak ada
    notice_seq: int        # naik 1 tiap peringatan baru; halaman memakainya untuk bunyi pendek


class Listener:
    def __init__(self, cfg: Config, words_path: Path, make_recognizer: Callable[[], Recognizer],
                 open_mic: Callable[[int | str | None], Mic], log: PredLog | None = None,
                 clip_dir: Path | None = None, on_toggle: Callable[[bool], None] | None = None,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 retry_s: float = 2.0) -> None:
        self.cfg = cfg
        self.words = load_words(words_path)
        self.prompt = build_prompt(self.words, cfg.words_prompt_limit)
        self.log = log
        self.clip_dir = Path(clip_dir) if clip_dir is not None else None   # hanya diisi di mode rekam uji
        self.retry_s = retry_s
        self.mic_name = ""
        self._make_recognizer = make_recognizer
        self._open_mic = open_mic
        self._on_toggle = on_toggle
        self._clock = clock
        self._wall = wall
        self._enabled = bool(cfg.words_enabled)
        self.microphone = cfg.microphone
        self._input_version = 0
        self._recognizer: Recognizer | None = None
        self._recognizer_failed = False
        self._recognizer_lock = threading.Lock()
        self._mic: Mic | None = None
        self._lock = threading.Lock()
        self._notice = ""
        self._notice_t = float("-inf")
        self._notice_seq = 0
        self._waiting: deque[tuple[int, np.ndarray]] = deque(maxlen=1)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._clips = 0
        self._last_report = float("-inf")

    # ---- yang dibaca layar ------------------------------------------------------------

    def snapshot(self) -> ListenState:
        with self._lock:
            status = self._status()
            shown = self._notice if self._clock() - self._notice_t < self.cfg.words_notice_s else ""
            return ListenState(status, STATUS_LABEL[status], shown, self._notice_seq)

    def _status(self) -> str:
        if not self._enabled:
            return "mati"
        if not self.words:
            return "daftar_kosong"
        if self._recognizer_failed:
            return "tidak_tersedia"
        if self._recognizer is None:
            return "memuat"
        if self._mic is None:
            return "tanpa_mikrofon"
        return "hidup"

    def set_enabled(self, on: bool) -> bool:
        """Tombol hidup/mati. Mati berarti utas mikrofon menutup alirannya, bukan sekadar menyembunyikan peringatan."""
        on = bool(on)
        with self._lock:
            changed = on != self._enabled
            self._enabled = on
            if not on:
                self._waiting.clear()
                self._notice, self._notice_t = "", float("-inf")     # MATI: tidak ada peringatan yang tersisa di layar
        if changed and self._on_toggle is not None:
            try:
                self._on_toggle(on)
            except Exception:                              # gagal menyimpan pilihan tidak boleh menggagalkan tombol
                pass
        self._wake.set()
        return on

    # ---- satu ucapan --------------------------------------------------------------------

    def set_microphone(self, device: int | str | None) -> None:
        """Minta utas mikrofon mengganti perangkat tanpa menyalakan pendengar yang sedang mati."""
        with self._lock:
            if self.microphone == device:
                return
            self.microphone = device
            self._input_version += 1
            self._mic, self.mic_name = None, ""
            self._waiting.clear()
            self._notice, self._notice_t = "", float("-inf")
        self._wake.set()

    def hear(self, utterance: np.ndarray, input_version: int | None = None) -> Verdict:
        """Mengenali satu ucapan (int16), menilai, mencatat, dan memperbarui peringatan."""
        recognizer = self._ensure_recognizer()
        if recognizer is None:
            return Verdict((), "tidak_tersedia", "")
        pcm = np.asarray(utterance, dtype=np.int16)
        try:
            verdict = judge(recognizer.transcribe(pcm.astype(np.float32) / 32768.0, self.prompt), self.words, self.cfg,
                            seconds=len(pcm) / RATE)
        except Exception:                                  # satu ucapan gagal: lewati, tetap mendengarkan
            self._report("pengenal ucapan")
            verdict = Verdict((), "galat", "")
        announced = False
        with self._lock:
            if input_version is not None and input_version != self._input_version:
                return verdict
            if verdict.matches and self._enabled:
                shown = ", ".join(censor(m.word) for m in verdict.matches[:self.cfg.words_max_shown])
                self._notice = NOTICE_TEXT.format(shown)
                self._notice_t = self._clock()
                self._notice_seq += 1
                announced = True
        if announced:
            if self.log is not None:                       # hanya kata tersensor; teks ucapan tidak pernah dicatat di sini
                self.log.write({"t": round(self._wall(), 1), "kata": [censor(m.word) for m in verdict.matches],
                                    "prob": [round(float(m.prob), 2) for m in verdict.matches]})
        self._save_clip(pcm, verdict)
        return verdict

    def _ensure_recognizer(self) -> Recognizer | None:
        with self._recognizer_lock:
            if self._recognizer is None and not self._recognizer_failed:
                try:
                    recognizer = self._make_recognizer()
                except Exception as e:                     # model tidak ada, pustaka tidak ada, atau gagal dimuat
                    print(f"Peringatan: pengenal ucapan tidak bisa dimuat ({e}). Peringatan kata kasar tidak tersedia.")
                    with self._lock:
                        self._recognizer_failed = True
                    return None
                with self._lock:
                    self._recognizer = recognizer
            return self._recognizer

    def _report(self, where: str) -> None:
        now = time.monotonic()
        if now - self._last_report >= REPORT_EVERY_S:
            self._last_report = now
            print(f"Galat di {where}:")
            traceback.print_exc()

    def _save_clip(self, pcm: np.ndarray, verdict: Verdict) -> None:
        """Mode rekam uji: tiap ucapan disimpan bersama teksnya supaya ambang bisa disetel tanpa mengulang."""
        if self.clip_dir is None:
            return
        try:
            self.clip_dir.mkdir(parents=True, exist_ok=True)
            self._clips += 1
            name = f"{self._clips:04d}.wav"
            with wave.open(str(self.clip_dir / name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(pcm.tobytes())
            row = {"berkas": name, "t": round(self._wall(), 1), "teks": verdict.text,
                   "kata": [m.word for m in verdict.matches], "alasan": verdict.reason}
            with open(self.clip_dir / "suara.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except (OSError, wave.Error):
            print("Peringatan: potongan suara uji tidak bisa disimpan; lanjut tanpa menyimpan.")
            self.clip_dir = None

    # ---- utas ---------------------------------------------------------------------------

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        self._threads = [threading.Thread(target=self._mic_loop, name="mikrofon", daemon=True),
                         threading.Thread(target=self._work_loop, name="pengenal-ucapan", daemon=True)]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        for thread in self._threads:
            thread.join(timeout=3.0)
        self._threads = []
        if self.log is not None:
            self.log.close()

    def _wants_mic(self) -> bool:
        with self._lock:
            return self._enabled and bool(self.words) and self._recognizer is not None

    def _mic_loop(self) -> None:
        mic: Mic | None = None
        version = -1
        segmenter = Segmenter(self.cfg, RATE)

        def release() -> None:
            nonlocal mic
            if mic is not None:
                mic.close()
                mic = None
                with self._lock:
                    self._mic = None
                    self.mic_name = ""

        try:
            while not self._stop.is_set():
                if version != self._input_version:
                    release()
                if not self._wants_mic():
                    release()
                    self._stop.wait(IDLE_S)
                    continue
                if mic is None:
                    with self._lock:
                        version, device = self._input_version, self.microphone
                    try:
                        mic = self._open_mic(device)
                    except Exception:                      # tidak ada perangkat, atau pustaka mikrofon tidak ada
                        self._stop.wait(self.retry_s)
                        continue
                    segmenter = Segmenter(self.cfg, RATE)
                    with self._lock:
                        self._mic = mic
                        self.mic_name = getattr(mic, "name", "")
                try:
                    block = mic.read()
                except Exception:                          # perangkat dicabut: lepas, lalu coba lagi nanti
                    release()
                    self._stop.wait(self.retry_s)
                    continue
                if block is None:
                    continue
                try:
                    if version != self._input_version:
                        continue
                    for utterance in segmenter.feed(block):
                        self._waiting.append((version, utterance))
                        self._wake.set()
                except Exception:                          # blok aneh dari perangkat: mulai pemotong dari awal
                    self._report("pendengar kata kasar")
                    segmenter = Segmenter(self.cfg, RATE)
        finally:
            release()

    def _work_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                should_load = self._enabled and bool(self.words)
            if should_load:
                self._ensure_recognizer()
            if not self._wake.wait(IDLE_S):
                continue
            self._wake.clear()
            while not self._stop.is_set():
                try:
                    version, utterance = self._waiting.pop()
                except IndexError:
                    break
                with self._lock:
                    on = self._enabled and version == self._input_version
                if on:
                    try:
                        self.hear(utterance, input_version=version)
                    except Exception:                      # apa pun yang tak terduga: utas ini tidak boleh mati,
                        self._report("pendengar kata kasar")   # kalau mati layar tetap bilang HIDUP padahal tuli
