"""Pengenal ucapan (Whisper base, lokal) dan pengaman terhadap salah dengar.

    python -m sahabat.speech --unduh    mengunduh model ke models/whisper-base (butuh internet sekali)

Aplikasi tidak pernah mengunduh saat berjalan: model hanya dimuat dari folder itu.

Pengenal diberi "petunjuk kata" (daftar kata kasar) supaya menulis "goblok" dan bukan "go blog". Harganya:
saat suara tidak jelas ia kadang mengulang-ulang kata dari petunjuk. `judge` membuang ucapan semacam itu.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Protocol, Sequence

import numpy as np

from .config import MODELS_DIR, Config
from .words import find_listed, tokens

MODEL_NAME = "base"
MODEL_DIR = MODELS_DIR / "whisper-base"
MODEL_FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
# Pencegah pengulangan di penyandi. Tanpa ini, pada ucapan yang dipotong per 6 detik pengenal sering terjebak
# ("asu, goblok, asu, goblok, ..." atau "ayam, ayam, ayam, ...") sampai seluruh ucapan dibuang pengaman dan
# satu ucapan memakan 4-8 detik. Terukur 15 November 2026 pada rekaman percobaan: dengan dua setelan ini rekaman
# dekat naik dari 4 ke 7 kata, waktu proses turun dari 3,8 ke 1,9 detik per ucapan, kontrol tanpa kata kasar tetap bersih.
NO_REPEAT_NGRAM = 3
REPETITION_PENALTY = 1.1
_MEL_FRAMES_PER_S = 100
_ENCODER_SILENCE_FRAMES = 100


@dataclass(frozen=True)
class HeardWord:
    text: str
    prob: float


@dataclass(frozen=True)
class Heard:
    """Satu segmen hasil pengenal."""
    text: str
    words: tuple[HeardWord, ...]
    no_speech: float        # peluang segmen ini bukan ucapan
    compression: float      # teks yang sangat mampat = pengulangan
    logprob: float          # keyakinan rata-rata segmen


@dataclass(frozen=True)
class Match:
    word: str
    prob: float


@dataclass(frozen=True)
class Verdict:
    matches: tuple[Match, ...]
    reason: str             # "" bila diterima; berulang | terlalu_cepat | bukan_ucapan | mampat | tidak_yakin | galat
    text: str


class Recognizer(Protocol):
    def transcribe(self, audio: np.ndarray, prompt: str) -> list[Heard]: ...


def _longest_run(items: list[str]) -> int:
    best = run = 0
    last = None
    for item in items:
        run = run + 1 if item == last else 1
        last = item
        best = max(best, run)
    return best


def _skip_reason(h: Heard, cfg: Config) -> str:
    if h.no_speech > cfg.speech_max_no_speech:
        return "bukan_ucapan"
    if h.compression > cfg.speech_max_compression:
        return "mampat"
    if h.logprob < cfg.speech_min_logprob:
        return "tidak_yakin"
    return ""


def judge(heard: Sequence[Heard], words: Collection[str], cfg: Config, seconds: float | None = None) -> Verdict:
    """Kata kasar yang boleh diumumkan dari satu ucapan, setelah pengaman. Lihat spec 6.5.

    `seconds`: lama ucapannya, bila diketahui; dipakai untuk membuang teks yang mustahil diucapkan secepat itu.
    """
    text = " ".join(h.text for h in heard)
    spoken = tokens(text)
    if _longest_run(spoken) >= cfg.word_repeat_max:
        return Verdict((), "berulang", text)
    if seconds is not None and len(spoken) > cfg.speech_max_words_per_s * seconds + cfg.speech_words_slack:
        return Verdict((), "terlalu_cepat", text)

    reasons = [_skip_reason(h, cfg) for h in heard]
    kept = [h for h, why in zip(heard, reasons) if not why]
    if heard and not kept:
        return Verdict((), reasons[0], text)

    wanted = tuple(words)
    found: dict[str, float] = {}
    for h in kept:
        spoken = [(tok, w.prob) for w in h.words for tok in tokens(w.text)]
        if not h.words:                                   # pengenal tidak memberi waktu per kata
            spoken = [(tok, math.exp(h.logprob)) for tok in tokens(h.text)]
        for tok, prob in spoken:
            listed = find_listed(tok, wanted)            # "kontoli" dan "anjiiir" dihitung sebagai kata terdaftarnya
            if listed is not None and prob >= cfg.word_min_prob and listed not in found:
                found[listed] = prob
    return Verdict(tuple(Match(w, p) for w, p in found.items()), "", text)


def _trim_encoder_padding(features: np.ndarray, min_frames: int) -> np.ndarray:
    """Memendekkan bantalan nol dari faster-whisper, dengan ruang untuk ujung ucapan."""
    if features.shape[-1] <= min_frames:
        return features
    occupied = np.flatnonzero(features.any(axis=tuple(range(features.ndim - 1))))
    actual = int(occupied[-1]) + 1 if occupied.size else 0
    size = max(min_frames, actual + _ENCODER_SILENCE_FRAMES)
    size += size % 2                       # encoder membagi jumlah bingkai mel dengan dua
    return features[..., :size] if size < features.shape[-1] else features


class WhisperRecognizer:
    """faster-whisper di CPU. Setelannya diukur 15 November 2026."""

    def __init__(self, model_dir: Path, threads: int, no_repeat_ngram: int = NO_REPEAT_NGRAM,
                 repetition_penalty: float = REPETITION_PENALTY, *,
                 encoder_min_s: float = Config.speech_encoder_min_s) -> None:
        # Semua berkas harus ada: bila tokenizer.json tidak ikut tersalin, pustakanya mencoba mengunduh pengganti.
        missing = [name for name in MODEL_FILES if not (Path(model_dir) / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Model ucapan di {model_dir} tidak lengkap (tidak ada: {', '.join(missing)}). "
                                    "Jalankan: python -m sahabat.speech --unduh")
        os.environ["HF_HUB_OFFLINE"] = "1"              # aplikasi tidak pernah mengunduh saat berjalan
        self._no_repeat_ngram = int(no_repeat_ngram)
        self._repetition_penalty = float(repetition_penalty)
        from faster_whisper import WhisperModel

        min_frames = round(encoder_min_s * _MEL_FRAMES_PER_S)

        class ShortAudioWhisper(WhisperModel):
            _short_input_supported = True

            def encode(self, features):
                if self._short_input_supported:
                    shortened = _trim_encoder_padding(features, min_frames)
                    if shortened is not features:
                        try:
                            return super().encode(shortened)
                        except (ValueError, RuntimeError):
                            # Versi encoder yang menolak panjang variabel kembali ke jalur bawaan.
                            self._short_input_supported = False
                return super().encode(features)

        model_cls = ShortAudioWhisper if 0 < encoder_min_s < 30 else WhisperModel
        self._model = model_cls(str(model_dir), device="cpu", compute_type="int8", cpu_threads=threads)

    def transcribe(self, audio: np.ndarray, prompt: str) -> list[Heard]:
        segments, _ = self._model.transcribe(
            audio, language="id", beam_size=1, temperature=0.0, condition_on_previous_text=False,
            initial_prompt=prompt or None, vad_filter=True, word_timestamps=True,
            no_repeat_ngram_size=self._no_repeat_ngram, repetition_penalty=self._repetition_penalty)
        return [Heard(s.text.strip(), tuple(HeardWord(w.word, float(w.probability)) for w in (s.words or ())),
                      float(s.no_speech_prob), float(s.compression_ratio), float(s.avg_logprob)) for s in segments]


def download(model_dir: Path = MODEL_DIR) -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)
    from faster_whisper import download_model
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    download_model(MODEL_NAME, output_dir=str(model_dir))


if __name__ == "__main__":
    if "--unduh" in sys.argv:
        download()
        size = sum(p.stat().st_size for p in MODEL_DIR.rglob("*") if p.is_file())
        print(f"model ucapan siap di {MODEL_DIR} ({size / 1e6:.0f} MB)")
    else:
        print(__doc__)
