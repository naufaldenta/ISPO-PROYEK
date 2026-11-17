"""Memutar ulang suara lewat jalur pendengar yang asli (pemotong ucapan -> pengenal -> pencocok -> pengaman).

    python -m latih.suara <berkas.wav | folder sesi rekam uji> [--set nama=nilai ...]

Dipakai untuk menyetel ambang peringatan kata kasar tanpa mengulang ucapan: rekam sekali lewat
"Mulai SAHABAT (rekam uji).bat", lalu putar ulang dengan ambang berbeda, misalnya
    --set speech_min_logprob=-0.8 word_repeat_max=4
    --set speech_no_repeat_ngram=0 speech_repetition_penalty=1     (pengenal tanpa pencegah pengulangan)

Berkas wav harus 16 kHz, mono, 16-bit. Folder sesi: tiap suara/NNNN.wav sudah satu ucapan dan tidak dipotong lagi.
Keluaran menampilkan teks ucapan dan kata apa adanya (tidak disensor): ini alat penyetelan, bukan layar.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
import wave
from pathlib import Path
from typing import Collection, Iterable

import numpy as np

from sahabat.config import WORDS_PATH, Config
from sahabat.speech import Recognizer, Verdict, judge
from sahabat.utterance import FRAME_S, Segmenter
from sahabat.words import build_prompt, load_words

from .putar import _overrides

RATE = 16000


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (RATE, 1, 2):
            raise ValueError(f"{Path(path).name}: harus 16 kHz, mono, 16-bit "
                             f"(ini {w.getframerate()} Hz, {w.getnchannels()} kanal, {8 * w.getsampwidth()}-bit)")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def utterances_of(pcm: np.ndarray, cfg: Config) -> list[tuple[float, np.ndarray]]:
    """(detik mulai, ucapan) dari satu rekaman, dengan pemotong yang sama seperti aplikasi."""
    segmenter, out = Segmenter(cfg, RATE), []
    frame = int(round(FRAME_S * RATE))
    fed = 0
    for i in range(0, len(pcm) // frame * frame, frame):       # satu bingkai tiap kali: akhir ucapan = sampel yang sudah masuk
        fed = i + frame
        for utterance in segmenter.feed(pcm[i:fed]):
            out.append(((fed - len(utterance)) / RATE, utterance))
    for utterance in segmenter.flush():
        out.append(((fed - len(utterance)) / RATE, utterance))
    return out


def replay(utterances: Iterable[tuple[float, np.ndarray]], recognizer: Recognizer, words: Collection[str],
           cfg: Config) -> list[tuple[float, Verdict]]:
    """Putusan untuk tiap ucapan, persis seperti yang akan diambil pendengar dengan ambang `cfg`."""
    prompt = build_prompt(list(words), cfg.words_prompt_limit)
    return [(start, judge(recognizer.transcribe(np.asarray(u, dtype=np.float32) / 32768.0, prompt), words, cfg,
                          seconds=len(u) / RATE))
            for start, u in utterances]


def session_utterances(folder: Path) -> list[tuple[float, np.ndarray]]:
    """Potongan ucapan dari satu sesi rekam uji; waktu mulai dihitung dari potongan pertama."""
    clips = Path(folder) / "suara" if (Path(folder) / "suara").is_dir() else Path(folder)
    times: dict[str, float] = {}
    try:
        for line in (clips / "suara.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            times[row["berkas"]] = float(row["t"])
    except (OSError, ValueError, KeyError):
        pass
    first = min(times.values(), default=0.0)
    return [(times.get(p.name, first) - first, read_wav(p)) for p in sorted(clips.glob("[0-9]*.wav"))]


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    path, sets = Path(argv[0]), []
    rest = argv[1:]
    while rest:
        flag = rest.pop(0)
        if flag != "--set":
            raise SystemExit(f"Argumen tidak dikenal: {flag}")
        while rest and not rest[0].startswith("--"):
            sets.append(rest.pop(0))
    cfg = dataclasses.replace(Config(), **_overrides(sets))
    words = load_words(WORDS_PATH)
    if not words:
        print("Daftar kata kosong (kata-kasar.txt).")
        return 1
    try:
        utterances = session_utterances(path) if path.is_dir() else utterances_of(read_wav(path), cfg)
    except (OSError, ValueError, wave.Error) as e:
        print(f"Tidak bisa dibaca: {e}")
        return 1
    if not utterances:
        print("Tidak ada ucapan yang ditemukan.")
        return 1

    from sahabat.speech import MODEL_DIR, WhisperRecognizer
    recognizer = WhisperRecognizer(MODEL_DIR, cfg.speech_threads, cfg.speech_no_repeat_ngram,
                                   cfg.speech_repetition_penalty, encoder_min_s=cfg.speech_encoder_min_s)
    started = time.monotonic()
    results = replay(utterances, recognizer, words, cfg)
    took = time.monotonic() - started
    notices = 0
    for (start, utterance), (_, verdict) in zip(utterances, results):
        found = ", ".join(f"{m.word} ({m.prob:.2f})" for m in verdict.matches)
        notices += bool(verdict.matches)
        outcome = f"DIBUANG ({verdict.reason})" if verdict.reason else (f"PERINGATAN: {found}" if found else "-")
        print(f"{start:7.1f} dtk  {len(utterance) / RATE:4.1f} dtk  {outcome}\n           {verdict.text or '(kosong)'}")
    print(f"{len(utterances)} ucapan, {notices} peringatan, rata-rata {took / len(utterances):.1f} dtk proses per ucapan")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
