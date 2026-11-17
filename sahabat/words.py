"""Daftar kata kasar: membaca berkasnya, memecah teks jadi kata, menyensor, dan membuat petunjuk untuk pengenal.

Pencocokan sengaja ketat: "kontrol" dan "kontol" hanya beda satu huruf, jadi tidak ada pencocokan "mirip".
Yang diterima selain kata persis hanya dua hal yang memang terjadi saat orang mengumpat (uji 15 November 2026):
huruf yang dipanjangkan ("anjiiir", "asuuu") dan akhiran pendek ("kontoli", "gobloknya", "asune").
"""
from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Collection, Sequence

_WORD = re.compile(r"[a-z]+")
_BOM = bytes((0xEF, 0xBB, 0xBF))        # tanda awal berkas yang ditambahkan Notepad
PROMPT_LEAD = "Percakapan anak sekolah dengan kata kasar: "
_RUN = re.compile(r"(.)\1+")            # satu huruf yang diulang berturut-turut
# Akhiran yang boleh menempel di kata terdaftar, yang terpanjang dulu: -nya, -lah, -ne (Jawa), -mu, -ku, -an,
# dan -i / -e yang sering hanya intonasi ("jancokiii"). Sengaja tidak ada -s dan -h: "asus" dan "asuh" kata biasa.
SUFFIXES = ("nya", "lah", "ne", "mu", "ku", "an", "i", "e")
MIN_STEM = 3


def load_words(path: Path) -> tuple[str, ...]:
    """Kata-kata dari berkas, menurut urutan baris. Berkas hilang, rusak, atau kosong menghasilkan ().

    Yang diterima hanya baris yang seluruhnya huruf a-z setelah dipangkas dan dijadikan huruf kecil;
    baris berawalan # adalah catatan. Kata dobel diambil sekali.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return ()
    words: list[str] = []
    for line in raw.removeprefix(_BOM).splitlines():
        try:
            word = line.decode("utf-8").strip().lower()
        except UnicodeDecodeError:              # baris berisi byte rusak: lewati barisnya, jangan menebak katanya
            continue
        if word and not word.startswith("#") and _WORD.fullmatch(word) and word not in words:
            words.append(word)
    return tuple(words)


def tokens(text: str) -> list[str]:
    """Kata-kata dalam teks: deretan huruf a-z, huruf kecil."""
    return _WORD.findall(text.lower())


def _squeeze(word: str) -> str:
    """Huruf berulang dirapatkan: anjiiir -> anjir, assshole -> ashole."""
    return _RUN.sub(r"\1", word)


@functools.lru_cache(maxsize=8)
def _index(words: tuple[str, ...]) -> dict[str, str]:
    """Bentuk rapat -> kata terdaftar; bila dua kata rapatnya sama, yang lebih atas di daftar yang dipakai."""
    return {_squeeze(w): w for w in reversed(words)}


def find_listed(token: str, words: Collection[str]) -> str | None:
    """Kata terdaftar yang dimaksud `token` (huruf kecil), atau None.

    Urutannya: sama persis; sama setelah huruf berulang dirapatkan; lalu kata terdaftar + satu akhiran pendek.
    """
    if token in words:
        return token
    index = _index(tuple(words))
    squeezed = _squeeze(token)
    if squeezed in index:
        return index[squeezed]
    for suffix in SUFFIXES:
        stem = squeezed[:-len(suffix)]
        if squeezed.endswith(suffix) and len(stem) >= MIN_STEM and stem in index:
            return index[stem]
    return None


def censor(word: str) -> str:
    """Huruf pertama dan terakhir, sisanya bintang: anjir -> a***r."""
    if len(word) <= 2:
        return word[:1] + "*"
    return word[0] + "*" * (len(word) - 2) + word[-1]


def build_prompt(words: Sequence[str], limit: int) -> str:
    """Petunjuk kata untuk pengenal ucapan; tanpa itu ia menulis "go blog" dan "kancuk"."""
    chosen = list(words)[:limit]
    return PROMPT_LEAD + ", ".join(chosen) + "." if chosen else ""
