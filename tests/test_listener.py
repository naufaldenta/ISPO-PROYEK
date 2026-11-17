import json
import threading
import time

import numpy as np
import pytest

from sahabat.config import Config
from sahabat.listener import NOTICE_TEXT, STATUS_LABEL, Listener
from sahabat.predlog import PredLog
from sahabat.speech import Heard, HeardWord

from .test_utterance import hiss, tone

WORD = np.concatenate([hiss(0.6), tone(0.8), hiss(0.9)])           # satu "ucapan" untuk pemotong


class FakeRecognizer:
    def __init__(self, texts=("dasar anjir",), delay=0.0, fail_on=()):
        self.texts, self.delay, self.fail_on, self.calls, self.prompts = list(texts), delay, set(fail_on), 0, []

    def transcribe(self, audio, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        if self.delay:
            time.sleep(self.delay)
        if self.calls in self.fail_on:
            raise RuntimeError("pengenal tersedak")
        text = self.texts[min(self.calls, len(self.texts)) - 1]
        return [Heard(text, tuple(HeardWord(w, 0.5) for w in text.split()), 0.01, 1.1, -0.4)]


class FakeMic:
    name = "Mikrofon tiruan"
    opened = 0

    def __init__(self, audio=WORD, loop=False, die_after=None):
        type(self).opened += 1
        self.blocks = [audio[i:i + 1600] for i in range(0, len(audio), 1600)]
        self.i, self.loop, self.die_after, self.closed = 0, loop, die_after, False

    def read(self):
        time.sleep(0.001)                                           # jangan memutar CPU di uji berutas
        if self.die_after is not None and self.i >= self.die_after:
            raise OSError("perangkat dicabut")
        if self.i >= len(self.blocks):
            if not self.loop:
                time.sleep(0.01)
                return None
            self.i = 0
        self.i += 1
        return self.blocks[self.i - 1]

    def close(self):
        self.closed = True


@pytest.fixture
def words(tmp_path):
    p = tmp_path / "kata.txt"
    p.write_text("anjir\nasu\nkontol\ngoblok\njancok\n", encoding="utf-8")
    return p


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def make(words, rec=None, mic=None, **kw):
    rec = rec or FakeRecognizer()
    return Listener(Config(), words, lambda: rec, mic or (lambda device: FakeMic()), retry_s=0.05, **kw), rec


def wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_select_microphone_releases_old_and_drops_inflight_recognition(words):
    entered, done = threading.Event(), threading.Event()
    class SlowRecognizer(FakeRecognizer):
        def transcribe(self, audio, prompt):
            entered.set()
            done.wait(3)
            return super().transcribe(audio, prompt)
    opened = []
    def open_selected(device):
        mic = FakeMic(audio=WORD if device is None else np.zeros(16000, np.int16))
        opened.append((device, mic))
        return mic
    lis, _ = make(words, rec=SlowRecognizer(), mic=open_selected)
    lis.start()
    try:
        assert entered.wait(2)
        lis.set_microphone("mic:usb")
        assert wait(lambda: len(opened) >= 2)
        assert opened[0][1].closed and opened[-1][0] == "mic:usb"
        done.set()
        time.sleep(0.1)
        assert lis.snapshot().notice_seq == 0
        assert lis.microphone == "mic:usb"
    finally:
        done.set()
        lis.stop()


def test_select_microphone_while_disabled_keeps_it_disabled(words):
    lis, _ = make(words)
    lis.set_enabled(False)
    lis.set_microphone("mic:usb")
    assert lis.snapshot().status == "mati" and lis.microphone == "mic:usb"


def test_labels_are_exact():
    assert STATUS_LABEL == {"hidup": "Kata kasar: HIDUP", "mati": "Kata kasar: MATI", "memuat": "Kata kasar: memuat…",
                            "tanpa_mikrofon": "Kata kasar: mikrofon tidak ada", "tidak_tersedia": "Kata kasar: tidak tersedia",
                            "daftar_kosong": "Kata kasar: daftar kata kosong"}
    assert NOTICE_TEXT.format("a***r") == "KATA KASAR TERDENGAR: a***r"


def test_heard_word_becomes_a_censored_notice_for_six_seconds(words):
    clock = Clock()
    lis, _ = make(words, clock=clock)
    assert lis.snapshot().notice == "" and lis.snapshot().notice_seq == 0
    lis.hear(WORD)
    s = lis.snapshot()
    assert (s.notice, s.notice_seq) == ("KATA KASAR TERDENGAR: a***r", 1)
    clock.t += 5.9
    assert lis.snapshot().notice != ""
    clock.t += 0.2
    assert lis.snapshot().notice == "" and lis.snapshot().notice_seq == 1


def test_notice_is_available_before_saving_test_audio(words, monkeypatch):
    lis, _ = make(words)
    seen = []
    monkeypatch.setattr(lis, "_save_clip", lambda pcm, verdict: seen.append(lis.snapshot().notice_seq))
    lis.hear(WORD)
    assert seen == [1]


def test_at_most_three_words_are_shown_and_a_new_notice_replaces_the_old(words):
    clock = Clock()
    lis, _ = make(words, FakeRecognizer(["anjir asu kontol goblok jancok", "kamu asu"]), clock=clock)
    lis.hear(WORD)
    assert lis.snapshot().notice == "KATA KASAR TERDENGAR: a***r, a*u, k****l"
    clock.t += 4.0
    lis.hear(WORD)
    assert (lis.snapshot().notice, lis.snapshot().notice_seq) == ("KATA KASAR TERDENGAR: a*u", 2)
    clock.t += 5.0
    assert lis.snapshot().notice != ""                              # 6 detik dihitung dari peringatan terakhir


def test_clean_speech_and_discarded_loops_give_no_notice(words):
    lis, _ = make(words, FakeRecognizer(["selamat pagi semua", "asu asu asu asu"]))
    assert lis.hear(WORD).matches == () and lis.hear(WORD).reason == "berulang"
    assert lis.snapshot().notice_seq == 0


def test_prompt_is_built_from_the_word_file(words):
    lis, rec = make(words)
    lis.hear(WORD)
    assert rec.prompts == ["Percakapan anak sekolah dengan kata kasar: anjir, asu, kontol, goblok, jancok."]


def test_normal_mode_logs_only_time_censored_word_and_confidence(words, tmp_path):
    lis, _ = make(words, FakeRecognizer(["dasar anjir kamu"]), log=PredLog(tmp_path / "log", prefix="kata"), wall=lambda: 1700000000.0)
    lis.hear(WORD)
    lis.stop()
    files = list((tmp_path / "log").glob("*"))
    assert [f.name[:5] for f in files] == ["kata-"] and not list(tmp_path.rglob("*.wav"))
    rec = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert rec == {"t": 1700000000.0, "kata": ["a***r"], "prob": [0.5]}
    assert "anjir" not in files[0].read_text(encoding="utf-8") and "dasar" not in files[0].read_text(encoding="utf-8")


def test_test_recording_mode_saves_every_utterance_with_its_text(words, tmp_path):
    lis, _ = make(words, FakeRecognizer(["selamat pagi", "dasar anjir"]), clip_dir=tmp_path / "suara")
    lis.hear(WORD)
    lis.hear(WORD)
    assert sorted(p.name for p in (tmp_path / "suara").glob("*.wav")) == ["0001.wav", "0002.wav"]
    rows = [json.loads(x) for x in (tmp_path / "suara" / "suara.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["berkas"], r["teks"], r["kata"], r["alasan"]) for r in rows] == [("0001.wav", "selamat pagi", [], ""),
                                                                              ("0002.wav", "dasar anjir", ["anjir"], "")]


def test_status_follows_what_is_missing(words, tmp_path):
    empty = tmp_path / "kosong.txt"
    empty.write_text("# tidak ada kata\n", encoding="utf-8")
    cases = [
        (make(empty)[0], "daftar_kosong"),
        (Listener(Config(), words, lambda: (_ for _ in ()).throw(FileNotFoundError("model")), lambda d: FakeMic(), retry_s=0.05), "tidak_tersedia"),
        (make(words, mic=lambda d: (_ for _ in ()).throw(OSError("tidak ada")))[0], "tanpa_mikrofon"),
        (make(words)[0], "hidup"),
    ]
    for lis, want in cases:
        lis.start()
        try:
            assert wait(lambda: lis.snapshot().status == want), (want, lis.snapshot().status)
            assert lis.snapshot().label == STATUS_LABEL[want]
        finally:
            lis.stop()


def test_running_listener_turns_microphone_speech_into_a_notice(words):
    lis, rec = make(words, mic=lambda d: FakeMic(loop=True))
    lis.start()
    try:
        assert wait(lambda: lis.snapshot().notice_seq >= 1) and lis.mic_name == "Mikrofon tiruan"
    finally:
        lis.stop()


def test_off_closes_the_microphone_and_on_reopens_it(words):
    mics, toggles = [], []
    lis, _ = make(words, mic=lambda d: mics.append(FakeMic(loop=True)) or mics[-1], on_toggle=toggles.append)
    lis.start()
    try:
        assert wait(lambda: lis.snapshot().status == "hidup")
        assert lis.set_enabled(False) is False
        assert wait(lambda: mics[0].closed) and lis.snapshot().status == "mati"
        time.sleep(0.2)
        assert len(mics) == 1                                       # selama mati, mikrofon tidak dibuka lagi
        assert lis.set_enabled(True) is True
        assert wait(lambda: len(mics) == 2 and lis.snapshot().status == "hidup") and toggles == [False, True]
    finally:
        lis.stop()
    assert mics[-1].closed


def test_microphone_unplugged_is_reported_then_recovers(words):             # Review Focus 1
    made = []

    def open_mic(device):
        made.append(FakeMic(loop=True, die_after=5 if not made else None))
        return made[-1]
    lis, _ = make(words, mic=open_mic)
    lis.start()
    try:
        assert wait(lambda: len(made) >= 2 and made[0].closed)
        assert wait(lambda: lis.snapshot().status == "hidup")
    finally:
        lis.stop()


def test_slow_recognizer_keeps_at_most_one_utterance_waiting(words):         # Review Focus 2
    rec = FakeRecognizer(delay=0.3)
    lis, _ = make(words, rec, mic=lambda d: FakeMic(loop=True))
    lis.start()
    try:
        time.sleep(1.5)                                             # ±2,3 dtk suara per putaran mikrofon tiruan, tanpa jeda
    finally:
        lis.stop()
    assert 1 <= rec.calls <= 7                                      # tidak semua ucapan diproses; tidak menumpuk


def test_recognizer_error_skips_that_utterance_and_keeps_listening(words, capsys):   # Review Focus 4
    lis, rec = make(words, FakeRecognizer(["dasar anjir"], fail_on={1}))
    assert lis.hear(WORD).matches == () and lis.snapshot().notice_seq == 0
    assert [m.word for m in lis.hear(WORD).matches] == ["anjir"] and lis.snapshot().notice_seq == 1
    assert capsys.readouterr().out.count("Galat di pengenal ucapan") == 1


def test_rapid_toggling_never_leaves_two_microphones_open(words):            # Review Focus 5
    mics = []
    slow_rec = threading.Event()

    def make_rec():
        slow_rec.wait(0.3)                                          # model "masih dimuat" saat tombol ditekan-tekan
        return FakeRecognizer()
    lis = Listener(Config(), words, make_rec, lambda d: mics.append(FakeMic(loop=True)) or mics[-1], retry_s=0.05)
    lis.start()
    try:
        for on in (False, True, False, True, False, True):
            lis.set_enabled(on)
        assert wait(lambda: lis.snapshot().status == "hidup")
        assert sum(not m.closed for m in mics) == 1
        lis.set_enabled(False)
        assert wait(lambda: all(m.closed for m in mics)) and lis.snapshot().status == "mati"
    finally:
        lis.stop()


def test_disabled_at_start_never_opens_the_microphone(words):
    import dataclasses
    mics = []
    lis = Listener(dataclasses.replace(Config(), words_enabled=False), words, lambda: FakeRecognizer(),
                   lambda d: mics.append(FakeMic()) or mics[-1], retry_s=0.05)
    lis.start()
    try:
        time.sleep(0.2)
        assert lis.snapshot().status == "mati" and mics == []
    finally:
        lis.stop()


# ---- temuan tinjauan akhir ----

def test_no_notice_appears_once_the_listener_is_switched_off(words):
    lis, _ = make(words, clock=Clock())
    lis.hear(WORD)
    assert lis.snapshot().notice != ""
    lis.set_enabled(False)
    off = lis.snapshot()
    assert (off.status, off.notice) == ("mati", "")                 # peringatan lama ikut hilang
    lis.hear(WORD)                                                  # ucapan yang masih diproses saat tombol ditekan
    assert (lis.snapshot().notice, lis.snapshot().notice_seq) == ("", off.notice_seq)
    lis.set_enabled(True)
    lis.hear(WORD)
    assert lis.snapshot().notice != "" and lis.snapshot().notice_seq == off.notice_seq + 1


def test_unexpected_error_in_the_worker_does_not_leave_a_deaf_listener(words, capsys):
    class BadLog:
        calls = 0

        def write(self, record):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("log rusak")

        def close(self):
            pass

    lis, _ = make(words, mic=lambda d: FakeMic(loop=True), log=BadLog())
    lis.start()
    try:
        assert wait(lambda: lis.snapshot().notice_seq >= 2)         # peringatan kedua: pekerja masih hidup
    finally:
        lis.stop()
    assert "Galat di pendengar kata kasar" in capsys.readouterr().out


def test_listener_tells_the_judge_how_long_the_utterance_was(words):
    made_up = "kembali ayo ayah ayuh ayu ayoh ayi ayur ayol ay ayat ayang ayyuh yuh asu ayoy yo ye ya"
    lis, _ = make(words, FakeRecognizer([made_up]))
    assert lis.hear(WORD[:16000]).reason == "terlalu_cepat"                        # 19 kata berbeda dalam 1 detik
    assert lis.snapshot().notice_seq == 0
