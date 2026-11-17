import wave

import numpy as np
import pytest

from latih.suara import read_wav, replay, utterances_of
from sahabat.config import Config

from .test_listener import FakeRecognizer
from .test_utterance import hiss, tone


def write_wav(path, pcm, rate=16000, channels=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def test_wav_round_trip_and_wrong_formats_are_refused(tmp_path):
    pcm = tone(0.5)
    write_wav(tmp_path / "a.wav", pcm)
    assert np.array_equal(read_wav(tmp_path / "a.wav"), pcm)
    write_wav(tmp_path / "b.wav", pcm, rate=44100)
    with pytest.raises(ValueError, match="16 kHz"):
        read_wav(tmp_path / "b.wav")


def test_utterances_come_with_their_start_time():
    pcm = np.concatenate([hiss(1.0), tone(0.8), hiss(1.5), tone(0.8), hiss(1.0)])
    got = utterances_of(pcm, Config())
    assert len(got) == 2 and got[0][0] == pytest.approx(0.8, abs=0.3) and got[1][0] == pytest.approx(3.1, abs=0.3)


def test_replay_gives_one_verdict_per_utterance():
    pcm = np.concatenate([hiss(1.0), tone(0.8), hiss(1.5), tone(0.8), hiss(1.0)])
    out = replay(utterances_of(pcm, Config()), FakeRecognizer(["selamat pagi", "dasar anjir"]), ("anjir",), Config())
    assert [[m.word for m in v.matches] for _, v in out] == [[], ["anjir"]]
