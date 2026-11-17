"""mic.py diuji dengan pustaka sounddevice tiruan; tidak ada mikrofon sungguhan yang dibuka."""
import sys
import types

import numpy as np
import pytest

from sahabat.mic import open_mic


class FakeStream:
    def __init__(self, fail_start=False, **kw):
        self.kw, self.active, self.closed, self.fail_start = kw, False, False, fail_start

    def start(self):
        if self.fail_start:
            raise FakeSD.PortAudioError("perangkat sibuk")
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True


class FakeSD(types.ModuleType):
    PortAudioError = type("PortAudioError", (Exception,), {})

    def __init__(self):
        super().__init__("sounddevice")
        self.streams, self.refreshed, self.fail_start = [], 0, False

    def query_devices(self, device=None, kind=None):
        return {"name": "Mik tiruan"}

    def InputStream(self, **kw):
        self.streams.append(FakeStream(self.fail_start, **kw))
        return self.streams[-1]

    def _terminate(self):
        self.refreshed += 1

    def _initialize(self):
        pass


@pytest.fixture
def sd(monkeypatch):
    fake = FakeSD()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    monkeypatch.setattr("sahabat.mic.READ_WAIT_S", 0.02)
    monkeypatch.setattr("sahabat.mic.STALL_S", 0.15)
    return fake


def test_blocks_from_the_callback_come_out_of_read(sd):
    mic = open_mic(None)
    sd.streams[0].kw["callback"](np.full((1600, 1), 7, dtype=np.int16), 1600, None, None)
    block = mic.read()
    assert mic.name == "Mik tiruan" and block.dtype == np.int16 and len(block) == 1600 and block[0] == 7
    mic.close()
    assert sd.streams[0].closed


def test_stream_that_stops_delivering_is_reported_as_lost(sd):
    # Mikrofon USB dicabut: Windows berhenti memanggil callback tetapi alirannya tetap "aktif".
    mic = open_mic(None)
    assert mic.read() is None                           # jeda pendek belum dianggap hilang
    with pytest.raises(OSError):
        for _ in range(100):
            mic.read()


def test_a_working_stream_is_never_reported_as_lost(sd):
    mic = open_mic(None)
    feed = sd.streams[0].kw["callback"]
    for _ in range(20):
        feed(np.ones((1600, 1), dtype=np.int16), 1600, None, None)
        assert mic.read() is not None
        assert mic.read() is None                       # sesekali kosong di antara blok: wajar


def test_reopening_refreshes_the_device_list(sd):
    open_mic(None).close()
    before = sd.refreshed
    open_mic(None).close()
    assert sd.refreshed == before + 1                   # mikrofon yang dicolok ulang baru terlihat setelah ini


def test_stream_that_fails_to_start_is_closed_and_reported(sd):
    sd.fail_start = True
    with pytest.raises(OSError):
        open_mic(None)
    assert sd.streams[0].closed
