"""Asisten pemasangan kamera: petunjuk dari pose yang terlihat (badan utuh, ukuran orang, sudut pandang)."""
import numpy as np
import pytest

from sahabat.asisten_kamera import KECIL, MENUNDUK, Asisten
from sahabat.types import Person

WAIT = "Belum ada orang terlihat. Minta satu orang berjalan di depan kamera."
SMALL = "Orang terlalu kecil, dekatkan kamera atau pakai lensa lebih sempit"
TILTED = "Kamera terlalu menunduk"
CUT = "Kaki sering terpotong, naikkan atau mundurkan kamera"


def standing(leg: float, torso: float, height_frac: float, frame_h: int = 480, legs_visible: bool = True) -> Person:
    """Orang tegak di tengah gambar: tinggi kotak = height_frac x tinggi gambar, kaki/badan = leg/torso."""
    total = height_frac * frame_h
    unit = total / (leg + torso + 0.4)           # kepala sekitar 0,4 badan
    top = 50.0
    sh_y = top + 0.4 * unit
    hip_y = sh_y + torso * unit
    ank_y = hip_y + leg * unit
    kp = np.zeros((17, 2), np.float32)
    kp[:, 0] = 320.0
    kp[0, 1] = top + 0.1 * unit
    kp[5, 1] = kp[6, 1] = sh_y
    for i in (7, 8, 9, 10):                       # siku dan pergelangan tangan di bawah bahu
        kp[i, 1] = sh_y + (i - 6) * 0.1 * unit
    kp[11, 1] = kp[12, 1] = hip_y
    kp[13, 1] = kp[14, 1] = (hip_y + ank_y) / 2
    kp[15, 1] = kp[16, 1] = ank_y
    kp[5, 0], kp[6, 0], kp[11, 0], kp[12, 0] = 300, 340, 305, 335
    conf = np.full(17, 0.9, np.float32)
    if not legs_visible:
        conf[13:] = 0.0
    return Person(1, kp, conf, (290.0, top, 350.0, top + total))


def feed(a: Asisten, person: Person, n: int, mode: str = "full", shape=(480, 640)) -> Asisten:
    for i in range(n):
        a.amati(i * 0.1, [person], shape, {1: mode})
    return a


def test_no_people_gives_waiting_message():                                    # Review Focus 4
    a = Asisten()
    a.amati(0.0, [], (480, 640), {})
    assert a.ringkas()["saran"] == [WAIT]
    assert feed(Asisten(), standing(1.5, 1.0, 0.6), n=5).ringkas()["saran"] == [WAIT]    # baru beberapa frame


def test_full_body_eye_level_person_gives_no_warning():
    s = feed(Asisten(), standing(leg=1.5, torso=1.0, height_frac=0.6), n=40).ringkas()
    assert s["saran"] == ["Badan terlihat utuh: 100%"]
    assert s["rasio_kaki_badan"] == pytest.approx(1.5, abs=0.05) and s["tinggi_orang"] == pytest.approx(0.6, abs=0.01)
    assert s["orang"] == 1 and s["utuh_persen"] == 100


def test_small_and_tilted_camera_warnings():
    s = feed(Asisten(), standing(leg=0.7, torso=1.0, height_frac=0.1), n=40).ringkas()
    assert 0.7 < MENUNDUK and 0.1 < KECIL
    assert SMALL in s["saran"] and TILTED in s["saran"]


def test_cut_legs_warning():
    s = feed(Asisten(), standing(1.5, 1.0, 0.6, legs_visible=False), n=40, mode="upper").ringkas()
    assert s["saran"][0] == "Badan terlihat utuh: 0%" and CUT in s["saran"] and s["rasio_kaki_badan"] is None
    assert TILTED not in s["saran"]                       # sudut tidak bisa dinilai tanpa kaki


def test_old_observations_expire():
    a = feed(Asisten(window_s=30.0), standing(0.7, 1.0, 0.6), n=40)        # t = 0..3,9 dtk: menunduk
    for i in range(40):
        a.amati(100.0 + i * 0.1, [standing(1.5, 1.0, 0.6)], (480, 640), {1: "full"})
    assert TILTED not in a.ringkas()["saran"]


class _OnePerson:
    def detect(self, frame):
        return [standing(1.5, 1.0, 0.6)]


def test_engine_feeds_the_assistant():
    from sahabat.config import Config
    from sahabat.engine import Engine
    engine = Engine(Config(), None, _OnePerson())
    frame = np.zeros((480, 640, 3), np.uint8)
    for i in range(25):
        engine.step(i * 0.1, frame)
    s = engine.asisten.ringkas()
    assert s["orang"] == 1 and s["saran"][0].startswith("Badan terlihat utuh: ")


def test_assistant_failure_never_skips_scoring():                            # Final #4
    from sahabat.config import Config
    from sahabat.engine import Engine
    engine = Engine(Config(), None, _OnePerson())
    ticks = []
    original = engine.fall.tick
    engine.fall.tick = lambda *a, **k: ticks.append(1) or original(*a, **k)

    def boom(*a, **k):
        raise IndexError("deque kosong")
    engine.asisten.amati = boom
    engine.step(0.0, np.zeros((480, 640, 3), np.uint8))
    assert ticks == [1]


def test_assistant_is_safe_across_threads():
    import threading
    a = Asisten(window_s=0.05)
    errors, stop = [], threading.Event()

    def feed_loop():
        t = 0.0
        while not stop.is_set():
            t += 0.01
            try:
                a.amati(t, [standing(1.5, 1.0, 0.6)], (480, 640), {1: "full"})
            except Exception as e:                  # noqa: BLE001
                errors.append(e)

    def read_loop():
        while not stop.is_set():
            try:
                a.ringkas()
            except Exception as e:                  # noqa: BLE001
                errors.append(e)
    threads = [threading.Thread(target=feed_loop), threading.Thread(target=read_loop)]
    for th in threads:
        th.start()
    import time
    time.sleep(0.5)
    stop.set()
    for th in threads:
        th.join()
    assert errors == []
