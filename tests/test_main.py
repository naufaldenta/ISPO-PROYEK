import socket
import threading

import uvicorn

from sahabat.__main__ import already_running, claim_port, find_browser, find_free_port, wait_until_running
from sahabat.server import create_app

from .test_server import StubEngine


def test_free_port_skips_occupied_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert find_free_port(busy) != busy


def test_free_port_keeps_preferred_when_free():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    assert find_free_port(free) == free


def test_already_running_false_for_silent_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        assert already_running(s.getsockname()[1]) is False


def test_already_running_false_for_closed_port():
    assert already_running(1) is False


def test_already_running_true_for_our_own_server():
    port = find_free_port(0)
    server = uvicorn.Server(uvicorn.Config(create_app(StubEngine()), host="127.0.0.1", port=port,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        assert any(already_running(port, timeout=1.0) for _ in range(20))
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_find_browser_returns_existing_file_or_none(tmp_path, monkeypatch):
    for var in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData"):
        monkeypatch.setenv(var, str(tmp_path / "kosong"))
    assert find_browser() is None
    edge = tmp_path / "pf" / "Microsoft" / "Edge" / "Application" / "msedge.exe"
    edge.parent.mkdir(parents=True)
    edge.write_bytes(b"")
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "pf"))
    assert find_browser() == edge


def test_open_browser_can_be_suppressed(monkeypatch):
    # untuk uji otomatis dan pemasangan tanpa layar: jangan membuka jendela apa pun
    import sahabat.__main__ as m

    def forbidden(*a, **k):
        raise AssertionError("browser tidak boleh dibuka")
    monkeypatch.setenv("SAHABAT_NO_BROWSER", "1")
    monkeypatch.setattr(m.subprocess, "Popen", forbidden)
    monkeypatch.setattr(m.webbrowser, "open", forbidden)
    m.open_browser("http://127.0.0.1:1/")


def test_only_one_instance_can_claim_the_port():
    first = claim_port(find_free_port(0))
    try:
        port = first.getsockname()[1]
        assert claim_port(port) is None                     # klik dua kali kedua: port sudah dipegang
        assert already_running(port) is False               # ...padahal instans pertama belum melayani
    finally:
        first.close()


def test_wait_until_running_gives_up_on_a_silent_port():
    import time
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        started = time.monotonic()
        assert wait_until_running(s.getsockname()[1], timeout_s=1.0) is False
        assert time.monotonic() - started < 4.0


def test_server_runs_on_the_claimed_socket_and_second_launch_finds_it():
    sock = claim_port(find_free_port(0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(StubEngine()), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        assert wait_until_running(port, timeout_s=10.0) is True
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_word_choice_is_saved_next_to_the_other_laptop_settings(tmp_path):
    from sahabat.__main__ import save_words_choice
    from sahabat.config import Config, load, save
    import dataclasses
    path = tmp_path / "pengaturan.json"
    save(dataclasses.replace(Config(), imgsz=416, port=9000), path)
    save_words_choice(False, path)
    c = load(path)
    assert (c.words_enabled, c.imgsz, c.port) == (False, 416, 9000)
    save_words_choice(True, path)
    assert load(path).words_enabled is True


def test_saving_the_choice_to_an_unwritable_place_does_not_raise(tmp_path):
    from sahabat.__main__ import save_words_choice
    blocker = tmp_path / "berkas"
    blocker.write_text("x")
    save_words_choice(False, blocker / "pengaturan.json")
