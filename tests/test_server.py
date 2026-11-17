import pytest
from fastapi.testclient import TestClient

from sahabat.engine import BAR_TEXT, SLOW_TEXT, Snapshot
from sahabat.listener import STATUS_LABEL, ListenState
from sahabat.server import create_app, mjpeg_chunk


class StubEngine:
    def __init__(self):
        self.ack_result = True
        self.ack_calls = 0
        self.marks = []
        self.set()

    def set(self, bar="aman", fps=10.0, people=0, slow=False, alarm_seq=0):
        self._snap = Snapshot(bar, BAR_TEXT[bar], fps, people, slow, alarm_seq, ())

    def snapshot(self):
        return self._snap

    def acknowledge(self):
        self.ack_calls += 1
        return self.ack_result

    def mark(self, label):
        self.marks.append(label)
        return True

    def latest_jpeg(self):
        return b"\xff\xd8JPEG"


@pytest.fixture
def stub():
    return StubEngine()


@pytest.fixture
def app(stub):
    return create_app(stub)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def test_index_has_required_controls(client):
    r = client.get("/")
    assert r.status_code == 200
    assert all(s in r.text for s in ("Mulai", "Sudah diperiksa", "Senyap"))


def test_static_files_are_served(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_page_and_static_files_are_never_cached(client):
    # folder aplikasi bisa diganti versi baru; browser tidak boleh memakai berkas lama
    for path in ("/", "/static/app.js", "/static/style.css"):
        assert client.get(path).headers["cache-control"] == "no-store"


def test_ws_sends_state_with_exact_text(client, stub):
    stub.set(bar="kekerasan", alarm_seq=2, people=2)
    with client.websocket_connect("/ws") as ws:
        m = ws.receive_json()
    assert m["text"] == "INDIKASI KEKERASAN TERDETEKSI" and m["alarm_seq"] == 2 and m["people"] == 2
    assert set(m) == {"bar", "text", "fps", "people", "slow", "slow_text", "alarm_seq", "recording", "words", "flip"}


def test_ws_slow_text_only_when_slow(client, stub):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["slow_text"] == ""
    stub.set(slow=True)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["slow_text"] == SLOW_TEXT == "laptop ini terlalu lambat, deteksi bisa meleset"


def test_ws_follows_state_changes(client, stub):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["bar"] == "aman"
        stub.set(bar="jatuh", alarm_seq=1)
        assert [ws.receive_json()["bar"] for _ in range(3)][-1] == "jatuh"


def test_two_tabs_get_the_same_state(client, stub):
    stub.set(bar="jatuh")
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        assert a.receive_json()["bar"] == b.receive_json()["bar"] == "jatuh"


def test_ack_reports_result(client, stub):
    stub.ack_result = True
    assert client.post("/api/ack").json() == {"ok": True}
    stub.ack_result = False
    r = client.post("/api/ack")
    assert (r.status_code, r.json()) == (200, {"ok": False}) and stub.ack_calls == 2


def test_mjpeg_chunk_format():                      # alirannya tak berujung, jadi yang diuji potongannya
    c = mjpeg_chunk(b"\xff\xd8JPEG")
    assert c.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n") and c.endswith(b"\xff\xd8JPEG\r\n")


def test_video_route_exists(app):
    assert "/video" in {r.path for r in app.routes}


def test_ping(client):
    assert client.get("/api/ping").json() == {"app": "sahabat"}


def test_ws_passes_the_engine_warning_text_through(client, stub):
    stub._snap = Snapshot("aman", "AMAN", 7.5, 0, True, 0, (), "kamera hanya mengirim 8 gambar per detik")
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["slow_text"] == "kamera hanya mengirim 8 gambar per detik"


def test_ws_tells_the_page_when_recording(client, stub):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["recording"] is False
    stub._snap = Snapshot("aman", "AMAN", 10.0, 0, False, 0, (), "", True)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["recording"] is True


def test_mark_passes_known_labels_to_the_engine(client, stub):
    for label in ("jatuh", "kekerasan", "biasa"):
        assert client.post("/api/tanda", json={"label": label}).json() == {"ok": True}
    assert stub.marks == ["jatuh", "kekerasan", "biasa"]


def test_mark_rejects_unknown_labels(client, stub):
    assert client.post("/api/tanda", json={"label": "<script>"}).json() == {"ok": False}
    assert client.post("/api/tanda", json={}).json() == {"ok": False}
    assert stub.marks == []


def test_page_has_the_recording_sign(client):
    assert "MEREKAM" in client.get("/").text


class StubListener:
    def __init__(self):
        self.on, self.calls = True, []
        self.state = ListenState("hidup", STATUS_LABEL["hidup"], "", 0)

    def snapshot(self):
        return self.state

    def set_enabled(self, on):
        self.calls.append(on)
        self.on = on
        return on


@pytest.fixture
def heard():
    return StubListener()


@pytest.fixture
def client2(stub, heard):
    with TestClient(create_app(stub, heard)) as c:
        yield c


def test_ws_carries_the_word_state(client2, heard):
    heard.state = ListenState("hidup", STATUS_LABEL["hidup"], "KATA KASAR TERDENGAR: a***r", 3)
    with client2.websocket_connect("/ws") as ws:
        m = ws.receive_json()
    assert m["words"] == {"status": "hidup", "label": "Kata kasar: HIDUP", "notice": "KATA KASAR TERDENGAR: a***r", "notice_seq": 3}
    assert m["bar"] == "aman"                                      # pita status tidak ikut berubah


def test_without_a_listener_the_page_is_told_it_is_unavailable(client):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["words"] == {"status": "tidak_tersedia", "label": "Kata kasar: tidak tersedia",
                                              "notice": "", "notice_seq": 0}
    assert client.post("/api/kata", json={"on": False}).json() == {"ok": False}


def test_toggle_endpoint_passes_the_choice_on(client2, heard):
    assert client2.post("/api/kata", json={"on": False}).json() == {"ok": True, "on": False}
    assert client2.post("/api/kata", json={"on": True}).json() == {"ok": True, "on": True}
    assert heard.calls == [False, True]
    for bad in ({}, {"on": "ya"}, {"on": 1}, [True]):
        assert client2.post("/api/kata", json=bad).json() == {"ok": False}
    assert heard.calls == [False, True]


def test_page_has_the_word_button_and_notice_strip(client):
    html = client.get("/").text
    assert 'id="words"' in html and 'id="words-notice"' in html


class FlipSource:
    def __init__(self):
        self.flip = False


def test_flip_endpoint_turns_the_picture_and_remembers_it(stub, monkeypatch, tmp_path):
    from sahabat import server
    saved = []
    monkeypatch.setattr(server, "remember_flip", lambda on: saved.append(on))
    stub.source = FlipSource()
    with TestClient(create_app(stub)) as c:
        assert c.post("/api/cermin", json={"on": True}).json() == {"ok": True, "on": True}
        assert stub.source.flip is True and saved == [True]
        for bad in ({}, {"on": "ya"}, {"on": 1}, [True]):
            assert c.post("/api/cermin", json=bad).json() == {"ok": False}
        assert saved == [True]


def test_flip_endpoint_without_camera_says_no(client):
    assert client.post("/api/cermin", json={"on": True}).json() == {"ok": False}


def test_page_has_the_flip_button(client):
    assert 'id="flip"' in client.get("/").text
