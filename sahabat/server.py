"""Server lokal: halaman layar guru, video MJPEG, keadaan lewat WebSocket."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .config import DATA_DIR, MODELS_DIR, SETTINGS_PATH, WORDS_PATH, Config, load
from .engine import SLOW_TEXT, Engine
from .jaringan import COOKIE, is_local, qr_png
from .listener import STATUS_LABEL, Listener

WEB_DIR = Path(__file__).resolve().parent / "web"
VIDEO_GAP_S = 0.05        # paling cepat 20 frame per detik
REMOTE_VIDEO_GAP_S = 0.1  # layar jarak jauh: paling cepat 10 frame per detik, hemat WiFi
STATE_GAP_S = 0.2
VIOLENCE_MODEL = MODELS_DIR / "violence_model.joblib"
FALL_MODEL = MODELS_DIR / "fall_model.joblib"
NO_CACHE = {"Cache-Control": "no-store"}
MARK_LABELS = ("jatuh", "kekerasan", "biasa")      # tanda adegan saat merekam uji
MAX_VIDEO_BYTES = 4 * 1024 ** 3                    # video cadangan yang diunggah dari layar
_BAD_NAME_CHARS = set(r'<>:"/\|?*')


class _FreshStaticFiles(StaticFiles):
    """Berkas halaman tidak boleh disimpan browser: folder aplikasi bisa diganti versi baru."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


def mjpeg_chunk(jpeg: bytes) -> bytes:
    return (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
            + b"\r\n\r\n" + jpeg + b"\r\n")


def _words(listener) -> dict:
    """Keadaan pendengar kata kasar untuk layar. Tanpa pendengar: tombolnya menulis "tidak tersedia"."""
    if listener is None:
        return {"status": "tidak_tersedia", "label": STATUS_LABEL["tidak_tersedia"], "notice": "", "notice_seq": 0}
    w = listener.snapshot()
    return {"status": w.status, "label": w.label, "notice": w.notice, "notice_seq": w.notice_seq}


def _state(engine, listener=None, inputs=None) -> dict:
    s = engine.snapshot()
    state = {"bar": s.bar, "text": s.text, "fps": s.fps, "people": s.people, "slow": s.slow,
            "slow_text": s.slow_text or (SLOW_TEXT if s.slow else ""), "alarm_seq": s.alarm_seq,
            "recording": bool(s.recording), "words": _words(listener),
            "flip": bool(getattr(getattr(engine, "source", None), "flip", False))}
    if inputs is not None:
        state["devices"] = inputs.status()
    riwayat = getattr(engine, "riwayat", None)
    if riwayat is not None:
        state["riwayat"] = {"seq": riwayat.seq, "klip": bool(riwayat.keep_clips)}
    if hasattr(engine, "set_zone"):
        zone = getattr(engine, "zone", None)
        state["zona"] = [list(p) for p in zone] if zone else None
    return state


def remember_flip(on: bool) -> None:
    from .config import SETTINGS_PATH, update_local
    update_local(SETTINGS_PATH, camera_flip=bool(on))


def video_name(name: str) -> str | None:
    """Nama berkas video yang aman untuk folder video/, atau None."""
    from .config import VIDEO_EXTENSIONS
    name = (name or "").strip()
    if (not name or len(name) > 120 or name in (".", "..") or any(c in _BAD_NAME_CHARS or ord(c) < 32 for c in name)
            or Path(name).suffix.lower() not in VIDEO_EXTENSIONS or name.startswith(".")):
        return None
    return name


def _free_name(folder: Path, name: str) -> Path:
    target = folder / name
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 2
    while target.exists():
        target = folder / f"{stem} ({n}){suffix}"
        n += 1
    return target


def open_folder(folder: Path) -> None:
    """Membuka folder di Windows Explorer supaya guru bisa menyalin video ke sana."""
    import os
    folder.mkdir(parents=True, exist_ok=True)
    os.startfile(str(folder))        # noqa: S606 - hanya folder aplikasi sendiri


def remember_lan(enabled: bool, code: str) -> None:
    from .config import SETTINGS_PATH, update_local
    update_local(SETTINGS_PATH, lan_enabled=bool(enabled), lan_code=code)


def remember_zone(zone) -> None:
    from .config import SETTINGS_PATH, update_local
    update_local(SETTINGS_PATH, zone=[list(p) for p in zone] if zone else None)


def remember_clips(on: bool) -> None:
    from .config import SETTINGS_PATH, update_local
    update_local(SETTINGS_PATH, clips_enabled=bool(on))


def _list_riwayat(riwayat) -> list[dict]:
    riwayat.prune()                     # aplikasi bisa menyala berhari-hari: bersihkan juga saat panel dibuka
    return riwayat.list()


def _local_only(method: str, path: str) -> bool:
    """Pengaturan yang hanya boleh diubah dari laptop SAHABAT sendiri, bukan dari HP atau laptop lain."""
    if path.startswith("/api/jaringan"):
        return True
    return (method, path) in {
        ("GET", "/api/perangkat"), ("POST", "/api/perangkat"), ("POST", "/api/video"), ("POST", "/api/video/folder"),
        ("POST", "/api/zona"), ("POST", "/api/riwayat/klip"), ("POST", "/api/cermin"), ("POST", "/api/kata"),
        ("POST", "/api/tanda")}


LOGIN_STYLE = (
    "body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#0b0f15;"
    "color:#fff;font-family:'Segoe UI',system-ui,sans-serif}form{width:min(360px,90vw);text-align:center}"
    "input{width:100%;box-sizing:border-box;font-size:28px;letter-spacing:6px;text-align:center;padding:12px;"
    "border-radius:10px;border:1px solid #536174;text-transform:uppercase}button{margin-top:14px;width:100%;"
    "font-size:20px;padding:12px;border:0;border-radius:10px;background:#17914a;color:#fff;font-weight:700}"
    "p{color:#ccd5df;line-height:1.5}.err{color:#ff8a80}")


def _login_page(message: str = "") -> str:
    note = '<p class="err">' + message + "</p>" if message else ""
    return ('<!doctype html><html lang="id"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1"><title>SAHABAT Sekolah</title>'
            "<style>" + LOGIN_STYLE + "</style></head><body>"
            '<form method="get" action="/"><h1>SAHABAT Sekolah</h1>'
            "<p>Masukkan kode akses yang tampil di laptop SAHABAT (tombol <b>HP / laptop lain</b>).</p>" + note +
            '<input name="kode" autocomplete="off" autofocus maxlength="12" aria-label="Kode akses">'
            '<button type="submit">Buka layar</button></form></body></html>')


def create_app(engine, listener=None, inputs=None, lan=None) -> FastAPI:
    if inputs is None and getattr(engine, "source", None) is not None:
        from .input_control import InputDevices
        inputs = InputDevices(engine, listener)
    app = FastAPI(title="SAHABAT Sekolah", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", _FreshStaticFiles(directory=WEB_DIR), name="static")

    def remote_allowed(cookie: str | None) -> bool:
        return lan is not None and lan.enabled and lan.check(cookie)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        """Laptop SAHABAT sendiri bebas. Perangkat lain butuh jaringan lokal hidup dan kode akses yang benar."""
        host = request.client.host if request.client else None
        if is_local(host):
            return await call_next(request)
        if lan is None or not lan.enabled:
            return Response("Akses dari jaringan lokal sedang dimatikan.", status_code=403,
                            media_type="text/plain; charset=utf-8")
        code = request.query_params.get("kode")
        if code is not None and request.url.path == "/":
            if not lan.check(code):
                return HTMLResponse(_login_page("Kode akses salah. Periksa lagi di laptop SAHABAT."), status_code=403)
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(COOKIE, lan.code, max_age=30 * 86400, httponly=True, samesite="lax")
            return response
        if not lan.check(request.cookies.get(COOKIE)):
            if request.url.path == "/":
                return HTMLResponse(_login_page(), status_code=401)
            return JSONResponse({"ok": False, "error": "Butuh kode akses."}, status_code=401)
        if _local_only(request.method, request.url.path):
            return JSONResponse({"ok": False, "error": "Pengaturan hanya bisa diubah dari laptop SAHABAT."},
                                status_code=403)
        return await call_next(request)

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html", headers=NO_CACHE)

    @app.get("/video")
    async def video(request: Request):
        gap = VIDEO_GAP_S if is_local(request.client.host if request.client else None) else REMOTE_VIDEO_GAP_S

        async def frames():
            while True:
                jpeg = await run_in_threadpool(engine.latest_jpeg)
                if jpeg:
                    yield mjpeg_chunk(jpeg)
                await asyncio.sleep(gap)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers=NO_CACHE)

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        local = is_local(socket.client.host if socket.client else None)
        if not local and not remote_allowed(socket.cookies.get(COOKIE)):
            await socket.close(code=1008)
            return
        await socket.accept()
        if not local:
            lan.viewers += 1
        try:
            while True:
                if not local and not remote_allowed(socket.cookies.get(COOKIE)):
                    await socket.close(code=1008)                  # kode diganti atau jaringan dimatikan
                    return
                state = _state(engine, listener, inputs)
                state["lokal"] = local
                if local and lan is not None:
                    state["jaringan"] = {"hidup": lan.enabled, "penonton": lan.viewers}
                await socket.send_json(state)
                await asyncio.sleep(STATE_GAP_S)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if not local:
                lan.viewers = max(0, lan.viewers - 1)

    @app.get("/api/jaringan")
    async def lan_status():
        if lan is None:
            return JSONResponse({"ok": False, "error": "Jaringan lokal tidak tersedia."}, status_code=503)
        return {"ok": True, **lan.status()}

    @app.get("/api/jaringan/qr.png")
    async def lan_qr():
        urls = lan.urls() if lan is not None else []
        if not urls:
            return JSONResponse({"ok": False, "error": "Jaringan lokal mati."}, status_code=404)
        png = await run_in_threadpool(qr_png, f"{urls[0]}?kode={lan.code}")
        return Response(png, media_type="image/png", headers=NO_CACHE)

    @app.post("/api/jaringan")
    async def lan_toggle(request: Request):
        if lan is None:
            return JSONResponse({"ok": False, "error": "Jaringan lokal tidak tersedia."}, status_code=503)
        try:
            body = await request.json()
        except ValueError:
            body = None
        on = body.get("hidup") if isinstance(body, dict) else None
        if not isinstance(on, bool):
            return JSONResponse({"ok": False, "error": "Kirim hidup: true/false."}, status_code=400)
        status = await run_in_threadpool(lan.set_enabled, on)
        try:
            await run_in_threadpool(remember_lan, lan.enabled, lan.code)
        except OSError:
            pass
        return {"ok": not status["galat"], **status}

    @app.post("/api/jaringan/kode")
    async def lan_new_code():
        if lan is None:
            return JSONResponse({"ok": False, "error": "Jaringan lokal tidak tersedia."}, status_code=503)
        lan.regenerate()
        try:
            await run_in_threadpool(remember_lan, lan.enabled, lan.code)
        except OSError:
            pass
        return {"ok": True, **lan.status()}

    @app.post("/api/ack")
    async def ack():
        return {"ok": bool(await run_in_threadpool(engine.acknowledge))}

    @app.post("/api/tanda")
    async def mark(request: Request):
        """Pencatat menandai adegan yang baru diperagakan (hanya berarti saat merekam uji)."""
        try:
            body = await request.json()
        except ValueError:
            body = None
        label = body.get("label") if isinstance(body, dict) else None
        if label not in MARK_LABELS:
            return {"ok": False}
        return {"ok": bool(await run_in_threadpool(engine.mark, label))}

    @app.post("/api/kata")
    async def words(request: Request):
        """Tombol hidup/mati pendengar kata kasar."""
        try:
            body = await request.json()
        except ValueError:
            body = None
        on = body.get("on") if isinstance(body, dict) else None
        if listener is None or not isinstance(on, bool):
            return {"ok": False}
        return {"ok": True, "on": bool(await run_in_threadpool(listener.set_enabled, on))}

    @app.post("/api/cermin")
    async def flip(request: Request):
        """Tombol balik kiri-kanan gambar kamera; pilihannya diingat per laptop."""
        source = getattr(engine, "source", None)
        try:
            body = await request.json()
        except ValueError:
            body = None
        on = body.get("on") if isinstance(body, dict) else None
        if source is None or not isinstance(on, bool):
            return {"ok": False}
        source.flip = on
        try:
            await run_in_threadpool(remember_flip, on)
        except OSError:
            pass                                            # tetap berlaku untuk sesi ini
        return {"ok": True, "on": on}

    @app.get("/api/ping")
    async def ping():
        return {"app": "sahabat"}

    @app.get("/api/riwayat")
    async def history():
        riwayat = getattr(engine, "riwayat", None)
        if riwayat is None or riwayat.disabled:
            return {"ok": False, "kejadian": [], "simpan_klip": False, "hari": 0}
        items = await run_in_threadpool(_list_riwayat, riwayat)
        return {"ok": True, "kejadian": items, "simpan_klip": bool(riwayat.keep_clips),
                "hari": riwayat.keep_days, "sebelum": riwayat.before_s, "sesudah": riwayat.after_s}

    @app.get("/api/riwayat/{record_id}/klip")
    async def history_clip(record_id: str):
        """Klip bukti diputar sekali dengan kecepatan aslinya, lewat MJPEG seperti video langsung."""
        riwayat = getattr(engine, "riwayat", None)
        frames = await run_in_threadpool(riwayat.clip, record_id) if riwayat is not None else None
        if not frames:
            return JSONResponse({"ok": False, "error": "Klip tidak ada."}, status_code=404)

        async def play():
            loop = asyncio.get_running_loop()
            start = loop.time()
            for t, jpeg in frames:
                delay = start + t - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                yield mjpeg_chunk(jpeg)
        return StreamingResponse(play(), media_type="multipart/x-mixed-replace; boundary=frame", headers=NO_CACHE)

    @app.post("/api/video")
    async def upload_video(request: Request, nama: str = ""):
        """Menyalin video cadangan ke folder video/. Isi berkas dikirim apa adanya sebagai badan permintaan."""
        folder = getattr(inputs, "video_dir", None)
        if folder is None:
            return JSONResponse({"ok": False, "error": "Pemilih video tidak tersedia."}, status_code=503)
        name = video_name(nama)
        if name is None:
            return JSONResponse({"ok": False, "error": "Pilih berkas video (mp4, avi, mkv, mov, wmv, webm)."},
                                status_code=400)
        part = None
        try:
            folder.mkdir(parents=True, exist_ok=True)
            target = _free_name(folder, name)
            part = target.with_name(target.name + ".part")
            size = 0
            with open(part, "wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_VIDEO_BYTES:
                        raise ValueError("Video terlalu besar (paling besar 4 GB).")
                    f.write(chunk)
            if size == 0:
                raise ValueError("Berkas video kosong.")
            part.replace(target)
        except ValueError as exc:
            if part is not None:
                part.unlink(missing_ok=True)
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        except OSError as exc:
            if part is not None:
                part.unlink(missing_ok=True)
            return JSONResponse({"ok": False, "error": "Video tidak bisa disimpan: " + str(exc)}, status_code=503)
        return {"ok": True, "id": "video:" + target.name, "nama": target.name}

    @app.post("/api/video/folder")
    async def video_folder():
        folder = getattr(inputs, "video_dir", None)
        if folder is None:
            return JSONResponse({"ok": False, "error": "Pemilih video tidak tersedia."}, status_code=503)
        try:
            await run_in_threadpool(open_folder, folder)
        except (OSError, AttributeError) as exc:
            return JSONResponse({"ok": False, "error": "Folder tidak bisa dibuka: " + str(exc)}, status_code=503)
        return {"ok": True}

    @app.post("/api/zona")
    async def set_zone(request: Request):
        """Zona pantau dari layar: {"titik": [[x, y], ...]} ternormalisasi, atau {"titik": null} = seluruh gambar."""
        if not hasattr(engine, "set_zone"):
            return JSONResponse({"ok": False, "error": "Zona tidak tersedia."}, status_code=503)
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or "titik" not in body:
            return JSONResponse({"ok": False, "error": "Kirim titik zona."}, status_code=400)
        try:
            zone = await run_in_threadpool(engine.set_zone, body["titik"])
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        try:
            await run_in_threadpool(remember_zone, zone)
        except OSError:
            pass                                            # tetap berlaku untuk sesi ini
        return {"ok": True, "titik": [list(p) for p in zone] if zone else None}

    @app.post("/api/riwayat/klip")
    async def keep_clips(request: Request):
        """Tombol hidup/mati penyimpanan klip bukti; pilihannya diingat per laptop."""
        riwayat = getattr(engine, "riwayat", None)
        try:
            body = await request.json()
        except ValueError:
            body = None
        on = body.get("on") if isinstance(body, dict) else None
        if riwayat is None or riwayat.disabled or not isinstance(on, bool):
            return {"ok": False}
        riwayat.set_keep_clips(on)
        try:
            await run_in_threadpool(remember_clips, on)
        except OSError:
            pass                                            # tetap berlaku untuk sesi ini
        return {"ok": True, "on": on}

    @app.get("/api/perangkat")
    async def devices():
        if inputs is None:
            return JSONResponse({"ok": False, "error": "Pemilih perangkat tidak tersedia."}, status_code=503)
        return await run_in_threadpool(inputs.scan)

    @app.post("/api/perangkat")
    async def select_devices(request: Request):
        if inputs is None:
            return JSONResponse({"ok": False, "error": "Pemilih perangkat tidak tersedia."}, status_code=503)
        try:
            values = await request.json()
            status = await run_in_threadpool(inputs.apply, values)
            return {"ok": True, **status}
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": "Pilihan belum diterapkan: " + str(exc)}, status_code=503)

    return app


def build_engine(cfg: Config, source=None, record: bool = False) -> Engine:
    """Merangkai mesin nyata: webcam (atau file), YOLO-pose, log, dan model bila ada.

    record=True menyalakan perekam uji (data/rekaman/); bawaannya tidak ada video yang disimpan.
    """
    from .pose import YoloPose
    from .predlog import PredLog
    from .source import VideoSource

    if source is None:
        source = VideoSource(cfg.camera, flip=cfg.camera_flip)
        source.start()
    pose = YoloPose(cfg.imgsz or cfg.imgsz_candidates[-1])
    from .riwayat import Riwayat

    recorder = None
    if record:
        from .recorder import Recorder
        recorder = Recorder(DATA_DIR / "rekaman")
    riwayat = Riwayat(DATA_DIR / "riwayat", keep_clips=cfg.clips_enabled, before_s=cfg.clip_before_s,
                      after_s=cfg.clip_after_s, keep_days=cfg.clip_keep_days)
    return Engine(cfg, source, pose, PredLog(DATA_DIR / "log"), cfg.slow, VIOLENCE_MODEL, FALL_MODEL, recorder,
                  riwayat)


def build_listener(cfg: Config, clip_dir: Path | None = None, on_toggle=None) -> Listener:
    """Merangkai pendengar nyata: mikrofon bawaan, Whisper base dari models/, log kata di data/log.

    clip_dir hanya diisi di mode rekam uji; di mode biasa tidak ada suara atau teks ucapan yang disimpan.
    """
    from .mic import open_mic
    from .predlog import PredLog
    from .speech import MODEL_DIR, WhisperRecognizer

    def make_recognizer():
        return WhisperRecognizer(MODEL_DIR, cfg.speech_threads, cfg.speech_no_repeat_ngram,
                                 cfg.speech_repetition_penalty, encoder_min_s=cfg.speech_encoder_min_s)

    return Listener(cfg, WORDS_PATH, make_recognizer, open_mic,
                    PredLog(DATA_DIR / "log", prefix="kata"), clip_dir, on_toggle)


def dev_app() -> FastAPI:
    """Untuk pengembangan: uvicorn --factory sahabat.server:dev_app"""
    cfg = load(SETTINGS_PATH)
    engine = build_engine(cfg)
    engine.start()
    listener = None
    try:
        listener = build_listener(cfg)
        listener.start()
    except Exception as e:
        print(f"Peringatan: pendengar kata kasar tidak bisa dinyalakan ({e}); bagian kamera tetap jalan.")
    return create_app(engine, listener)
