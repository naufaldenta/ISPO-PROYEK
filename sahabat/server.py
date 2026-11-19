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
from .cctv_uji import uji_sambungan
from .jaringan import COOKIE, allow_firewall, firewall_status, is_local, qr_png
from .kamera import IPCAM_PREFIX, default_store
from .merek import MEREK
from .onvif import OnvifError
from .onvif import discover as onvif_discover
from .onvif import stream_uri as onvif_stream_uri
from .listener import STATUS_LABEL, Listener

WEB_DIR = Path(__file__).resolve().parent / "web"
VIDEO_GAP_S = 0.05        # paling cepat 20 frame per detik
REMOTE_VIDEO_GAP_S = 0.1  # layar jarak jauh: paling cepat 10 frame per detik, hemat WiFi
TILE_VIDEO_GAP_S = 0.2    # ubin grid Mode CCTV: 5 frame per detik, gambar kecil
LAST_CAMERA = "Ini kamera aktif terakhir. Aktifkan kamera lain atau matikan Mode CCTV dulu."
STATE_GAP_S = 0.2
VIOLENCE_MODEL = MODELS_DIR / "violence_model.joblib"
FALL_MODEL = MODELS_DIR / "fall_model.joblib"
NO_CACHE = {"Cache-Control": "no-store"}
MARK_LABELS = ("jatuh", "kekerasan", "biasa")      # tanda adegan saat merekam uji
MAX_VIDEO_BYTES = 4 * 1024 ** 3                    # video cadangan yang diunggah dari layar
_BAD_NAME_CHARS = set(r'<>:"/\|?*')
# Nama host yang sah untuk permintaan dari laptop sendiri. Situs lain yang terbuka di browser laptop bisa memakai
# DNS rebinding (nama situsnya menunjuk ke 127.0.0.1); Host-nya tetap nama situs itu, jadi ditolak.
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "testserver"}


def _host_name(host_header: str) -> str:
    host = (host_header or "").strip().lower()
    if host.startswith("["):
        return host[1:host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


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


def _video_gap(local: bool, small: bool) -> float:
    if small:
        return TILE_VIDEO_GAP_S
    return VIDEO_GAP_S if local else REMOTE_VIDEO_GAP_S


CLIP_MISSING = "Klip tidak ada atau sudah terhapus."
CLIP_SAVING = "Klip sedang disimpan, coba lagi sebentar lagi."


def _safe_file_part(name: str) -> str:
    """Bagian nama berkas yang aman untuk header unduhan: hanya huruf Latin, angka, dan tanda hubung."""
    cleaned = "".join(c if (c.isascii() and c.isalnum()) or c == "-" else "-" for c in name or "")[:40]
    return cleaned or "Kamera"


def _other_cameras(riwayat, item: dict) -> list[dict]:
    """Kejadian dari kamera lain yang klipnya ada dan terjadi dalam rentang klip ini (sebelum + sesudah)."""
    span = riwayat.before_s + riwayat.after_s
    out = []
    for other in riwayat.list(limit=10 ** 6):
        if other["id"] == item["id"] or other.get("kamera", "") == item.get("kamera", "") or not other["klip"]:
            continue
        diff = other["waktu"] - item["waktu"]
        if abs(diff) <= span:
            out.append({"id": other["id"], "kamera_nama": other.get("kamera_nama", ""), "jenis": other["jenis"],
                        "selisih_s": round(diff, 3)})
    return sorted(out, key=lambda x: abs(x["selisih_s"]))


def remember_cctv(on: bool) -> None:
    from .config import SETTINGS_PATH, update_local
    update_local(SETTINGS_PATH, cctv_enabled=bool(on))


def build_mode(cfg: Config, engine):
    """ModeControl nyata: model pose kedua untuk Mode CCTV dibuat sekali, saat Mode CCTV pertama kali dinyalakan."""
    from .cctv import CctvRunner, ModeControl
    from .pose import SharedPose, YoloPose
    shared: dict = {}

    def make_runner():
        if "pose" not in shared:
            shared["pose"] = SharedPose(YoloPose(cfg.imgsz or cfg.imgsz_candidates[-1]))
        return CctvRunner(cfg, default_store(), shared["pose"], getattr(engine, "riwayat", None))
    return ModeControl(engine, make_runner, remember_cctv)


def remember_flip(on: bool) -> None:
    from .config import SETTINGS_PATH, update_local
    update_local(SETTINGS_PATH, camera_flip=bool(on))


def video_name(name: str) -> str | None:
    """Nama berkas video yang aman untuk folder video/, atau None."""
    from .devices import safe_video_name
    return safe_video_name(name or "")


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


def _list_riwayat(riwayat, kamera: str | None = None) -> list[dict]:
    riwayat.prune()                     # aplikasi bisa menyala berhari-hari: bersihkan juga saat panel dibuka
    return riwayat.list() if kamera is None else riwayat.list(kamera=kamera)


def _local_only(method: str, path: str) -> bool:
    """Pengaturan yang hanya boleh diubah dari laptop SAHABAT sendiri, bukan dari HP atau laptop lain."""
    if path.startswith("/api/jaringan") or path.startswith("/api/kamera"):
        return True
    if method == "POST" and path == "/api/cctv":
        return True
    if path.startswith("/api/riwayat/") and ((method == "POST" and path.endswith("/tindak"))
                                             or (method == "GET" and path.endswith("/unduh"))):
        return True                         # tanda guru dan unduhan klip hanya dari laptop SAHABAT
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


def create_app(engine, listener=None, inputs=None, lan=None, kamera=None, mode=None) -> FastAPI:
    kamera = kamera if kamera is not None else default_store()      # daftar kamera CCTV (data/kamera.json)
    if inputs is None and getattr(engine, "source", None) is not None:
        from .input_control import InputDevices
        inputs = InputDevices(engine, listener, kamera=kamera)
    app = FastAPI(title="SAHABAT Sekolah", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", _FreshStaticFiles(directory=WEB_DIR), name="static")

    def remote_allowed(cookie: str | None) -> bool:
        return lan is not None and lan.enabled and lan.check(cookie)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        """Laptop SAHABAT sendiri bebas. Perangkat lain butuh jaringan lokal hidup dan kode akses yang benar."""
        host = request.client.host if request.client else None
        if is_local(host):
            if _host_name(request.headers.get("host", "")) not in LOCAL_HOSTS:
                return Response("Alamat tidak dikenal.", status_code=403, media_type="text/plain; charset=utf-8")
            # POST kamera hanya JSON: form atau text/plain dari situs lain tidak memicu pemeriksaan CORS browser
            if (request.method == "POST" and request.url.path.startswith("/api/kamera")
                    and not request.headers.get("content-type", "").lower().startswith("application/json")):
                return JSONResponse({"ok": False, "error": "Kirim sebagai JSON."}, status_code=415)
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

    def cctv_on() -> bool:
        return mode is not None and mode.hidup

    def last_active(cam_id: str) -> bool:
        """Mode CCTV tanpa kamera aktif tidak memantau apa pun, jadi kamera aktif terakhir tidak boleh dilepas."""
        return cctv_on() and [c["id"] for c in kamera.active()] == [cam_id]

    def with_cctv(state: dict) -> dict:
        """Mode CCTV: pita, FPS, jumlah orang, dan nomor alarm diambil dari semua kamera."""
        if mode is None:
            return state
        if not cctv_on():
            state["cctv"] = {"aktif": False}
            return state
        snap = mode.cctv.snapshot()
        state.update(bar=snap["bar"], text=snap["text"], fps=snap["fps"], people=snap["people"],
                     alarm_seq=snap["alarm_seq"], slow=snap["lambat"],
                     slow_text="Komputer tidak sanggup menilai semua kamera penuh" if snap["lambat"] else "")
        state["cctv"] = snap
        return state

    @app.get("/video/gambar")
    async def video_frame(kamera: str | None = None, kecil: int = 0):
        """Satu JPEG (ubin grid Mode CCTV). Ubin diminta ulang oleh layar tiap 0,2 dtk; tidak memakai stream, karena
        browser hanya membuka 6 sambungan per alamat dan stream ubin akan menahan tombol "Sudah diperiksa"."""
        if kamera and cctv_on():
            jpeg = await run_in_threadpool(mode.cctv.latest_jpeg, kamera, bool(kecil))
        else:
            jpeg = await run_in_threadpool(engine.latest_jpeg)
        if not jpeg:
            return Response(status_code=404, headers=NO_CACHE)
        return Response(jpeg, media_type="image/jpeg", headers=NO_CACHE)

    @app.get("/video")
    async def video(request: Request, kamera: str | None = None, kecil: int = 0):
        local = is_local(request.client.host if request.client else None)
        gap = _video_gap(local, bool(kecil) and bool(kamera))

        def grab():
            if kamera and cctv_on():
                return mode.cctv.latest_jpeg(kamera, small=bool(kecil))
            return engine.latest_jpeg()

        async def frames():
            while True:
                jpeg = await run_in_threadpool(grab)
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
                state = with_cctv(_state(engine, listener, inputs))
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

    @app.get("/api/jaringan/firewall")
    async def lan_firewall():
        """Apakah Windows Firewall mengizinkan HP masuk. Butuh beberapa detik, jadi terpisah dari /api/jaringan."""
        return {"ok": True, "firewall": await run_in_threadpool(firewall_status)}

    @app.post("/api/jaringan/firewall")
    async def lan_firewall_allow():
        """Memasang aturan izin firewall; Windows meminta persetujuan admin di laptop ini."""
        if not await run_in_threadpool(allow_firewall):
            return {"ok": False, "error": "Izin admin tidak diberikan. Coba lagi dan pilih Ya."}
        return {"ok": True}

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
    async def ack(request: Request):
        """Mode CCTV: {"kamera": "<id>"} atau {"kamera": "semua"}; tanpa badan = alarm yang sedang tampil."""
        target = None
        try:
            body = await request.json()
            if isinstance(body, dict) and isinstance(body.get("kamera"), str):
                target = body["kamera"]
        except ValueError:
            pass
        if cctv_on():
            if target == "semua":
                return {"ok": bool(await run_in_threadpool(mode.cctv.acknowledge_all))}
            return {"ok": bool(await run_in_threadpool(mode.cctv.acknowledge, target))}
        return {"ok": bool(await run_in_threadpool(engine.acknowledge))}

    @app.get("/api/cctv")
    async def cctv_status():
        if mode is None:
            return {"ok": True, "hidup": False, "pesan": "Mode CCTV tidak tersedia."}
        return {"ok": True, **await run_in_threadpool(mode.status)}

    @app.post("/api/cctv")
    async def cctv_switch(request: Request):
        if mode is None:
            return JSONResponse({"ok": False, "error": "Mode CCTV tidak tersedia."}, status_code=503)
        try:
            body = await request.json()
        except ValueError:
            body = None
        on = body.get("hidup") if isinstance(body, dict) else None
        if not isinstance(on, bool):
            return JSONResponse({"ok": False, "error": "Kirim hidup: true/false."}, status_code=400)
        try:
            return {"ok": True, **await run_in_threadpool(mode.set_cctv, on)}
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

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
    async def history(kamera: str | None = None):
        riwayat = getattr(engine, "riwayat", None)
        if riwayat is None or riwayat.disabled:
            return {"ok": False, "kejadian": [], "simpan_klip": False, "hari": 0}
        items = await run_in_threadpool(_list_riwayat, riwayat, kamera)
        summary = await run_in_threadpool(riwayat.ringkasan, kamera)
        return {"ok": True, "kejadian": items, "ringkasan": summary, "simpan_klip": bool(riwayat.keep_clips),
                "hari": riwayat.keep_days, "sebelum": riwayat.before_s, "sesudah": riwayat.after_s}

    def clip_target(record_id: str):
        """(riwayat, kejadian) atau JSONResponse 404/409 yang ramah."""
        riwayat = getattr(engine, "riwayat", None)
        item = riwayat.find(record_id) if riwayat is not None else None
        if item is None:
            return None, JSONResponse({"ok": False, "error": CLIP_MISSING}, status_code=404)
        if item.get("menyimpan"):
            return None, JSONResponse({"ok": False, "error": CLIP_SAVING}, status_code=409)
        return (riwayat, item), None

    @app.get("/api/riwayat/{record_id}/info")
    async def history_info(record_id: str):
        target, error = await run_in_threadpool(clip_target, record_id)
        if error is not None:
            return error
        riwayat, item = target
        info = await run_in_threadpool(riwayat.clip_info, record_id)
        if info is None:
            return JSONResponse({"ok": False, "error": CLIP_MISSING}, status_code=404)
        from .types import SKELETON_EDGES
        return {"ok": True, **info, "jumlah": len(info["waktu"]), "tulang": [list(e) for e in SKELETON_EDGES],
                "kejadian": item, "lain": await run_in_threadpool(_other_cameras, riwayat, item)}

    @app.get("/api/riwayat/{record_id}/frame/{n}")
    async def history_frame(record_id: str, n: int):
        riwayat = getattr(engine, "riwayat", None)
        jpeg = await run_in_threadpool(riwayat.clip_frame, record_id, n) if riwayat is not None else None
        if not jpeg:
            return JSONResponse({"ok": False, "error": CLIP_MISSING}, status_code=404)
        return Response(jpeg, media_type="image/jpeg", headers=NO_CACHE)

    @app.get("/api/riwayat/{record_id}/unduh")
    async def history_download(record_id: str):
        target, error = await run_in_threadpool(clip_target, record_id)
        if error is not None:
            return error
        riwayat, item = target
        frames = await run_in_threadpool(riwayat.clip, record_id)
        info = await run_in_threadpool(riwayat.clip_info, record_id)
        if not frames or info is None:
            return JSONResponse({"ok": False, "error": CLIP_MISSING}, status_code=404)
        import time as _time
        from . import klip_mp4
        name = item.get("kamera_nama") or "Kamera"
        stamp = _time.strftime("%d-%m-%Y %H.%M.%S", _time.localtime(item["waktu"]))
        caption = f"{name} - {stamp} - SAHABAT Sekolah"
        zone = (info.get("meta") or {}).get("zona")
        try:
            data = await run_in_threadpool(klip_mp4.render, frames, info["overlay"], zone, caption)
        except Exception as exc:                        # termasuk MemoryError: jawab jelas, bukan galat 500
            reason = str(exc) or type(exc).__name__
            return JSONResponse({"ok": False, "error": "MP4 belum bisa dibuat: " + reason}, status_code=503)
        filename = f"SAHABAT-{_safe_file_part(name)}-{record_id}.mp4"
        return Response(data, media_type="video/mp4",
                        headers={**NO_CACHE, "Content-Disposition": f'attachment; filename="{filename}"'})

    @app.post("/api/riwayat/{record_id}/tindak")
    async def history_mark(record_id: str, request: Request):
        """Tanda tindak lanjut guru: {"tanda": "benar"|"keliru"|"tindak"|"", "catatan": str}."""
        riwayat = getattr(engine, "riwayat", None)
        try:
            body = await request.json()
        except ValueError:
            body = None
        if riwayat is None or not isinstance(body, dict):
            return JSONResponse({"ok": False, "error": "Isian tidak valid."}, status_code=400)
        try:
            item = await run_in_threadpool(riwayat.set_tindak, record_id, body.get("tanda"), body.get("catatan"))
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        if item is None:
            return JSONResponse({"ok": False, "error": CLIP_MISSING}, status_code=404)
        return {"ok": True, "kejadian": item}

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
        """Zona pantau dari layar: {"titik": [[x, y], ...]} ternormalisasi, atau {"titik": null} = seluruh gambar.
        Mode CCTV: {"titik": ..., "kamera": "<id>"} untuk kamera yang sedang diperbesar."""
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or "titik" not in body:
            return JSONResponse({"ok": False, "error": "Kirim titik zona."}, status_code=400)
        if not (cctv_on() and isinstance(body.get("kamera"), str)) and not hasattr(engine, "set_zone"):
            return JSONResponse({"ok": False, "error": "Zona tidak tersedia."}, status_code=503)
        if cctv_on() and isinstance(body.get("kamera"), str):
            try:
                zone = await run_in_threadpool(mode.cctv.set_zone, body["kamera"], body["titik"])
                points = [list(p) for p in zone] if zone else None
                await run_in_threadpool(kamera.set_zone, body["kamera"], points)
            except ValueError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
            except OSError:
                points = [list(p) for p in zone] if zone else None
            return {"ok": True, "titik": points, "kamera": body["kamera"]}
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

    # ---- kamera CCTV (hanya dari laptop SAHABAT; kata sandi tidak pernah dikirim balik) ----
    async def _body(request: Request) -> dict:
        try:
            body = await request.json()
        except ValueError:
            raise ValueError("Isian tidak valid.") from None
        if not isinstance(body, dict):
            raise ValueError("Isian tidak valid.")
        return body

    def _bad(exc: Exception, status: int = 400) -> JSONResponse:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=status)

    @app.get("/api/kamera")
    async def kamera_list():
        brands = [{"id": key, "nama": info["nama"], "port": info["port"], "channel": info["channel"]}
                  for key, info in MEREK.items()]
        return {"ok": True, "kamera": await run_in_threadpool(kamera.list_public), "merek": brands}

    @app.post("/api/kamera")
    async def kamera_save(request: Request):
        try:
            item = await run_in_threadpool(kamera.save, await _body(request))
        except ValueError as exc:
            return _bad(exc)
        except OSError as exc:
            return _bad(OSError("Daftar kamera tidak bisa disimpan: " + str(exc)), 503)
        source = getattr(engine, "source", None)
        if getattr(source, "spec", None) == IPCAM_PREFIX + item["id"] and hasattr(source, "reopen"):
            source.reopen()                 # kamera yang sedang dipakai langsung memakai alamat/sandi baru
        if cctv_on():
            await run_in_threadpool(mode.cctv.reload)
        return {"ok": True, "kamera": item}

    @app.post("/api/kamera/{cam_id}/tanda")
    async def kamera_flags(cam_id: str, request: Request):
        """Tanda Mode CCTV dari panel Kamera: {"aktif": bool?, "prioritas": bool?}."""
        try:
            body = await _body(request)
            aktif, prioritas = body.get("aktif"), body.get("prioritas")
            if not all(v is None or isinstance(v, bool) for v in (aktif, prioritas)):
                raise ValueError("Tanda kamera harus true/false.")
            if aktif is False and cctv_on() and mode.cctv.has_unacked(cam_id):
                return _bad(ValueError("Periksa alarm kamera ini dulu."), 409)
            if aktif is False and last_active(cam_id):
                return _bad(ValueError(LAST_CAMERA), 409)
            item = await run_in_threadpool(kamera.set_flags, cam_id, aktif, prioritas)
        except ValueError as exc:
            return _bad(exc)
        if cctv_on():
            await run_in_threadpool(mode.cctv.reload)
        return {"ok": True, "kamera": item}

    @app.delete("/api/kamera/{cam_id}")
    async def kamera_delete(cam_id: str):
        source = getattr(engine, "source", None)
        if getattr(source, "spec", None) == IPCAM_PREFIX + cam_id:
            return _bad(ValueError("Kamera ini sedang dipakai. Pilih kamera lain dulu di Perangkat."), 409)
        if cctv_on() and mode.cctv.has_unacked(cam_id):
            return _bad(ValueError("Periksa alarm kamera ini dulu."), 409)
        if last_active(cam_id):
            return _bad(ValueError(LAST_CAMERA), 409)
        if not await run_in_threadpool(kamera.delete, cam_id):
            return _bad(ValueError("Kamera tidak ditemukan."), 404)
        if cctv_on():
            await run_in_threadpool(mode.cctv.reload)
        return {"ok": True}

    @app.post("/api/kamera/uji")
    async def kamera_test(request: Request):
        try:
            stream = kamera.form_stream(await _body(request))
        except ValueError as exc:
            return _bad(exc)
        return await run_in_threadpool(uji_sambungan, stream)

    @app.post("/api/kamera/cari")
    async def kamera_search():
        return {"ok": True, "ditemukan": await run_in_threadpool(onvif_discover)}

    @app.post("/api/kamera/onvif")
    async def kamera_onvif(request: Request):
        try:
            body = await _body(request)
            xaddr = body.get("xaddr")
            if not isinstance(xaddr, str) or not xaddr.lower().startswith(("http://", "https://")):
                raise ValueError("Alamat ONVIF kamera tidak valid.")
            user, password = body.get("pengguna") or "", body.get("sandi") or ""
            if not isinstance(user, str) or not isinstance(password, str):
                raise ValueError("Nama pengguna atau kata sandi tidak valid.")
            uri = await run_in_threadpool(onvif_stream_uri, xaddr, user, password,
                                          substream=bool(body.get("substream", True)))
        except (ValueError, OnvifError) as exc:
            return _bad(exc)
        return {"ok": True, "alamat": uri}

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
        source = VideoSource(cfg.camera, flip=cfg.camera_flip, resolve=default_store().resolve)
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
    return create_app(engine, listener, mode=build_mode(cfg, engine))
