"""Kalibrasi model kekerasan untuk profil "Dekat / lomba" dari rekaman di posisi kamera lomba.

    python -m latih.kalibrasi_dekat cek             periksa rekaman: lama, laju, jumlah tanda (cepat, tanpa melatih)
    python -m latih.kalibrasi_dekat latih           pindahkan rekaman ke D:, uji silang, simpan kandidat bila lolos gerbang
    python -m latih.kalibrasi_dekat pasang          pasang kandidat yang lolos ke models/
    python -m latih.kalibrasi_dekat lepas           kembali ke model biasa (model kalibrasi disimpan sebagai cadangan)
    python -m latih.kalibrasi_dekat hapus-video --ya  hapus video dan suara rekaman; titik tubuh dan tanda tetap ada

Rekaman dibuat dengan "Mulai SAHABAT (rekam uji).bat" (adegan, tanda K/B) dan "Mulai SAHABAT (rekam kegiatan biasa).bat",
dengan Posisi kamera "Dekat / lomba". Tiap ulangan dinilai model yang tidak melihatnya (3 lipatan). Angkanya tetap optimistis
karena orang, ruangan, dan hari sama, jadi model ini hanya berlaku untuk posisi kamera saat direkam.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from sahabat.config import DATA_DIR, MODELS_DIR
from sahabat.profil_kamera import CONFIRM_DEKAT_S, VIOLENCE_DEKAT_NAME
from sahabat.recorder import read_marks

LOMBA = Path("D:/sahabat-data/lomba")
KANDIDAT = LOMBA / "kandidat"
LAPORAN = Path(__file__).resolve().parent.parent / "docs" / "uji"
LIPATAN = 3
AMBANG = (0.8, 0.85, 0.9, 0.95)
BOBOT = 10.0
GRACE_S = 3.0
JARAK_TANDA_S = 4.0             # tombol yang tertekan berulang dalam selang ini = satu ulangan
# Gerbang pemasangan
MIN_POSITIF, MIN_NEGATIF, MIN_MENIT_BIASA = 20, 10, 5.0
MIN_TANGKAP = 0.70
MAKS_KLIP_KELIRU = 0.20
MAKS_KELIRU_PER_JAM = 34.1      # alarm keliru kekerasan aplikasi pada kamera 3–4 m
LAJU_MINIMAL = 8.0


def gabung_tanda(marks: list[tuple[float, str]], jarak_s: float = JARAK_TANDA_S) -> list[tuple[float, str]]:
    """Tombol yang sama ditekan berulang setelah satu adegan dihitung satu ulangan, memakai tekanan pertama."""
    out, last = [], {}
    for t, label in sorted(marks):
        repeated = label in last and t - last[label] < jarak_s
        last[label] = t
        if not repeated:
            out.append((t, label))
    return out


def alasan_gagal(sekarang: dict, kandidat: dict) -> list[str]:
    """Kosong = lolos. Model hanya dipasang bila datanya cukup, lebih baik dari sekarang, dan alarm kelirunya wajar."""
    out = []
    if kandidat["positif"] < MIN_POSITIF:
        out.append(f"ulangan kekerasan baru {kandidat['positif']}, minimal {MIN_POSITIF}")
    if kandidat["negatif"] < MIN_NEGATIF:
        out.append(f"ulangan biasa baru {kandidat['negatif']}, minimal {MIN_NEGATIF}")
    if kandidat["menit_biasa"] < MIN_MENIT_BIASA:
        out.append(f"rekaman kegiatan biasa baru {kandidat['menit_biasa']:.1f} menit, minimal {MIN_MENIT_BIASA:g}")
    if kandidat["tertangkap"] <= sekarang["tertangkap"]:
        out.append(f"tidak lebih baik dari model sekarang ({kandidat['tertangkap']} vs {sekarang['tertangkap']})")
    if kandidat["positif"] and kandidat["tertangkap"] / kandidat["positif"] < MIN_TANGKAP:
        out.append(f"tertangkap {kandidat['tertangkap']}/{kandidat['positif']}, minimal {MIN_TANGKAP:.0%}")
    if kandidat["negatif"] and kandidat["keliru_klip"] / kandidat["negatif"] > MAKS_KLIP_KELIRU:
        out.append(f"adegan biasa beralarm {kandidat['keliru_klip']}/{kandidat['negatif']}, maksimal {MAKS_KLIP_KELIRU:.0%}")
    if kandidat["keliru_per_jam"] > MAKS_KELIRU_PER_JAM:
        out.append(f"alarm keliru {kandidat['keliru_per_jam']} per jam, maksimal {MAKS_KELIRU_PER_JAM}")
    return out


def pilih_ambang(sekarang: dict, per_ambang: dict[float, dict]) -> float | None:
    """Ambang yang lolos gerbang dengan tangkapan terbanyak; bila seri, alarm keliru tersedikit lalu ambang tertinggi."""
    ok = [th for th, m in per_ambang.items() if not alasan_gagal(sekarang, m)]
    return max(ok, key=lambda th: (per_ambang[th]["tertangkap"], -per_ambang[th]["keliru_per_jam"], th), default=None)


# ---- berkas rekaman -----------------------------------------------------------------------

def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _meta(folder: Path) -> dict:
    try:
        return json.loads((Path(folder) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def rekaman_dekat(root: Path) -> list[Path]:
    """Folder rekaman yang dibuat dengan Posisi kamera "Dekat / lomba"."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and _meta(p).get("posisi") == "dekat")


def pindahkan(folder: Path, root: Path) -> Path:
    """Salin ke root, cocokkan isi tiap berkas, baru hapus aslinya. Rekaman berisi wajah tidak boleh tertinggal di C:."""
    folder, root = Path(folder), Path(root)
    target = root / folder.name
    if target.exists():
        raise FileExistsError(f"{target} sudah ada")
    part = root / (folder.name + ".part")
    if part.exists():
        shutil.rmtree(part)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(folder, part)
    for source in folder.rglob("*"):
        if source.is_file() and _sha(source) != _sha(part / source.relative_to(folder)):
            shutil.rmtree(part)
            raise OSError(f"Salinan {source.name} tidak sama; rekaman asli tidak dihapus.")
    part.rename(target)
    shutil.rmtree(folder)
    return target


def _aplikasi_menyala(port: int = 8765) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def ringkas_rekaman(folder: Path) -> dict:
    """Lama, laju, dan jumlah tanda dari cap waktu; tidak menjalankan model."""
    times = []
    for f in sorted(Path(folder).glob("[0-9]*.txt")):
        times += [float(x) for x in f.read_text(encoding="utf-8").split()]
    t = np.array(times)
    gaps = np.diff(t) if len(t) > 1 else np.zeros(0)
    moving = gaps[(gaps > 0) & (gaps < 1.0)]
    marks = gabung_tanda(read_marks(folder))
    return {"nama": Path(folder).name, "jenis": _meta(folder).get("jenis", "adegan"),
            "menit": round(float(t[-1] - t[0]) / 60.0, 1) if len(t) > 1 else 0.0,
            "laju": round(1.0 / float(moving.mean()), 1) if len(moving) else 0.0,
            "kekerasan": sum(label == "kekerasan" for _, label in marks),
            "biasa": sum(label == "biasa" for _, label in marks)}


def cek(folders: list[Path]) -> int:
    if not folders:
        print('Belum ada rekaman dengan Posisi kamera "Dekat / lomba".')
        return 1
    rows = [ringkas_rekaman(f) for f in folders]
    for r in rows:
        print(f"{r['nama']} ({r['jenis']}): {r['menit']} menit, {r['laju']} gambar/detik, "
              f"tanda K {r['kekerasan']}, tanda B {r['biasa']}")
        if r["laju"] < LAJU_MINIMAL:
            print(f"  Peringatan: laju di bawah {LAJU_MINIMAL:g}. Matikan Kata kasar dan tutup aplikasi lain, lalu rekam ulang.")
    k, b = sum(r["kekerasan"] for r in rows), sum(r["biasa"] for r in rows)
    menit = sum(r["menit"] for r in rows if r["jenis"] == "biasa")
    kurang = []
    if k < MIN_POSITIF:
        kurang.append(f"{MIN_POSITIF - k} ulangan kekerasan")
    if b < MIN_NEGATIF:
        kurang.append(f"{MIN_NEGATIF - b} ulangan biasa")
    if menit < MIN_MENIT_BIASA:
        kurang.append(f"{MIN_MENIT_BIASA - menit:.1f} menit kegiatan biasa")
    print("Data cukup untuk dilatih." if not kurang else "Masih kurang: " + ", ".join(kurang) + ".")
    return 0 if not kurang else 1


# ---- pelatihan ----------------------------------------------------------------------------

@dataclass
class Sesi:
    nama: str
    biasa: bool                 # rekaman kegiatan biasa
    replays: dict               # langkah -> Replay (1 = laju asli, 2 = setengahnya)
    rentang: list[dict]         # {"t0", "t1", "kekerasan", "lipatan"} dalam detik jejak


def muat_sesi(folder: Path, cfg) -> Sesi:
    from .kamera_tinggi import label_dari_tanda
    from .latih_kekerasan_baru import prepare
    from .putar import load_or_trace
    tr = load_or_trace(Path(folder), 416, cfg.rate_hz)
    jenis = _meta(folder).get("jenis", "adegan")
    t0, durasi = float(tr.t[0]), float(tr.t[-1]) - float(tr.t[0])
    ranges = label_dari_tanda(gabung_tanda(read_marks(folder)), jenis, t0, durasi)
    rentang = [{"t0": t0 + r["mulai"], "t1": t0 + r["selesai"], "kekerasan": r["kekerasan"],
                "lipatan": r["grup"] % LIPATAN} for r in ranges]
    replays = {}
    for step in (1, 2):
        frames = list(zip(map(float, np.asarray(tr.t)[::step]), list(tr.people)[::step]))
        replays[step] = prepare(Path(folder).name, "lomba", False, 0, "sesi", frames, (tr.frame_h, tr.frame_w), cfg)
    return Sesi(Path(folder).name, jenis == "biasa", replays, rentang)


def hitung(sesi: list[Sesi], events_of) -> dict:
    """Hitungan per ulangan. events_of(sesi, lipatan) = alarm dari model yang tidak melihat lipatan itu."""
    tp = pos = fp = neg = 0
    alarm_adegan = alarm_biasa = 0
    detik_adegan = detik_biasa = 0.0
    for s in sesi:
        for r in s.rentang:
            evs = events_of(s, r["lipatan"])
            if r["kekerasan"]:
                pos += 1
                tp += any(r["t0"] <= e["t"] <= r["t1"] + GRACE_S for e in evs)
                continue
            inside = sum(r["t0"] <= e["t"] <= r["t1"] for e in evs)
            if s.biasa:
                alarm_biasa += inside
                detik_biasa += r["t1"] - r["t0"]
            else:
                neg += 1
                fp += bool(inside)
                alarm_adegan += inside
                detik_adegan += r["t1"] - r["t0"]
    alarm, detik = (alarm_biasa, detik_biasa) if detik_biasa > 0 else (alarm_adegan, detik_adegan)
    return {"tertangkap": tp, "positif": pos, "keliru_klip": fp, "negatif": neg,
            "keliru_per_jam": round(alarm / (detik / 3600.0), 1) if detik > 0 else 0.0,
            "menit_biasa": round(detik_biasa / 60.0, 1)}


def _baris(sesi: list[Sesi], keep) -> list[tuple]:
    from .latih_dekat import session_rows
    parts = []
    for s in sesi:
        segs = [r for r in s.rentang if keep(r)]
        if segs:
            for replay in s.replays.values():
                parts.append(session_rows(replay, segs, BOBOT))
    return parts


def uji_silang(sesi: list[Sesi], cfg, base, installed) -> dict:
    """Tangkapan model sekarang dan kandidat per ambang, keduanya dengan konfirmasi berbasis waktu."""
    from .latih_dekat import events, probs
    from .violence_train import fit
    timed = replace(cfg, confirm_window_s=CONFIRM_DEKAT_S)
    now = {s.nama: events(s.replays[1], probs(installed, s.replays[1], cfg, masked=False), timed) for s in sesi}
    sekarang = hitung(sesi, lambda s, _k: now[s.nama])
    print("model sekarang:", sekarang, flush=True)
    oof = {}
    for k in range(LIPATAN):
        parts = [base] + _baris(sesi, lambda r: r["lipatan"] != k)
        x, y, w = [np.concatenate(p) for p in zip(*parts, strict=True)]
        model = fit(x, y, w)
        for s in sesi:
            oof[s.nama, k] = probs(model, s.replays[1], cfg)
        print(f"lipatan {k + 1}/{LIPATAN} selesai", flush=True)
    per_ambang = {}
    for th in AMBANG:
        c = replace(timed, violence_threshold=th)
        got = {(s.nama, k): events(s.replays[1], oof[s.nama, k], c) for s in sesi for k in range(LIPATAN)}
        per_ambang[th] = hitung(sesi, lambda s, k: got[s.nama, k])
        print(f"ambang {th:g}:", per_ambang[th], flush=True)
    return {"sekarang": sekarang, "per_ambang": per_ambang}


def laporan(hasil: dict, rows: list[dict]) -> str:
    lines = [f"# Kalibrasi model kamera lomba, {time.strftime('%Y-%m-%d')}", "",
             "Dibuat oleh `python -m latih.kalibrasi_dekat latih`. Tiap ulangan dinilai model yang tidak melihatnya.", "",
             "| Rekaman | Jenis | Menit | Gambar/detik | Tanda K | Tanda B |", "|---|---|---:|---:|---:|---:|"]
    lines += [f"| {r['nama']} | {r['jenis']} | {r['menit']} | {r['laju']} | {r['kekerasan']} | {r['biasa']} |" for r in rows]
    lines += ["", "| Model | Tertangkap | Adegan biasa beralarm | Alarm keliru per jam |", "|---|---:|---:|---:|"]
    def row(name, m):
        return f"| {name} | {m['tertangkap']}/{m['positif']} | {m['keliru_klip']}/{m['negatif']} | {m['keliru_per_jam']} |"
    lines.append(row("Sekarang", hasil["sekarang"]))
    lines += [row(f"Kandidat, ambang {float(th):g}", m) for th, m in hasil["per_ambang"].items()]
    if hasil["lolos"]:
        lines += ["", f"**Lolos gerbang** pada ambang {hasil['ambang']:g}. Dipasang hanya lewat perintah `pasang`."]
    else:
        lines += ["", "**Tidak lolos gerbang**, tidak ada model yang disimpan:"] + [f"- {x}" for x in hasil["alasan"]]
    lines += ["", "Angka ini dari orang, ruangan, dan hari yang sama, jadi cenderung optimistis. Model hanya berlaku untuk "
              "posisi kamera saat direkam."]
    return "\n".join(lines) + "\n"


def latih() -> int:
    if _aplikasi_menyala():
        print("Tutup dulu aplikasi SAHABAT supaya rekamannya selesai ditulis.")
        return 2
    if not Path("D:/").exists():
        print("Drive D: tidak ada. Rekaman berisi wajah harus disimpan di sana.")
        return 2
    for folder in rekaman_dekat(DATA_DIR / "rekaman"):
        print("Memindahkan ke D:", pindahkan(folder, LOMBA), flush=True)
    folders = rekaman_dekat(LOMBA)
    if cek(folders) != 0:
        return 2

    import joblib
    from .latih_dekat import P3, base_rows
    from .putar import load_config_snapshot
    from .violence_train import fit
    cfg = load_config_snapshot(P3 / "config-baseline.json")
    sesi = [muat_sesi(f, cfg) for f in folders]
    base = base_rows(cfg)
    hasil = uji_silang(sesi, cfg, base, joblib.load(MODELS_DIR / "violence_model.joblib"))
    ambang = pilih_ambang(hasil["sekarang"], hasil["per_ambang"])
    hasil["ambang"], hasil["lolos"] = ambang, ambang is not None
    KANDIDAT.mkdir(parents=True, exist_ok=True)
    target = KANDIDAT / VIOLENCE_DEKAT_NAME
    if ambang is None:
        best = max(hasil["per_ambang"].values(), key=lambda m: m["tertangkap"] - m["keliru_klip"])
        hasil["alasan"] = alasan_gagal(hasil["sekarang"], best)
        target.unlink(missing_ok=True)
    else:
        x, y, w = [np.concatenate(p) for p in zip(*([base] + _baris(sesi, lambda r: True)), strict=True)]
        model = fit(x, y, w)
        model.upper_only_ = True
        model.own_threshold_ = float(ambang)
        joblib.dump(model, target, compress=3)
        hasil["sha256"] = _sha(target)
    (KANDIDAT / "hasil.json").write_text(json.dumps(hasil, indent=1), encoding="utf-8")
    LAPORAN.mkdir(parents=True, exist_ok=True)
    report = LAPORAN / f"hasil-kalibrasi-lomba-{time.strftime('%Y%m%d')}.md"
    report.write_text(laporan(hasil, [ringkas_rekaman(f) for f in folders]), encoding="utf-8")
    print("Laporan:", report)
    print("LOLOS gerbang." if hasil["lolos"] else "TIDAK LOLOS: " + "; ".join(hasil["alasan"]))
    return 0 if hasil["lolos"] else 3


# ---- pasang, lepas, hapus -------------------------------------------------------------------

def pasang(kandidat: Path = KANDIDAT, models: Path = MODELS_DIR) -> bool:
    try:
        hasil = json.loads((Path(kandidat) / "hasil.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("Belum ada hasil pelatihan.")
        return False
    source = Path(kandidat) / VIOLENCE_DEKAT_NAME
    if not hasil.get("lolos") or not source.is_file():
        print("Tidak dipasang: kandidat tidak lolos gerbang.")
        return False
    if _sha(source) != hasil.get("sha256"):
        print("Tidak dipasang: berkas kandidat berbeda dari yang diuji.")
        return False
    target = Path(models) / VIOLENCE_DEKAT_NAME
    if target.is_file():
        shutil.copy2(target, Path(kandidat) / f"sebelumnya-{time.strftime('%Y%m%d-%H%M%S')}.joblib")
    part = target.with_suffix(".part")
    shutil.copyfile(source, part)
    part.replace(target)
    print("Dipasang. Pilih Posisi kamera: Dekat / lomba di aplikasi.")
    return True


def lepas(kandidat: Path = KANDIDAT, models: Path = MODELS_DIR) -> bool:
    target = Path(models) / VIOLENCE_DEKAT_NAME
    if not target.is_file():
        print("Tidak ada model kalibrasi yang terpasang.")
        return False
    Path(kandidat).mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), Path(kandidat) / f"dilepas-{time.strftime('%Y%m%d-%H%M%S')}.joblib")
    print("Model kalibrasi dilepas. Profil Dekat / lomba kembali memakai model biasa.")
    return True


def hapus_video(root: Path = LOMBA) -> int:
    """Menghapus video dan suara dari rekaman yang titik tubuhnya sudah tersimpan."""
    count = 0
    for folder in rekaman_dekat(root):
        if not (folder / "jejak_416.npz").is_file():
            print(f"{folder.name}: dilewati, titik tubuhnya belum diambil.")
            continue
        for video in folder.glob("*.avi"):
            video.unlink()
            count += 1
        if (folder / "suara").is_dir():
            shutil.rmtree(folder / "suara")
    print(f"{count} potongan video dihapus.")
    return count


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tahap", choices=("cek", "latih", "pasang", "lepas", "hapus-video"))
    ap.add_argument("--ya", action="store_true", help="konfirmasi hapus-video")
    args = ap.parse_args(argv)
    if args.tahap == "cek":
        return cek(rekaman_dekat(DATA_DIR / "rekaman") + rekaman_dekat(LOMBA))
    if args.tahap == "latih":
        return latih()
    if args.tahap == "pasang":
        return 0 if pasang() else 1
    if args.tahap == "lepas":
        return 0 if lepas() else 1
    if not args.ya:
        print("Video tidak bisa dikembalikan. Jalankan lagi dengan --ya bila sudah yakin.")
        return 2
    hapus_video()
    return 0


if __name__ == "__main__":
    sys.exit(main())
