"""Seberapa turun akurasi bila satu kamera hanya dinilai 5 atau ±3,3 kali per detik (Mode CCTV saat komputer lambat)?

    python -m latih.laju_rendah        ukur model terpasang + semua aturan pada 10 / 5 / 3,3 Hz; tulis hasil.json

Hanya laporan: model, ambang, dan aturan tidak diubah. Laju rendah ditiru dengan mengambil setiap tick ke-n dari jejak
pose 10 Hz yang sudah ada (latih.putar.PoseTrace), lalu menjalankan penilai aplikasi pada jejak itu.

Data (sama dengan baris "aplikasi" eksperimen A/B/C, sejauh jejak pose mentahnya tersedia):
- jatuh: CAUCAFall subjek 9-10 dan Le2i (F1, latensi);
- kekerasan dan alarm keliru per jam: rekaman sesi 17 November 2026 (37 ulangan kekerasan, kegiatan biasa kontinu).
RLVS, AIRTLab, dan Bus Violence tidak diukur ulang: data ujinya tersimpan sebagai replay yang sudah dihitung pada 10 Hz.

Keterbatasan: nomor orang berasal dari pelacakan 10 Hz. Pada laju rendah sungguhan pelacak bisa lebih sering
kehilangan orang, jadi hasil sebenarnya bisa sedikit lebih buruk dari angka di sini.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np

from .putar import PoseTrace

OUT = Path(__file__).resolve().parent.parent / "docs" / "uji" / "laju-rendah"
STEPS = {"10 Hz": 1, "5 Hz": 2, "3,3 Hz": 3}


def subsample(trace: PoseTrace, step: int) -> PoseTrace:
    """Setiap tick ke-step dari jejak pose: step 2 = 5 Hz, step 3 = ±3,3 Hz dari jejak 10 Hz."""
    return PoseTrace(np.asarray(trace.t)[::step], trace.frame_h, trace.frame_w, list(trace.people)[::step])


def measure_rate(step: int) -> dict:
    from . import eksperimen_abc as abc
    from .latih_jatuh_baru import measure as app_fall_measure

    cfg = abc._cfg()
    installed_fall = joblib.load(abc.INSTALLED / "fall_model.joblib")
    installed_viol = joblib.load(abc.INSTALLED / "violence_model.joblib")
    original_load = PoseTrace.load.__func__
    original_out = abc.OUT
    PoseTrace.load = classmethod(lambda cls, path: subsample(original_load(cls, path), step))
    try:
        # ---- jatuh (CAUCAFall 9-10 + Le2i), model terpasang + aturan jatuh ----
        clips = abc.fall_test_clips(cfg)
        report = app_fall_measure(clips, cfg, {"aplikasi": installed_fall}, {"aplikasi": 0.5})["aplikasi"]
        rows = report["rows"]
        tp = sum(r["delay_s"] is not None for r in rows if r["fall"])
        fn = sum(r["delay_s"] is None for r in rows if r["fall"])
        fp = sum(bool(r["events"]) for r in rows if not r["fall"]) + sum(r["early"] for r in rows)
        delays = [r["delay_s"] for r in rows if r["delay_s"] is not None]
        fall = abc._prf(tp, fn, fp)
        fall["latensi_median"] = round(float(np.median(delays)), 2) if delays else None

        # ---- kekerasan: sesi 17 November, replay dihitung ulang dari jejak yang di-subsample ----
        with tempfile.TemporaryDirectory() as tmp:
            abc.OUT = Path(tmp)                         # jangan memakai cache replay 10 Hz
            s = abc.session(cfg)
        replay = s["replay"]
        probs = abc._violence_probs(installed_viol, [replay], cfg, True)[0]
        events = abc.violence_events(replay, probs, cfg, True, ack_after_s=3.0)
        k_hit = k_n = b_hit = b_n = 0
        for a, b, violent in s["takes"]:
            if a < s["start"]:
                continue
            got = any(a <= t <= b for t in events)
            if violent:
                k_n, k_hit = k_n + 1, k_hit + got
            else:
                b_n, b_hit = b_n + 1, b_hit + got
        gap = [t for t in events if abc._in_normal(s, t) and not any(a <= t <= b for a, b, v in s["takes"] if not v)]
        normal_alarms = [t for t in events if abc._in_normal(s, t)]
        violence = abc._prf(k_hit, k_n - k_hit, b_hit + len(gap))
        violence.update({"kekerasan_tertangkap": f"{k_hit}/{k_n}", "ulangan_biasa_beralarm": f"{b_hit}/{b_n}",
                         "alarm_keliru_per_jam": round(len(normal_alarms) / s["normal_hours"], 2)})
    finally:
        PoseTrace.load = classmethod(original_load)
        abc.OUT = original_out
    return {"jatuh": fall, "kekerasan_sesi17": violence}


def main() -> int:
    result = {}
    for name, step in STEPS.items():
        result[name] = measure_rate(step)
        print(name, json.dumps(result[name], ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hasil.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"{'laju':8} {'F1 jatuh':>9} {'lat. jatuh':>11} {'F1 kekerasan':>13} {'tertangkap':>11} {'keliru/jam':>11}")
    for name, r in result.items():
        f, v = r["jatuh"], r["kekerasan_sesi17"]
        print(f"{name:8} {f['f1']:>9} {str(f['latensi_median']):>11} {v['f1']:>13} {v['kekerasan_tertangkap']:>11} "
              f"{v['alarm_keliru_per_jam']:>11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
