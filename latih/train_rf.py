"""Melatih penilai kekerasan (Random Forest) dari titik tubuh hasil extract_pose.

    python -m latih.train_rf <folder_pose> [violent-action-classes.csv] [motion_min]

Label dari nama folder: yang diawali "non" = bukan kekerasan, yang memuat "violent"/"fight" = kekerasan.

Bila daftar aksi per klip diberikan (bawaan dataset AIRTLab), klip kekerasan yang tidak memuat
pukulan, tendangan, dorongan, tamparan, perkelahian, atau cekikan dilewati: tongkat, tusukan, dan
tembakan berada di luar cakupan sistem ini.

Yang dilaporkan adalah angka yang penting bagi pengguna: berapa klip yang memunculkan ALARM, dengan
aturan konfirmasi yang sama seperti aplikasi. Angka itu diukur dengan validasi silang per adegan:
tiap klip dinilai oleh model yang tidak pernah melihat adegan itu (termasuk dari kamera kedua).
Model yang disimpan dilatih dari semua klip.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path, PurePosixPath

import numpy as np

from sahabat.alarm import AlarmManager
from sahabat.config import MODELS_DIR, Config
from sahabat.pair_features import FEATURE_NAMES
from sahabat.violence import PairScore

from .windows import ClipPairs, clip_pairs, training_rows

MODEL_PATH = MODELS_DIR / "violence_rf.joblib"
REPORT_PATH = MODELS_DIR / "violence_rf.laporan.txt"
TARGET_ACTIONS = {"punch", "kick", "push", "slap", "fight", "choke"}
N_FOLDS = 5
THRESHOLDS = (0.7, 0.8, 0.85, 0.9)
CONFIRMS = ((4, 6), (6, 10))


def _parts(rel_path: str) -> list[str]:
    return [p.lower() for p in PurePosixPath(str(rel_path).replace("\\", "/")).parts]


def label_of(rel_path: str) -> int | None:
    folders = _parts(rel_path)[:-1]
    if any(p.startswith("non") for p in folders):
        return 0
    if any("violent" in p or "fight" in p for p in folders):
        return 1
    return None


def group_of(rel_path: str) -> str:
    """Satu adegan = label + nama berkas; folder kamera (cam1/cam2) diabaikan."""
    parts = _parts(rel_path)
    folders = [p for p in parts[:-1] if not p.startswith("cam")]
    return "/".join(folders + [PurePosixPath(parts[-1]).stem])


def load_actions(csv: Path) -> dict[str, set[str]]:
    """{nama klip tanpa ekstensi: {aksi, ...}} dari berkas 'FILE; ACTION CLASSES'."""
    out = {}
    for line in Path(csv).read_text(encoding="utf-8").splitlines()[1:]:
        name, _, actions = line.partition(";")
        if name.strip():
            out[PurePosixPath(name.strip()).stem] = {a.strip().lower() for a in actions.split(",") if a.strip()}
    return out


def keep_clip(label: int, stem: str, actions: dict[str, set[str]] | None) -> bool:
    if label == 0 or actions is None:
        return True
    return bool(actions.get(stem, set()) & TARGET_ACTIONS)


def folds_by_clip(groups: np.ndarray, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Lipatan (indeks latih, indeks uji); satu adegan tidak pernah terbagi dua."""
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=n_folds).split(np.zeros(len(groups)), groups=groups))


def clip_alarm(clip: ClipPairs, probs: np.ndarray, cfg: Config) -> bool:
    """Apakah klip ini memunculkan alarm kekerasan, dengan konfirmasi yang sama seperti aplikasi.

    probs: satu peluang per baris clip.rows (sudah nilai tertinggi dari dua arah).
    """
    manager = AlarmManager(cfg)
    by_frame: dict[int, list[PairScore]] = {}
    for (frame, a, b, _, _), p in zip(clip.rows, probs):
        by_frame.setdefault(frame, []).append(PairScore((a, b), float(p)))
    return any(manager.tick(t, [], by_frame.get(i, []), {}) for i, t in enumerate(clip.t))


def _fit(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                                  random_state=0, n_jobs=-1).fit(X, y)


def _pair_probs(model, clips: list[ClipPairs]) -> list[np.ndarray]:
    """Peluang per baris untuk tiap klip; model dipanggil sekali untuk semuanya."""
    ab = [r[3] for c in clips for r in c.rows]
    ba = [r[4] for c in clips for r in c.rows]
    if not ab:
        return [np.zeros(0) for _ in clips]
    p = np.maximum(model.predict_proba(np.array(ab))[:, 1], model.predict_proba(np.array(ba))[:, 1])
    out, k = [], 0
    for c in clips:
        out.append(p[k:k + len(c.rows)])
        k += len(c.rows)
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    import joblib

    src = Path(argv[0])
    actions = load_actions(Path(argv[1])) if len(argv) > 1 else None
    motion_min = float(argv[2]) if len(argv) > 2 else 1.0
    cfg = Config()

    clips, labels, group_names, skipped = [], [], [], 0
    for npz in sorted(src.rglob("*.npz")):
        rel = npz.relative_to(src).as_posix()
        label = label_of(rel)
        if label is None:
            continue
        if not keep_clip(label, npz.stem, actions):
            skipped += 1
            continue
        clips.append(clip_pairs(npz, cfg))
        labels.append(label)
        group_names.append(group_of(rel))
    labels = np.array(labels)
    ids = {g: i for i, g in enumerate(sorted(set(group_names)))}
    groups = np.array([ids[g] for g in group_names])
    rows = [training_rows(c, int(lab), motion_min) for c, lab in zip(clips, labels)]

    def stack(idx):
        return np.concatenate([rows[i] for i in idx]), np.concatenate([np.full(len(rows[i]), labels[i]) for i in idx])

    # Validasi silang: peluang tiap klip berasal dari model yang tidak melihat adegannya.
    probs: list[np.ndarray] = [np.zeros(0)] * len(clips)
    for train, test in folds_by_clip(groups):
        model = _fit(*stack(train))
        for i, p in zip(test, _pair_probs(model, [clips[i] for i in test])):
            probs[i] = p

    lines = [
        f"sumber: {src}",
        f"klip: bukan={int((labels == 0).sum())} kekerasan={int((labels == 1).sum())}  adegan={len(ids)}"
        f"  dilewati (aksi di luar cakupan)={skipped}",
        f"jendela latih: bukan={sum(len(r) for r, lab in zip(rows, labels) if lab == 0)}"
        f" kekerasan={sum(len(r) for r, lab in zip(rows, labels) if lab == 1)}  (motion_min={motion_min})",
        "",
        f"Klip yang memunculkan ALARM (validasi silang {N_FOLDS} lipatan per adegan, konfirmasi seperti aplikasi):",
        "  ambang  konfirmasi   kekerasan beralarm   bukan-kekerasan beralarm",
    ]
    for thr in THRESHOLDS:
        for k, n in CONFIRMS:
            c2 = dataclasses.replace(cfg, violence_threshold=thr, confirm_k=k, confirm_n=n)
            hit = np.array([clip_alarm(c, p, c2) for c, p in zip(clips, probs)])
            v, nv = hit[labels == 1], hit[labels == 0]
            mark = "  <- bawaan aplikasi" if (thr, k, n) == (cfg.violence_threshold, cfg.confirm_k, cfg.confirm_n) else ""
            lines.append(f"  {thr:<6}  {k} dari {n:<3}   {int(v.sum()):3d}/{len(v)} ({100 * v.mean():3.0f}%)"
                         f"        {int(nv.sum()):3d}/{len(nv)} ({100 * nv.mean():3.0f}%){mark}")
    lines += [
        "",
        "Catatan: klip bukan-kekerasan di dataset ini sengaja dibuat mengecoh (pelukan, tos, sorakan,",
        "gerak tangan cepat), dan kameranya di pojok atas ruangan. Angka di webcam sejajar badan bisa",
        "berbeda ke dua arah; hanya uji adegan langsung yang menentukan.",
    ]

    final = _fit(*stack(range(len(clips))))
    top = sorted(zip(final.feature_importances_, FEATURE_NAMES), reverse=True)[:5]
    lines += ["", "5 fitur terpenting (model akhir, dilatih dari semua klip):", *[f"  {n}: {v:.3f}" for v, n in top]]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH, compress=3)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nmodel disimpan: {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
