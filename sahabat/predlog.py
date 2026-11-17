"""Log prediksi: satu baris JSON per putaran penilaian. Bahan evaluasi (alarm keliru per jam, waktu peringatan)."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

KEEP_DAYS = 14


class PredLog:
    def __init__(self, folder: Path, prefix: str = "prediksi") -> None:
        self.folder = Path(folder)
        self.prefix = prefix                         # prediksi (mesin kamera) | kata (pendengar)
        self._file = None
        self._day = None
        self._disabled = False
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
        except OSError:                              # folder hanya-baca (misalnya dijalankan dari flashdisk terkunci)
            self._disabled = True
            print("Peringatan: log prediksi tidak bisa ditulis; lanjut tanpa log.")
            return
        cutoff = time.time() - KEEP_DAYS * 86400
        for old in self.folder.glob(f"{self.prefix}-*.jsonl"):
            try:
                if old.stat().st_mtime < cutoff:
                    old.unlink()
            except OSError:
                pass

    def write(self, record: dict) -> None:
        """Gagal menulis (misalnya disk penuh) tidak boleh menghentikan penilaian."""
        if self._disabled:
            return
        try:
            day = datetime.now().strftime("%Y%m%d")
            if day != self._day:
                self.close()
                self._file = open(self.folder / f"{self.prefix}-{day}.jsonl", "a", encoding="utf-8")
                self._day = day
            self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._file.flush()
        except OSError:
            pass

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None
