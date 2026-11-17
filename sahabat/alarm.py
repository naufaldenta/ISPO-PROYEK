"""Pengelola alarm: konfirmasi k-dari-n untuk kekerasan, kunci sampai "Sudah diperiksa"."""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from .config import Config
from .gestures import GestureHit
from .types import ViewMode
from .violence import PairScore

AlarmKind = Literal["fall", "violence"]
_MODE_RANK = {"none": 0, "upper": 1, "full": 2}
FORGET_S = 5.0          # jendela konfirmasi pasangan yang tak terlihat lagi dibuang setelah ini


@dataclass(frozen=True)
class Event:
    kind: AlarmKind
    t: float
    ids: tuple[int, ...]
    mode: ViewMode          # mode terlemah di antara orang yang terlibat
    source: str = "model"


def _weakest(ids, modes: dict[int, ViewMode]) -> ViewMode:
    return min((modes.get(i, "none") for i in ids), key=_MODE_RANK.__getitem__)


class AlarmManager:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._pending: list[Event] = []
        self._windows: dict[tuple[int, int], deque[bool]] = {}
        self._last_seen: dict[tuple[int, int], float] = {}
        self._cooldown_until: dict[tuple[int, int], float] = {}
        self._evidence: dict[tuple[int, int], deque[tuple[float, bool]]] = {}

    @property
    def active(self) -> Event | None:
        """Kejadian tertua yang belum diperiksa."""
        return self._pending[0] if self._pending else None

    @property
    def red_ids(self) -> set[int]:
        return {i for e in self._pending for i in e.ids}

    def tracked_pairs(self) -> int:
        return len(self._windows)

    def reset_input(self) -> None:
        """Kamera berubah: ID/konfirmasi lama tidak berlaku, tetapi alarm belum diperiksa tetap terkunci."""
        self._windows.clear()
        self._last_seen.clear()
        self._cooldown_until.clear()
        self._evidence.clear()
        self._pending = [replace(event, ids=()) for event in self._pending]

    def tick(self, t: float, fall_ids: list[int], pairs: list[PairScore],
             modes: dict[int, ViewMode], hits: Sequence[GestureHit] = (),
             seen: set[int] | None = None, fall_sources: dict[int, str] | None = None,
             exiting: set[int] | None = None) -> list[Event]:
        """Kejadian baru pada panggilan ini."""
        cfg = self.cfg
        new = [Event("fall", t, (i,), modes.get(i, "none"), (fall_sources or {}).get(i, "model"))
               for i in fall_ids]

        latched = {e.ids for e in self._pending if e.kind == "violence"}

        def add(ids, source):
            if ids not in latched and t >= self._cooldown_until.get(ids, float("-inf")):
                new.append(Event("violence", t, ids, _weakest(ids, modes), source))
                latched.add(ids)
                self._evidence.pop(ids, None)
                self._windows.pop(ids, None)

        for hit in hits:
            add(hit.ids, hit.kind)
        scored = set()
        for ps in pairs:
            scored.add(ps.ids)
            self._last_seen[ps.ids] = t
            window = self._windows.setdefault(ps.ids, deque(maxlen=cfg.confirm_n))
            if t < self._cooldown_until.get(ps.ids, float("-inf")):
                window.clear()
                continue
            present = seen is None or all(i in seen for i in ps.ids)
            high = ps.prob >= cfg.violence_threshold and present
            window.append(high)
            if ps.ids not in latched and seen is not None and present:
                evidence = self._evidence.get(ps.ids)
                if evidence is None and high:
                    evidence = self._evidence[ps.ids] = deque(maxlen=cfg.confirm_n)
                if evidence is not None:
                    evidence.append((t, high))
            if sum(window) >= cfg.confirm_k:
                add(ps.ids, "model")

        for ids, votes in list(self._evidence.items()):
            if seen is not None and ids not in scored and all(i in seen for i in ids):
                votes.append((t, False))
            while votes and t - votes[0][0] > cfg.lost_s:
                votes.popleft()
            count = sum(high for _, high in votes)
            if not count:
                del self._evidence[ids]
            elif seen is not None and count >= cfg.lost_votes:
                missing = set(ids) - seen
                if len(missing) == 1 and (exiting is None or missing <= exiting):
                    add(ids, "keluar_gambar")

        for ids in list(self._windows):
            if ids in scored:
                continue
            self._windows[ids].append(False)             # pasangan yang berpisah meluruh
            if t - self._last_seen.get(ids, t) > FORGET_S:
                del self._windows[ids]
                self._last_seen.pop(ids, None)
        for ids in [k for k, until in self._cooldown_until.items() if t >= until]:
            del self._cooldown_until[ids]

        self._pending.extend(new)
        return new

    def acknowledge(self, t: float) -> bool:
        """Guru menekan "Sudah diperiksa". False bila tidak ada alarm."""
        if not self._pending:
            return False
        for e in self._pending:
            if e.kind == "violence" and e.ids:
                self._cooldown_until[e.ids] = t + self.cfg.pair_cooldown_s
                self._windows.pop(e.ids, None)
                self._evidence.pop(e.ids, None)
        self._pending = []
        return True
