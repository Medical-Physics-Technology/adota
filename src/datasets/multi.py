"""Composite CT dataset: one unified index over several sub-datasets.

Lets a single object hold patients from one collection or many (e.g. all four
anatomies), preserving each record's provenance. This is the object the MC
generator iterates and the AL sampler will sample from.
"""
from __future__ import annotations

import bisect
from typing import Dict, List, Sequence

from src.datasets.base import CTDataset, CTRecord


class MultiDataset(CTDataset):
    """Concatenation of :class:`CTDataset`s with a shared, contiguous index."""

    def __init__(self, datasets: Sequence[CTDataset], name: str = "multi"):
        self.datasets = list(datasets)
        self.name = name
        self.anatomy = "mixed"
        # cumulative lengths for O(log n) index -> (dataset, local index)
        self._cum: List[int] = []
        total = 0
        for d in self.datasets:
            total += len(d)
            self._cum.append(total)

    def __len__(self) -> int:
        return self._cum[-1] if self._cum else 0

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        d = bisect.bisect_right(self._cum, idx)
        prev = self._cum[d - 1] if d > 0 else 0
        return d, idx - prev

    def record(self, idx: int) -> CTRecord:
        d, local = self._locate(idx)
        return self.datasets[d].record(local)

    def counts_by_anatomy(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for d in self.datasets:
            out[d.anatomy] = out.get(d.anatomy, 0) + len(d)
        return out
