"""Index-range partitioning for batched energy evaluation.

Kept torch-free and dependency-free so it can be unit-tested in the
lightweight CI matrix (no torch / nvalchemi).
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple


def chunk_ranges(
    n_items: int, chunk_size: Optional[int]
) -> Iterator[Tuple[int, int]]:
    """Yield ``(start, stop)`` ranges partitioning ``range(n_items)`` into
    consecutive chunks of at most ``chunk_size`` items.

    ``chunk_size`` of ``None`` or any non-positive value yields a single full
    range ``(0, n_items)`` (i.e. no chunking). ``n_items == 0`` yields nothing.
    """
    if n_items <= 0:
        return
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n_items:
        yield (0, n_items)
        return
    for start in range(0, n_items, chunk_size):
        yield (start, min(start + chunk_size, n_items))
