"""User-facing logging setup for mcpy.

Library modules only ever call ``logging.getLogger(__name__)``; no handlers
are attached, no global state is mutated on import. Applications that want
log output call :func:`configure` once at startup.
"""
from __future__ import annotations

import logging
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s %(name)s %(levelname)s: %(message)s"
_RANK_FORMAT = "%(asctime)s [rank {rank}] %(name)s %(levelname)s: %(message)s"


def configure(level: int = logging.INFO,
              file: Optional[str] = None,
              mpi_rank: Optional[int] = None,
              stream: bool = True) -> logging.Logger:
    """Attach handlers to the top-level ``mcpy`` logger.

    Args:
        level: Log level for the mcpy logger.
        file: Optional path to a log file.
        mpi_rank: If provided, included in the format string. Use one log
            file per rank by passing ``file=f"mcpy_rank_{rank}.log"``.
        stream: If True, also emit to stderr.

    Returns:
        The configured ``mcpy`` logger.
    """
    logger = logging.getLogger("mcpy")
    logger.setLevel(level)
    logger.propagate = False

    fmt = _RANK_FORMAT.format(rank=mpi_rank) if mpi_rank is not None else _DEFAULT_FORMAT
    formatter = logging.Formatter(fmt)

    # Drop any handlers we previously attached so re-configure is idempotent.
    for h in list(logger.handlers):
        if getattr(h, "_mcpy_managed", False):
            logger.removeHandler(h)

    if stream:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh._mcpy_managed = True
        logger.addHandler(sh)

    if file is not None:
        fh = logging.FileHandler(file)
        fh.setFormatter(formatter)
        fh._mcpy_managed = True
        logger.addHandler(fh)

    return logger
