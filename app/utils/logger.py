"""Structured logging setup using loguru."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger


def configure_logging(service: str = "api") -> None:
    """Configure loguru with JSON output to stderr and rotating log files.

    Produces two file sinks per service:
      logs/<service>.out.log  — all levels (INFO and above by default)
      logs/<service>.err.log  — ERROR and above, with full traceback
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        serialize=True,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    logger.add(
        str(log_dir / f"{service}.out.log"),
        level=level,
        serialize=True,
        rotation="10 MB",
        retention=3,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        encoding="utf-8",
    )

    logger.add(
        str(log_dir / f"{service}.err.log"),
        level="ERROR",
        serialize=True,
        rotation="10 MB",
        retention=3,
        enqueue=True,
        backtrace=True,
        diagnose=False,
        encoding="utf-8",
    )


def get_logger(name: str) -> "Logger":
    """Return a loguru logger bound to the given module name."""
    return logger.bind(module=name)
