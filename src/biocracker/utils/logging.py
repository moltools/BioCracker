"""Utility functions for logging."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass


PACKAGE_LOGGER = "biocracker"

STANDARD_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
STANDARD_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str | int = "INFO",
    *,
    fmt: str = STANDARD_FMT,
    datefmt: str = STANDARD_DATEFMT,
    stream: None | int | str | object = None,
) -> None:
    """
    Set up logging for the biocracker package.

    :param level: log level for console output
    :param fmt: log message format
    :param datefmt: date format for log messages
    :param stream: output stream for console logs; defaults to sys.stderr
    .. note:: safe to call multiple times; library code should not call this function;
        it is intended for use by applications using the library
    """
    if stream is None:
        stream = sys.stderr

    if isinstance(level, str):
        level = level.upper()

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    # Avoid duplicate handlers if called repeatedly (common in notebooks)
    # Keep it simple: remove existing handlers created by previous setup calls.
    root.handlers = [handler]

    # Make sure package logger propagates to root
    logging.getLogger(PACKAGE_LOGGER).propagate = True


def add_file_handler(
    logfile: str,
    *,
    level: str | int = "DEBUG",
    fmt: str = STANDARD_FMT,
    datefmt: str = STANDARD_DATEFMT,
) -> None:
    """
    Add a file handler to the root logger.

    :param logfile: path to log file
    :param level: log level for file output
    .. note:: intended to be called after setup_logginer(); safe to call multiple times
        for the same logfile
    """
    if isinstance(level, str):
        level = level.upper()

    root = logging.getLogger()

    # Prevent duplicate file handlers for the same path
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == logfile:
            return

    fh = logging.FileHandler(logfile)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root.addHandler(fh)


@dataclass(frozen=True)
class Ctx:
    """
    Context information for logging.
    
    :cvar region: optional region identifier
    :cvar gene: optional gene identifier
    :cvar domain: optional domain identifier
    :cvar model: optional model identifier

    .. note::
        Usage example:
            ctx = Ctx(region="chr1", gene="BRCA1")
            logger.info(f"{ctx.prefix()}This is a log message.")
    """

    region: str | None = None
    gene: str | None = None
    domain: str | None = None
    model: str | None = None

    def prefix(self) -> str:
        """
        Generate a log prefix string based on the context.
        
        :return: formatted prefix string
        """
        parts = []
        if self.region: parts.append(f"region={self.region}")
        if self.gene: parts.append(f"gene={self.gene}")
        if self.domain: parts.append(f"domain={self.domain}")
        if self.model: parts.append(f"model={self.model}")
        return ("[" + " ".join(parts) + "] ") if parts else ""
