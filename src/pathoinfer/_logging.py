"""loguru setup, shared with patchology and pathotrain through ``patchology.log``.

One stderr handler serves the whole process. Its filter gives each record the level set
for its top-level package by ``<PACKAGE>_LOG_LEVEL`` (``INFO`` when unset, ``ALL`` for
everything), so ``PATHOINFER_LOG_LEVEL`` controls ``pathoinfer.*`` and
``PATCHOLOGY_LOG_LEVEL`` controls ``patchology.*``. pathoinfer adds no stderr handler of
its own, so nothing prints twice, and never removes a handler it did not add.

``set_log_level`` changes a package's level at runtime by updating its variable, so
DataLoader workers started afterwards inherit it.
"""

import functools
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from patchology.log import configure_logging, level_for

if TYPE_CHECKING:
    from loguru import Record

LOG_LEVEL = os.getenv("PATHOINFER_LOG_LEVEL", "INFO")
PACKAGES = ("pathoinfer", "patchology")


def set_log_level(level: str, package: str = "pathoinfer") -> None:
    """Log ``package`` at ``level`` (a loguru level name, or ``ALL``) from now on.

    Raises:
        ValueError: ``level`` names no loguru level.
    """
    global LOG_LEVEL
    level = level.upper()
    logger.level("TRACE" if level == "ALL" else level)
    os.environ[f"{package.upper()}_LOG_LEVEL"] = level
    # shortcut: patchology caches each package's level; upstream a set_log_level there.
    level_for.cache_clear()
    if package == "pathoinfer":
        LOG_LEVEL = level


def add_file_sink(path: str | Path, level: str = "DEBUG", rotation: str = "50 MB") -> int:
    """Also log pathoinfer's and patchology's records to ``path``; returns the handler id
    to pass to ``logger.remove`` when the run ends. Records are queued, so the run's
    threads never block on the file."""
    return logger.add(
        path,
        level=level,
        rotation=rotation,
        enqueue=True,
        filter=lambda record: (record["name"] or "").partition(".")[0] in PACKAGES,
    )


def log_call[F: Callable[..., Any]](fn: F) -> F:
    """Log each call of ``fn`` at DEBUG with its arguments, under ``fn``'s own module and
    name. Arrays and tensors are logged as shape and dtype, other values cut to 80
    characters. Costs one level check when DEBUG is off."""

    def at_fn(record: "Record") -> None:
        record["name"] = fn.__module__
        record["function"] = fn.__qualname__
        record["line"] = fn.__code__.co_firstlineno

    bound = logger.patch(at_fn)
    debug = logger.level("DEBUG").no
    package = fn.__module__.partition(".")[0]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if level_for(package) <= debug:
            shown = [_short(a) for a in args] + [f"{k}={_short(v)}" for k, v in kwargs.items()]
            bound.debug("{}({})", fn.__qualname__, ", ".join(shown))
        return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def _short(value: Any) -> str:
    shape, dtype = getattr(value, "shape", None), getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return f"{type(value).__name__}(shape={tuple(shape)}, dtype={dtype})"
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


configure_logging("pathoinfer")
