"""Lightweight in-process cache for FastAPI route helpers.

Two helpers:
- file_cached(path, ttl)  — re-reads only when the file mtime changes or TTL expires
- timed_cached(ttl)       — decorator for functions whose result should be cached by TTL
"""
import time
import functools
import os
from typing import Any, Callable, Optional

_store: dict[str, dict] = {}


def file_cached(path: str, ttl: float = 5.0) -> Optional[str]:
    """Return cached file content, refreshing when mtime changes or TTL expires."""
    now = time.monotonic()
    entry = _store.get(path)
    try:
        mtime = os.path.getmtime(path) if os.path.exists(path) else None
    except OSError:
        mtime = None

    if entry and (now - entry["ts"] < ttl) and entry.get("mtime") == mtime:
        return entry["value"]

    try:
        with open(path, "r", encoding="utf-8") as f:
            value = f.read()
    except (OSError, FileNotFoundError):
        value = None

    _store[path] = {"value": value, "ts": now, "mtime": mtime}
    return value


def timed_cached(ttl: float = 30.0):
    """Decorator: cache the return value for `ttl` seconds (keyed by func + args)."""
    def decorator(fn: Callable) -> Callable:
        cache: dict[tuple, dict] = {}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            entry = cache.get(key)
            if entry and (now - entry["ts"]) < ttl:
                return entry["value"]
            result = fn(*args, **kwargs)
            cache[key] = {"value": result, "ts": now}
            return result

        wrapper.cache_clear = lambda: cache.clear()
        return wrapper
    return decorator


def invalidate(path: str) -> None:
    """Remove a file cache entry (call after writes)."""
    _store.pop(path, None)
