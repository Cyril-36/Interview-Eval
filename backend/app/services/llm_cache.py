"""
SQLite-backed cache for LLM responses.

Keyed on a stable hash of normalized inputs. Failures are swallowed so the
cache never breaks the calling pipeline — a cache miss / error simply causes
the LLM to be invoked.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_DB_PATH = DATA_DIR / "llm_cache.db"

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", (text or "").strip().lower())


def make_key(*parts: str) -> str:
    """Build a stable cache key from arbitrary string parts."""
    h = hashlib.sha256()
    for part in parts:
        h.update(_normalize(part).encode("utf-8"))
        h.update(b"\x1f")  # unit-separator delimiter
    return h.hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0
    errors: int = 0


class LLMCache:
    """SQLite key-value cache with TTL. Thread-safe via a per-instance lock."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        default_ttl_seconds: int = 60 * 60 * 24 * 30,
    ):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.default_ttl_seconds = default_ttl_seconds
        self._lock = threading.Lock()
        self.stats = CacheStats()
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level=None,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    value TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_cache_expires"
                " ON llm_cache(expires_at)"
            )

    def get(self, namespace: str, key: str) -> Optional[Any]:
        try:
            with self._lock:
                conn = self._connect()
                row = conn.execute(
                    "SELECT value, expires_at FROM llm_cache"
                    " WHERE key = ? AND namespace = ?",
                    (key, namespace),
                ).fetchone()
            if row is None:
                self.stats.misses += 1
                return None
            value_json, expires_at = row
            if expires_at < time.time():
                self.stats.misses += 1
                self._delete(namespace, key)
                return None
            self.stats.hits += 1
            return json.loads(value_json)
        except (sqlite3.Error, json.JSONDecodeError, OSError) as e:
            self.stats.errors += 1
            logger.warning(f"LLM cache get failed: {e}")
            return None

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        now = time.time()
        try:
            payload = json.dumps(value)
            with self._lock:
                conn = self._connect()
                conn.execute(
                    "INSERT OR REPLACE INTO llm_cache"
                    " (key, namespace, value, expires_at, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (key, namespace, payload, now + ttl, now),
                )
            self.stats.sets += 1
        except (sqlite3.Error, TypeError, ValueError, OSError) as e:
            self.stats.errors += 1
            logger.warning(f"LLM cache set failed: {e}")

    def _delete(self, namespace: str, key: str) -> None:
        try:
            with self._lock:
                conn = self._connect()
                conn.execute(
                    "DELETE FROM llm_cache WHERE key = ? AND namespace = ?",
                    (key, namespace),
                )
        except sqlite3.Error as e:
            logger.warning(f"LLM cache delete failed: {e}")

    def purge_expired(self) -> int:
        try:
            with self._lock:
                conn = self._connect()
                cur = conn.execute(
                    "DELETE FROM llm_cache WHERE expires_at < ?", (time.time(),)
                )
                return cur.rowcount or 0
        except sqlite3.Error as e:
            logger.warning(f"LLM cache purge failed: {e}")
            return 0

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


_cache: Optional[LLMCache] = None
_cache_lock = threading.Lock()


def get_cache() -> LLMCache:
    """Return the process-wide cache instance, creating it on first use."""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                from app.config import get_settings
                settings = get_settings()
                _cache = LLMCache(
                    default_ttl_seconds=settings.LLM_CACHE_TTL_SECONDS,
                )
    return _cache


def reset_cache_for_tests(db_path: Optional[Path] = None) -> LLMCache:
    """Test helper: swap the global cache with a fresh instance."""
    global _cache
    with _cache_lock:
        if _cache is not None:
            _cache.close()
        _cache = LLMCache(db_path=db_path)
    return _cache
