"""Cache + checkpoint layer (PRD §7 "cache + checkpoint from run one").

Two guarantees this buys us:

1. **Re-extraction and the verification re-run are nearly free.** Raw searches and
   fetched pages are cached on disk per app, so Pass 2 (and any re-score) reuses
   Pass 1's network I/O instead of re-hitting Composio/DuckDuckGo — quota-safe.
2. **Resume, don't restart.** Each finished app is checkpointed per pass, so a crash
   at app 73 resumes at 73.

Layout (matches PRD §13)::

    data/cache/{slug}/searches/{hash}.json   raw search results
    data/cache/{slug}/pages/{hash}.json      raw fetched pages
    data/checkpoints/{pass}/{id}.json        finished AppResult per app+pass

All writes are atomic (temp file + os.replace) so a crash mid-write can't corrupt
the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CACHE_DIR, CHECKPOINT_DIR


def slugify(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "app"


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class Cache:
    def __init__(self, cache_dir: Path = CACHE_DIR, checkpoint_dir: Path = CHECKPOINT_DIR):
        self.cache_dir = cache_dir
        self.checkpoint_dir = checkpoint_dir

    # ------------------------------------------------------------------ pages
    def _page_path(self, slug: str, url: str) -> Path:
        return self.cache_dir / slug / "pages" / f"{_hash(url)}.json"

    def get_page(self, slug: str, url: str) -> dict | None:
        return _read_json(self._page_path(slug, url))

    def put_page(self, slug: str, page: dict) -> None:
        _write_json(self._page_path(slug, page["url"]), page)

    # --------------------------------------------------------------- searches
    def _search_path(self, slug: str, query: str) -> Path:
        return self.cache_dir / slug / "searches" / f"{_hash(query)}.json"

    def get_search(self, slug: str, query: str) -> list | None:
        obj = _read_json(self._search_path(slug, query))
        return obj.get("results") if isinstance(obj, dict) else None

    def put_search(self, slug: str, query: str, results: list) -> None:
        _write_json(
            self._search_path(slug, query),
            {"query": query, "fetched_at": utcnow(), "results": results},
        )

    # ------------------------------------------------------------ checkpoints
    def _checkpoint_path(self, pass_label: str, app_id: int) -> Path:
        return self.checkpoint_dir / pass_label / f"{app_id:03d}.json"

    def has_checkpoint(self, pass_label: str, app_id: int) -> bool:
        return self._checkpoint_path(pass_label, app_id).exists()

    def get_checkpoint(self, pass_label: str, app_id: int) -> dict | None:
        return _read_json(self._checkpoint_path(pass_label, app_id))

    def put_checkpoint(self, pass_label: str, app_id: int, result: dict) -> None:
        _write_json(self._checkpoint_path(pass_label, app_id), result)

    def all_checkpoints(self, pass_label: str) -> list[dict]:
        d = self.checkpoint_dir / pass_label
        if not d.exists():
            return []
        rows = [_read_json(p) for p in sorted(d.glob("*.json"))]
        return [r for r in rows if r is not None]
