"""PRIMARY backend: Composio SDK (on-brand). Search + scrape via the no-auth
``COMPOSIO_SEARCH`` toolkit, so it works with only a Composio API key — no per-user
OAuth / connected account.

Verified against composio 0.17.1:
  * ``from composio import Composio``; ``Composio(api_key=...)`` (or reads COMPOSIO_API_KEY).
  * search:  ``client.tools.execute("COMPOSIO_SEARCH_WEB", user_id=..., arguments={"query": q})``
  * scrape:  ``client.tools.execute("COMPOSIO_SEARCH_FETCH_URL_CONTENT", user_id=..., arguments={"url": u})``
  * result envelope has ``.successful`` / ``.data`` / ``.error`` — check ``.successful``
    before reading ``.data`` (tool failures don't raise, they come back here).
  * ``execute`` is synchronous → the orchestrator wraps it in ``asyncio.to_thread``.

Response *shapes* under ``.data`` vary by engine, so parsing here is defensive and is
smoke-tested live in scripts/check_env.py.
"""

from __future__ import annotations

import json
from typing import Any

from ..cache import utcnow
from ..config import SETTINGS
from .base import Backend, FetchedPage, QuotaExceeded, SearchResult

SEARCH_TOOL = "COMPOSIO_SEARCH_WEB"
FETCH_TOOL = "COMPOSIO_SEARCH_FETCH_URL_CONTENT"


def _looks_like_quota(err: Any) -> bool:
    s = str(err).lower()
    status = getattr(err, "status_code", None)
    return status == 429 or any(k in s for k in ("429", "rate limit", "ratelimit", "quota", "too many requests"))


def _successful(res: Any) -> bool:
    val = getattr(res, "successful", None)
    if val is None and isinstance(res, dict):
        val = res.get("successful")
    return True if val is None else bool(val)


def _data(res: Any) -> Any:
    if hasattr(res, "data"):
        return res.data
    if isinstance(res, dict):
        return res.get("data")
    return res


def _error(res: Any) -> str:
    val = getattr(res, "error", None)
    if val is None and isinstance(res, dict):
        val = res.get("error")
    return str(val) if val else ""


def _as_dict(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    if hasattr(x, "model_dump"):
        try:
            return x.model_dump()
        except Exception:
            pass
    return {}


def _parse_search(data: Any, num: int) -> list[SearchResult]:
    """Pull URL-bearing items out of whatever shape the search engine returned."""
    data = _as_dict(data) or {}
    container = data.get("results", data) if isinstance(data, dict) else {}
    container = _as_dict(container) or {}

    items: list[Any] = []
    for key in ("organic_results", "organic", "citations", "results", "sources", "webPages"):
        v = container.get(key)
        if isinstance(v, dict):  # e.g. Bing-style {"value": [...]}
            v = v.get("value") or v.get("results")
        if isinstance(v, list) and v:
            items = v
            break

    out: list[SearchResult] = []
    for it in items:
        if isinstance(it, str):
            out.append(SearchResult(title=it, url=it))
        elif isinstance(it, dict):
            url = it.get("url") or it.get("link") or it.get("href") or ""
            if not url:
                continue
            out.append(
                SearchResult(
                    title=it.get("title") or it.get("name") or url,
                    url=url,
                    snippet=it.get("snippet") or it.get("content") or it.get("description") or it.get("text") or "",
                )
            )
        if len(out) >= num:
            break
    return out


def _parse_page(data: Any) -> tuple[str, str]:
    """Return ``(title, text)``. Verified shape (COMPOSIO_SEARCH_FETCH_URL_CONTENT):
    ``data["results"][0]`` = {"title": ..., "text": <markdown>, "url": ...}. Falls back
    through other common shapes so an engine change doesn't break us."""
    if isinstance(data, str):
        return "", data
    d = _as_dict(data)
    if not d:
        return "", (str(data) if data else "")
    # verified: results is a list of page objects
    nested = d.get("results") or d.get("response") or d.get("data")
    if isinstance(nested, list) and nested and isinstance(nested[0], dict):
        page = nested[0]
        text = page.get("text") or page.get("content") or page.get("markdown") or ""
        return page.get("title", ""), text
    if isinstance(nested, (dict, str)):
        return _parse_page(nested)
    # flat single-page keys
    for key in ("content", "markdown", "text", "result", "page_content", "body"):
        v = d.get(key)
        if isinstance(v, str) and v.strip():
            return d.get("title", ""), v
    return "", json.dumps(d)[:20000]


class ComposioBackend(Backend):
    name = "composio"

    def __init__(self) -> None:
        if not SETTINGS.has_composio:
            raise RuntimeError("COMPOSIO_API_KEY is not set; cannot use the Composio backend.")
        from composio import Composio

        self._client = Composio(api_key=SETTINGS.composio_api_key)
        self._user_id = SETTINGS.composio_user_id or "default"

    def _execute(self, slug: str, arguments: dict) -> Any:
        try:
            # skip_version_check => use the toolkit's latest version without pinning
            # (manual execution otherwise rejects "latest"). Returns a plain dict.
            return self._client.tools.execute(
                slug, user_id=self._user_id, arguments=arguments,
                dangerously_skip_version_check=True,
            )
        except QuotaExceeded:
            raise
        except Exception as e:
            if _looks_like_quota(e):
                raise QuotaExceeded(f"Composio quota/rate limit on {slug}: {e}") from e
            raise

    def search(self, query: str, num: int = 5) -> list[SearchResult]:
        res = self._execute(SEARCH_TOOL, {"query": query})
        if not _successful(res):
            err = _error(res)
            if _looks_like_quota(err):
                raise QuotaExceeded(f"Composio search quota: {err}")
            return []
        return _parse_search(_data(res), num)

    def fetch(self, url: str) -> FetchedPage:
        res = self._execute(FETCH_TOOL, {"url": url})
        if not _successful(res):
            err = _error(res)
            if _looks_like_quota(err):
                raise QuotaExceeded(f"Composio fetch quota: {err}")
            return FetchedPage(url=url, ok=False, status=f"composio error: {err}", fetched_at=utcnow(), backend=self.name)
        title, text = _parse_page(_data(res))
        return FetchedPage(
            url=url, title=title, text=text, ok=bool(text), status="ok" if text else "empty",
            fetched_at=utcnow(), backend=self.name,
        )
