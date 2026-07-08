"""Plain-pipeline fallback backend: DuckDuckGo search + httpx fetch + BeautifulSoup.

Needs no API key at all, so the agent still finishes if Composio's free tier is
exhausted (PRD §7, §15). Search rate-limits surface as ``QuotaExceeded`` too, but for
the fallback that just means "back off" — there's nowhere further to fall.
"""

from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup

from ..cache import utcnow
from .base import Backend, FetchedPage, QuotaExceeded, SearchResult

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
_STRIP_TAGS = ("script", "style", "noscript", "svg", "template", "iframe", "form", "nav", "footer")


class FallbackBackend(Backend):
    name = "fallback"

    def __init__(self) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml"},
            timeout=httpx.Timeout(20.0, connect=10.0),
            follow_redirects=True,
        )

    def search(self, query: str, num: int = 5) -> list[SearchResult]:
        from ddgs import DDGS

        try:
            raw = DDGS().text(query, max_results=num)
        except Exception as e:  # ddgs raises its own exception types
            if "ratelimit" in type(e).__name__.lower() or "202" in str(e):
                raise QuotaExceeded(f"DuckDuckGo rate limit: {e}") from e
            return []
        return [
            SearchResult(title=r.get("title", ""), url=r.get("href", ""), snippet=r.get("body", ""))
            for r in raw
            if r.get("href")
        ]

    def fetch(self, url: str) -> FetchedPage:
        try:
            resp = self._client.get(url)
        except httpx.HTTPError as e:
            return FetchedPage(url=url, ok=False, status=f"error: {e}", fetched_at=utcnow(), backend=self.name)

        if resp.status_code >= 400:
            return FetchedPage(
                url=str(resp.url), ok=False, status=f"http {resp.status_code}",
                fetched_at=utcnow(), backend=self.name,
            )

        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype:
            return FetchedPage(
                url=str(resp.url), ok=False, status=f"non-html ({ctype})",
                fetched_at=utcnow(), backend=self.name,
            )

        title, text = _extract(resp.text)
        return FetchedPage(
            url=str(resp.url), title=title, text=text, ok=True, status="ok",
            fetched_at=utcnow(), backend=self.name,
        )

    def close(self) -> None:
        self._client.close()


def _extract(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = root.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return title, text
