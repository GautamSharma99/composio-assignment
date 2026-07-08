"""Backend interface: two methods, ``search`` and ``fetch``, with two implementations.

The whole point (PRD §7) is that the research step is swappable: Composio is primary
and on-brand; the free plain pipeline is the fallback so a free-tier quota never holds
the run hostage. ``QuotaExceeded`` is the signal the resilient retriever uses to flip
from Composio to fallback mid-run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SearchResult":
        return cls(title=d.get("title", ""), url=d.get("url", ""), snippet=d.get("snippet", ""))


@dataclass
class FetchedPage:
    url: str
    title: str = ""
    text: str = ""
    ok: bool = True
    status: str = "ok"
    fetched_at: str = ""
    backend: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FetchedPage":
        return cls(**{k: d.get(k, getattr(cls, k, "")) for k in
                      ("url", "title", "text", "ok", "status", "fetched_at", "backend")})


class QuotaExceeded(Exception):
    """Raised by a backend when it hits a rate/quota ceiling — triggers auto-fallback."""


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def search(self, query: str, num: int = 5) -> list[SearchResult]:
        ...

    @abstractmethod
    def fetch(self, url: str) -> FetchedPage:
        ...

    def close(self) -> None:  # optional cleanup
        pass


def make_backend(name: str):
    """Factory so callers don't import concrete classes (keeps the abstraction thin)."""
    if name == "composio":
        from .composio_backend import ComposioBackend

        return ComposioBackend()
    if name == "fallback":
        from .fallback_backend import FallbackBackend

        return FallbackBackend()
    raise ValueError(f"unknown backend: {name!r}")
