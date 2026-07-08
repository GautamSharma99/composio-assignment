"""Interchangeable search/scrape backends (PRD §7)."""

from .base import Backend, FetchedPage, QuotaExceeded, SearchResult, make_backend

__all__ = ["Backend", "FetchedPage", "SearchResult", "QuotaExceeded", "make_backend"]
