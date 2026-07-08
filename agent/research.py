"""Per-app research agent: A discover · B retrieve · C extract · D persist (PRD §7).

Also home to:
  * ``Retriever`` — cached search/fetch over the interchangeable backends, with
    auto-fallback: the moment Composio raises ``QuotaExceeded`` it degrades to the free
    pipeline for the rest of the run (the ceiling itself becomes page content).
  * ``ResearchAgent.run_pass1`` — the naive baseline (1 search, 1 fetch, 1 ungrounded
    extraction, no checks).
  * ``ResearchAgent.run_pass2`` — the same discovery widened, plus loops L1 (grounded
    extraction), L2 (cross-check), L3 (MCP-existence probe), L4 (confidence gating).
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from urllib.parse import urlparse

from .backends.base import Backend, FetchedPage, QuotaExceeded, SearchResult, make_backend
from .cache import Cache, slugify
from .config import SETTINGS
from .llm import LLM
from .schema import (
    AppExtraction,
    AppResult,
    AppSeed,
    Buildability,
    CrossCheck,
    ExistingMcp,
    SelfServe,
    VerificationStatus,
    build_result,
)

log = logging.getLogger("agent.research")

DOC_HINTS = ("docs", "developer", "developers", "api", "reference", "auth", "oauth",
             "pricing", "plans", "rest", "graphql", "getting-started")
NOISE_HOSTS = ("youtube.com", "reddit.com", "medium.com", "stackoverflow.com",
               "facebook.com/watch", "twitter.com", "x.com/", "pinterest.com")
# Registry hosts count as MCP evidence ONLY when the app's own name is in the path.
MCP_REGISTRY_HOSTS = ("mcp.so", "smithery.ai", "glama.ai", "pulsemcp.com", "mcp-get.com", "mcpservers.org")
# Generic tokens we must NOT treat as an app-name match when scanning MCP hits.
MCP_STOPWORDS = {"ads", "api", "app", "the", "inc", "cloud", "business", "selling", "partner",
                 "payment", "payments", "exchange", "data", "server", "mcp", "get", "platform",
                 "group", "tech", "connect", "commerce", "marketing", "internet"}
# Third-party hosting labels we must NOT derive an app-name token from (e.g. a docs page
# hosted on notion.site or github.io says nothing about the app's own name).
HOSTING_LABELS = {"notion", "github", "mintlify", "gitbook", "readme", "herokuapp", "vercel",
                  "netlify", "atlassian", "google", "microsoft", "amazonaws", "cloudfront",
                  "pages", "site", "gitlab", "web"}


def _domain(url: str | None) -> str:
    if not url:
        return ""
    try:
        net = urlparse(url).netloc.lower()
        return net[4:] if net.startswith("www.") else net
    except ValueError:
        return ""


def _root_domain(host: str) -> str:
    """example: 'docs.github.com' -> 'github.com' (last two labels; good enough here)."""
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


# ---------------------------------------------------------------------------
# Resilient, cached retriever
# ---------------------------------------------------------------------------
class Retriever:
    def __init__(self, cache: Cache, prefer: str):
        self.cache = cache
        self.prefer = prefer  # "composio" | "fallback"
        self._degraded = False
        self._degrade_reason = ""
        self._backends: dict[str, Backend] = {}
        self._lock = threading.Lock()

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def mode(self) -> str:
        if self.prefer == "fallback":
            return "fallback"
        return "composio→fallback" if self._degraded else "composio"

    def _get(self, name: str) -> Backend:
        with self._lock:
            if name not in self._backends:
                self._backends[name] = make_backend(name)
            return self._backends[name]

    def _active_name(self) -> str:
        return "fallback" if (self.prefer == "fallback" or self._degraded) else "composio"

    def _degrade(self, reason: str) -> None:
        with self._lock:
            if not self._degraded:
                self._degraded = True
                self._degrade_reason = reason
                log.warning("Backend degraded to fallback: %s", reason)

    # -- search --
    def search(self, slug: str, query: str, num: int) -> list[SearchResult]:
        cached = self.cache.get_search(slug, query)
        if cached is not None:
            return [SearchResult.from_dict(x) for x in cached]
        results = self._run_search(query, num)
        self.cache.put_search(slug, query, [r.to_dict() for r in results])
        return results

    def _run_search(self, query: str, num: int) -> list[SearchResult]:
        name = self._active_name()
        try:
            return self._get(name).search(query, num)
        except QuotaExceeded as e:
            if name == "composio":
                self._degrade(str(e))
                return self._get("fallback").search(query, num)
            raise
        except Exception as e:  # network / parsing hiccup on one query shouldn't kill the app
            log.debug("search failed (%s): %s", name, e)
            if name == "composio":
                self._degrade(f"composio search error: {e}")
                try:
                    return self._get("fallback").search(query, num)
                except Exception:
                    return []
            return []

    # -- fetch --
    def fetch(self, slug: str, url: str) -> FetchedPage:
        cached = self.cache.get_page(slug, url)
        if cached is not None:
            return FetchedPage.from_dict(cached)
        page = self._run_fetch(url)
        self.cache.put_page(slug, page.to_dict())
        return page

    def _run_fetch(self, url: str) -> FetchedPage:
        name = self._active_name()
        try:
            return self._get(name).fetch(url)
        except QuotaExceeded as e:
            if name == "composio":
                self._degrade(str(e))
                return self._get("fallback").fetch(url)
            return FetchedPage(url=url, ok=False, status=f"quota: {e}")
        except Exception as e:
            log.debug("fetch failed (%s): %s", name, e)
            if name == "composio":
                self._degrade(f"composio fetch error: {e}")
                try:
                    return self._get("fallback").fetch(url)
                except Exception as e2:
                    return FetchedPage(url=url, ok=False, status=f"error: {e2}")
            return FetchedPage(url=url, ok=False, status=f"error: {e}")

    def close(self) -> None:
        for b in self._backends.values():
            try:
                b.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Research agent
# ---------------------------------------------------------------------------
class ResearchAgent:
    def __init__(self, retriever: Retriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm

    # thread-wrapped I/O so the sync backends don't block the event loop
    async def _search(self, slug: str, query: str, num: int) -> list[SearchResult]:
        return await asyncio.to_thread(self.retriever.search, slug, query, num)

    async def _fetch(self, slug: str, url: str) -> FetchedPage:
        return await asyncio.to_thread(self.retriever.fetch, slug, url)

    # ------------------------------------------------------------------ Pass 1
    async def run_pass1(self, seed: AppSeed) -> AppResult:
        """Naive baseline (PRD §9): a SINGLE search, a SINGLE fetch, a single extraction,
        no targeted discovery and no verification loops. It uses the same grounded extractor
        as Pass 2 — the only differences are evidence breadth (one page vs. many targeted
        pages) and the absence of L2/L3/L4. So the delta isolates exactly what the agent's
        pipeline adds: gathering the right evidence and checking it. With one page, fields
        that live on *other* pages (pricing → self_serve, the auth doc → full auth list, an
        MCP registry → existing_mcp) are simply not determinable — which is the point."""
        slug = slugify(seed.name)
        try:
            results = await self._search(slug, f"{seed.name} developer API documentation", num=3)
            top_url = results[0].url if results else seed.hint_url
            pages: list[FetchedPage] = []
            if top_url:
                pages = [await self._fetch(slug, top_url)]
            ext = await self.llm.extract(seed, pages, grounded=True)
            row = build_result(seed, ext, pass_label="pass1", backend=self.retriever.mode)
            row.verification_status = VerificationStatus.unverified
            return row
        except Exception as e:
            log.warning("pass1 failed for %s: %s", seed.name, e)
            return _error_row(seed, "pass1", str(e), self.retriever.mode)

    # ------------------------------------------------------------------ Pass 2
    async def run_pass2(self, seed: AppSeed) -> AppResult:
        """Widened discovery + loops L1–L4."""
        slug = slugify(seed.name)
        try:
            pages = await self._discover_and_retrieve(seed, slug)

            # L1: grounded, evidence-quoting extraction over multiple sources
            ext = await self.llm.extract(seed, pages, grounded=True)
            row = build_result(seed, ext, pass_label="pass2", backend=self.retriever.mode)

            # L2: independent cross-check of the 3 hallucination-prone fields
            cc = await self.llm.cross_check(seed, pages)
            disagreements = _apply_cross_check(row, ext, cc)
            row.cross_check = {
                "auth_methods": [a.value for a in cc.auth_methods],
                "self_serve": cc.self_serve.value,
                "existing_mcp": cc.existing_mcp.value,
                "reasoning": cc.reasoning,
                "disagreements": disagreements,
            }

            # L3: targeted MCP-existence probe (don't trust model memory)
            probe = await self._mcp_probe(seed, slug)
            row.mcp_probe = probe
            _apply_mcp_probe(row, probe)

            # L4: confidence gating → human queue
            _apply_confidence_gate(row, disagreements, probe)

            row.verification_status = VerificationStatus.agent_verified
            return row
        except Exception as e:
            log.warning("pass2 failed for %s: %s", seed.name, e)
            return _error_row(seed, "pass2", str(e), self.retriever.mode)

    async def _discover_and_retrieve(self, seed: AppSeed, slug: str) -> list[FetchedPage]:
        queries = [
            f"{seed.name} developer API documentation",
            f"{seed.name} API authentication OAuth API key",
            f"{seed.name} API access pricing free plan developer signup",
        ]
        candidates: list[str] = []
        if seed.hint_url:
            candidates.append(seed.hint_url)
        for q in queries:
            for r in await self._search(slug, q, num=SETTINGS.search_results):
                if r.url:
                    candidates.append(r.url)

        ranked = _prioritize(candidates, seed, limit=6)
        pages = await asyncio.gather(*[self._fetch(slug, u) for u in ranked])
        return list(pages)

    async def _mcp_probe(self, seed: AppSeed, slug: str) -> dict:
        """Targeted, PRECISE MCP-existence check. A hit only counts as evidence for THIS
        app if it names the app AND mentions MCP — either on the vendor's own domain
        (official) or in an app-specific GitHub repo / registry entry (community). Generic
        "mcp" pages that merely list servers are ignored, which is what kept the naive
        version from inventing community MCPs for gated apps."""
        query = f"{seed.name} MCP server"
        results = await self._search(slug, query, num=8)
        app_root = _root_domain(_domain(seed.hint_url)) if seed.hint_url else ""
        tokens = _mcp_match_tokens(seed)

        evidence: list[dict] = []
        for r in results:
            kind = _classify_mcp_hit(r, tokens, app_root)
            if kind:
                evidence.append({"url": r.url, "title": r.title, "kind": kind})

        if any(e["kind"] == "official" for e in evidence):
            verdict = ExistingMcp.official
        elif any(e["kind"] == "community" for e in evidence):
            verdict = ExistingMcp.community
        else:
            verdict = ExistingMcp.none

        return {"query": query, "n_hits": len(evidence), "verdict": verdict.value, "evidence": evidence[:5]}


# ---------------------------------------------------------------------------
# Discovery ranking
# ---------------------------------------------------------------------------
def _classify_mcp_hit(r: SearchResult, tokens: set[str], app_root: str) -> str | None:
    """Classify one MCP search hit as 'official', 'community', or None (not this app's MCP).

    official  = the vendor's own MCP: a real MCP endpoint on the vendor domain
                (mcp.X.com / X.com/mcp / …-mcp-server), OR a GitHub repo whose OWNER org
                matches the app (github.com/firecrawl/firecrawl-mcp-server).
    community = a GitHub/registry MCP that names the app but is NOT under the vendor's org
                (github.com/eadm/grain-mcp-server, smithery.ai/server/<app>).
    None      = generic/unrelated — merely mentions MCP, or a registry index page.
    """
    url = r.url or ""
    ul, tl = url.lower(), (r.title or "").lower()
    if not url or ("mcp" not in ul and "mcp" not in tl and "model context protocol" not in tl):
        return None
    host = _domain(url)
    path = urlparse(url).path.lower()
    mcp_in_ref = "mcp" in ul or "mcp" in tl or "model context protocol" in tl
    name_in_ref = any(t in ul or t in tl for t in tokens)
    strong_mcp = host.startswith("mcp.") or "/mcp" in path or "-mcp" in ul or "mcp-server" in ul
    is_vendor = bool(app_root) and (host == app_root or host.endswith("." + app_root))

    if "github.com" in host:
        if not (mcp_in_ref and name_in_ref):
            return None
        parts = [p for p in path.split("/") if p]
        owner = parts[0] if parts else ""
        owner_match = any(t in owner for t in tokens)
        return "official" if owner_match else "community"
    if is_vendor and strong_mcp:
        return "official"
    if name_in_ref and mcp_in_ref and any(h in host for h in MCP_REGISTRY_HOSTS):
        return "community"
    return None


def _mcp_match_tokens(seed: AppSeed) -> set[str]:
    """Significant name/domain tokens that must appear in an MCP hit for it to count."""
    toks: set[str] = set()
    for t in re.split(r"[^a-z0-9]+", seed.name.lower()):
        if len(t) >= 4 and t not in MCP_STOPWORDS:
            toks.add(t)
    dom = _domain(seed.hint_url)
    if dom:
        label = _root_domain(dom).split(".")[0]
        if len(label) >= 4 and label not in MCP_STOPWORDS and label not in HOSTING_LABELS:
            toks.add(label)
    return toks


def _prioritize(urls: list[str], seed: AppSeed, limit: int) -> list[str]:
    app_root = _root_domain(_domain(seed.hint_url)) if seed.hint_url else ""
    seen: set[str] = set()
    scored: list[tuple[int, int, str]] = []
    for i, url in enumerate(urls):
        if not url or url in seen:
            continue
        seen.add(url)
        u = url.lower()
        host = _domain(url)
        score = sum(2 for h in DOC_HINTS if h in u)
        if app_root and app_root in host:
            score += 6
        if any(n in u for n in NOISE_HOSTS):
            score -= 4
        scored.append((score, -i, url))  # -i keeps original order as tiebreak
    scored.sort(reverse=True)
    return [url for _, _, url in scored[:limit]]


# ---------------------------------------------------------------------------
# Loop reconciliation helpers
# ---------------------------------------------------------------------------
def _apply_cross_check(row: AppResult, ext: AppExtraction, cc: CrossCheck) -> dict:
    """L2: the blind second pass is used to DETECT disagreement, not to overwrite values.
    Disagreements lower confidence and route the row to the human queue (L4). We do not
    auto-union auth methods — unioning was found to inject the second pass's own errors
    (e.g. a spurious OAuth2). The grounded multi-page extraction already handles multi-auth;
    MCP is arbitrated by the stronger L3 probe below."""
    dis: dict = {}

    if set(cc.auth_methods) != set(ext.auth_methods) and cc.auth_methods:
        dis["auth_methods"] = {"extract": [a.value for a in ext.auth_methods],
                               "crosscheck": [a.value for a in cc.auth_methods]}
        row.confidence["auth_methods"] = min(row.confidence.get("auth_methods", 0.5), 0.55)

    if cc.self_serve != ext.self_serve:
        dis["self_serve"] = {"extract": ext.self_serve.value, "crosscheck": cc.self_serve.value}
        row.confidence["self_serve"] = min(row.confidence.get("self_serve", 0.5), 0.5)

    if cc.existing_mcp != ext.api_surface.existing_mcp:
        dis["existing_mcp"] = {"extract": ext.api_surface.existing_mcp.value,
                               "crosscheck": cc.existing_mcp.value}
    return dis


def _apply_mcp_probe(row: AppResult, probe: dict) -> None:
    """L3: the targeted probe carries concrete, app-specific evidence, so it is the
    authoritative signal for MCP existence — stronger than the model's memory.

    * Probe found an app-specific MCP → adopt it (fixes both hallucinated 'official' claims
      and missed real MCPs), attach the evidence URL.
    * Probe found nothing but the extractor claimed one → downgrade to 'unknown' (H1 catch).
    """
    verdict = ExistingMcp(probe["verdict"])
    claimed = row.api_surface.existing_mcp
    note = None

    if verdict in (ExistingMcp.official, ExistingMcp.community):
        if claimed != verdict:
            note = f"MCP set to '{verdict.value}' by targeted probe (was '{claimed.value}')."
        row.api_surface.existing_mcp = verdict
        row.confidence["existing_mcp"] = 0.8
        if probe["evidence"]:
            row.evidence_urls["existing_mcp"] = probe["evidence"][0]["url"]
    elif claimed in (ExistingMcp.official, ExistingMcp.community):
        # H1 catch: claimed an MCP the app-specific probe can't corroborate → downgrade.
        row.api_surface.existing_mcp = ExistingMcp.unknown
        row.confidence["existing_mcp"] = 0.3
        note = (f"MCP claim '{claimed.value}' NOT corroborated by targeted probe "
                f"({probe['n_hits']} app-specific hits) → downgraded to 'unknown'.")
    else:
        row.api_surface.existing_mcp = ExistingMcp.none
        row.confidence["existing_mcp"] = max(row.confidence.get("existing_mcp", 0.5), 0.6)

    if note:
        row.notes = (row.notes + " " + note).strip() if row.notes else note


def _apply_confidence_gate(row: AppResult, disagreements: dict, probe: dict) -> None:
    """L4: low confidence or unresolved disagreement → human review queue (PRD §8.2)."""
    reasons: list[str] = []
    low = [f for f, c in row.confidence.items() if c < SETTINGS.confidence_threshold]
    if low:
        reasons.append(f"low confidence on: {', '.join(sorted(low))}")
    if disagreements.get("self_serve"):
        reasons.append("cross-check disagreed on self_serve")
    if disagreements.get("existing_mcp"):
        reasons.append("cross-check disagreed on existing_mcp")
    # gated apps with no usable API surface are worth a human glance
    if row.buildability == Buildability.blocked and not row.evidence_urls:
        reasons.append("blocked with no evidence URL captured")

    if reasons:
        row.needs_human_review = True
        row.human_reason = "; ".join(reasons)


def _error_row(seed: AppSeed, pass_label: str, err: str, backend: str) -> AppResult:
    return AppResult(
        id=seed.id,
        name=seed.name,
        category=seed.category,
        buildability=Buildability.buildable_with_effort,
        self_serve=SelfServe.contact_sales,
        pass_label=pass_label,
        backend=backend,
        needs_human_review=True,
        human_reason="research error — needs manual research",
        error=err,
        notes=f"Automated research failed: {err}",
    )
