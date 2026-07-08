"""Pydantic models — the strict target that makes verification measurable (PRD §6).

Two model families:

* ``AppExtraction`` (+ nested field-evidence models) is what the LLM is forced to
  emit via OpenAI structured outputs. It uses *explicit* per-field evidence/quote/
  confidence sub-objects rather than open dicts, because strict JSON-schema mode
  forbids arbitrary-key dicts — and, more importantly, forcing a quote per field is
  exactly loop L1 (evidence grounding).
* ``AppResult`` is the storage/serialization model written to ``results.json``. It
  matches PRD §6 one-to-one (dict-keyed ``evidence_urls`` / ``confidence``) and is
  assembled from an ``AppExtraction`` plus the app seed and pipeline metadata.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# The 10 categories (PRD: "one of the 10").
# ---------------------------------------------------------------------------
CATEGORIES: list[str] = [
    "Dev & Infrastructure",
    "Productivity & Docs",
    "Communication & Messaging",
    "CRM & Sales",
    "Finance & Fintech",
    "Marketing & Ads",
    "Commerce & Payments",
    "Data, SEO & Scraping",
    "AI & Research",
    "Social & Media",
]

# The fields we score accuracy on (must exist in ground_truth.json rows).
SCORED_FIELDS: list[str] = [
    "auth_methods",
    "self_serve",
    "api_type",
    "api_breadth",
    "public_docs",
    "existing_mcp",
    "buildability",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class AuthMethod(str, Enum):
    OAUTH2 = "OAuth2"
    API_KEY = "API_key"
    BASIC = "Basic"
    TOKEN = "Token"
    OTHER = "Other"


class SelfServe(str, Enum):
    self_serve = "self_serve"
    free_trial = "free_trial"
    paid_gated = "paid_gated"
    admin_approval = "admin_approval"
    partner_gated = "partner_gated"
    contact_sales = "contact_sales"


class ApiType(str, Enum):
    REST = "REST"
    GraphQL = "GraphQL"
    both = "both"
    SDK_only = "SDK_only"
    none = "none"
    unknown = "unknown"


class Breadth(str, Enum):
    broad = "broad"
    moderate = "moderate"
    narrow = "narrow"
    unknown = "unknown"


class ExistingMcp(str, Enum):
    official = "official"
    community = "community"
    none = "none"
    unknown = "unknown"


class Buildability(str, Enum):
    easy_win = "easy_win"
    buildable_with_effort = "buildable_with_effort"
    blocked = "blocked"


class VerificationStatus(str, Enum):
    unverified = "unverified"
    agent_verified = "agent_verified"
    human_verified = "human_verified"


# ---------------------------------------------------------------------------
# LLM extraction models (structured output target)
# ---------------------------------------------------------------------------
class ApiSurfaceExtraction(BaseModel):
    """API surface sub-object as extracted by the model."""

    type: ApiType
    breadth: Breadth
    public_docs: bool
    existing_mcp: ExistingMcp
    evidence_url: str = Field(description="URL of the page that supports this API-surface assessment.")
    evidence_quote: str = Field(description="Exact text quoted from that page. Empty string if none found.")
    confidence: float = Field(description="0.0-1.0 confidence in the API-surface assessment.")


class AppExtraction(BaseModel):
    """Exactly what the extractor LLM returns. Every claim carries a quote + URL (loop L1)."""

    one_liner: str = Field(description="What the app does, in one line.")

    auth_methods: list[AuthMethod] = Field(
        description="ALL developer auth methods the API supports. Do not collapse OAuth2+API_key to one."
    )
    auth_evidence_url: str
    auth_evidence_quote: str = Field(description="Exact quote naming the auth method(s). Empty if none found.")
    auth_confidence: float = Field(description="0.0-1.0 confidence in the auth_methods answer.")

    self_serve: SelfServe = Field(
        description="How a developer obtains API access: self_serve (sign up + keys instantly), "
        "free_trial, paid_gated (must pay), admin_approval, partner_gated (apply/be approved), "
        "or contact_sales."
    )
    self_serve_evidence_url: str
    self_serve_evidence_quote: str
    self_serve_confidence: float = Field(description="0.0-1.0 confidence in the self_serve answer.")

    api_surface: ApiSurfaceExtraction

    buildability: Buildability = Field(
        description="easy_win (public REST/GraphQL + self-serve keys + docs), "
        "buildable_with_effort (works but auth/approval/limited surface adds effort), "
        "blocked (no public API, or docs/access are gated behind login/partner/sales)."
    )
    blocker: Optional[str] = Field(
        default=None, description="Main blocker if not an easy_win, else null."
    )
    notes: str = Field(
        default="",
        description="Honest caveats, disambiguation done, or an explicit 'this defeated me'.",
    )


class CrossCheck(BaseModel):
    """Loop L2: an independent second pass re-derives the three most hallucination-prone
    fields (auth, gating, MCP) from the same sources, without seeing the first extraction."""

    auth_methods: list[AuthMethod]
    self_serve: SelfServe
    existing_mcp: ExistingMcp
    reasoning: str = Field(description="One or two sentences citing what in the sources drove this.")


def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Storage model (results.json) — PRD §6 one-to-one
# ---------------------------------------------------------------------------
class ApiSurface(BaseModel):
    type: ApiType = ApiType.unknown
    breadth: Breadth = Breadth.unknown
    public_docs: bool = False
    existing_mcp: ExistingMcp = ExistingMcp.unknown


class AppResult(BaseModel):
    id: int
    name: str
    category: str
    one_liner: str = ""
    auth_methods: list[AuthMethod] = Field(default_factory=list)
    self_serve: SelfServe = SelfServe.contact_sales
    api_surface: ApiSurface = Field(default_factory=ApiSurface)
    buildability: Buildability = Buildability.buildable_with_effort
    blocker: Optional[str] = None
    evidence_urls: dict[str, str] = Field(default_factory=dict)
    confidence: dict[str, float] = Field(default_factory=dict)
    needs_human_review: bool = False
    verification_status: VerificationStatus = VerificationStatus.unverified
    notes: str = ""

    # --- pipeline metadata (not in the graded schema, but honest provenance) ---
    pass_label: str = "pass1"  # "pass1" | "pass2"
    backend: str = ""  # which search/scrape backend produced this row
    cross_check: Optional[dict] = None  # L2 result
    mcp_probe: Optional[dict] = None  # L3 result
    human_reason: Optional[str] = None  # why it hit the human queue, if it did
    error: Optional[str] = None  # populated if research failed for this app


# ---------------------------------------------------------------------------
# App seed (apps.json row)
# ---------------------------------------------------------------------------
class AppSeed(BaseModel):
    id: int
    name: str
    category: str
    hint_url: Optional[str] = None
    disambiguation: Optional[str] = None


def build_result(
    seed: AppSeed,
    ext: AppExtraction,
    *,
    pass_label: str,
    backend: str,
) -> AppResult:
    """Merge an LLM extraction with the app seed into a storable ``AppResult``."""
    evidence_urls: dict[str, str] = {}
    confidence: dict[str, float] = {}

    if ext.auth_evidence_url:
        evidence_urls["auth_methods"] = ext.auth_evidence_url
    confidence["auth_methods"] = clamp01(ext.auth_confidence)

    if ext.self_serve_evidence_url:
        evidence_urls["self_serve"] = ext.self_serve_evidence_url
    confidence["self_serve"] = clamp01(ext.self_serve_confidence)

    if ext.api_surface.evidence_url:
        for f in ("api_type", "api_breadth", "public_docs", "existing_mcp"):
            evidence_urls[f] = ext.api_surface.evidence_url
    for f in ("api_type", "api_breadth", "public_docs", "existing_mcp"):
        confidence[f] = clamp01(ext.api_surface.confidence)

    return AppResult(
        id=seed.id,
        name=seed.name,
        category=seed.category,
        one_liner=ext.one_liner,
        auth_methods=ext.auth_methods,
        self_serve=ext.self_serve,
        api_surface=ApiSurface(
            type=ext.api_surface.type,
            breadth=ext.api_surface.breadth,
            public_docs=ext.api_surface.public_docs,
            existing_mcp=ext.api_surface.existing_mcp,
        ),
        buildability=ext.buildability,
        blocker=ext.blocker,
        evidence_urls=evidence_urls,
        confidence=confidence,
        notes=ext.notes,
        pass_label=pass_label,
        backend=backend,
    )
