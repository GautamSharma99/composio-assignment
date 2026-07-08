"""OpenAI structured-output extraction (Pass 1 & 2) + the L2 independent cross-check.

Pass 1 vs Pass 2 differ *here* in exactly two ways, and nowhere sneaky:
  * Pass 1 (``grounded=False``) gets ONE page and a permissive "use your knowledge,
    best-guess every field" prompt — this is the naive baseline that triggers the
    pre-registered failure modes (hallucinated MCP, unread pricing, collapsed auth).
  * Pass 2 (``grounded=True``) gets several targeted pages and a strict "quote your
    evidence or answer unknown, never fabricate MCP" prompt (loop L1).

The cross-check (``cross_check``) is loop L2: a second, independent derivation of the
three most hallucination-prone fields, blind to the first extraction.
"""

from __future__ import annotations

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .backends.base import FetchedPage
from .config import SETTINGS
from .schema import AppExtraction, AppSeed, CrossCheck

_TRANSIENT = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def format_sources(pages: list[FetchedPage], max_chars: int) -> str:
    """Concatenate fetched pages into a bounded SOURCE block, budget split across pages."""
    usable = [p for p in pages if p.ok and p.text]
    if not usable:
        return "(no source pages could be fetched)"
    per_page = max(800, max_chars // len(usable))
    blocks = []
    for i, p in enumerate(usable, 1):
        text = p.text[:per_page]
        header = f"### SOURCE {i}: {p.url}"
        if p.title:
            header += f"\n(title: {p.title})"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


_GROUNDED_SYS = """You are a meticulous API-integration research analyst at Composio. \
Composio turns apps into tools that AI agents can call, so you assess how build-ready an \
app's developer API is.

From ONLY the SOURCE pages provided, extract a strict structured profile for the app.

RULES (follow exactly):
1. Ground every field in the SOURCE text. Do NOT use prior knowledge or guess.
2. For each field, put the exact supporting quote in the matching *_evidence_quote and \
the page URL in the matching *_evidence_url. If the sources don't support a value, pick \
the 'unknown' option (or an empty auth list), set that field's confidence <= 0.3, and say \
so in notes. Never fabricate.
3. existing_mcp: only 'official' or 'community' if a source EXPLICITLY mentions an MCP \
server for this app. Otherwise 'none' or 'unknown'. Hallucinated MCPs are the #1 failure — \
do not invent one.
4. auth_methods: list ALL developer auth methods the API supports (e.g. OAuth2 AND API_key). \
Do not collapse multiple methods into one.
5. self_serve — how a developer actually obtains WORKING API access. Choose precisely:
   - self_serve: sign up and get working credentials instantly, free, no approval or payment.
   - paid_gated: you must add a payment method / prepaid credits / a paid plan before the API \
works, EVEN IF key creation looks instant (e.g. no free tier).
   - free_trial: usable only during a trial or on a paid account that offers a trial.
   - admin_approval: a workspace/account admin must enable it.
   - partner_gated: you must apply to and be approved for a developer/partner program, pass \
app review, or complete business verification before real access.
   - contact_sales: access is sold via sales / enterprise contract only.
   When in doubt between self_serve and a gated value, the presence of "requires payment", \
"app review", "business verification", "apply", or "contact us" makes it gated — not self_serve.
6. buildability:
   - easy_win: public REST/GraphQL + self_serve credentials + usable docs.
   - buildable_with_effort: there IS a public, documented API, but auth/approval/partner/\
payment/limited-surface adds real friction. Use this even for approval-gated apps AS LONG AS \
the docs and an API genuinely exist.
   - blocked: reserve for apps you effectively cannot build against — NO public API, or the \
API docs themselves are not publicly viewable (login/partner-walled), or access is enterprise-\
sales-only with no developer path. Radical honesty is correct here: name the blocker and cite \
the gate. A well-evidenced "blocked" beats a confident guess."""

_NAIVE_SYS = """You are an assistant that fills in a developer-API profile for the given app. \
Use the SOURCE page below plus your general knowledge to complete every field of the schema \
with your best answer. Provide your best guess for each field even if the page doesn't cover \
it, and fill the evidence/confidence fields as best you can."""

_CROSSCHECK_SYS = """You independently verify three facts about an app's developer API, using \
ONLY the SOURCE pages provided. Determine: (a) auth_methods — every developer auth method \
supported; (b) self_serve — how a developer obtains access; (c) existing_mcp — whether an MCP \
server exists (only 'official'/'community' if a source explicitly says so, else 'none'/'unknown'). \
Be conservative: if the sources don't explicitly show something, choose unknown/none. Do not \
assume or use outside knowledge. Briefly cite what in the sources drove each answer."""


def _user_msg(seed: AppSeed, sources: str) -> str:
    dis = f"\nDisambiguation (critical — research the RIGHT product): {seed.disambiguation}" if seed.disambiguation else ""
    return f"App: {seed.name}\nCategory: {seed.category}{dis}\n\n{sources}"


class LLM:
    def __init__(self) -> None:
        if not SETTINGS.has_openai:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot run extraction.")
        self.client = AsyncOpenAI(api_key=SETTINGS.openai_api_key)
        self.model = SETTINGS.openai_model

    @retry(
        retry=retry_if_exception_type(_TRANSIENT),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(SETTINGS.max_retries),
        reraise=True,
    )
    async def _parse(self, messages: list[dict], response_format):
        kwargs = dict(model=self.model, messages=messages, response_format=response_format)
        try:
            resp = await self.client.chat.completions.parse(temperature=0, **kwargs)
        except BadRequestError as e:
            if "temperature" in str(e).lower():  # some models reject custom temperature
                resp = await self.client.chat.completions.parse(**kwargs)
            else:
                raise
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            refusal = getattr(resp.choices[0].message, "refusal", None)
            raise ValueError(f"model returned no parsed object (refusal={refusal!r})")
        return parsed

    async def extract(self, seed: AppSeed, pages: list[FetchedPage], *, grounded: bool) -> AppExtraction:
        sys = _GROUNDED_SYS if grounded else _NAIVE_SYS
        sources = format_sources(pages, SETTINGS.max_page_chars)
        return await self._parse(
            [{"role": "system", "content": sys}, {"role": "user", "content": _user_msg(seed, sources)}],
            AppExtraction,
        )

    async def cross_check(self, seed: AppSeed, pages: list[FetchedPage]) -> CrossCheck:
        sources = format_sources(pages, SETTINGS.max_page_chars)
        return await self._parse(
            [{"role": "system", "content": _CROSSCHECK_SYS}, {"role": "user", "content": _user_msg(seed, sources)}],
            CrossCheck,
        )
