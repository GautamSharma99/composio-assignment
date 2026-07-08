"""Runtime configuration, loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project layout
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
APPS_JSON = ROOT / "apps.json"
RESULTS_JSON = DATA_DIR / "results.json"
GROUND_TRUTH_JSON = DATA_DIR / "ground_truth.json"
ACCURACY_JSON = DATA_DIR / "accuracy_report.json"
SITE_DIR = ROOT / "site"

load_dotenv(ROOT / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    composio_api_key: str | None
    use_fallback: bool
    openai_model: str
    proof_app: str
    composio_user_id: str

    # Orchestrator knobs
    concurrency: int = 6
    per_app_timeout: float = 90.0
    max_retries: int = 3
    search_results: int = 5
    max_page_chars: int = 12_000  # cap page text sent to the LLM (cost control)
    confidence_threshold: float = 0.6  # loop L4: below this -> human queue

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_composio(self) -> bool:
        return bool(self.composio_api_key)

    def resolve_backend(self) -> str:
        """Which backend research should *prefer*. 'composio' unless forced/absent."""
        if self.use_fallback or not self.has_composio:
            return "fallback"
        return "composio"


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        composio_api_key=os.getenv("COMPOSIO_API_KEY") or None,
        use_fallback=_as_bool(os.getenv("USE_FALLBACK"), default=False),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        proof_app=os.getenv("PROOF_APP", "github"),
        composio_user_id=os.getenv("COMPOSIO_USER_ID", "composio-assignment-user"),
    )


SETTINGS = load_settings()
