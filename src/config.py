from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    # Memory quality guardrails (see STEP8.md, bonus section)
    confidence_threshold: float = 0.6
    decay_half_life_turns: float = 200.0
    decay_min_score: float = 0.25


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load configuration and environment variables."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load .env file if available
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()

    data_dir = root / "data"
    state_dir = root / "state"

    # Ensure state directory exists
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "600"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key = os.getenv("LLM_API_KEY") or os.getenv(f"{provider.upper()}_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("CUSTOM_BASE_URL")

    model_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        api_key=api_key,
        base_url=base_url,
    )

    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)
    judge_api_key = os.getenv("JUDGE_API_KEY") or api_key
    judge_base_url = os.getenv("JUDGE_BASE_URL") or base_url

    judge_model_config = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_config,
        judge_model=judge_model_config,
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.6")),
        decay_half_life_turns=float(os.getenv("DECAY_HALF_LIFE_TURNS", "200")),
        decay_min_score=float(os.getenv("DECAY_MIN_SCORE", "0.25")),
    )
