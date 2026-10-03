from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Configuration for LLM providers shared by agents."""

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider name, handling aliases and common typos."""
    normalized = value.strip().lower()
    aliases = {
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "google": "gemini",
        "google-genai": "gemini",
        "gemini-pro": "gemini",
        "open-ai": "openai",
        "chatgpt": "openai",
        "open_router": "openrouter",
        "router": "openrouter",
        "local": "ollama",
        "ollama-llm": "ollama",
        "openai-compatible": "custom",
    }
    return aliases.get(normalized, normalized)


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate chat model for the selected provider.
    
    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter (using ChatOpenAI with OpenRouter base URL)
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("CUSTOM_API_KEY", "custom-key")
        base_url = config.base_url or os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
            base_url=base_url,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        base_url = config.base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://localhost",
                "X-Title": "Memory Systems Lab",
            },
        )


    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model_name=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = config.base_url or os.getenv("OLLAMA_BASE_URL")
        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)

    raise ValueError(f"Unsupported provider: '{config.provider}' (normalized: '{provider}')")
