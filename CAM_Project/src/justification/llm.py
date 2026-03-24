"""wrapper around the OpenAI Python SDK with automatic provider selection."""

import os

from openai import OpenAI


def _client() -> OpenAI:
    if key := os.getenv("OPENROUTER_API_KEY"):
        return OpenAI(
            api_key=key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        )
    if key := os.getenv("OPENAI_API_KEY"):
        return OpenAI(api_key=key, timeout=float(os.getenv("LLM_TIMEOUT", "60")))
    raise RuntimeError("No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")


def call_llm(system: str, user: str) -> str:
    """Send a system + user message and return a raw JSON string."""
    model = os.getenv("LLM_MODEL")
    if not model:
        raise ValueError("LLM_MODEL environment variable is not set.")
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content
