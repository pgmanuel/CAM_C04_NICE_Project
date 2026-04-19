"""
llm.py
------
Wrapper around the OpenAI Python SDK with automatic provider selection.

This file is responsible for ONE thing only: creating a configured LLM
client and making a single call to it. All three providers (Ollama,
OpenRouter, OpenAI) speak the OpenAI-compatible API format, so the
same client code works for all three. The only difference is the
base_url and the api_key.

Provider priority order (first matching key wins):
  1. Ollama  — local model, no API key needed, best for development
  2. OpenRouter — cloud gateway to many models, one API key
  3. OpenAI direct — standard OpenAI API

To switch providers, change your .env file. No code changes needed.
"""

from __future__ import annotations

import os

from openai import OpenAI


def _client() -> OpenAI:
    """
    Build and return a configured OpenAI client.

    Why does this work for Ollama?
    Ollama exposes an OpenAI-compatible REST API at localhost:11434/v1.
    The OpenAI Python SDK just needs a base_url and any non-empty
    string as api_key (Ollama ignores it but the SDK requires one).
    This means we can reuse all the existing call_llm() code unchanged
    when switching between local and cloud providers.
    """

    # ── Option 1: Ollama (local model, no internet required) ──────────
    # Set OLLAMA_BASE_URL=http://localhost:11434/v1 in your .env
    # and LLM_MODEL=llama3.2 (or whichever model you have pulled).
    # Ollama must be running: https://ollama.com
    # Pull a model first: ollama pull llama3.2:1b
    #
    # llama3.2:1b  — fast, lightweight, good for testing (1GB download)
    # llama3.2     — better quality, slower (2GB download)
    # mistral      — good balance of speed and quality (4GB download)
    if base_url := os.getenv("OLLAMA_BASE_URL"):
        return OpenAI(
            api_key="ollama",  # Ollama ignores this but the SDK requires a non-empty value
            base_url=base_url,
            timeout=float(os.getenv("LLM_TIMEOUT", "120")),  # Local models can be slower
        )

    # ── Option 2: OpenRouter (cloud gateway) ──────────────────────────
    if key := os.getenv("OPENROUTER_API_KEY"):
        return OpenAI(
            api_key=key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai"),
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        )

    # ── Option 3: OpenAI direct ────────────────────────────────────────
    if key := os.getenv("OPENAI_API_KEY"):
        return OpenAI(
            api_key=key,
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        )

    raise RuntimeError(
        "No LLM provider configured.\n"
        "For local testing:  set OLLAMA_BASE_URL=http://localhost:11434\n"
        "For cloud:          set OPENROUTER_API_KEY or OPENAI_API_KEY"
    )


def call_llm(system: str, user: str) -> str:
    """
    Send a system + user message and return a raw JSON string.

    Why does this return a raw string rather than a parsed dict?
    Because this function's only job is to communicate with the LLM.
    Parsing the response into structured data is service.py's job.
    Keeping concerns separate means you can change how parsing works
    without touching the LLM call, and vice versa.

    response_format={"type": "json_object"} tells the model to return
    only valid JSON — no explanatory prose, no markdown code fences.
    NOTE: not all Ollama models support this parameter. If you get an
    error, remove response_format from the kwargs dict below and add
    "Return only valid JSON." to the system prompt instead.
    """
    model = os.getenv("LLM_MODEL")
    if not model:
        raise ValueError(
            "LLM_MODEL environment variable is not set.\n"
            "Example: LLM_MODEL=llama3.2:1b  (for Ollama)\n"
            "         LLM_MODEL=gpt-4o-mini   (for OpenAI)"
        )

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Build kwargs separately so we can conditionally include
    # response_format — some local models do not support it
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
    }

    # Only add response_format if not using a local Ollama model,
    # or if the user has explicitly opted in via env var.
    # Most Ollama models handle JSON reliably via prompt instructions alone.
    use_json_mode = os.getenv("USE_JSON_RESPONSE_FORMAT", "true").lower() == "true"
    if use_json_mode and not os.getenv("OLLAMA_BASE_URL"):
        kwargs["response_format"] = {"type": "json_object"}

    client = _client()
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content
