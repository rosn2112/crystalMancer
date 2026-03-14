"""OpenRouter LLM client for free model inference.

Supports automatic fallback through multiple free models and
structured JSON extraction from scientific text.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any

import requests

from crystalmancer.config import BACKOFF_BASE, BACKOFF_FACTOR, BACKOFF_MAX, JITTER_MAX, MAX_RETRIES

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free models — ordered by quality (strongest first, then fallbacks)
# Update this list periodically; free models rotate on OpenRouter.
# Check https://openrouter.ai/models?q=free for current availability.
FREE_MODELS: list[str] = [
    # ── Tier 1: Strongest reasoning ──
    "qwen/qwen3-235b-a22b:free",                          # Qwen3 235B MoE
    "meta-llama/llama-4-maverick:free",                   # Llama 4 Maverick
    # ── Tier 2: Strong instruction-following ──
    "google/gemma-3-27b-it:free",                         # Gemma 3 27B
    "mistralai/mistral-small-3.1-24b-instruct:free",      # Mistral Small 3.1
    "qwen/qwen3-32b:free",                                # Qwen3 32B
    # ── Tier 3: Fast fallbacks ──
    "google/gemma-3-12b-it:free",                         # Gemma 3 12B
    "meta-llama/llama-4-scout:free",                      # Llama 4 Scout
    "deepseek/deepseek-chat-v3-0324:free",                # DeepSeek V3
]


def _get_api_key() -> str:
    """Retrieve the OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Get a free key at https://openrouter.ai/keys"
        )
    return key


def _backoff_sleep(attempt: int) -> None:
    delay = min(BACKOFF_BASE * (BACKOFF_FACTOR ** attempt), BACKOFF_MAX)
    delay += random.uniform(0, JITTER_MAX)
    time.sleep(delay)


def chat_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    json_mode: bool = False,
) -> str:
    """Send a chat completion request to OpenRouter.

    Parameters
    ----------
    messages : list[dict]
        Chat messages in OpenAI format: [{"role": "user", "content": "..."}]
    model : str | None
        Model ID. If None, tries free models in order.
    temperature : float
        Sampling temperature (low for extraction tasks).
    max_tokens : int
        Maximum output tokens.
    json_mode : bool
        If True, request JSON response format.

    Returns
    -------
    str
        The assistant's response text.

    Raises
    ------
    RuntimeError
        If all models fail after retries.
    """
    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/crystalmancer",
        "X-Title": "Crystal Mancer",
    }

    models_to_try = [model] if model else FREE_MODELS

    for model_id in models_to_try:
        body: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=body,
                    timeout=60,
                )

                if resp.status_code == 429:
                    logger.warning("Rate limited on %s (attempt %d).", model_id, attempt + 1)
                    _backoff_sleep(attempt)
                    continue

                if resp.status_code >= 500:
                    logger.warning("Server error %d on %s.", resp.status_code, model_id)
                    _backoff_sleep(attempt)
                    continue

                resp.raise_for_status()
                data = resp.json()

                choices = data.get("choices", [])
                if not choices:
                    logger.warning("Empty choices from %s.", model_id)
                    break  # try next model

                content = choices[0].get("message", {}).get("content", "")
                if content:
                    logger.debug("Got response from %s (%d chars).", model_id, len(content))
                    return content

            except requests.exceptions.RequestException as exc:
                if attempt == MAX_RETRIES - 1:
                    logger.warning("Model %s failed: %s. Trying next.", model_id, exc)
                    break
                _backoff_sleep(attempt)

    raise RuntimeError("All OpenRouter models failed after retries.")


def extract_json(
    prompt: str,
    system_prompt: str = "",
    model: str | None = None,
) -> dict[str, Any]:
    """Send a prompt and parse the response as JSON.

    Parameters
    ----------
    prompt : str
        The user prompt (typically containing text to analyze).
    system_prompt : str
        System instructions (e.g., extraction schema).
    model : str | None
        Model ID override.

    Returns
    -------
    dict
        Parsed JSON from the model's response.

    Raises
    ------
    ValueError
        If the response cannot be parsed as JSON.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = chat_completion(messages, model=model, json_mode=True)

    # Try to extract JSON from response (handle markdown code blocks)
    text = response.strip()
    if text.startswith("```"):
        # Strip markdown code fences
        lines = text.split("\n")
        # Remove first and last lines if they are fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object within the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}...")
