"""Environment-backed configuration for optional Gemini features."""

import os


DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
# Checked against Google's Gemini API pricing page on 2026-11-17. Keep this
# allowlist narrow; an unrecognized/paid-only override must fail closed.
FREE_TIER_GEMINI_MODELS = frozenset({DEFAULT_GEMINI_MODEL})
DEFAULT_GEMINI_TIMEOUT_MS = 25000
DEFAULT_GEMINI_REQUESTS_PER_MINUTE = 30


def gemini_api_key():
    return os.environ.get("GEMINI_API_KEY", "").strip()


def gemini_model():
    model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    if model not in FREE_TIER_GEMINI_MODELS:
        raise ValueError("GEMINI_MODEL is not in the verified free-tier model allowlist.")
    return model


def gemini_timeout_ms():
    try:
        value = int(os.environ.get("GEMINI_TIMEOUT_MS", str(DEFAULT_GEMINI_TIMEOUT_MS)))
    except (TypeError, ValueError) as exc:
        raise ValueError("GEMINI_TIMEOUT_MS must be an integer.") from exc
    if value < 1000 or value > 120000:
        raise ValueError("GEMINI_TIMEOUT_MS must be between 1000 and 120000.")
    return value


def gemini_requests_per_minute():
    try:
        value = int(
            os.environ.get(
                "GEMINI_REQUESTS_PER_MINUTE",
                str(DEFAULT_GEMINI_REQUESTS_PER_MINUTE),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("GEMINI_REQUESTS_PER_MINUTE must be an integer.") from exc
    if value < 1 or value > 600:
        raise ValueError("GEMINI_REQUESTS_PER_MINUTE must be between 1 and 600.")
    return value
