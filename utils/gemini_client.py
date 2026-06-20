"""
utils/gemini_client.py
Thin shared wrapper around google-generativeai providing:
  - lazy, single-point model initialization
  - rate limiting (sleep-based token spacing)
  - exponential-backoff retries on transient errors
  - robust JSON extraction from model responses
  - lightweight token/call usage tracking for the evaluation report

agents/claim_parser.py and agents/vision_analyzer.py both call through here
so retry/rate-limit/parsing behavior is implemented exactly once.
"""

import hashlib
import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai

import config
from utils.logger import get_logger

logger = get_logger(__name__)

_model = None
_last_call_ts = 0.0

# ---------------------------------------------------------------------------
# Disk-based response cache.
# Keyed on a hash of (prompt text + image bytes), so re-running the pipeline
# on unchanged claims/images never re-spends API calls or tokens. This is the
# caching strategy referenced in evaluation/evaluation_report.md.
# ---------------------------------------------------------------------------
CACHE_DIR = config.OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ENABLED = os.environ.get("GEMINI_DISABLE_CACHE", "") != "1"


def _cache_key(prompt: str, image) -> str:
    digest = hashlib.sha256()
    digest.update(prompt.encode("utf-8"))
    if image is not None:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        digest.update(buf.getvalue())
    return digest.hexdigest()


def _cache_path(key: str):
    return CACHE_DIR / f"{key}.json"


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _cache_set(key: str, value: Dict[str, Any]):
    try:
        _cache_path(key).write_text(json.dumps(value), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write cache entry %s: %s", key, exc)


class GeminiCallError(RuntimeError):
    """Raised when a Gemini call fails after all retries are exhausted."""


class UsageTracker:
    """Process-wide counters used to build the evaluation report."""

    def __init__(self):
        self.text_calls = 0
        self.vision_calls = 0
        self.failed_calls = 0
        self.retried_calls = 0
        self.cache_hits = 0
        self.estimated_input_tokens = 0
        self.estimated_output_tokens = 0

    def record_call(self, kind: str, prompt_chars: int, response_chars: int):
        # Rough, conservative heuristic: ~4 characters per token.
        self.estimated_input_tokens += max(prompt_chars // 4, 1)
        self.estimated_output_tokens += max(response_chars // 4, 1)
        if kind == "text":
            self.text_calls += 1
        elif kind == "vision":
            self.vision_calls += 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text_calls": self.text_calls,
            "vision_calls": self.vision_calls,
            "total_calls": self.text_calls + self.vision_calls,
            "cache_hits": self.cache_hits,
            "failed_calls": self.failed_calls,
            "retried_calls": self.retried_calls,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
        }


usage = UsageTracker()


def _get_model():
    global _model
    if _model is not None:
        return _model

    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your environment or a .env "
            "file before running the pipeline."
        )

    genai.configure(api_key=config.GEMINI_API_KEY)
    _model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
    return _model


def _respect_rate_limit():
    global _last_call_ts
    now = time.monotonic()
    elapsed = now - _last_call_ts
    wait = config.MIN_SECONDS_BETWEEN_CALLS - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.monotonic()


def _extract_json(raw_text: str) -> Dict[str, Any]:
    """
    Best-effort extraction of a JSON object from a model response.
    Handles the common failure modes: markdown code fences, leading/trailing
    prose, or minor trailing commas.
    """
    text = raw_text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: grab the largest {...} block.
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)  # trailing commas
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise GeminiCallError(f"Could not parse JSON from model response: {exc}") from exc

    raise GeminiCallError("Model response contained no JSON object.")


def generate_structured(
    prompt: str,
    image=None,
    kind: str = "text",
) -> Dict[str, Any]:
    """
    Calls Gemini with the given prompt (and optional PIL image), enforcing
    JSON-mode generation, rate limiting, and retry-with-backoff. Returns the
    parsed JSON dict. Raises GeminiCallError if all retries fail.
    """
    contents: List[Union[str, Any]] = [prompt] if image is None else [prompt, image]

    cache_key = _cache_key(prompt, image) if CACHE_ENABLED else None
    if cache_key:
        cached = _cache_get(cache_key)
        if cached is not None:
            usage.cache_hits += 1
            return cached

    model = _get_model()

    last_exc: Optional[Exception] = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        _respect_rate_limit()
        try:
            response = model.generate_content(
                contents,
                generation_config=config.GENERATION_CONFIG_JSON,
            )
            raw_text = response.text or ""
            parsed = _extract_json(raw_text)
            usage.record_call(kind, prompt_chars=len(prompt), response_chars=len(raw_text))
            if cache_key:
                _cache_set(cache_key, parsed)
            return parsed

        except Exception as exc:  # noqa: BLE001 - intentionally broad: SDK raises various error types
            last_exc = exc
            usage.retried_calls += 1
            wait = config.RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Gemini call failed (attempt %d/%d, kind=%s): %s -- retrying in %.1fs",
                attempt, config.MAX_RETRIES, kind, exc, wait,
            )
            time.sleep(wait)

    usage.failed_calls += 1
    raise GeminiCallError(
        f"Gemini call failed after {config.MAX_RETRIES} attempts: {last_exc}"
    ) from last_exc
