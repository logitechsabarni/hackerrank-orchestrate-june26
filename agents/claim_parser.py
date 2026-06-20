"""
agents/claim_parser.py
Stage 1 of the pipeline: extract a structured, normalized claim
(issue_type, object_part, language, injection signals) from the raw
user_claim conversation text. Pure text -- no images here.
"""

import re
from typing import Any, Dict

import config
from utils.gemini_client import generate_structured, GeminiCallError
from utils.logger import get_logger

logger = get_logger(__name__)

with open(config.PROMPTS_DIR / "claim_prompt.txt", "r", encoding="utf-8") as f:
    _CLAIM_PROMPT_TEMPLATE = f.read()


def _keyword_injection_scan(user_claim: str) -> list:
    """
    Deterministic backstop for injection detection. Runs regardless of what
    the model reports, so a model miss can never let manipulative text slip
    through unflagged.
    """
    text = (user_claim or "").lower()
    found = []
    for phrase in config.INJECTION_PHRASES:
        if phrase in text:
            found.append(phrase)
    return found


def _clean_enum(value: Any, allowed: list, default: str = "unknown") -> str:
    if not isinstance(value, str):
        return default
    value = value.strip().lower().replace(" ", "_")
    return value if value in allowed else default


class ClaimParser:
    """Extracts structured claim fields from raw conversation text."""

    def parse(self, user_claim: str, claim_object: str, history_summary: str = "") -> Dict[str, Any]:
        claim_object = (claim_object or "").strip().lower()
        object_part_options = config.OBJECT_PARTS_BY_CLAIM_OBJECT.get(
            claim_object, config.OBJECT_PARTS_BY_CLAIM_OBJECT["car"]
        )

        prompt = _CLAIM_PROMPT_TEMPLATE.format(
            claim_object=claim_object or "unknown",
            user_claim=user_claim or "",
            history_summary=history_summary or "(none provided)",
            issue_type_options=", ".join(config.ISSUE_TYPES),
            object_part_options=", ".join(object_part_options),
        )

        keyword_hits = _keyword_injection_scan(user_claim)

        try:
            raw = generate_structured(prompt, kind="text")
        except GeminiCallError as exc:
            logger.error("Claim parsing failed for claim_object=%s: %s", claim_object, exc)
            return {
                "language_detected": "unknown",
                "normalized_claim_text": user_claim or "",
                "extracted_issue_type": "unknown",
                "extracted_object_part": "unknown",
                "injection_detected": len(keyword_hits) > 0,
                "injection_phrases": keyword_hits,
                "confidence": 0.0,
                "parser_error": str(exc),
            }

        model_phrases = raw.get("injection_phrases", [])
        if not isinstance(model_phrases, list):
            model_phrases = []
        # Union of model-reported and keyword-scanned phrases -- never trust
        # only the model for a security-relevant signal.
        all_phrases = sorted(set([p.strip() for p in model_phrases if isinstance(p, str)] + keyword_hits))

        injection_detected = bool(raw.get("injection_detected")) or len(keyword_hits) > 0

        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        return {
            "language_detected": raw.get("language_detected", "unknown"),
            "normalized_claim_text": raw.get("normalized_claim_text") or (user_claim or ""),
            "extracted_issue_type": _clean_enum(raw.get("extracted_issue_type"), config.ISSUE_TYPES),
            "extracted_object_part": _clean_enum(raw.get("extracted_object_part"), object_part_options),
            "injection_detected": injection_detected,
            "injection_phrases": all_phrases,
            "confidence": confidence,
            "parser_error": None,
        }
