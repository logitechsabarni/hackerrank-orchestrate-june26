"""
agents/vision_analyzer.py
Stage 2 of the pipeline: per-image visual analysis against the
extracted claim. Images are treated as the primary source of truth.
"""

from pathlib import Path
from typing import Any, Dict, List

import config
from utils.gemini_client import generate_structured, GeminiCallError
from utils.image_utils import (
    assess_image_quality,
    load_image_for_model,
    resolve_image_path,
)
from utils.logger import get_logger

logger = get_logger(__name__)

with open(config.PROMPTS_DIR / "vision_prompt.txt", "r", encoding="utf-8") as f:
    _VISION_PROMPT_TEMPLATE = f.read()


def _clean_enum(value: Any, allowed: List[str], default: str = "unknown") -> str:
    if not isinstance(value, str):
        return default
    value = value.strip().lower().replace(" ", "_")
    return value if value in allowed else default


def _clean_flag_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned = []
    for v in values:
        if isinstance(v, str):
            v = v.strip().lower().replace(" ", "_")
            if v in config.RISK_FLAGS:
                cleaned.append(v)
    return sorted(set(cleaned))


def _clean_bool_or_none(value: Any):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
            return False
    return None


class VisionAnalyzer:
    """Runs per-image visual evidence analysis for a single claim."""

    def analyze_claim_images(
        self,
        image_paths: List[str],
        claim_object: str,
        normalized_claim_text: str,
        extracted_issue_type: str,
        extracted_object_part: str,
    ) -> List[Dict[str, Any]]:
        claim_object = (claim_object or "").strip().lower()
        object_part_options = config.OBJECT_PARTS_BY_CLAIM_OBJECT.get(
            claim_object, config.OBJECT_PARTS_BY_CLAIM_OBJECT["car"]
        )

        results = []
        for idx, raw_path in enumerate(image_paths):
            resolved_path = resolve_image_path(raw_path)
            results.append(
                self._analyze_single_image(
                    resolved_path=resolved_path,
                    raw_path=raw_path,
                    claim_object=claim_object,
                    object_part_options=object_part_options,
                    normalized_claim_text=normalized_claim_text,
                    extracted_issue_type=extracted_issue_type,
                    extracted_object_part=extracted_object_part,
                )
            )
        return results

    def _analyze_single_image(
        self,
        resolved_path: Path,
        raw_path: str,
        claim_object: str,
        object_part_options: List[str],
        normalized_claim_text: str,
        extracted_issue_type: str,
        extracted_object_part: str,
    ) -> Dict[str, Any]:
        quality = assess_image_quality(resolved_path)

        base_result = {
            "raw_path": raw_path,
            "resolved_path": str(resolved_path),
            "is_valid_image": quality["is_loadable"],
            "claim_object_present": None,
            "object_part_detected": "unknown",
            "issue_type_detected": "unknown",
            "damage_matches_claim": None,
            "severity": "unknown",
            "image_quality_flags": list(quality["local_quality_flags"]),
            "confidence": 0.0,
            "notes": "",
            "analyzer_error": None,
        }

        if not quality["is_loadable"]:
            base_result["notes"] = "Image could not be loaded (missing or corrupt file)."
            return base_result

        model_image = load_image_for_model(resolved_path)
        if model_image is None:
            base_result["is_valid_image"] = False
            base_result["notes"] = "Image failed to prepare for model input."
            return base_result

        prompt = _VISION_PROMPT_TEMPLATE.format(
            claim_object=claim_object or "unknown",
            normalized_claim_text=normalized_claim_text or "",
            extracted_issue_type=extracted_issue_type or "unknown",
            extracted_object_part=extracted_object_part or "unknown",
            object_part_options=", ".join(object_part_options),
            issue_type_options=", ".join(config.ISSUE_TYPES),
            risk_flag_options=", ".join(config.RISK_FLAGS),
        )

        try:
            raw = generate_structured(prompt, image=model_image, kind="vision")
        except GeminiCallError as exc:
            logger.error("Vision analysis failed for %s: %s", resolved_path, exc)
            base_result["analyzer_error"] = str(exc)
            base_result["notes"] = "Automated vision analysis failed after retries."
            return base_result

        model_flags = _clean_flag_list(raw.get("image_quality_flags"))
        merged_flags = sorted(set(quality["local_quality_flags"] + model_flags))

        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        is_valid = raw.get("is_valid_image")
        is_valid = True if is_valid is None else bool(is_valid)

        return {
            "raw_path": raw_path,
            "resolved_path": str(resolved_path),
            "is_valid_image": is_valid and quality["is_loadable"],
            "claim_object_present": _clean_bool_or_none(raw.get("claim_object_present")),
            "object_part_detected": _clean_enum(raw.get("object_part_detected"), object_part_options),
            "issue_type_detected": _clean_enum(raw.get("issue_type_detected"), config.ISSUE_TYPES),
            "damage_matches_claim": _clean_bool_or_none(raw.get("damage_matches_claim")),
            "severity": _clean_enum(raw.get("severity"), config.SEVERITY_LEVELS, default="unknown"),
            "image_quality_flags": merged_flags,
            "confidence": confidence,
            "notes": raw.get("notes", "") if isinstance(raw.get("notes", ""), str) else "",
            "analyzer_error": None,
        }
