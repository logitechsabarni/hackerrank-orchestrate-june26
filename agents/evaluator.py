"""
agents/evaluator.py
Stage 3 of the pipeline: deterministic adjudication.

This module NEVER calls an LLM. It only combines the already-structured
outputs of ClaimParser, VisionAnalyzer and HistoryAnalyzer using the
explicit decision rules from the spec. Keeping this rule-based (instead of
asking a model "should this be approved?") is the core of the prompt-
injection defense: injected text can, at worst, flip a few risk-flag
booleans -- it can never reach the claim_status branch logic below.
"""

from typing import Any, Dict, List, Tuple

import config


def _max_severity(severities: List[str]) -> str:
    filtered = [s for s in severities if s in config.SEVERITY_ORDER]
    if not filtered:
        return "unknown"
    return max(filtered, key=lambda s: config.SEVERITY_ORDER[s])


def _most_common(values: List[str], default: str = "unknown") -> str:
    candidates = [v for v in values if v and v not in ("unknown",)]
    if not candidates:
        return default
    return max(set(candidates), key=candidates.count)


class Evaluator:
    """Combines claim parse + per-image vision results + history risk into
    a final adjudicated row."""

    def evaluate(
        self,
        claim_object: str,
        image_ids: List[str],
        vision_results: List[Dict[str, Any]],
        claim_parse: Dict[str, Any],
        history_result: Dict[str, Any],
        evidence_req: Dict[str, Any],
    ) -> Dict[str, Any]:
        extracted_object_part = claim_parse.get("extracted_object_part", "unknown")
        extracted_issue_type = claim_parse.get("extracted_issue_type", "unknown")

        indexed = list(zip(image_ids, vision_results))

        # --- 1. Validity -----------------------------------------------------
        valid_pairs = [(i, v) for i, v in indexed if v.get("is_valid_image")]
        valid_image = len(valid_pairs) > 0

        # --- 2. Usable evidence (valid + actually shows the claimed object) --
        usable_pairs = [
            (i, v) for i, v in valid_pairs if v.get("claim_object_present") is True
        ]
        evidence_count = len(usable_pairs)
        min_required = int(evidence_req.get("minimum_image_evidence", 1))
        evidence_standard_met = evidence_count >= min_required

        evidence_standard_met_reason = (
            f"{evidence_count} of {len(indexed)} submitted image(s) were valid and "
            f"clearly depicted the claimed object ('{claim_object}'); minimum required "
            f"is {min_required} (rule: {evidence_req.get('matched_rule', 'default')})."
        )
        if evidence_count == 0:
            evidence_standard_met_reason += " No usable image evidence of the claimed object was found."

        # --- 3. Is the claimed area (object_part) actually visible? ----------
        if extracted_object_part != "unknown":
            area_pairs = [
                (i, v) for i, v in usable_pairs
                if v.get("object_part_detected") == extracted_object_part
            ]
        else:
            # Claim text didn't specify a part -> any usable image of the
            # object counts as showing "the claimed area".
            area_pairs = usable_pairs

        claimed_area_visible = len(area_pairs) > 0

        # --- 4. Does visible damage match the claim? --------------------------
        matching_pairs = [
            (i, v) for i, v in area_pairs
            if v.get("damage_matches_claim") is True
            or (extracted_issue_type != "unknown" and v.get("issue_type_detected") == extracted_issue_type)
        ]
        contradicting_pairs = [p for p in area_pairs if p not in matching_pairs]

        # --- 5. Provisional status from visual evidence alone ------------------
        if evidence_count == 0 or not claimed_area_visible:
            provisional_status = "not_enough_information"
        elif matching_pairs:
            provisional_status = "supported"
        else:
            provisional_status = "contradicted"

        # --- 6. Evidence sufficiency gates the final status --------------------
        if not evidence_standard_met:
            claim_status = "not_enough_information"
        else:
            claim_status = provisional_status

        # --- 7. Reported issue_type / object_part -------------------------------
        if claim_status == "supported":
            object_part = extracted_object_part if extracted_object_part != "unknown" else _most_common(
                [v.get("object_part_detected", "unknown") for _, v in matching_pairs]
            )
            issue_type = extracted_issue_type if extracted_issue_type != "unknown" else _most_common(
                [v.get("issue_type_detected", "unknown") for _, v in matching_pairs]
            )
            severity = _max_severity([v.get("severity", "unknown") for _, v in matching_pairs])
            supporting_image_ids = [i for i, _ in matching_pairs]
        elif claim_status == "contradicted":
            object_part = _most_common([v.get("object_part_detected", "unknown") for _, v in area_pairs]) \
                if area_pairs else extracted_object_part
            issue_type = _most_common([v.get("issue_type_detected", "unknown") for _, v in area_pairs]) \
                if area_pairs else "unknown"
            severity = _max_severity([v.get("severity", "unknown") for _, v in area_pairs])
            supporting_image_ids = []
        else:  # not_enough_information
            object_part = extracted_object_part
            issue_type = extracted_issue_type
            severity = "unknown"
            supporting_image_ids = [i for i, _ in matching_pairs]  # usually empty

        # --- 8. Justification text ----------------------------------------------
        if claim_status == "supported":
            claim_status_justification = (
                f"Claimed '{issue_type}' on '{object_part}' is corroborated by "
                f"{len(matching_pairs)} image(s) showing matching damage, and the "
                f"evidence standard was met ({evidence_count}/{min_required} required images)."
            )
        elif claim_status == "contradicted":
            claim_status_justification = (
                f"The claimed area ('{extracted_object_part}') is visible in "
                f"{len(area_pairs)} image(s), but the visible condition "
                f"('{issue_type}', severity={severity}) does not match the claimed "
                f"issue ('{extracted_issue_type}')."
            )
        else:
            if evidence_count == 0:
                claim_status_justification = (
                    "No valid, usable image clearly shows the claimed object "
                    f"('{claim_object}'); cannot evaluate the claim."
                )
            elif not claimed_area_visible:
                claim_status_justification = (
                    f"The claimed object is visible but the specific claimed area "
                    f"('{extracted_object_part}') is not clearly shown in any submitted image."
                )
            else:
                claim_status_justification = (
                    f"Only {evidence_count} usable image(s) were available against a "
                    f"minimum requirement of {min_required}; insufficient evidence to "
                    "confirm or contradict the claim."
                )

        # --- 9. Risk flags --------------------------------------------------------
        risk_flags = set(history_result.get("risk_flags", []))

        for _, v in indexed:
            risk_flags.update(v.get("image_quality_flags", []))

        if claim_parse.get("injection_detected"):
            risk_flags.add("text_instruction_present")
            risk_flags.add("manual_review_required")

        if claim_status == "contradicted":
            risk_flags.add("claim_mismatch")
            risk_flags.add("manual_review_required")

        if claim_status == "not_enough_information" and evidence_count > 0 and not evidence_standard_met:
            risk_flags.add("manual_review_required")

        if claimed_area_visible and not matching_pairs and area_pairs:
            any_none = any(v.get("issue_type_detected") == "none" for _, v in area_pairs)
            if any_none:
                risk_flags.add("damage_not_visible")

        if history_result.get("risk_score", 0.0) >= 0.8:
            risk_flags.add("manual_review_required")

        risk_flags = {f for f in risk_flags if f in config.RISK_FLAGS}
        risk_flags_str = ", ".join(sorted(risk_flags)) if risk_flags else config.NO_RISK_FLAG

        return {
            "evidence_standard_met": evidence_standard_met,
            "evidence_standard_met_reason": evidence_standard_met_reason,
            "risk_flags": risk_flags_str,
            "issue_type": issue_type,
            "object_part": object_part,
            "claim_status": claim_status,
            "claim_status_justification": claim_status_justification,
            "supporting_image_ids": ", ".join(supporting_image_ids),
            "valid_image": valid_image,
            "severity": severity,
        }
