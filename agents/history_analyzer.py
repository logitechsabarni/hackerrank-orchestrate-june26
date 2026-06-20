"""
agents/history_analyzer.py
Adds user-history risk context. No LLM call -- this is a deterministic
rules pass over user_history.csv fields, kept separate from the
claim/vision agents so risk scoring logic is easy to audit and tune.
"""

from typing import Any, Dict, List

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class HistoryAnalyzer:
    """Derives risk flags / risk notes from a user's claim history record."""

    def analyze(self, history_row: Dict[str, Any]) -> Dict[str, Any]:
        past_count = int(history_row.get("past_claim_count", 0) or 0)
        rejected = int(history_row.get("rejected_claim", 0) or 0)
        recent = int(history_row.get("last_90_days_claim_count", 0) or 0)
        flags_field = str(history_row.get("history_flags", "") or "")
        summary = str(history_row.get("history_summary", "") or "")

        reasons: List[str] = []
        risk_flags: List[str] = []

        rejected_ratio = (rejected / past_count) if past_count > 0 else 0.0
        if past_count > 0 and rejected_ratio >= config.HISTORY_REJECTED_RATIO_THRESHOLD:
            risk_flags.append("user_history_risk")
            reasons.append(
                f"{rejected}/{past_count} past claims rejected "
                f"({rejected_ratio:.0%}, threshold {config.HISTORY_REJECTED_RATIO_THRESHOLD:.0%})"
            )

        if recent >= config.HISTORY_RECENT_CLAIM_COUNT_THRESHOLD:
            risk_flags.append("user_history_risk")
            reasons.append(
                f"{recent} claims filed in the last 90 days "
                f"(threshold {config.HISTORY_RECENT_CLAIM_COUNT_THRESHOLD})"
            )

        flags_lower = flags_field.lower()
        matched_keywords = [kw for kw in config.HISTORY_FLAG_KEYWORDS if kw in flags_lower]
        if matched_keywords:
            risk_flags.append("user_history_risk")
            reasons.append(f"history_flags contains: {', '.join(matched_keywords)}")

        risk_score = min(1.0, 0.4 * len(reasons)) if reasons else 0.0

        return {
            "risk_flags": sorted(set(risk_flags)),
            "risk_score": risk_score,
            "reason": "; ".join(reasons) if reasons else "No elevated risk signals in history.",
            "history_summary": summary,
        }
