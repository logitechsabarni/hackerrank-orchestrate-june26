"""
evaluation/evaluate.py
Runs the full pipeline against evaluation/sample_claims.csv (which carries
ground-truth labels: gt_claim_status, gt_issue_type, gt_object_part),
scores accuracy, and regenerates:
  - outputs/sample_predictions.csv
  - outputs/sample_metrics.json
  - evaluation/evaluation_report.md

Usage:
    python -m evaluation.evaluate
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

# Allow running as `python evaluation/evaluate.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from main import run_pipeline
from utils.gemini_client import usage as gemini_usage
from utils.logger import get_logger

logger = get_logger(__name__)

SAMPLE_CLAIMS_CSV = Path(__file__).resolve().parent / "sample_claims.csv"
SAMPLE_PREDICTIONS_CSV = config.OUTPUT_DIR / "sample_predictions.csv"
SAMPLE_METRICS_JSON = config.OUTPUT_DIR / "sample_metrics.json"
EVALUATION_REPORT_MD = Path(__file__).resolve().parent / "evaluation_report.md"

# Approximate Gemini 2.5 Flash list pricing (USD per 1K tokens) at time of
# writing. These are ESTIMATES for planning purposes only -- check current
# Google AI pricing before relying on them for budgeting.
PRICE_PER_1K_INPUT_TOKENS_USD = 0.00035
PRICE_PER_1K_OUTPUT_TOKENS_USD = 0.00105


def compute_accuracy(merged: pd.DataFrame) -> dict:
    n = len(merged)
    if n == 0:
        return {
            "claim_status_accuracy": None,
            "issue_type_accuracy": None,
            "object_part_accuracy": None,
            "overall_accuracy": None,
            "n_samples": 0,
        }

    claim_status_acc = (merged["claim_status"] == merged["gt_claim_status"]).mean()
    issue_type_acc = (merged["issue_type"] == merged["gt_issue_type"]).mean()
    object_part_acc = (merged["object_part"] == merged["gt_object_part"]).mean()
    overall_acc = (claim_status_acc + issue_type_acc + object_part_acc) / 3.0

    return {
        "claim_status_accuracy": round(float(claim_status_acc), 4),
        "issue_type_accuracy": round(float(issue_type_acc), 4),
        "object_part_accuracy": round(float(object_part_acc), 4),
        "overall_accuracy": round(float(overall_acc), 4),
        "n_samples": n,
    }


def count_total_images(claims_df: pd.DataFrame) -> int:
    from utils.csv_loader import parse_image_paths
    return int(claims_df["image_paths"].apply(lambda c: len(parse_image_paths(c))).sum())


def build_report(metrics: dict, usage_stats: dict, image_count: int, n_claims: int, runtime_seconds: float) -> str:
    total_tokens = usage_stats["estimated_input_tokens"] + usage_stats["estimated_output_tokens"]
    estimated_cost = (
        usage_stats["estimated_input_tokens"] / 1000.0 * PRICE_PER_1K_INPUT_TOKENS_USD
        + usage_stats["estimated_output_tokens"] / 1000.0 * PRICE_PER_1K_OUTPUT_TOKENS_USD
    )
    per_claim_runtime = runtime_seconds / n_claims if n_claims else 0.0

    return f"""# Evaluation Report -- Multi-Modal Evidence Review

Generated automatically by `evaluation/evaluate.py`.

## Accuracy (vs. `evaluation/sample_claims.csv` ground truth)

| Metric | Value |
|---|---|
| claim_status accuracy | {metrics['claim_status_accuracy']} |
| issue_type accuracy | {metrics['issue_type_accuracy']} |
| object_part accuracy | {metrics['object_part_accuracy']} |
| **Overall accuracy** (macro-average of the three above) | **{metrics['overall_accuracy']}** |
| Sample size | {metrics['n_samples']} claims |

> Note: the bundled sample images are synthetic, abstract illustrations
> (see `scripts/generate_sample_images.py`), not real damage photos. These
> numbers demonstrate the evaluation harness, not real-world model accuracy.
> Re-run against a real labeled photo set for a meaningful accuracy figure.

## Cost & Usage Estimate

| Metric | Value |
|---|---|
| Claims processed | {n_claims} |
| Images processed | {image_count} |
| Gemini text (claim-parsing) calls | {usage_stats['text_calls']} |
| Gemini vision (image-analysis) calls | {usage_stats['vision_calls']} |
| Total model calls | {usage_stats['total_calls']} |
| Cache hits (calls avoided on rerun) | {usage_stats['cache_hits']} |
| Retried calls | {usage_stats['retried_calls']} |
| Failed calls (after exhausting retries) | {usage_stats['failed_calls']} |
| Estimated input tokens | {usage_stats['estimated_input_tokens']:,} |
| Estimated output tokens | {usage_stats['estimated_output_tokens']:,} |
| Estimated total tokens | {total_tokens:,} |
| **Estimated cost (USD)** | **${estimated_cost:.4f}** |
| Total runtime | {runtime_seconds:.1f}s |
| Avg. runtime per claim | {per_claim_runtime:.1f}s |

Cost estimate uses an illustrative Gemini 2.5 Flash rate of
${PRICE_PER_1K_INPUT_TOKENS_USD}/1K input tokens and
${PRICE_PER_1K_OUTPUT_TOKENS_USD}/1K output tokens, and a conservative
~4-characters-per-token heuristic for token counting (actual tokenization
will differ slightly). Verify against current published Google AI pricing
before using this for budgeting at scale.

## Scaling Estimate

For a production batch of **N** claims averaging **k** images each:
- Model calls ≈ `N * (1 + k)` (1 text call for claim parsing + 1 vision
  call per image), minus whatever fraction hits the cache on reruns.
- At {config.REQUESTS_PER_MINUTE} requests/minute (current rate-limit
  setting), throughput ≈ `{config.REQUESTS_PER_MINUTE}` calls/minute, so
  `N * (1 + k) / {config.REQUESTS_PER_MINUTE}` minutes minimum wall-clock
  time for a cold run with no cache hits.

## Rate Limit Strategy

- A single shared call-spacing gate (`utils/gemini_client._respect_rate_limit`)
  enforces a minimum interval of `{config.MIN_SECONDS_BETWEEN_CALLS:.2f}s`
  between any two Gemini calls, derived from `GEMINI_RPM_LIMIT`
  (currently {config.REQUESTS_PER_MINUTE} requests/minute).
- All Gemini calls -- text and vision -- funnel through this single gate,
  so the limit holds regardless of which agent is calling.
- The limit is configurable via the `GEMINI_RPM_LIMIT` environment variable
  to match the caller's actual API tier.

## Retry Strategy

- Each call is retried up to `GEMINI_MAX_RETRIES` (currently
  {config.MAX_RETRIES}) times on any exception (network errors, transient
  5xx, rate-limit 429s, malformed/non-JSON responses).
- Backoff is exponential: `GEMINI_RETRY_BASE_DELAY * 2^(attempt-1)` seconds
  (currently base {config.RETRY_BASE_DELAY_SECONDS}s, so ~2s, 4s, 8s, 16s).
- If all retries are exhausted, the failure is surfaced honestly: the
  affected claim is marked `not_enough_information` with a
  `manual_review_required` risk flag and a justification noting the
  automated-analysis failure -- it is never silently replaced with
  fabricated or guessed content.

## Caching Strategy

- `utils/gemini_client.py` maintains a disk-based cache under
  `outputs/cache/`, keyed on a SHA-256 hash of the exact prompt text plus
  image bytes (when an image is attached).
- Before any API call, the cache is checked; on a hit, the previously
  parsed JSON result is returned with zero additional API calls or tokens
  spent (`cache_hits` in the usage table above).
- This makes it safe to re-run `main.py` or `evaluate.py` repeatedly while
  iterating on the rules engine (`agents/evaluator.py`) without re-paying
  for unchanged claims/images. Set `GEMINI_DISABLE_CACHE=1` to force fresh
  calls (e.g. when validating prompt changes).
"""


def main():
    start = time.time()
    output_df = run_pipeline(claims_path=SAMPLE_CLAIMS_CSV)
    runtime_seconds = time.time() - start

    gt_df = pd.read_csv(SAMPLE_CLAIMS_CSV, dtype=str, keep_default_na=False)
    merged = output_df.merge(
        gt_df[["user_id", "gt_claim_status", "gt_issue_type", "gt_object_part"]],
        on="user_id",
        how="left",
    )
    merged["claim_status_correct"] = merged["claim_status"] == merged["gt_claim_status"]
    merged["issue_type_correct"] = merged["issue_type"] == merged["gt_issue_type"]
    merged["object_part_correct"] = merged["object_part"] == merged["gt_object_part"]

    metrics = compute_accuracy(merged)
    usage_stats = gemini_usage.as_dict()
    image_count = count_total_images(gt_df)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(SAMPLE_PREDICTIONS_CSV, index=False)

    metrics_payload = {
        "accuracy": metrics,
        "usage": usage_stats,
        "image_count": image_count,
        "n_claims": len(gt_df),
        "runtime_seconds": round(runtime_seconds, 2),
    }
    SAMPLE_METRICS_JSON.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    report = build_report(
        metrics=metrics,
        usage_stats=usage_stats,
        image_count=image_count,
        n_claims=len(gt_df),
        runtime_seconds=runtime_seconds,
    )
    EVALUATION_REPORT_MD.write_text(report, encoding="utf-8")

    logger.info("Evaluation complete.")
    logger.info("  Predictions -> %s", SAMPLE_PREDICTIONS_CSV)
    logger.info("  Metrics     -> %s", SAMPLE_METRICS_JSON)
    logger.info("  Report      -> %s", EVALUATION_REPORT_MD)
    logger.info("  Overall accuracy: %s", metrics["overall_accuracy"])


if __name__ == "__main__":
    main()
