"""
main.py
Entry point for the Multi-Modal Evidence Review pipeline.

Usage:
    python main.py
    python main.py --claims dataset/claims.csv --output outputs/predictions.csv
    python main.py --limit 5          # quick smoke test on the first 5 rows

Pipeline per claim row:
    1. ClaimParser.parse()        -> normalized claim + injection signals
    2. VisionAnalyzer.analyze_*() -> per-image structured visual findings
    3. HistoryAnalyzer.analyze()  -> user-history risk context
    4. Evaluator.evaluate()       -> final deterministic decision
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from agents.claim_parser import ClaimParser
from agents.evaluator import Evaluator
from agents.history_analyzer import HistoryAnalyzer
from agents.vision_analyzer import VisionAnalyzer
from utils.csv_loader import (
    get_evidence_requirement,
    get_user_history_row,
    load_claims,
    load_evidence_requirements,
    load_user_history,
)
from utils.gemini_client import usage as gemini_usage
from utils.image_utils import make_image_id
from utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(
    claims_path: Path = config.CLAIMS_CSV,
    history_path: Path = config.USER_HISTORY_CSV,
    requirements_path: Path = config.EVIDENCE_REQUIREMENTS_CSV,
    limit: int = None,
) -> pd.DataFrame:
    claims_df = load_claims(claims_path)
    history_df = load_user_history(history_path)
    requirements_df = load_evidence_requirements(requirements_path)

    if limit:
        claims_df = claims_df.head(limit)

    claim_parser = ClaimParser()
    vision_analyzer = VisionAnalyzer()
    history_analyzer = HistoryAnalyzer()
    evaluator = Evaluator()

    output_rows = []

    for _, row in tqdm(claims_df.iterrows(), total=len(claims_df), desc="Reviewing claims"):
        user_id = str(row["user_id"])
        claim_object = str(row["claim_object"]).strip().lower()
        user_claim = str(row["user_claim"])
        image_paths = row["image_path_list"]
        image_paths_display = row["image_paths"]

        history_row = get_user_history_row(user_id, history_df)
        history_result = history_analyzer.analyze(history_row)

        try:
            claim_parse = claim_parser.parse(
                user_claim=user_claim,
                claim_object=claim_object,
                history_summary=str(history_row.get("history_summary", "")),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error parsing claim for user %s", user_id)
            claim_parse = {
                "extracted_issue_type": "unknown",
                "extracted_object_part": "unknown",
                "normalized_claim_text": user_claim,
                "injection_detected": False,
                "injection_phrases": [],
                "confidence": 0.0,
                "parser_error": str(exc),
            }

        if image_paths:
            try:
                vision_results = vision_analyzer.analyze_claim_images(
                    image_paths=image_paths,
                    claim_object=claim_object,
                    normalized_claim_text=claim_parse.get("normalized_claim_text", user_claim),
                    extracted_issue_type=claim_parse.get("extracted_issue_type", "unknown"),
                    extracted_object_part=claim_parse.get("extracted_object_part", "unknown"),
                )
            except Exception:  # noqa: BLE001
                logger.exception("Unexpected error analyzing images for user %s", user_id)
                vision_results = [
                    {
                        "is_valid_image": False,
                        "claim_object_present": None,
                        "object_part_detected": "unknown",
                        "issue_type_detected": "unknown",
                        "damage_matches_claim": None,
                        "severity": "unknown",
                        "image_quality_flags": [],
                        "confidence": 0.0,
                        "notes": "Unhandled error during vision analysis.",
                    }
                    for _ in image_paths
                ]
        else:
            vision_results = []

        image_ids = [make_image_id(user_id, idx) for idx in range(len(image_paths))]

        evidence_req = get_evidence_requirement(
            claim_object=claim_object,
            object_part=claim_parse.get("extracted_object_part", "unknown"),
            requirements_df=requirements_df,
        )

        decision = evaluator.evaluate(
            claim_object=claim_object,
            image_ids=image_ids,
            vision_results=vision_results,
            claim_parse=claim_parse,
            history_result=history_result,
            evidence_req=evidence_req,
        )

        output_rows.append(
            {
                "user_id": user_id,
                "image_paths": image_paths_display,
                "user_claim": user_claim,
                "claim_object": claim_object,
                **decision,
            }
        )

    output_df = pd.DataFrame(output_rows, columns=config.REQUIRED_OUTPUT_COLUMNS)
    return output_df


def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review pipeline")
    parser.add_argument("--claims", type=Path, default=config.CLAIMS_CSV)
    parser.add_argument("--history", type=Path, default=config.USER_HISTORY_CSV)
    parser.add_argument("--requirements", type=Path, default=config.EVIDENCE_REQUIREMENTS_CSV)
    parser.add_argument("--output", type=Path, default=config.DEFAULT_OUTPUT_CSV)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N claims")
    args = parser.parse_args()

    if not config.GEMINI_API_KEY:
        logger.error(
            "GEMINI_API_KEY is not set. Set it in your environment or a .env file "
            "(see .env.example) before running the pipeline."
        )
        sys.exit(1)

    start = time.time()
    output_df = run_pipeline(
        claims_path=args.claims,
        history_path=args.history,
        requirements_path=args.requirements,
        limit=args.limit,
    )
    elapsed = time.time() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)

    stats = gemini_usage.as_dict()
    logger.info("Done in %.1fs. Wrote %d rows to %s", elapsed, len(output_df), args.output)
    logger.info(
        "Gemini usage -- text calls: %d, vision calls: %d, failed: %d, retried: %d",
        stats["text_calls"], stats["vision_calls"], stats["failed_calls"], stats["retried_calls"],
    )


if __name__ == "__main__":
    main()
