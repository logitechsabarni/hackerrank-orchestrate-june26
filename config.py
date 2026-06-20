"""
config.py
Central configuration for the Multi-Modal Evidence Review pipeline.

All allowed vocabularies, file paths, model settings, and tunable
thresholds live here so the rest of the codebase never hardcodes them.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs"
PROMPTS_DIR = BASE_DIR / "prompts"
LOG_DIR = OUTPUT_DIR / "logs"

CLAIMS_CSV = DATASET_DIR / "claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"

DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "predictions.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Gemini model configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# Generation settings used for every structured-output call.
GENERATION_CONFIG_JSON = {
    "temperature": 0.1,
    "top_p": 0.9,
    "max_output_tokens": 1024,
    "response_mime_type": "application/json",
}

# ---------------------------------------------------------------------------
# Rate limiting / retry strategy
# ---------------------------------------------------------------------------
REQUESTS_PER_MINUTE = int(os.environ.get("GEMINI_RPM_LIMIT", "12"))
MIN_SECONDS_BETWEEN_CALLS = 60.0 / max(REQUESTS_PER_MINUTE, 1)

MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "4"))
RETRY_BASE_DELAY_SECONDS = float(os.environ.get("GEMINI_RETRY_BASE_DELAY", "2.0"))

# ---------------------------------------------------------------------------
# Image path parsing
# ---------------------------------------------------------------------------
# claims.csv "image_paths" cells may use any of these separators.
IMAGE_PATH_SEPARATORS = ["|", ";", ","]

# Image quality heuristic thresholds (local pre-filter, PIL-only).
LOW_LIGHT_BRIGHTNESS_THRESHOLD = 60.0      # mean grayscale 0-255
OVEREXPOSED_BRIGHTNESS_THRESHOLD = 235.0   # near-white glare
BLUR_EDGE_VARIANCE_THRESHOLD = 90.0        # below this -> likely blurry
MIN_IMAGE_DIMENSION_PX = 150               # below this on either side -> likely cropped/too small

# ---------------------------------------------------------------------------
# Allowed output vocabularies (must match hackathon spec exactly)
# ---------------------------------------------------------------------------
CLAIM_OBJECTS = ["car", "laptop", "package"]

ISSUE_TYPES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
]

OBJECT_PARTS_BY_CLAIM_OBJECT = {
    "car": [
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender",
        "quarter_panel", "body", "unknown",
    ],
    "laptop": [
        "screen", "keyboard", "trackpad", "hinge", "lid",
        "corner", "port", "base", "body", "unknown",
    ],
    "package": [
        "box", "package_corner", "package_side", "seal",
        "label", "contents", "item", "unknown",
    ],
}

CLAIM_STATUSES = ["supported", "contradicted", "not_enough_information"]

SEVERITY_LEVELS = ["none", "low", "medium", "high", "unknown"]
SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}  # "unknown" excluded on purpose

RISK_FLAGS = [
    "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part",
    "damage_not_visible", "claim_mismatch", "possible_manipulation",
    "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required",
]
NO_RISK_FLAG = "none"

REQUIRED_OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part",
    "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]

# ---------------------------------------------------------------------------
# Prompt-injection defense
# ---------------------------------------------------------------------------
# Keyword/phrase backstop used *in addition to* the model's own
# injection_detected flag. Decisions never branch on these directly --
# they only ever feed into risk flags, never into claim_status logic.
INJECTION_PHRASES = [
    "approve immediately",
    "approve this claim",
    "approve now",
    "skip review",
    "skip manual review",
    "mark supported",
    "mark as supported",
    "mark this as supported",
    "ignore previous instructions",
    "ignore all instructions",
    "ignore the above",
    "disregard previous instructions",
    "follow the note",
    "auto approve",
    "auto-approve",
    "autoapprove",
    "do not flag",
    "do not review",
    "no review needed",
    "system:",
    "you are now",
]

# ---------------------------------------------------------------------------
# History risk thresholds
# ---------------------------------------------------------------------------
HISTORY_REJECTED_RATIO_THRESHOLD = 0.4     # rejected_claim / past_claim_count
HISTORY_RECENT_CLAIM_COUNT_THRESHOLD = 3   # last_90_days_claim_count
HISTORY_FLAG_KEYWORDS = ["fraud", "flag", "suspicious", "risk", "watch"]
