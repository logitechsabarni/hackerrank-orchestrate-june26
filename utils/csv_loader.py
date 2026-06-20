from pathlib import Path
import pandas as pd


class CSVLoader:
    """Utility class for loading and validating CSV datasets."""

    REQUIRED_CLAIMS_COLUMNS = [
        "user_id",
        "image_paths",
        "user_claim",
        "claim_object"
    ]

    REQUIRED_HISTORY_COLUMNS = [
        "user_id",
        "past_claim_count",
        "accept_claim",
        "manual_review_claim",
        "rejected_claim",
        "last_90_days_claim_count",
        "history_flags",
        "history_summary"
    ]

    REQUIRED_EVIDENCE_COLUMNS = [
        "requirement_id",
        "claim_object",
        "applies_to",
        "minimum_image_evidence"
    ]

    @staticmethod
    def load_csv(csv_path: str) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame.

        Args:
            csv_path (str): Path to CSV file.

        Returns:
            pd.DataFrame
        """
        path = Path(csv_path)

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        return pd.read_csv(path)

    @classmethod
    def load_claims(cls, csv_path: str) -> pd.DataFrame:
        """
        Load claims.csv and validate required columns.
        """
        df = cls.load_csv(csv_path)

        missing = [
            col for col in cls.REQUIRED_CLAIMS_COLUMNS
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns in claims file: {missing}"
            )

        return df

    @classmethod
    def load_user_history(cls, csv_path: str) -> pd.DataFrame:
        """
        Load user_history.csv and validate required columns.
        """
        df = cls.load_csv(csv_path)

        missing = [
            col for col in cls.REQUIRED_HISTORY_COLUMNS
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns in user history file: {missing}"
            )

        return df

    @classmethod
    def load_evidence_requirements(cls, csv_path: str) -> pd.DataFrame:
        """
        Load evidence_requirements.csv and validate required columns.
        """
        df = cls.load_csv(csv_path)

        missing = [
            col for col in cls.REQUIRED_EVIDENCE_COLUMNS
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns in evidence requirements file: {missing}"
            )

        return df

    @staticmethod
    def get_user_history(user_id: str, history_df: pd.DataFrame) -> dict:
        """
        Fetch a user's history record as a dictionary.
        """
        row = history_df[history_df["user_id"] == user_id]

        if row.empty:
            return {}

        return row.iloc[0].to_dict()

    @staticmethod
    def split_image_paths(image_paths: str) -> list:
        """
        Convert semicolon-separated image paths into list.

        Example:
        'img1.jpg;img2.jpg'
        ->
        ['img1.jpg', 'img2.jpg']
        """
        if pd.isna(image_paths):
            return []

        return [
            path.strip()
            for path in str(image_paths).split(";")
            if path.strip()
        ]


if __name__ == "__main__":
    claims_df = CSVLoader.load_claims("dataset/claims.csv")
    print(f"Loaded {len(claims_df)} claims")

    image_list = CSVLoader.split_image_paths(
        claims_df.iloc[0]["image_paths"]
    )
    print("Images:", image_list)
