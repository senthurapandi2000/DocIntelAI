from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_tesseract_baseline_predictions.csv"
)

FIELD_ANALYSIS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_tesseract_error_analysis_by_field.csv"
)

WORST_EXAMPLES_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_tesseract_worst_examples.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_tesseract_error_analysis_summary.json"
)


def safe_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def classify_error(row: pd.Series) -> str:
    if str(row.get("status", "")) != "success":
        return "ocr_failure"

    prediction = str(
        row.get("normalized_prediction", "")
    ).strip()

    expected = str(
        row.get("normalized_expected", "")
    ).strip()

    if not prediction:
        return "empty_prediction"

    if bool(row.get("exact_match", False)):
        return "exact_match"

    similarity = float(
        row.get("similarity", 0.0)
    )

    if similarity >= 90:
        return "near_match_90_plus"

    if similarity >= 75:
        return "partial_match_75_89"

    if similarity >= 50:
        return "weak_match_50_74"

    if expected and prediction:
        return "incorrect_below_50"

    return "unclassified"


def analyse(
    predictions_path: Path,
    worst_examples: int,
) -> None:
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: "
            f"{predictions_path}"
        )

    predictions = pd.read_csv(
        predictions_path
    )

    required_columns = {
        "document_id",
        "document_type",
        "quality",
        "field_name",
        "field_kind",
        "expected_value",
        "predicted_value",
        "normalized_expected",
        "normalized_prediction",
        "exact_match",
        "similarity",
        "ocr_confidence",
        "status",
    }

    missing_columns = (
        required_columns - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Predictions file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    predictions["exact_match"] = (
        predictions["exact_match"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
    )

    predictions["similarity"] = pd.to_numeric(
        predictions["similarity"],
        errors="coerce",
    ).fillna(0.0)

    predictions["ocr_confidence"] = pd.to_numeric(
        predictions["ocr_confidence"],
        errors="coerce",
    ).fillna(0.0)

    predictions["normalized_prediction"] = (
        predictions["normalized_prediction"]
        .fillna("")
        .astype(str)
    )

    predictions["normalized_expected"] = (
        predictions["normalized_expected"]
        .fillna("")
        .astype(str)
    )

    predictions["empty_prediction"] = (
        predictions["normalized_prediction"]
        .str.strip()
        .eq("")
    )

    predictions["high_similarity_mismatch"] = (
        (~predictions["exact_match"])
        & (predictions["similarity"] >= 80)
    )

    predictions["error_category"] = (
        predictions.apply(
            classify_error,
            axis=1,
        )
    )

    field_analysis = (
        predictions
        .groupby(
            [
                "quality",
                "document_type",
                "field_name",
                "field_kind",
            ],
            dropna=False,
        )
        .agg(
            field_predictions=(
                "field_name",
                "size",
            ),
            exact_match_rate=(
                "exact_match",
                "mean",
            ),
            mean_similarity=(
                "similarity",
                "mean",
            ),
            median_similarity=(
                "similarity",
                "median",
            ),
            mean_ocr_confidence=(
                "ocr_confidence",
                "mean",
            ),
            empty_prediction_rate=(
                "empty_prediction",
                "mean",
            ),
            high_similarity_mismatch_rate=(
                "high_similarity_mismatch",
                "mean",
            ),
        )
        .reset_index()
    )

    percentage_columns = [
        "exact_match_rate",
        "empty_prediction_rate",
        "high_similarity_mismatch_rate",
    ]

    for column in percentage_columns:
        field_analysis[column] = (
            field_analysis[column]
            .astype(float)
            .round(4)
        )

    field_analysis["mean_similarity"] = (
        field_analysis["mean_similarity"]
        .astype(float)
        .round(2)
    )

    field_analysis["median_similarity"] = (
        field_analysis["median_similarity"]
        .astype(float)
        .round(2)
    )

    field_analysis["mean_ocr_confidence"] = (
        field_analysis["mean_ocr_confidence"]
        .astype(float)
        .round(2)
    )

    field_analysis["priority_score"] = (
        (
            1.0
            - field_analysis["exact_match_rate"]
        )
        * 0.55
        + (
            1.0
            - field_analysis["mean_similarity"] / 100.0
        )
        * 0.30
        + field_analysis[
            "empty_prediction_rate"
        ]
        * 0.15
    ).round(4)

    field_analysis = field_analysis.sort_values(
        [
            "quality",
            "priority_score",
            "exact_match_rate",
        ],
        ascending=[True, False, True],
    )

    FIELD_ANALYSIS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    field_analysis.to_csv(
        FIELD_ANALYSIS_PATH,
        index=False,
    )

    mismatches = predictions[
        ~predictions["exact_match"]
    ].copy()

    mismatches = mismatches.sort_values(
        [
            "similarity",
            "ocr_confidence",
        ],
        ascending=[True, True],
    )

    worst_df = mismatches.head(
        max(1, worst_examples)
    )[
        [
            "document_id",
            "document_type",
            "quality",
            "field_name",
            "field_kind",
            "expected_value",
            "predicted_value",
            "normalized_expected",
            "normalized_prediction",
            "similarity",
            "ocr_confidence",
            "status",
            "error_category",
        ]
    ]

    worst_df.to_csv(
        WORST_EXAMPLES_PATH,
        index=False,
    )

    by_quality: dict[str, Any] = {}

    for quality, quality_df in predictions.groupby(
        "quality"
    ):
        by_quality[str(quality)] = {
            "field_predictions": int(
                len(quality_df)
            ),
            "exact_match_rate": round(
                safe_rate(
                    quality_df["exact_match"]
                ),
                4,
            ),
            "mean_similarity": round(
                float(
                    quality_df[
                        "similarity"
                    ].mean()
                ),
                2,
            ),
            "empty_prediction_rate": round(
                safe_rate(
                    quality_df[
                        "empty_prediction"
                    ]
                ),
                4,
            ),
        }

    error_category_counts = {
        str(category): int(count)
        for category, count in (
            predictions[
                "error_category"
            ]
            .value_counts()
            .items()
        )
    }

    summary = {
        "predictions_file": str(
            predictions_path
        ),
        "field_predictions": int(
            len(predictions)
        ),
        "documents": int(
            predictions[
                "document_id"
            ].nunique()
        ),
        "overall_exact_match_rate": round(
            safe_rate(
                predictions["exact_match"]
            ),
            4,
        ),
        "overall_mean_similarity": round(
            float(
                predictions[
                    "similarity"
                ].mean()
            ),
            2,
        ),
        "overall_empty_prediction_rate": round(
            safe_rate(
                predictions[
                    "empty_prediction"
                ]
            ),
            4,
        ),
        "by_quality": by_quality,
        "error_category_counts": (
            error_category_counts
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\nTesseract error analysis completed"
    )
    print(
        f"Documents analysed: "
        f"{summary['documents']}"
    )
    print(
        f"Field predictions: "
        f"{summary['field_predictions']}"
    )
    print(
        "Overall exact-match rate: "
        f"{summary['overall_exact_match_rate']:.2%}"
    )
    print(
        "Overall mean similarity: "
        f"{summary['overall_mean_similarity']:.2f}"
    )
    print(
        "Overall empty-prediction rate: "
        f"{summary['overall_empty_prediction_rate']:.2%}"
    )

    print("\nWorst fields by scan quality")

    for quality in [
        "clean",
        "mild",
        "hard",
    ]:
        subset = field_analysis[
            field_analysis["quality"]
            == quality
        ].head(8)

        if subset.empty:
            continue

        print(f"\n{quality.upper()}")

        for _, row in subset.iterrows():
            print(
                f"{row['document_type']} | "
                f"{row['field_name']} | "
                f"exact {row['exact_match_rate']:.2%} | "
                f"similarity {row['mean_similarity']:.2f} | "
                f"empty {row['empty_prediction_rate']:.2%}"
            )

    print(
        f"\nField analysis: "
        f"{FIELD_ANALYSIS_PATH}"
    )
    print(
        f"Worst examples: "
        f"{WORST_EXAMPLES_PATH}"
    )
    print(
        f"Summary: {SUMMARY_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse field-level errors from the "
            "IRS Tesseract OCR baseline."
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
    )

    parser.add_argument(
        "--worst-examples",
        type=int,
        default=100,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    analyse(
        predictions_path=(
            arguments.predictions
        ),
        worst_examples=(
            arguments.worst_examples
        ),
    )
