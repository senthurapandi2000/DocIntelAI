from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.classification.train_irs_field_confidence_model import (
    ALL_FEATURES,
    CROSS_FIELD_MAP,
    character_ratios,
    field_risk_tier,
)
from src.validation.irs_extraction_validator import (
    clean_text,
    engine_confidence,
    field_format_validation,
    required_fields_for,
    validate_cross_fields,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_ocr_final_test_predictions.csv"
)

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "irs_field_confidence_model.joblib"
)

FIELD_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_field_confidence_final_test_fields.csv"
)

DOCUMENT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_field_confidence_final_test_documents.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_field_confidence_final_test_summary.json"
)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return False

    return clean_text(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def add_cross_field_flags(
    dataframe: pd.DataFrame,
) -> pd.Series:
    flags = pd.Series(
        0.0,
        index=dataframe.index,
        dtype=float,
    )

    for _, document_df in dataframe.groupby(
        "document_id",
        sort=False,
    ):
        document_type = clean_text(
            document_df["document_type"].iloc[0]
        )

        field_values = {
            clean_text(row["field_name"]): clean_text(
                row["final_normalized_prediction"]
            )
            for _, row in document_df.iterrows()
        }

        issues = validate_cross_fields(
            document_type,
            field_values,
        )

        affected_fields: set[str] = set()

        for issue in issues:
            affected_fields.update(
                CROSS_FIELD_MAP.get(
                    issue,
                    set(),
                )
            )

        if affected_fields:
            affected_mask = (
                document_df["field_name"]
                .astype(str)
                .isin(affected_fields)
            )

            flags.loc[
                document_df.index[affected_mask]
            ] = 1.0

    return flags


def engineer_features(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    data = predictions.copy()

    required_columns = {
        "document_id",
        "parent_document_id",
        "document_type",
        "split",
        "quality",
        "field_name",
        "final_engine",
        "final_prediction",
        "final_normalized_prediction",
        "predicted_value",
        "normalized_prediction",
        "ocr_confidence",
        "router_probability",
        "final_exact_match",
        "final_similarity",
    }

    missing_columns = (
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Final OCR predictions are missing columns: "
            f"{sorted(missing_columns)}"
        )

    for column in [
        "document_id",
        "parent_document_id",
        "document_type",
        "split",
        "quality",
        "field_name",
        "final_engine",
        "final_prediction",
        "final_normalized_prediction",
        "predicted_value",
        "normalized_prediction",
    ]:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
        )

    if "trocr_prediction" not in data.columns:
        data["trocr_prediction"] = ""

    if "trocr_normalized" not in data.columns:
        data["trocr_normalized"] = ""

    data["trocr_prediction"] = (
        data["trocr_prediction"]
        .fillna("")
        .astype(str)
    )

    data["trocr_normalized"] = (
        data["trocr_normalized"]
        .fillna("")
        .astype(str)
    )

    for column in [
        "ocr_confidence",
        "router_probability",
        "final_similarity",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        ).fillna(0.0)

    data["final_exact_match"] = (
        data["final_exact_match"]
        .map(to_bool)
    )

    data["required_field"] = data.apply(
        lambda row: float(
            clean_text(row["field_name"])
            in required_fields_for(
                clean_text(
                    row["document_type"]
                )
            )
        ),
        axis=1,
    )

    validation_results = data.apply(
        lambda row: field_format_validation(
            clean_text(row["field_name"]),
            clean_text(
                row[
                    "final_normalized_prediction"
                ]
            ),
            required=bool(
                row["required_field"]
            ),
        ),
        axis=1,
    )

    validation_frame = pd.DataFrame(
        validation_results.tolist(),
        index=data.index,
        columns=[
            "format_valid",
            "format_score",
            "format_message",
        ],
    )

    data = pd.concat(
        [
            data,
            validation_frame,
        ],
        axis=1,
    )

    data["format_valid"] = (
        data["format_valid"]
        .astype(float)
    )

    data["format_score"] = pd.to_numeric(
        data["format_score"],
        errors="coerce",
    ).fillna(0.0)

    data["model_confidence"] = data.apply(
        engine_confidence,
        axis=1,
    )

    data["risk_tier"] = (
        data["field_name"]
        .map(field_risk_tier)
    )

    data["prediction_length"] = (
        data[
            "final_normalized_prediction"
        ]
        .str.len()
        .astype(float)
    )

    ratios = data[
        "final_prediction"
    ].map(character_ratios)

    data[
        [
            "digit_ratio",
            "alpha_ratio",
            "punctuation_ratio",
        ]
    ] = pd.DataFrame(
        ratios.tolist(),
        index=data.index,
    )

    data["has_trocr_candidate"] = (
        data["trocr_normalized"]
        .str.strip()
        .ne("")
        .astype(float)
    )

    data[
        "tesseract_trocr_agreement"
    ] = data.apply(
        lambda row: float(
            fuzz.ratio(
                row["normalized_prediction"],
                row["trocr_normalized"],
            )
        )
        if (
            clean_text(
                row["normalized_prediction"]
            )
            and clean_text(
                row["trocr_normalized"]
            )
        )
        else 0.0,
        axis=1,
    )

    data["cross_field_flag"] = (
        add_cross_field_flags(data)
    )

    return data


def policy_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    review = dataframe[
        "field_review_required"
    ].astype(bool)

    accepted = ~review

    correct = dataframe[
        "final_exact_match"
    ].astype(bool)

    incorrect = ~correct

    return {
        "fields": int(len(dataframe)),
        "fields_auto_accepted": int(
            accepted.sum()
        ),
        "fields_reviewed": int(
            review.sum()
        ),
        "field_review_rate": round(
            float(review.mean()),
            4,
        ),
        "auto_accept_coverage": round(
            float(accepted.mean()),
            4,
        ),
        "auto_accepted_exact_match_rate": round(
            float(
                correct[accepted].mean()
            )
            if accepted.any()
            else 0.0,
            4,
        ),
        "incorrect_field_capture_rate": round(
            float(
                review[incorrect].mean()
            )
            if incorrect.any()
            else 0.0,
            4,
        ),
        "incorrect_fields_auto_accepted": int(
            (
                incorrect
                & accepted
            ).sum()
        ),
    }


def build_document_report(
    fields: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for document_id, document_df in fields.groupby(
        "document_id",
        sort=False,
    ):
        critical_review = (
            (
                document_df["risk_tier"]
                == "critical"
            )
            & document_df[
                "field_review_required"
            ]
        )

        structured_review = (
            (
                document_df["risk_tier"]
                == "structured"
            )
            & document_df[
                "field_review_required"
            ]
        )

        descriptive_review = (
            (
                document_df["risk_tier"]
                == "descriptive"
            )
            & document_df[
                "field_review_required"
            ]
        )

        required_invalid = (
            document_df[
                "required_field"
            ].astype(bool)
            & ~document_df[
                "format_valid"
            ].astype(bool)
        )

        cross_field_review = (
            document_df[
                "cross_field_flag"
            ]
            > 0
        )

        if (
            critical_review.any()
            or required_invalid.any()
            or cross_field_review.any()
        ):
            status = "critical_review"
        elif structured_review.any():
            status = "structured_review"
        elif descriptive_review.any():
            status = "descriptive_review"
        else:
            status = "auto_accept"

        rows.append(
            {
                "document_id": document_id,
                "parent_document_id": clean_text(
                    document_df[
                        "parent_document_id"
                    ].iloc[0]
                ),
                "document_type": clean_text(
                    document_df[
                        "document_type"
                    ].iloc[0]
                ),
                "quality": clean_text(
                    document_df[
                        "quality"
                    ].iloc[0]
                ),
                "fields_checked": int(
                    len(document_df)
                ),
                "fields_reviewed": int(
                    document_df[
                        "field_review_required"
                    ].sum()
                ),
                "critical_fields_reviewed": int(
                    critical_review.sum()
                ),
                "structured_fields_reviewed": int(
                    structured_review.sum()
                ),
                "descriptive_fields_reviewed": int(
                    descriptive_review.sum()
                ),
                "required_invalid_fields": int(
                    required_invalid.sum()
                ),
                "cross_field_flags": int(
                    cross_field_review.sum()
                ),
                "document_status": status,
            }
        )

    return pd.DataFrame(rows)


def run_evaluation(
    *,
    predictions_path: Path,
    model_path: Path,
) -> None:
    if not predictions_path.exists():
        raise FileNotFoundError(
            "Final OCR predictions not found: "
            f"{predictions_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            "Field-confidence model not found: "
            f"{model_path}"
        )

    raw_predictions = pd.read_csv(
        predictions_path
    )

    test_data = engineer_features(
        raw_predictions
    )

    bundle = joblib.load(model_path)

    pipeline = bundle["pipeline"]
    thresholds = bundle["thresholds"]

    correctness_probabilities = (
        pipeline.predict_proba(
            test_data[ALL_FEATURES]
        )[:, 1]
    )

    test_data[
        "correctness_probability"
    ] = correctness_probabilities

    test_data[
        "risk_threshold"
    ] = (
        test_data["risk_tier"]
        .map(thresholds)
    )

    test_data[
        "field_review_required"
    ] = (
        ~test_data[
            "format_valid"
        ].astype(bool)
        | (
            test_data[
                "correctness_probability"
            ]
            < test_data[
                "risk_threshold"
            ]
        )
        | (
            test_data[
                "cross_field_flag"
            ]
            > 0
        )
    )

    document_report = build_document_report(
        test_data
    )

    FIELD_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_data.to_csv(
        FIELD_OUTPUT_PATH,
        index=False,
    )

    document_report.to_csv(
        DOCUMENT_OUTPUT_PATH,
        index=False,
    )

    overall = policy_metrics(
        test_data
    )

    by_tier = {
        str(tier): policy_metrics(
            tier_df
        )
        for tier, tier_df in (
            test_data.groupby(
                "risk_tier"
            )
        )
    }

    by_quality = {
        str(quality): policy_metrics(
            quality_df
        )
        for quality, quality_df in (
            test_data.groupby(
                "quality"
            )
        )
    }

    by_document_type = {
        str(document_type): policy_metrics(
            document_df
        )
        for document_type, document_df in (
            test_data.groupby(
                "document_type"
            )
        )
    }

    summary = {
        "evaluation_split": "test",
        "predictions_path": str(
            predictions_path
        ),
        "model_path": str(model_path),
        "risk_thresholds": thresholds,
        "documents": int(
            test_data[
                "document_id"
            ].nunique()
        ),
        "parent_documents": int(
            test_data[
                "parent_document_id"
            ].nunique()
        ),
        "overall": overall,
        "by_risk_tier": by_tier,
        "by_quality": by_quality,
        "by_document_type": by_document_type,
        "document_status_counts": {
            str(status): int(count)
            for status, count in (
                document_report[
                    "document_status"
                ]
                .value_counts()
                .items()
            )
        },
    }

    with SUMMARY_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\nIRS field-confidence final-test "
        "evaluation completed"
    )
    print(
        f"Documents: "
        f"{summary['documents']}"
    )
    print(
        f"Parent documents: "
        f"{summary['parent_documents']}"
    )
    print(
        f"Fields: {overall['fields']}"
    )
    print(
        f"Fields auto accepted: "
        f"{overall['fields_auto_accepted']}"
    )
    print(
        f"Fields reviewed: "
        f"{overall['fields_reviewed']}"
    )
    print(
        f"Field review rate: "
        f"{overall['field_review_rate']:.2%}"
    )
    print(
        "Auto-accepted exact match: "
        f"{overall['auto_accepted_exact_match_rate']:.2%}"
    )
    print(
        "Incorrect-field capture rate: "
        f"{overall['incorrect_field_capture_rate']:.2%}"
    )
    print(
        "Incorrect fields auto accepted: "
        f"{overall['incorrect_fields_auto_accepted']}"
    )

    print("\nBy risk tier")

    for tier in [
        "critical",
        "structured",
        "descriptive",
    ]:
        metrics = by_tier.get(
            tier,
            {},
        )

        print(
            f"{tier}: coverage "
            f"{metrics.get('auto_accept_coverage', 0.0):.2%}, "
            f"accepted exact "
            f"{metrics.get('auto_accepted_exact_match_rate', 0.0):.2%}, "
            f"incorrect capture "
            f"{metrics.get('incorrect_field_capture_rate', 0.0):.2%}"
        )

    print(
        "\nDocument statuses: "
        f"{summary['document_status_counts']}"
    )
    print(
        f"\nField report: "
        f"{FIELD_OUTPUT_PATH}"
    )
    print(
        f"Document report: "
        f"{DOCUMENT_OUTPUT_PATH}"
    )
    print(
        f"Summary: "
        f"{SUMMARY_OUTPUT_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen risk-aware IRS "
            "field-confidence model on final OCR "
            "test predictions."
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_evaluation(
        predictions_path=(
            arguments.predictions
        ),
        model_path=arguments.model,
    )
