from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.validation.irs_extraction_validator import (
    PROJECT_ROOT,
    clean_text,
    engine_confidence,
    field_format_validation,
    required_fields_for,
    validate_cross_fields,
)


DEFAULT_HYBRID_VALIDATION = (
    PROJECT_ROOT
    / "reports"
    / "irs_tesseract_hybrid_validation_predictions.csv"
)

DEFAULT_ROUTER_HOLDOUT = (
    PROJECT_ROOT
    / "reports"
    / "irs_ocr_router_holdout_predictions.csv"
)

THRESHOLD_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_review_policy_threshold_search.csv"
)

FIELD_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_review_policy_holdout_fields.csv"
)

DOCUMENT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_review_policy_holdout_documents.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_review_policy_calibration_summary.json"
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


def load_parent_holdout_predictions(
    *,
    hybrid_validation_path: Path,
    router_holdout_path: Path,
) -> pd.DataFrame:
    """
    Build an unbiased validation subset.

    Parent documents in the OCR-router holdout were not used to train the
    holdout router. Non-fallback fields retain their hybrid-Tesseract outputs;
    fallback fields use the holdout router decision.
    """

    if not hybrid_validation_path.exists():
        raise FileNotFoundError(
            "Hybrid validation predictions not found: "
            f"{hybrid_validation_path}"
        )

    if not router_holdout_path.exists():
        raise FileNotFoundError(
            "Router holdout predictions not found: "
            f"{router_holdout_path}"
        )

    hybrid = pd.read_csv(
        hybrid_validation_path
    )

    router = pd.read_csv(
        router_holdout_path
    )

    required_hybrid_columns = {
        "document_id",
        "parent_document_id",
        "document_type",
        "split",
        "quality",
        "field_name",
        "expected_value",
        "predicted_value",
        "normalized_prediction",
        "exact_match",
        "similarity",
        "ocr_confidence",
    }

    required_router_columns = {
        "document_id",
        "parent_document_id",
        "field_name",
        "router_probability",
        "router_choose_trocr",
        "router_prediction",
        "router_exact_match",
        "router_similarity",
        "trocr_normalized",
        "tesseract_normalized",
    }

    missing_hybrid = (
        required_hybrid_columns
        - set(hybrid.columns)
    )

    missing_router = (
        required_router_columns
        - set(router.columns)
    )

    if missing_hybrid:
        raise ValueError(
            "Hybrid validation file is missing columns: "
            f"{sorted(missing_hybrid)}"
        )

    if missing_router:
        raise ValueError(
            "Router holdout file is missing columns: "
            f"{sorted(missing_router)}"
        )

    holdout_parents = set(
        router["parent_document_id"]
        .fillna("")
        .astype(str)
        .unique()
    )

    holdout = hybrid[
        hybrid["parent_document_id"]
        .fillna("")
        .astype(str)
        .isin(holdout_parents)
    ].copy()

    router_subset = router[
        [
            "document_id",
            "field_name",
            "router_probability",
            "router_choose_trocr",
            "router_prediction",
            "router_exact_match",
            "router_similarity",
            "trocr_normalized",
            "tesseract_normalized",
        ]
    ].copy()

    router_subset = router_subset.rename(
        columns={
            "router_prediction": (
                "holdout_router_prediction"
            ),
            "router_exact_match": (
                "holdout_router_exact_match"
            ),
            "router_similarity": (
                "holdout_router_similarity"
            ),
            "trocr_normalized": (
                "holdout_trocr_normalized"
            ),
            "tesseract_normalized": (
                "holdout_tesseract_normalized"
            ),
        }
    )

    merged = holdout.merge(
        router_subset,
        on=[
            "document_id",
            "field_name",
        ],
        how="left",
        validate="one_to_one",
    )

    merged["router_probability"] = (
        pd.to_numeric(
            merged["router_probability"],
            errors="coerce",
        )
    )

    merged["router_choose_trocr"] = (
        merged["router_choose_trocr"]
        .map(to_bool)
    )

    has_holdout_router = (
        merged["holdout_router_prediction"]
        .notna()
    )

    merged["final_engine"] = np.where(
        has_holdout_router
        & merged["router_choose_trocr"],
        "trocr",
        "tesseract",
    )

    merged["final_prediction"] = np.where(
        has_holdout_router,
        merged[
            "holdout_router_prediction"
        ],
        merged["predicted_value"],
    )

    merged[
        "final_normalized_prediction"
    ] = np.where(
        has_holdout_router
        & merged["router_choose_trocr"],
        merged[
            "holdout_trocr_normalized"
        ],
        merged["normalized_prediction"],
    )

    merged["final_exact_match"] = np.where(
        has_holdout_router,
        merged[
            "holdout_router_exact_match"
        ].map(to_bool),
        merged["exact_match"].map(
            to_bool
        ),
    ).astype(bool)

    merged["final_similarity"] = np.where(
        has_holdout_router,
        pd.to_numeric(
            merged[
                "holdout_router_similarity"
            ],
            errors="coerce",
        ).fillna(0.0),
        pd.to_numeric(
            merged["similarity"],
            errors="coerce",
        ).fillna(0.0),
    )

    merged["router_probability"] = (
        merged["router_probability"]
        .fillna(0.0)
    )

    return merged


def build_field_scores(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in dataframe.iterrows():
        document_type = clean_text(
            row["document_type"]
        )

        field_name = clean_text(
            row["field_name"]
        )

        required = (
            field_name
            in required_fields_for(
                document_type
            )
        )

        final_value = clean_text(
            row[
                "final_normalized_prediction"
            ]
        )

        (
            format_valid,
            format_score,
            format_message,
        ) = field_format_validation(
            field_name,
            final_value,
            required=required,
        )

        model_score = engine_confidence(
            row
        )

        validation_score = (
            0.60 * format_score
            + 0.40 * model_score
        )

        rows.append(
            {
                "document_id": clean_text(
                    row["document_id"]
                ),
                "parent_document_id": (
                    clean_text(
                        row[
                            "parent_document_id"
                        ]
                    )
                ),
                "document_type": (
                    document_type
                ),
                "split": clean_text(
                    row["split"]
                ),
                "quality": clean_text(
                    row["quality"]
                ),
                "field_name": field_name,
                "required_field": required,
                "final_engine": clean_text(
                    row["final_engine"]
                ),
                "final_prediction": clean_text(
                    row["final_prediction"]
                ),
                "final_normalized_prediction": (
                    final_value
                ),
                "format_valid": (
                    format_valid
                ),
                "format_score": round(
                    float(format_score),
                    4,
                ),
                "format_message": (
                    format_message
                ),
                "model_confidence": round(
                    float(model_score),
                    4,
                ),
                "validation_score": round(
                    float(validation_score),
                    4,
                ),
                "final_exact_match": (
                    bool(
                        row[
                            "final_exact_match"
                        ]
                    )
                ),
                "final_similarity": float(
                    row[
                        "final_similarity"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def threshold_metrics(
    fields: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    review = (
        ~fields["format_valid"]
        | (
            fields[
                "validation_score"
            ]
            < threshold
        )
    )

    accepted = ~review
    correct = fields[
        "final_exact_match"
    ].astype(bool)
    incorrect = ~correct

    return {
        "threshold": round(
            float(threshold),
            2,
        ),
        "fields": int(
            len(fields)
        ),
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
                correct[
                    accepted
                ].mean()
            )
            if accepted.any()
            else 0.0,
            4,
        ),
        "incorrect_field_capture_rate": round(
            float(
                review[
                    incorrect
                ].mean()
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


def choose_threshold(
    search: pd.DataFrame,
    *,
    target_auto_accept_accuracy: float,
) -> pd.Series:
    eligible = search[
        search[
            "auto_accepted_exact_match_rate"
        ]
        >= target_auto_accept_accuracy
    ].copy()

    if not eligible.empty:
        return eligible.sort_values(
            [
                "auto_accept_coverage",
                "incorrect_field_capture_rate",
                "threshold",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        ).iloc[0]

    return search.sort_values(
        [
            "auto_accepted_exact_match_rate",
            "auto_accept_coverage",
            "incorrect_field_capture_rate",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).iloc[0]


def build_document_policy(
    fields: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    fields = fields.copy()

    fields["field_review_required"] = (
        ~fields["format_valid"]
        | (
            fields[
                "validation_score"
            ]
            < threshold
        )
    )

    document_rows: list[
        dict[str, Any]
    ] = []

    for document_id, document_df in (
        fields.groupby(
            "document_id",
            sort=False,
        )
    ):
        document_type = clean_text(
            document_df[
                "document_type"
            ].iloc[0]
        )

        field_values = {
            clean_text(row["field_name"]): (
                clean_text(
                    row[
                        "final_normalized_prediction"
                    ]
                )
            )
            for _, row in (
                document_df.iterrows()
            )
        }

        cross_field_issues = (
            validate_cross_fields(
                document_type,
                field_values,
            )
        )

        required_invalid = (
            document_df[
                "required_field"
            ]
            & ~document_df[
                "format_valid"
            ]
        )

        required_review = (
            document_df[
                "required_field"
            ]
            & document_df[
                "field_review_required"
            ]
        )

        optional_review = (
            ~document_df[
                "required_field"
            ]
            & document_df[
                "field_review_required"
            ]
        )

        if (
            required_invalid.any()
            or cross_field_issues
        ):
            status = "full_review"

        elif required_review.any():
            status = (
                "required_fields_review"
            )

        elif optional_review.any():
            status = "partial_review"

        else:
            status = "auto_accept"

        document_rows.append(
            {
                "document_id": (
                    document_id
                ),
                "parent_document_id": (
                    clean_text(
                        document_df[
                            "parent_document_id"
                        ].iloc[0]
                    )
                ),
                "document_type": (
                    document_type
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
                "required_fields_reviewed": int(
                    required_review.sum()
                ),
                "optional_fields_reviewed": int(
                    optional_review.sum()
                ),
                "required_invalid_fields": int(
                    required_invalid.sum()
                ),
                "cross_field_issue_count": int(
                    len(cross_field_issues)
                ),
                "cross_field_issues": (
                    "; ".join(
                        cross_field_issues
                    )
                ),
                "document_status": (
                    status
                ),
            }
        )

    return pd.DataFrame(
        document_rows
    )


def run_calibration(
    *,
    hybrid_validation_path: Path,
    router_holdout_path: Path,
    target_auto_accept_accuracy: float,
) -> None:
    routed_holdout = (
        load_parent_holdout_predictions(
            hybrid_validation_path=(
                hybrid_validation_path
            ),
            router_holdout_path=(
                router_holdout_path
            ),
        )
    )

    fields = build_field_scores(
        routed_holdout
    )

    thresholds = np.arange(
        0.50,
        0.951,
        0.01,
    )

    search = pd.DataFrame(
        [
            threshold_metrics(
                fields,
                float(threshold),
            )
            for threshold in thresholds
        ]
    )

    selected = choose_threshold(
        search,
        target_auto_accept_accuracy=(
            target_auto_accept_accuracy
        ),
    )

    selected_threshold = float(
        selected["threshold"]
    )

    fields[
        "field_review_required"
    ] = (
        ~fields["format_valid"]
        | (
            fields[
                "validation_score"
            ]
            < selected_threshold
        )
    )

    documents = build_document_policy(
        fields,
        threshold=selected_threshold,
    )

    THRESHOLD_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    search.to_csv(
        THRESHOLD_REPORT_PATH,
        index=False,
    )

    fields.to_csv(
        FIELD_REPORT_PATH,
        index=False,
    )

    documents.to_csv(
        DOCUMENT_REPORT_PATH,
        index=False,
    )

    summary = {
        "calibration_source": (
            "parent-held-out validation documents"
        ),
        "hybrid_validation_path": str(
            hybrid_validation_path
        ),
        "router_holdout_path": str(
            router_holdout_path
        ),
        "documents": int(
            fields[
                "document_id"
            ].nunique()
        ),
        "parent_documents": int(
            fields[
                "parent_document_id"
            ].nunique()
        ),
        "fields": int(
            len(fields)
        ),
        "target_auto_accept_accuracy": (
            target_auto_accept_accuracy
        ),
        "selected_threshold": (
            selected_threshold
        ),
        "selected_policy": {
            key: (
                int(value)
                if key in {
                    "fields",
                    "fields_auto_accepted",
                    "fields_reviewed",
                    "incorrect_fields_auto_accepted",
                }
                else float(value)
            )
            for key, value in (
                selected.to_dict().items()
            )
        },
        "document_status_counts": {
            str(status): int(count)
            for status, count in (
                documents[
                    "document_status"
                ]
                .value_counts()
                .items()
            )
        },
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
        "\nIRS review-policy calibration completed"
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
        f"Fields: "
        f"{summary['fields']}"
    )
    print(
        "Target auto-accepted accuracy: "
        f"{target_auto_accept_accuracy:.2%}"
    )
    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )
    print(
        "Auto-accepted fields: "
        f"{int(selected['fields_auto_accepted'])}"
    )
    print(
        "Reviewed fields: "
        f"{int(selected['fields_reviewed'])}"
    )
    print(
        "Field review rate: "
        f"{float(selected['field_review_rate']):.2%}"
    )
    print(
        "Auto-accepted exact match: "
        f"{float(selected['auto_accepted_exact_match_rate']):.2%}"
    )
    print(
        "Incorrect-field capture rate: "
        f"{float(selected['incorrect_field_capture_rate']):.2%}"
    )
    print(
        "Incorrect fields auto accepted: "
        f"{int(selected['incorrect_fields_auto_accepted'])}"
    )
    print(
        "Document statuses: "
        f"{summary['document_status_counts']}"
    )
    print(
        f"\nThreshold search: "
        f"{THRESHOLD_REPORT_PATH}"
    )
    print(
        f"Field report: "
        f"{FIELD_REPORT_PATH}"
    )
    print(
        f"Document report: "
        f"{DOCUMENT_REPORT_PATH}"
    )
    print(
        f"Summary: {SUMMARY_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate IRS field-review thresholds "
            "using only parent-held-out validation data."
        )
    )

    parser.add_argument(
        "--hybrid-validation",
        type=Path,
        default=(
            DEFAULT_HYBRID_VALIDATION
        ),
    )

    parser.add_argument(
        "--router-holdout",
        type=Path,
        default=(
            DEFAULT_ROUTER_HOLDOUT
        ),
    )

    parser.add_argument(
        "--target-auto-accept-accuracy",
        type=float,
        default=0.90,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_calibration(
        hybrid_validation_path=(
            arguments.hybrid_validation
        ),
        router_holdout_path=(
            arguments.router_holdout
        ),
        target_auto_accept_accuracy=(
            arguments.target_auto_accept_accuracy
        ),
    )
