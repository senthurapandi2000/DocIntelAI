from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.validation.calibrate_irs_review_policy import (
    load_parent_holdout_predictions,
)
from src.validation.irs_extraction_validator import (
    EIN_FIELDS,
    IDENTIFIER_FIELDS,
    MONEY_FIELDS,
    SSN_FIELDS,
    STATE_FIELDS,
    ZIP_FIELDS,
    clean_text,
    engine_confidence,
    field_format_validation,
    required_fields_for,
    validate_cross_fields,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "irs_field_confidence_model.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "irs_field_confidence_model_metadata.json"
)

OOF_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_field_confidence_oof_predictions.csv"
)

THRESHOLD_SEARCH_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_field_confidence_threshold_search.csv"
)

COEFFICIENTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_field_confidence_coefficients.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_field_confidence_summary.json"
)

RANDOM_SEED = 42

CATEGORICAL_FEATURES = [
    "document_type",
    "field_name",
    "final_engine",
    "risk_tier",
    "format_message",
]

NUMERIC_FEATURES = [
    "required_field",
    "format_valid",
    "format_score",
    "model_confidence",
    "ocr_confidence",
    "router_probability",
    "prediction_length",
    "digit_ratio",
    "alpha_ratio",
    "punctuation_ratio",
    "has_trocr_candidate",
    "tesseract_trocr_agreement",
    "cross_field_flag",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

CROSS_FIELD_MAP = {
    "social_security_tax_inconsistent": {
        "social_security_wages",
        "social_security_tax_withheld",
    },
    "medicare_tax_inconsistent": {
        "medicare_wages",
        "medicare_tax_withheld",
    },
    "federal_tax_exceeds_wages": {
        "wages",
        "federal_tax_withheld",
    },
    "federal_tax_exceeds_compensation": {
        "nonemployee_compensation",
        "federal_income_tax_withheld",
    },
    "state_tax_exceeds_state_income": {
        "state_income",
        "state_tax_withheld",
    },
}


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


def field_risk_tier(field_name: str) -> str:
    if (
        field_name in MONEY_FIELDS
        or field_name in SSN_FIELDS
        or field_name in EIN_FIELDS
    ):
        return "critical"

    if (
        field_name in ZIP_FIELDS
        or field_name in STATE_FIELDS
        or field_name in IDENTIFIER_FIELDS
    ):
        return "structured"

    return "descriptive"


def character_ratios(value: Any) -> tuple[float, float, float]:
    text = clean_text(value)

    if not text:
        return 0.0, 0.0, 0.0

    length = len(text)

    digit_ratio = sum(
        character.isdigit()
        for character in text
    ) / length

    alpha_ratio = sum(
        character.isalpha()
        for character in text
    ) / length

    punctuation_ratio = sum(
        not character.isalnum()
        and not character.isspace()
        for character in text
    ) / length

    return (
        round(digit_ratio, 6),
        round(alpha_ratio, 6),
        round(punctuation_ratio, 6),
    )


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
            document_df[
                "document_type"
            ].iloc[0]
        )

        field_values = {
            clean_text(row["field_name"]): clean_text(
                row[
                    "final_normalized_prediction"
                ]
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
                document_df[
                    "field_name"
                ]
                .astype(str)
                .isin(affected_fields)
            )

            flags.loc[
                document_df.index[
                    affected_mask
                ]
            ] = 1.0

    return flags


def engineer_features(
    routed_holdout: pd.DataFrame,
) -> pd.DataFrame:
    data = routed_holdout.copy()

    for column in [
        "document_id",
        "parent_document_id",
        "document_type",
        "field_name",
        "final_engine",
        "final_prediction",
        "final_normalized_prediction",
        "holdout_tesseract_normalized",
        "holdout_trocr_normalized",
    ]:
        if column not in data.columns:
            data[column] = ""

        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
        )

    for column in [
        "ocr_confidence",
        "router_probability",
    ]:
        data[column] = pd.to_numeric(
            data.get(column, 0.0),
            errors="coerce",
        ).fillna(0.0)

    data["final_exact_match"] = (
        data["final_exact_match"]
        .map(to_bool)
    )

    data["required_field"] = data.apply(
        lambda row: float(
            clean_text(
                row["field_name"]
            )
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
            clean_text(
                row["field_name"]
            ),
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
        data[
            "holdout_trocr_normalized"
        ]
        .str.strip()
        .ne("")
        .astype(float)
    )

    data[
        "tesseract_trocr_agreement"
    ] = data.apply(
        lambda row: float(
            fuzz.ratio(
                row[
                    "holdout_tesseract_normalized"
                ],
                row[
                    "holdout_trocr_normalized"
                ],
            )
        )
        if (
            clean_text(
                row[
                    "holdout_tesseract_normalized"
                ]
            )
            and clean_text(
                row[
                    "holdout_trocr_normalized"
                ]
            )
        )
        else 0.0,
        axis=1,
    )

    data["cross_field_flag"] = (
        add_cross_field_flags(data)
    )

    return data


def build_pipeline(
    regularization_c: float,
) -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
        ]
    )

    classifier = LogisticRegression(
        C=regularization_c,
        max_iter=4000,
        solver="liblinear",
        random_state=RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def grouped_oof_probabilities(
    data: pd.DataFrame,
    *,
    regularization_c: float,
) -> np.ndarray:
    group_count = int(
        data[
            "parent_document_id"
        ].nunique()
    )

    split_count = min(
        5,
        group_count,
    )

    if split_count < 2:
        raise ValueError(
            "At least two parent-document groups "
            "are required."
        )

    cv = GroupKFold(
        n_splits=split_count
    )

    pipeline = build_pipeline(
        regularization_c
    )

    return cross_val_predict(
        pipeline,
        data[ALL_FEATURES],
        data[
            "final_exact_match"
        ],
        groups=data[
            "parent_document_id"
        ],
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]


def choose_regularization(
    data: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_c = 1.0
    best_key: tuple[float, float] | None = None

    for regularization_c in [
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
    ]:
        probabilities = (
            grouped_oof_probabilities(
                data,
                regularization_c=(
                    regularization_c
                ),
            )
        )

        target = data[
            "final_exact_match"
        ].astype(bool)

        brier = float(
            brier_score_loss(
                target,
                probabilities,
            )
        )

        clipped = np.clip(
            probabilities,
            1e-6,
            1.0 - 1e-6,
        )

        loss = float(
            log_loss(
                target,
                clipped,
                labels=[
                    False,
                    True,
                ],
            )
        )

        try:
            auc = float(
                roc_auc_score(
                    target,
                    probabilities,
                )
            )
        except ValueError:
            auc = 0.5

        rows.append(
            {
                "regularization_c": (
                    regularization_c
                ),
                "brier_score": round(
                    brier,
                    6,
                ),
                "log_loss": round(
                    loss,
                    6,
                ),
                "roc_auc": round(
                    auc,
                    6,
                ),
            }
        )

        comparison_key = (
            -brier,
            -loss,
        )

        if (
            best_key is None
            or comparison_key > best_key
        ):
            best_key = comparison_key
            best_c = regularization_c

    return best_c, pd.DataFrame(rows)


def threshold_search(
    data: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    risk_tier: str,
    target_accuracy: float,
) -> pd.DataFrame:
    tier_mask = (
        data["risk_tier"]
        == risk_tier
    )

    tier_data = data[
        tier_mask
    ].copy()

    tier_probabilities = probabilities[
        tier_mask.to_numpy()
    ]

    rows: list[dict[str, Any]] = []

    if tier_data.empty:
        return pd.DataFrame(rows)

    correct = (
        tier_data[
            "final_exact_match"
        ]
        .astype(bool)
        .to_numpy()
    )

    format_valid = (
        tier_data[
            "format_valid"
        ]
        .astype(bool)
        .to_numpy()
    )

    incorrect = ~correct

    minimum_accept_count = max(
        10,
        int(
            round(
                len(tier_data)
                * 0.03
            )
        ),
    )

    for threshold in np.arange(
        0.05,
        1.0,
        0.01,
    ):
        accepted = (
            tier_probabilities
            >= threshold
        ) & format_valid

        reviewed = ~accepted
        accepted_count = int(
            accepted.sum()
        )

        auto_accuracy = (
            float(
                correct[
                    accepted
                ].mean()
            )
            if accepted_count > 0
            else 0.0
        )

        capture_rate = (
            float(
                reviewed[
                    incorrect
                ].mean()
            )
            if incorrect.any()
            else 0.0
        )

        rows.append(
            {
                "risk_tier": (
                    risk_tier
                ),
                "target_accuracy": (
                    target_accuracy
                ),
                "threshold": round(
                    float(threshold),
                    2,
                ),
                "tier_fields": int(
                    len(tier_data)
                ),
                "fields_auto_accepted": (
                    accepted_count
                ),
                "fields_reviewed": int(
                    reviewed.sum()
                ),
                "auto_accept_coverage": round(
                    float(
                        accepted.mean()
                    ),
                    4,
                ),
                "auto_accepted_exact_match_rate": round(
                    auto_accuracy,
                    4,
                ),
                "incorrect_field_capture_rate": round(
                    capture_rate,
                    4,
                ),
                "incorrect_fields_auto_accepted": int(
                    (
                        incorrect
                        & accepted
                    ).sum()
                ),
                "minimum_accept_count_met": bool(
                    accepted_count
                    >= minimum_accept_count
                ),
                "target_met": bool(
                    accepted_count
                    >= minimum_accept_count
                    and auto_accuracy
                    >= target_accuracy
                ),
            }
        )

    return pd.DataFrame(rows)


def choose_threshold(
    search: pd.DataFrame,
) -> tuple[float, bool]:
    target_met = search[
        "target_met"
    ].astype(bool)

    eligible = search[
        target_met
    ].copy()

    if not eligible.empty:
        selected = eligible.sort_values(
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

        return (
            float(
                selected["threshold"]
            ),
            True,
        )

    fallback = search[
        search[
            "minimum_accept_count_met"
        ].astype(bool)
    ].copy()

    if fallback.empty:
        fallback = search.copy()

    selected = fallback.sort_values(
        [
            "auto_accepted_exact_match_rate",
            "auto_accept_coverage",
            "incorrect_field_capture_rate",
            "threshold",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).iloc[0]

    return (
        float(
            selected["threshold"]
        ),
        False,
    )


def apply_policy(
    data: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    output = data.copy()

    output[
        "correctness_probability"
    ] = probabilities

    output[
        "risk_threshold"
    ] = output[
        "risk_tier"
    ].map(thresholds)

    output[
        "field_review_required"
    ] = (
        ~output[
            "format_valid"
        ].astype(bool)
        | (
            output[
                "correctness_probability"
            ]
            < output[
                "risk_threshold"
            ]
        )
    )

    return output


def policy_metrics(
    data: pd.DataFrame,
) -> dict[str, Any]:
    review = data[
        "field_review_required"
    ].astype(bool)

    accepted = ~review

    correct = data[
        "final_exact_match"
    ].astype(bool)

    incorrect = ~correct

    return {
        "fields": int(
            len(data)
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
            float(
                accepted.mean()
            ),
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


def save_coefficients(
    fitted_pipeline: Pipeline,
) -> None:
    preprocessor = (
        fitted_pipeline.named_steps[
            "preprocessor"
        ]
    )

    classifier = (
        fitted_pipeline.named_steps[
            "classifier"
        ]
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    coefficients = (
        classifier.coef_[0]
    )

    coefficient_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": (
                np.abs(coefficients)
            ),
        }
    ).sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    COEFFICIENTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coefficient_frame.to_csv(
        COEFFICIENTS_PATH,
        index=False,
    )


def run_training(
    *,
    hybrid_validation_path: Path,
    router_holdout_path: Path,
    critical_target: float,
    structured_target: float,
    descriptive_target: float,
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

    data = engineer_features(
        routed_holdout
    )

    best_c, model_search = (
        choose_regularization(
            data
        )
    )

    oof_probabilities = (
        grouped_oof_probabilities(
            data,
            regularization_c=best_c,
        )
    )

    targets = {
        "critical": critical_target,
        "structured": (
            structured_target
        ),
        "descriptive": (
            descriptive_target
        ),
    }

    threshold_frames: list[
        pd.DataFrame
    ] = []

    thresholds: dict[
        str,
        float
    ] = {}

    target_status: dict[
        str,
        bool
    ] = {}

    for tier, target in (
        targets.items()
    ):
        search = threshold_search(
            data,
            oof_probabilities,
            risk_tier=tier,
            target_accuracy=target,
        )

        threshold, met = (
            choose_threshold(search)
        )

        thresholds[tier] = (
            threshold
        )

        target_status[tier] = (
            met
        )

        threshold_frames.append(
            search
        )

    threshold_report = pd.concat(
        threshold_frames,
        ignore_index=True,
    )

    oof_output = apply_policy(
        data,
        oof_probabilities,
        thresholds,
    )

    overall_policy = (
        policy_metrics(
            oof_output
        )
    )

    by_tier = {
        str(tier): (
            policy_metrics(
                tier_df
            )
        )
        for tier, tier_df in (
            oof_output.groupby(
                "risk_tier"
            )
        )
    }

    final_model = build_pipeline(
        best_c
    )

    final_model.fit(
        data[ALL_FEATURES],
        data[
            "final_exact_match"
        ],
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "pipeline": final_model,
            "thresholds": thresholds,
            "categorical_features": (
                CATEGORICAL_FEATURES
            ),
            "numeric_features": (
                NUMERIC_FEATURES
            ),
            "all_features": (
                ALL_FEATURES
            ),
            "regularization_c": (
                best_c
            ),
        },
        MODEL_PATH,
    )

    save_coefficients(
        final_model
    )

    OOF_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    oof_output.to_csv(
        OOF_PREDICTIONS_PATH,
        index=False,
    )

    threshold_report.to_csv(
        THRESHOLD_SEARCH_PATH,
        index=False,
    )

    try:
        overall_auc = float(
            roc_auc_score(
                data[
                    "final_exact_match"
                ],
                oof_probabilities,
            )
        )
    except ValueError:
        overall_auc = 0.5

    summary = {
        "calibration_source": (
            "parent-held-out validation "
            "documents"
        ),
        "hybrid_validation_path": str(
            hybrid_validation_path
        ),
        "router_holdout_path": str(
            router_holdout_path
        ),
        "rows": int(
            len(data)
        ),
        "documents": int(
            data[
                "document_id"
            ].nunique()
        ),
        "parent_documents": int(
            data[
                "parent_document_id"
            ].nunique()
        ),
        "selected_regularization_c": (
            best_c
        ),
        "oof_brier_score": round(
            float(
                brier_score_loss(
                    data[
                        "final_exact_match"
                    ],
                    oof_probabilities,
                )
            ),
            6,
        ),
        "oof_roc_auc": round(
            overall_auc,
            6,
        ),
        "risk_targets": targets,
        "risk_thresholds": (
            thresholds
        ),
        "target_met": (
            target_status
        ),
        "overall_policy": (
            overall_policy
        ),
        "by_risk_tier": (
            by_tier
        ),
        "model_search": (
            model_search.to_dict(
                orient="records"
            )
        ),
        "model_path": str(
            MODEL_PATH
        ),
    }

    metadata = {
        "model_type": (
            "LogisticRegression field "
            "correctness model"
        ),
        "training_source": (
            "parent-held-out validation "
            "OCR outputs"
        ),
        "rows": int(
            len(data)
        ),
        "parent_documents": int(
            data[
                "parent_document_id"
            ].nunique()
        ),
        "regularization_c": (
            best_c
        ),
        "risk_thresholds": (
            thresholds
        ),
        "risk_targets": (
            targets
        ),
        "target_met": (
            target_status
        ),
        "categorical_features": (
            CATEGORICAL_FEATURES
        ),
        "numeric_features": (
            NUMERIC_FEATURES
        ),
        "excluded_from_features": [
            "ground-truth value",
            "ground-truth similarity",
            "test-set labels",
            "synthetic quality label",
        ],
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

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        "\nIRS field-confidence model "
        "training completed"
    )
    print(
        f"Rows: {summary['rows']}"
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
        f"Selected C: "
        f"{best_c}"
    )
    print(
        f"OOF Brier score: "
        f"{summary['oof_brier_score']:.4f}"
    )
    print(
        f"OOF ROC AUC: "
        f"{summary['oof_roc_auc']:.4f}"
    )

    print(
        "\nRisk-aware thresholds"
    )

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
            f"{tier}: threshold "
            f"{thresholds[tier]:.2f}, "
            f"target met "
            f"{target_status[tier]}, "
            f"coverage "
            f"{metrics.get('auto_accept_coverage', 0.0):.2%}, "
            f"accepted exact "
            f"{metrics.get('auto_accepted_exact_match_rate', 0.0):.2%}, "
            f"incorrect capture "
            f"{metrics.get('incorrect_field_capture_rate', 0.0):.2%}"
        )

    print(
        "\nOverall grouped-OOF policy"
    )
    print(
        f"Fields auto accepted: "
        f"{overall_policy['fields_auto_accepted']}"
    )
    print(
        f"Fields reviewed: "
        f"{overall_policy['fields_reviewed']}"
    )
    print(
        f"Review rate: "
        f"{overall_policy['field_review_rate']:.2%}"
    )
    print(
        "Auto-accepted exact match: "
        f"{overall_policy['auto_accepted_exact_match_rate']:.2%}"
    )
    print(
        "Incorrect-field capture rate: "
        f"{overall_policy['incorrect_field_capture_rate']:.2%}"
    )
    print(
        "Incorrect fields auto accepted: "
        f"{overall_policy['incorrect_fields_auto_accepted']}"
    )

    print(
        f"\nModel: {MODEL_PATH}"
    )
    print(
        f"Metadata: {METADATA_PATH}"
    )
    print(
        f"OOF predictions: "
        f"{OOF_PREDICTIONS_PATH}"
    )
    print(
        f"Threshold search: "
        f"{THRESHOLD_SEARCH_PATH}"
    )
    print(
        f"Coefficients: "
        f"{COEFFICIENTS_PATH}"
    )
    print(
        f"Summary: {SUMMARY_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a grouped-validation field "
            "correctness model and select "
            "risk-aware human-review thresholds."
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
        "--critical-target",
        type=float,
        default=0.97,
    )

    parser.add_argument(
        "--structured-target",
        type=float,
        default=0.92,
    )

    parser.add_argument(
        "--descriptive-target",
        type=float,
        default=0.85,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_training(
        hybrid_validation_path=(
            arguments.hybrid_validation
        ),
        router_holdout_path=(
            arguments.router_holdout
        ),
        critical_target=(
            arguments.critical_target
        ),
        structured_target=(
            arguments.structured_target
        ),
        descriptive_target=(
            arguments.descriptive_target
        ),
    )
