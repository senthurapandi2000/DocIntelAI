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
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.extraction.irs_tesseract_baseline import (
    PROJECT_ROOT,
    clean_text,
)
from src.extraction.irs_tesseract_enhanced import (
    format_score,
)


DEFAULT_TROCR_PREDICTIONS = (
    PROJECT_ROOT
    / "reports"
    / "irs_trocr_fallback_benchmark_predictions.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "irs_ocr_router.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "irs_ocr_router_metadata.json"
)

HOLDOUT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_ocr_router_holdout_predictions.csv"
)

COEFFICIENTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_ocr_router_coefficients.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_ocr_router_summary.json"
)

RANDOM_SEED = 42

CATEGORICAL_FEATURES = [
    "document_type",
    "field_name",
]

NUMERIC_FEATURES = [
    "tesseract_confidence",
    "tesseract_length",
    "trocr_length",
    "length_difference",
    "tesseract_format_score",
    "trocr_format_score",
    "model_agreement_similarity",
    "tesseract_digit_ratio",
    "tesseract_alpha_ratio",
    "tesseract_punctuation_ratio",
    "trocr_digit_ratio",
    "trocr_alpha_ratio",
    "trocr_punctuation_ratio",
    "tesseract_empty",
    "trocr_empty",
    "same_normalized_output",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def to_bool(value: Any) -> bool:
    """Convert common CSV boolean representations safely."""

    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def derive_parent_document_id(
    document_id: str,
) -> str:
    """Map clean and augmented variants back to one base document."""

    return re.sub(
        r"_scan_(?:mild|hard)_\d+$",
        "",
        document_id,
        flags=re.IGNORECASE,
    )


def character_ratios(
    value: Any,
) -> tuple[float, float, float]:
    """Return digit, alphabetic, and punctuation ratios."""

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


def make_target(row: pd.Series) -> bool:
    """
    Label TrOCR as preferred when it improves exact correctness, or when
    neither engine is exact but TrOCR improves similarity by at least 3 points.
    """

    tesseract_exact = to_bool(
        row["tesseract_exact_match"]
    )

    trocr_exact = to_bool(
        row["trocr_exact_match"]
    )

    if trocr_exact and not tesseract_exact:
        return True

    if tesseract_exact and not trocr_exact:
        return False

    return bool(
        float(row["trocr_similarity"])
        >= float(row["tesseract_similarity"]) + 3.0
    )


def engineer_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create only features that are available during real inference."""

    result = dataframe.copy()

    result["document_id"] = (
        result["document_id"]
        .fillna("")
        .astype(str)
    )

    result["parent_document_id"] = (
        result["document_id"]
        .map(derive_parent_document_id)
    )

    for column in [
        "tesseract_prediction",
        "tesseract_normalized",
        "trocr_prediction",
        "trocr_normalized",
        "field_name",
        "document_type",
    ]:
        result[column] = (
            result[column]
            .fillna("")
            .astype(str)
        )

    for column in [
        "tesseract_confidence",
        "tesseract_similarity",
        "trocr_similarity",
    ]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).fillna(0.0)

    result["tesseract_length"] = (
        result["tesseract_normalized"]
        .str.len()
        .astype(float)
    )

    result["trocr_length"] = (
        result["trocr_normalized"]
        .str.len()
        .astype(float)
    )

    result["length_difference"] = (
        result["trocr_length"]
        - result["tesseract_length"]
    ).abs()

    result["tesseract_format_score"] = (
        result.apply(
            lambda row: format_score(
                row["tesseract_prediction"],
                row["field_name"],
            ),
            axis=1,
        )
    )

    result["trocr_format_score"] = (
        result.apply(
            lambda row: format_score(
                row["trocr_prediction"],
                row["field_name"],
            ),
            axis=1,
        )
    )

    result["model_agreement_similarity"] = (
        result.apply(
            lambda row: float(
                fuzz.ratio(
                    row["tesseract_normalized"],
                    row["trocr_normalized"],
                )
            ),
            axis=1,
        )
    )

    tesseract_ratios = result[
        "tesseract_prediction"
    ].map(character_ratios)

    trocr_ratios = result[
        "trocr_prediction"
    ].map(character_ratios)

    result[
        [
            "tesseract_digit_ratio",
            "tesseract_alpha_ratio",
            "tesseract_punctuation_ratio",
        ]
    ] = pd.DataFrame(
        tesseract_ratios.tolist(),
        index=result.index,
    )

    result[
        [
            "trocr_digit_ratio",
            "trocr_alpha_ratio",
            "trocr_punctuation_ratio",
        ]
    ] = pd.DataFrame(
        trocr_ratios.tolist(),
        index=result.index,
    )

    result["tesseract_empty"] = (
        result["tesseract_normalized"]
        .str.strip()
        .eq("")
        .astype(float)
    )

    result["trocr_empty"] = (
        result["trocr_normalized"]
        .str.strip()
        .eq("")
        .astype(float)
    )

    result["same_normalized_output"] = (
        result["tesseract_normalized"]
        .eq(result["trocr_normalized"])
        .astype(float)
    )

    result["choose_trocr_target"] = (
        result.apply(
            make_target,
            axis=1,
        )
    )

    result["tesseract_exact_match"] = (
        result["tesseract_exact_match"]
        .map(to_bool)
    )

    result["trocr_exact_match"] = (
        result["trocr_exact_match"]
        .map(to_bool)
    )

    return result


def build_pipeline(
    *,
    regularization_c: float,
) -> Pipeline:
    """Build a mixed categorical/numeric logistic router."""

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
                    handle_unknown="ignore",
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
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def routed_metrics(
    dataframe: pd.DataFrame,
    choose_trocr: np.ndarray,
) -> dict[str, float | int]:
    """Calculate OCR metrics after applying router decisions."""

    choose_trocr = np.asarray(
        choose_trocr,
        dtype=bool,
    )

    routed_exact = np.where(
        choose_trocr,
        dataframe[
            "trocr_exact_match"
        ].to_numpy(dtype=bool),
        dataframe[
            "tesseract_exact_match"
        ].to_numpy(dtype=bool),
    )

    routed_similarity = np.where(
        choose_trocr,
        dataframe[
            "trocr_similarity"
        ].to_numpy(dtype=float),
        dataframe[
            "tesseract_similarity"
        ].to_numpy(dtype=float),
    )

    return {
        "field_predictions": int(
            len(dataframe)
        ),
        "exact_match_rate": round(
            float(routed_exact.mean()),
            4,
        ),
        "mean_similarity": round(
            float(routed_similarity.mean()),
            2,
        ),
        "trocr_selection_rate": round(
            float(choose_trocr.mean()),
            4,
        ),
    }


def single_engine_metrics(
    dataframe: pd.DataFrame,
    *,
    engine: str,
) -> dict[str, float | int]:
    exact_column = (
        f"{engine}_exact_match"
    )

    similarity_column = (
        f"{engine}_similarity"
    )

    return {
        "field_predictions": int(
            len(dataframe)
        ),
        "exact_match_rate": round(
            float(
                dataframe[
                    exact_column
                ].mean()
            ),
            4,
        ),
        "mean_similarity": round(
            float(
                dataframe[
                    similarity_column
                ].mean()
            ),
            2,
        ),
    }


def oracle_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, float | int]:
    oracle_exact = (
        dataframe["tesseract_exact_match"]
        | dataframe["trocr_exact_match"]
    )

    oracle_similarity = dataframe[
        [
            "tesseract_similarity",
            "trocr_similarity",
        ]
    ].max(axis=1)

    return {
        "field_predictions": int(
            len(dataframe)
        ),
        "exact_match_rate": round(
            float(oracle_exact.mean()),
            4,
        ),
        "mean_similarity": round(
            float(oracle_similarity.mean()),
            2,
        ),
    }


def tune_threshold(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, float | int]]:
    """Choose threshold by exact match, then similarity, then lower cost."""

    best_threshold = 0.5
    best_metrics: dict[str, float | int] | None = None
    best_key: tuple[float, float, float] | None = None

    for threshold in np.arange(
        0.10,
        0.91,
        0.02,
    ):
        metrics = routed_metrics(
            dataframe,
            probabilities >= threshold,
        )

        comparison_key = (
            float(
                metrics["exact_match_rate"]
            ),
            float(
                metrics["mean_similarity"]
            ),
            -float(
                metrics["trocr_selection_rate"]
            ),
        )

        if (
            best_key is None
            or comparison_key > best_key
        ):
            best_key = comparison_key
            best_threshold = float(
                round(threshold, 2)
            )
            best_metrics = metrics

    assert best_metrics is not None

    return best_threshold, best_metrics


def cross_validated_probabilities(
    dataframe: pd.DataFrame,
    *,
    regularization_c: float,
) -> np.ndarray:
    group_count = int(
        dataframe[
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
            "are required for router cross-validation."
        )

    pipeline = build_pipeline(
        regularization_c=regularization_c,
    )

    group_cv = GroupKFold(
        n_splits=split_count
    )

    return cross_val_predict(
        pipeline,
        dataframe[ALL_FEATURES],
        dataframe["choose_trocr_target"],
        groups=dataframe[
            "parent_document_id"
        ],
        cv=group_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]


def select_model_configuration(
    training_data: pd.DataFrame,
) -> tuple[float, float, dict[str, Any]]:
    """Tune logistic regularization and decision threshold by group CV."""

    candidates = [
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
    ]

    search_rows: list[dict[str, Any]] = []

    best_c = candidates[0]
    best_threshold = 0.5
    best_key: tuple[float, float, float] | None = None

    for regularization_c in candidates:
        probabilities = (
            cross_validated_probabilities(
                training_data,
                regularization_c=(
                    regularization_c
                ),
            )
        )

        threshold, metrics = tune_threshold(
            training_data,
            probabilities,
        )

        search_row = {
            "regularization_c": (
                regularization_c
            ),
            "threshold": threshold,
            **metrics,
        }

        search_rows.append(search_row)

        comparison_key = (
            float(
                metrics["exact_match_rate"]
            ),
            float(
                metrics["mean_similarity"]
            ),
            -float(
                metrics["trocr_selection_rate"]
            ),
        )

        if (
            best_key is None
            or comparison_key > best_key
        ):
            best_key = comparison_key
            best_c = regularization_c
            best_threshold = threshold

    return (
        best_c,
        best_threshold,
        {
            "candidates": search_rows,
        },
    )


def save_coefficients(
    fitted_pipeline: Pipeline,
) -> None:
    """Save interpretable logistic feature coefficients."""

    preprocessor = fitted_pipeline.named_steps[
        "preprocessor"
    ]

    classifier = fitted_pipeline.named_steps[
        "classifier"
    ]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    coefficients = classifier.coef_[0]

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(
                coefficients
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

    coefficient_df.to_csv(
        COEFFICIENTS_PATH,
        index=False,
    )


def run_training(
    *,
    predictions_path: Path,
    holdout_size: float,
) -> None:
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"TrOCR predictions not found: "
            f"{predictions_path}"
        )

    raw_data = pd.read_csv(
        predictions_path
    )

    required_columns = {
        "document_id",
        "document_type",
        "quality",
        "field_name",
        "tesseract_prediction",
        "tesseract_normalized",
        "tesseract_confidence",
        "tesseract_exact_match",
        "tesseract_similarity",
        "trocr_prediction",
        "trocr_normalized",
        "trocr_exact_match",
        "trocr_similarity",
    }

    missing_columns = (
        required_columns - set(raw_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "TrOCR benchmark file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    data = engineer_features(
        raw_data
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=holdout_size,
        random_state=RANDOM_SEED,
    )

    train_indices, holdout_indices = next(
        splitter.split(
            data,
            data["choose_trocr_target"],
            groups=data[
                "parent_document_id"
            ],
        )
    )

    training_data = data.iloc[
        train_indices
    ].copy()

    holdout_data = data.iloc[
        holdout_indices
    ].copy()

    (
        best_c,
        best_threshold,
        search_details,
    ) = select_model_configuration(
        training_data
    )

    holdout_model = build_pipeline(
        regularization_c=best_c,
    )

    holdout_model.fit(
        training_data[ALL_FEATURES],
        training_data[
            "choose_trocr_target"
        ],
    )

    holdout_probabilities = (
        holdout_model.predict_proba(
            holdout_data[ALL_FEATURES]
        )[:, 1]
    )

    holdout_choose_trocr = (
        holdout_probabilities
        >= best_threshold
    )

    target_predictions = (
        holdout_choose_trocr
    )

    target_true = holdout_data[
        "choose_trocr_target"
    ].to_numpy(dtype=bool)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            target_true,
            target_predictions,
            average="binary",
            zero_division=0,
        )
    )

    holdout_output = holdout_data.copy()

    holdout_output[
        "router_probability"
    ] = holdout_probabilities

    holdout_output[
        "router_choose_trocr"
    ] = holdout_choose_trocr

    holdout_output[
        "router_prediction"
    ] = np.where(
        holdout_choose_trocr,
        holdout_output[
            "trocr_prediction"
        ],
        holdout_output[
            "tesseract_prediction"
        ],
    )

    holdout_output[
        "router_exact_match"
    ] = np.where(
        holdout_choose_trocr,
        holdout_output[
            "trocr_exact_match"
        ],
        holdout_output[
            "tesseract_exact_match"
        ],
    )

    holdout_output[
        "router_similarity"
    ] = np.where(
        holdout_choose_trocr,
        holdout_output[
            "trocr_similarity"
        ],
        holdout_output[
            "tesseract_similarity"
        ],
    )

    HOLDOUT_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    holdout_output.to_csv(
        HOLDOUT_PREDICTIONS_PATH,
        index=False,
    )

    # Retune threshold using grouped out-of-fold probabilities on all
    # validation fallback data, then fit the production router on all rows.
    full_oof_probabilities = (
        cross_validated_probabilities(
            data,
            regularization_c=best_c,
        )
    )

    final_threshold, final_oof_metrics = (
        tune_threshold(
            data,
            full_oof_probabilities,
        )
    )

    final_model = build_pipeline(
        regularization_c=best_c,
    )

    final_model.fit(
        data[ALL_FEATURES],
        data[
            "choose_trocr_target"
        ],
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "pipeline": final_model,
            "threshold": final_threshold,
            "categorical_features": (
                CATEGORICAL_FEATURES
            ),
            "numeric_features": (
                NUMERIC_FEATURES
            ),
            "all_features": ALL_FEATURES,
        },
        MODEL_PATH,
    )

    save_coefficients(
        final_model
    )

    holdout_metrics = {
        "tesseract": single_engine_metrics(
            holdout_data,
            engine="tesseract",
        ),
        "trocr": single_engine_metrics(
            holdout_data,
            engine="trocr",
        ),
        "router": routed_metrics(
            holdout_data,
            holdout_choose_trocr,
        ),
        "oracle": oracle_metrics(
            holdout_data
        ),
        "router_target_accuracy": round(
            float(
                accuracy_score(
                    target_true,
                    target_predictions,
                )
            ),
            4,
        ),
        "router_target_precision": round(
            float(precision),
            4,
        ),
        "router_target_recall": round(
            float(recall),
            4,
        ),
        "router_target_f1": round(
            float(f1),
            4,
        ),
        "router_target_confusion_matrix": (
            confusion_matrix(
                target_true,
                target_predictions,
                labels=[False, True],
            ).tolist()
        ),
        "router_target_report": (
            classification_report(
                target_true,
                target_predictions,
                labels=[False, True],
                target_names=[
                    "choose_tesseract",
                    "choose_trocr",
                ],
                output_dict=True,
                zero_division=0,
            )
        ),
    }

    by_quality: dict[str, Any] = {}

    for quality, quality_df in (
        holdout_output.groupby(
            "quality"
        )
    ):
        decisions = quality_df[
            "router_choose_trocr"
        ].to_numpy(dtype=bool)

        by_quality[str(quality)] = {
            "tesseract": (
                single_engine_metrics(
                    quality_df,
                    engine="tesseract",
                )
            ),
            "trocr": (
                single_engine_metrics(
                    quality_df,
                    engine="trocr",
                )
            ),
            "router": routed_metrics(
                quality_df,
                decisions,
            ),
            "oracle": oracle_metrics(
                quality_df
            ),
        }

    summary = {
        "source_predictions": str(
            predictions_path
        ),
        "rows": int(len(data)),
        "parent_documents": int(
            data[
                "parent_document_id"
            ].nunique()
        ),
        "training_rows": int(
            len(training_data)
        ),
        "holdout_rows": int(
            len(holdout_data)
        ),
        "training_parent_documents": int(
            training_data[
                "parent_document_id"
            ].nunique()
        ),
        "holdout_parent_documents": int(
            holdout_data[
                "parent_document_id"
            ].nunique()
        ),
        "holdout_size": float(
            holdout_size
        ),
        "positive_target_rate": round(
            float(
                data[
                    "choose_trocr_target"
                ].mean()
            ),
            4,
        ),
        "selected_regularization_c": (
            best_c
        ),
        "holdout_threshold": (
            best_threshold
        ),
        "final_oof_threshold": (
            final_threshold
        ),
        "configuration_search": (
            search_details
        ),
        "holdout": holdout_metrics,
        "holdout_by_quality": (
            by_quality
        ),
        "full_validation_oof_router": (
            final_oof_metrics
        ),
        "model_path": str(MODEL_PATH),
        "metadata_path": str(
            METADATA_PATH
        ),
    }

    metadata = {
        "model_type": (
            "LogisticRegression OCR router"
        ),
        "training_source": str(
            predictions_path
        ),
        "training_rows": int(
            len(data)
        ),
        "parent_documents": int(
            data[
                "parent_document_id"
            ].nunique()
        ),
        "regularization_c": best_c,
        "decision_threshold": (
            final_threshold
        ),
        "categorical_features": (
            CATEGORICAL_FEATURES
        ),
        "numeric_features": (
            NUMERIC_FEATURES
        ),
        "target_definition": (
            "Choose TrOCR when it uniquely achieves exact "
            "match, or when neither engine is exact and "
            "TrOCR similarity exceeds Tesseract by at least "
            "3 points."
        ),
        "excluded_from_features": [
            "ground truth",
            "exact-match labels",
            "similarity-to-ground-truth",
            "synthetic quality label",
        ],
    }

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    print("\nOCR router training completed")
    print(
        f"Rows: {summary['rows']}"
    )
    print(
        f"Parent documents: "
        f"{summary['parent_documents']}"
    )
    print(
        f"Training rows: "
        f"{summary['training_rows']}"
    )
    print(
        f"Holdout rows: "
        f"{summary['holdout_rows']}"
    )
    print(
        f"Selected C: "
        f"{summary['selected_regularization_c']}"
    )
    print(
        f"Holdout threshold: "
        f"{summary['holdout_threshold']:.2f}"
    )
    print(
        f"Final OOF threshold: "
        f"{summary['final_oof_threshold']:.2f}"
    )

    print("\nParent-held-out fallback metrics")

    for name in [
        "tesseract",
        "trocr",
        "router",
        "oracle",
    ]:
        metrics = holdout_metrics[name]

        print(
            f"{name}: exact match "
            f"{metrics['exact_match_rate']:.2%}, "
            f"similarity "
            f"{metrics['mean_similarity']:.2f}"
            + (
                ", TrOCR selected "
                f"{metrics['trocr_selection_rate']:.2%}"
                if name == "router"
                else ""
            )
        )

    print(
        "\nRouter target classification: "
        f"accuracy "
        f"{holdout_metrics['router_target_accuracy']:.2%}, "
        f"precision "
        f"{holdout_metrics['router_target_precision']:.2%}, "
        f"recall "
        f"{holdout_metrics['router_target_recall']:.2%}, "
        f"F1 "
        f"{holdout_metrics['router_target_f1']:.2%}"
    )

    print(
        "\nFull-validation grouped OOF router: "
        f"exact match "
        f"{final_oof_metrics['exact_match_rate']:.2%}, "
        f"similarity "
        f"{final_oof_metrics['mean_similarity']:.2f}, "
        f"TrOCR selected "
        f"{final_oof_metrics['trocr_selection_rate']:.2%}"
    )

    print(f"\nModel: {MODEL_PATH}")
    print(
        f"Metadata: {METADATA_PATH}"
    )
    print(
        f"Holdout predictions: "
        f"{HOLDOUT_PREDICTIONS_PATH}"
    )
    print(
        f"Coefficients: "
        f"{COEFFICIENTS_PATH}"
    )
    print(f"Summary: {SUMMARY_PATH}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a parent-held-out router that chooses "
            "between hybrid Tesseract and TrOCR."
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=(
            DEFAULT_TROCR_PREDICTIONS
        ),
    )

    parser.add_argument(
        "--holdout-size",
        type=float,
        default=0.25,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_training(
        predictions_path=(
            arguments.predictions
        ),
        holdout_size=(
            arguments.holdout_size
        ),
    )
