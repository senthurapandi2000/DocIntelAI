from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification"
    / "sec_document_classification_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "sec_form_tfidf_linearsvc.joblib"

VALIDATION_METRICS_PATH = (
    REPORT_DIR / "sec_baseline_validation_metrics.json"
)

TEST_METRICS_PATH = (
    REPORT_DIR / "sec_baseline_test_metrics.json"
)

PREDICTIONS_PATH = (
    REPORT_DIR / "sec_baseline_test_predictions.csv"
)

CONFUSION_MATRIX_PATH = (
    REPORT_DIR / "sec_baseline_confusion_matrix.csv"
)

LABELS = ["10-K", "10-Q", "8-K"]

# Use a fixed amount of content from each filing so the classifier
# does not depend mainly on document length.
MAX_WORDS = 6000


def read_text_file(relative_path: str) -> str:
    """Read a processed SEC text file."""

    path = PROJECT_ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(
            f"Processed text file not found: {path}"
        )

    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def mask_explicit_form_labels(text: str) -> str:
    """
    Mask obvious document labels to reduce target leakage.

    The model should learn filing structure and language rather than
    simply detecting a phrase such as 'Form 10-K'.
    """

    patterns = [
        r"\bFORM\s+10[\s\-]?K\b",
        r"\bFORM\s+10[\s\-]?Q\b",
        r"\bFORM\s+8[\s\-]?K\b",
        r"\b10[\s\-]?K\b",
        r"\b10[\s\-]?Q\b",
        r"\b8[\s\-]?K\b",
        r"\bANNUAL\s+REPORT\b",
        r"\bQUARTERLY\s+REPORT\b",
        r"\bCURRENT\s+REPORT\b",
        r"\bANNUAL\s+REPORT\s+PURSUANT\s+TO\b",
        r"\bQUARTERLY\s+REPORT\s+PURSUANT\s+TO\b",
        r"\bCURRENT\s+REPORT\s+PURSUANT\s+TO\b",
    ]

    cleaned_text = text

    for pattern in patterns:
        cleaned_text = re.sub(
            pattern,
            " FORM_TYPE ",
            cleaned_text,
            flags=re.IGNORECASE,
        )

    return cleaned_text


def sample_document_words(
    text: str,
    max_words: int = MAX_WORDS,
) -> str:
    """
    Keep the beginning, middle, and end of long documents.

    This preserves multiple sections while limiting extreme length
    differences between 10-K, 10-Q, and 8-K filings.
    """

    words = text.split()

    if len(words) <= max_words:
        return " ".join(words)

    first_size = max_words // 3
    middle_size = max_words // 3
    last_size = max_words - first_size - middle_size

    middle_index = len(words) // 2
    middle_start = max(
        middle_index - middle_size // 2,
        first_size,
    )
    middle_end = middle_start + middle_size

    selected_words = (
        words[:first_size]
        + words[middle_start:middle_end]
        + words[-last_size:]
    )

    return " ".join(selected_words)


def prepare_document(text: str) -> str:
    """Apply leakage control and fixed-length sampling."""

    text = mask_explicit_form_labels(text)
    text = sample_document_words(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_dataset() -> pd.DataFrame:
    """Load metadata and processed filing text."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Classification dataset not found: {DATASET_PATH}"
        )

    dataset = pd.read_csv(DATASET_PATH)

    required_columns = {
        "document_id",
        "ticker",
        "company_name",
        "sector",
        "label",
        "split",
        "output_path",
    }

    missing_columns = (
        required_columns - set(dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            "Classification dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataset["text"] = dataset["output_path"].apply(
        read_text_file
    )

    dataset["prepared_text"] = dataset["text"].apply(
        prepare_document
    )

    empty_documents = (
        dataset["prepared_text"].str.strip() == ""
    ).sum()

    if empty_documents:
        raise ValueError(
            f"Prepared dataset contains "
            f"{empty_documents} empty documents."
        )

    return dataset


def build_pipeline() -> Pipeline:
    """Create the TF-IDF and Linear SVM baseline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=60000,
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def calculate_confidence_margin(
    model: Pipeline,
    texts: pd.Series,
) -> np.ndarray:
    """
    Calculate the difference between the two highest decision scores.

    This is a ranking margin, not a calibrated probability.
    """

    decision_scores = model.decision_function(texts)

    if decision_scores.ndim == 1:
        return np.abs(decision_scores)

    sorted_scores = np.sort(
        decision_scores,
        axis=1,
    )

    return (
        sorted_scores[:, -1]
        - sorted_scores[:, -2]
    )


def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, Any]:
    """Calculate classification performance metrics."""

    report = classification_report(
        actual,
        predicted,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(
            float(
                accuracy_score(
                    actual,
                    predicted,
                )
            ),
            4,
        ),
        "macro_f1": round(
            float(
                f1_score(
                    actual,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "weighted_f1": round(
            float(
                f1_score(
                    actual,
                    predicted,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
        "documents": int(len(actual)),
        "classification_report": report,
    }


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Save a dictionary as JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def train_baseline() -> None:
    """Train, validate, refit, and evaluate the classifier."""

    dataset = load_dataset()

    train_df = dataset[
        dataset["split"] == "train"
    ].copy()

    validation_df = dataset[
        dataset["split"] == "validation"
    ].copy()

    test_df = dataset[
        dataset["split"] == "test"
    ].copy()

    if train_df.empty:
        raise ValueError("Training split is empty.")

    if validation_df.empty:
        raise ValueError("Validation split is empty.")

    if test_df.empty:
        raise ValueError("Test split is empty.")

    print("\nDataset loaded")
    print(f"Training documents: {len(train_df)}")
    print(
        f"Validation documents: "
        f"{len(validation_df)}"
    )
    print(f"Test documents: {len(test_df)}")

    print("\nTraining validation-stage model...")

    validation_model = build_pipeline()

    validation_model.fit(
        train_df["prepared_text"],
        train_df["label"],
    )

    validation_predictions = validation_model.predict(
        validation_df["prepared_text"]
    )

    validation_metrics = calculate_metrics(
        validation_df["label"],
        validation_predictions,
    )

    save_json(
        validation_metrics,
        VALIDATION_METRICS_PATH,
    )

    print(
        "Validation accuracy: "
        f"{validation_metrics['accuracy']:.4f}"
    )
    print(
        "Validation macro-F1: "
        f"{validation_metrics['macro_f1']:.4f}"
    )

    print(
        "\nRefitting final model using "
        "training and validation data..."
    )

    development_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    )

    final_model = build_pipeline()

    final_model.fit(
        development_df["prepared_text"],
        development_df["label"],
    )

    test_predictions = final_model.predict(
        test_df["prepared_text"]
    )

    confidence_margins = calculate_confidence_margin(
        final_model,
        test_df["prepared_text"],
    )

    test_metrics = calculate_metrics(
        test_df["label"],
        test_predictions,
    )

    save_json(
        test_metrics,
        TEST_METRICS_PATH,
    )

    prediction_report = test_df[
        [
            "document_id",
            "ticker",
            "company_name",
            "sector",
            "label",
            "output_path",
        ]
    ].copy()

    prediction_report["predicted_label"] = (
        test_predictions
    )

    prediction_report["correct"] = (
        prediction_report["label"]
        == prediction_report["predicted_label"]
    )

    prediction_report["confidence_margin"] = (
        np.round(
            confidence_margins,
            4,
        )
    )

    prediction_report.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    matrix = confusion_matrix(
        test_df["label"],
        test_predictions,
        labels=LABELS,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"actual_{label}"
            for label in LABELS
        ],
        columns=[
            f"predicted_{label}"
            for label in LABELS
        ],
    )

    matrix_df.to_csv(
        CONFUSION_MATRIX_PATH
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_bundle = {
        "model": final_model,
        "labels": LABELS,
        "max_words": MAX_WORDS,
        "training_documents": int(
            len(development_df)
        ),
        "test_documents": int(
            len(test_df)
        ),
        "test_metrics": test_metrics,
        "preprocessing": {
            "mask_explicit_form_labels": True,
            "sampling_strategy": (
                "beginning_middle_end"
            ),
        },
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    print("\nFinal test evaluation")
    print(
        f"Test accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"Test macro-F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )
    print(
        f"Test weighted-F1: "
        f"{test_metrics['weighted_f1']:.4f}"
    )
    print(f"Model saved: {MODEL_PATH}")
    print(
        f"Test predictions: "
        f"{PREDICTIONS_PATH}"
    )
    print(
        f"Confusion matrix: "
        f"{CONFUSION_MATRIX_PATH}"
    )


if __name__ == "__main__":
    train_baseline()