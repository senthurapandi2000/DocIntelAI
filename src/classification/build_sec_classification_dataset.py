from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUALITY_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_text_quality_report.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification"
)

DATASET_PATH = (
    OUTPUT_DIR
    / "sec_document_classification_dataset.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "sec_classification_dataset_summary.json"
)

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def assign_company_splits(
    tickers: list[str],
) -> dict[str, str]:
    """
    Assign entire companies to train, validation, or test.

    This prevents documents from the same company from appearing
    in multiple dataset splits.
    """

    shuffled_tickers = tickers.copy()

    random.Random(RANDOM_SEED).shuffle(
        shuffled_tickers
    )

    total_companies = len(shuffled_tickers)

    train_count = int(
        total_companies * TRAIN_RATIO
    )

    validation_count = int(
        total_companies * VALIDATION_RATIO
    )

    train_tickers = set(
        shuffled_tickers[:train_count]
    )

    validation_tickers = set(
        shuffled_tickers[
            train_count:
            train_count + validation_count
        ]
    )

    test_tickers = set(
        shuffled_tickers[
            train_count + validation_count:
        ]
    )

    split_mapping: dict[str, str] = {}

    for ticker in train_tickers:
        split_mapping[ticker] = "train"

    for ticker in validation_tickers:
        split_mapping[ticker] = "validation"

    for ticker in test_tickers:
        split_mapping[ticker] = "test"

    return split_mapping


def build_dataset() -> None:
    """Create the SEC document-classification dataset."""

    if not QUALITY_REPORT.exists():
        raise FileNotFoundError(
            f"Quality report not found: "
            f"{QUALITY_REPORT}"
        )

    quality_df = pd.read_csv(
        QUALITY_REPORT
    )

    required_columns = {
        "ticker",
        "company_name",
        "sector",
        "form",
        "output_path",
        "word_count",
        "quality_status",
        "is_duplicate",
    }

    missing_columns = (
        required_columns - set(quality_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Quality report is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataset_df = quality_df[
        (quality_df["quality_status"] == "pass")
        & (~quality_df["is_duplicate"].astype(bool))
    ].copy()

    if dataset_df.empty:
        raise ValueError(
            "No valid documents were available."
        )

    unique_tickers = sorted(
        dataset_df["ticker"].unique()
    )

    split_mapping = assign_company_splits(
        unique_tickers
    )

    dataset_df["split"] = (
        dataset_df["ticker"]
        .map(split_mapping)
    )

    dataset_df["label"] = dataset_df["form"]

    dataset_df["document_id"] = (
        dataset_df["ticker"].astype(str)
        + "_"
        + dataset_df["form"]
        .str.replace("-", "", regex=False)
        + "_"
        + dataset_df.index.astype(str)
    )

    final_columns = [
        "document_id",
        "ticker",
        "company_name",
        "sector",
        "label",
        "split",
        "output_path",
        "word_count",
    ]

    dataset_df = dataset_df[
        final_columns
    ].sort_values(
        by=["split", "label", "ticker"]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_df.to_csv(
        DATASET_PATH,
        index=False,
    )

    split_counts = {
        str(split): int(count)
        for split, count in (
            dataset_df["split"]
            .value_counts()
            .items()
        )
    }

    label_counts = {
        str(label): int(count)
        for label, count in (
            dataset_df["label"]
            .value_counts()
            .items()
        )
    }

    split_label_counts = {}

    grouped_counts = (
        dataset_df
        .groupby(["split", "label"])
        .size()
    )

    for (split, label), count in grouped_counts.items():
        split_label_counts.setdefault(
            str(split),
            {},
        )[str(label)] = int(count)

    companies_by_split = {
        split: sorted(
            dataset_df.loc[
                dataset_df["split"] == split,
                "ticker",
            ].unique().tolist()
        )
        for split in (
            "train",
            "validation",
            "test",
        )
    }

    summary = {
        "total_documents": int(
            len(dataset_df)
        ),
        "total_companies": int(
            dataset_df["ticker"].nunique()
        ),
        "split_counts": split_counts,
        "label_counts": label_counts,
        "split_label_counts": split_label_counts,
        "companies_by_split": companies_by_split,
        "company_overlap": False,
        "random_seed": RANDOM_SEED,
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

    print(
        "\nSEC classification dataset created"
    )
    print(
        f"Documents: "
        f"{summary['total_documents']}"
    )
    print(
        f"Companies: "
        f"{summary['total_companies']}"
    )
    print(
        f"Split counts: "
        f"{summary['split_counts']}"
    )
    print(
        f"Label counts: "
        f"{summary['label_counts']}"
    )
    print(
        f"Split-label counts: "
        f"{summary['split_label_counts']}"
    )
    print(
        f"Dataset: {DATASET_PATH}"
    )
    print(
        f"Summary: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    build_dataset()