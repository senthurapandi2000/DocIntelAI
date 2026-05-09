from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.classification.train_sec_baseline import prepare_document
from src.ingestion.sec_ingestion import (
    create_session,
    load_company_directory,
    request_with_retry,
)
from src.preprocessing.sec_html_preprocessor import extract_clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sec_form_tfidf_linearsvc.joblib"
)

TRAINING_COMPANIES_PATH = (
    PROJECT_ROOT
    / "config"
    / "company_universe.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "external_holdout"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "sec_external_holdout_predictions.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "sec_external_holdout_summary.json"
)

TARGET_FORMS = ("10-K", "10-Q", "8-K")

# These companies must not exist in company_universe.csv.
EXTERNAL_TICKERS = [
    "GOOGL",
    "META",
    "NFLX",
    "KO",
    "F",
]


class ExternalEvaluationError(RuntimeError):
    """Raised when external evaluation cannot be completed."""


def load_training_tickers() -> set[str]:
    """Load tickers already used in model development."""

    with TRAINING_COMPANIES_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        return {
            str(row["ticker"]).strip().upper()
            for row in reader
            if row.get("ticker")
        }


def verify_no_company_overlap() -> None:
    """Ensure external companies were not used during training."""

    training_tickers = load_training_tickers()

    overlap = training_tickers.intersection(
        EXTERNAL_TICKERS
    )

    if overlap:
        raise ExternalEvaluationError(
            "External evaluation contains training companies: "
            f"{sorted(overlap)}"
        )


def find_latest_filings(
    submissions: dict[str, Any],
) -> list[dict[str, str]]:
    """Find the latest filing for every target form."""

    recent = submissions.get(
        "filings",
        {},
    ).get(
        "recent",
        {},
    )

    forms = recent.get("form", [])
    accession_numbers = recent.get(
        "accessionNumber",
        [],
    )
    filing_dates = recent.get(
        "filingDate",
        [],
    )
    primary_documents = recent.get(
        "primaryDocument",
        [],
    )

    selected_forms: set[str] = set()
    selected_records: list[dict[str, str]] = []

    record_count = min(
        len(forms),
        len(accession_numbers),
        len(filing_dates),
        len(primary_documents),
    )

    for index in range(record_count):
        form = forms[index]

        if form not in TARGET_FORMS:
            continue

        if form in selected_forms:
            continue

        primary_document = primary_documents[index]

        if not primary_document:
            continue

        selected_records.append(
            {
                "form": form,
                "accession_number": accession_numbers[index],
                "filing_date": filing_dates[index],
                "primary_document": primary_document,
            }
        )

        selected_forms.add(form)

        if selected_forms == set(TARGET_FORMS):
            break

    return selected_records


def calculate_margin(
    model: Any,
    prepared_text: str,
) -> float:
    """Calculate the margin between the two highest SVM scores."""

    scores = model.decision_function(
        [prepared_text]
    )

    scores = np.asarray(scores)

    if scores.ndim == 1:
        sorted_scores = np.sort(scores)
    else:
        sorted_scores = np.sort(scores[0])

    if len(sorted_scores) < 2:
        return float(abs(sorted_scores[-1]))

    return float(
        sorted_scores[-1] - sorted_scores[-2]
    )


def evaluate_external_filings() -> None:
    """Download and evaluate filings from unseen companies."""

    verify_no_company_overlap()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}"
        )

    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]

    company_directory = load_company_directory()
    session = create_session()

    prediction_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for ticker in EXTERNAL_TICKERS:
        try:
            if ticker not in company_directory:
                raise ExternalEvaluationError(
                    f"Ticker not found: {ticker}"
                )

            company_record = company_directory[ticker]

            cik = str(
                company_record["cik_str"]
            ).zfill(10)

            company_name = str(
                company_record["title"]
            )

            submissions_url = (
                "https://data.sec.gov/submissions/"
                f"CIK{cik}.json"
            )

            print(
                f"\nFetching external company: "
                f"{ticker} — {company_name}"
            )

            submissions_response = request_with_retry(
                session,
                submissions_url,
            )

            submissions = submissions_response.json()

            filing_records = find_latest_filings(
                submissions
            )

            cik_without_zeros = str(int(cik))

            for record in filing_records:
                form = record["form"]
                accession_number = (
                    record["accession_number"]
                )
                accession_compact = (
                    accession_number.replace("-", "")
                )
                primary_document = (
                    record["primary_document"]
                )
                filing_date = record["filing_date"]

                filing_url = (
                    "https://www.sec.gov/Archives/"
                    "edgar/data/"
                    f"{cik_without_zeros}/"
                    f"{accession_compact}/"
                    f"{primary_document}"
                )

                print(
                    f"Evaluating {ticker} {form} "
                    f"filed {filing_date}"
                )

                response = request_with_retry(
                    session,
                    filing_url,
                )

                raw_document = response.content.decode(
                    "utf-8",
                    errors="ignore",
                )

                cleaned_text = extract_clean_text(
                    raw_document
                )

                prepared_text = prepare_document(
                    cleaned_text
                )

                predicted_label = model.predict(
                    [prepared_text]
                )[0]

                confidence_margin = calculate_margin(
                    model,
                    prepared_text,
                )

                output_path = (
                    OUTPUT_DIR
                    / ticker
                    / form.replace("/", "_")
                    / (
                        f"{filing_date}_"
                        f"{accession_number}.txt"
                    )
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                output_path.write_text(
                    cleaned_text,
                    encoding="utf-8",
                )

                prediction_rows.append(
                    {
                        "ticker": ticker,
                        "company_name": company_name,
                        "actual_label": form,
                        "predicted_label": predicted_label,
                        "correct": (
                            form == predicted_label
                        ),
                        "confidence_margin": round(
                            confidence_margin,
                            4,
                        ),
                        "filing_date": filing_date,
                        "accession_number": accession_number,
                        "word_count": len(
                            cleaned_text.split()
                        ),
                        "source_url": filing_url,
                        "output_path": str(
                            output_path.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                    }
                )

        except Exception as exc:
            failures.append(
                {
                    "ticker": ticker,
                    "error": str(exc),
                }
            )

            print(
                f"Failed to evaluate {ticker}: {exc}"
            )

    if not prediction_rows:
        raise ExternalEvaluationError(
            "No external filings were evaluated."
        )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    accuracy = accuracy_score(
        predictions_df["actual_label"],
        predictions_df["predicted_label"],
    )

    macro_f1 = f1_score(
        predictions_df["actual_label"],
        predictions_df["predicted_label"],
        average="macro",
        zero_division=0,
    )

    summary = {
        "external_companies_requested": len(
            EXTERNAL_TICKERS
        ),
        "external_companies_evaluated": int(
            predictions_df["ticker"].nunique()
        ),
        "documents_evaluated": int(
            len(predictions_df)
        ),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "average_confidence_margin": round(
            float(
                predictions_df[
                    "confidence_margin"
                ].mean()
            ),
            4,
        ),
        "incorrect_predictions": int(
            (~predictions_df["correct"]).sum()
        ),
        "training_company_overlap": False,
        "failed_companies": failures,
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

    print("\nExternal holdout evaluation completed")
    print(
        f"Companies evaluated: "
        f"{summary['external_companies_evaluated']}"
    )
    print(
        f"Documents evaluated: "
        f"{summary['documents_evaluated']}"
    )
    print(
        f"Accuracy: "
        f"{summary['accuracy']:.4f}"
    )
    print(
        f"Macro-F1: "
        f"{summary['macro_f1']:.4f}"
    )
    print(
        f"Incorrect predictions: "
        f"{summary['incorrect_predictions']}"
    )
    print(
        f"Predictions report: {PREDICTIONS_PATH}"
    )
    print(
        f"Summary report: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    evaluate_external_filings()