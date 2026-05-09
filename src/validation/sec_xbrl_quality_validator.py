from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXTRACTION_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_xbrl_extraction_report.csv"
)

DETAIL_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_xbrl_quality_report.csv"
)

SUMMARY_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_xbrl_quality_summary.json"
)

METRICS = [
    "revenue",
    "net_income",
    "operating_income",
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "cash_and_cash_equivalents",
]

CORE_METRICS = [
    "revenue",
    "net_income",
    "total_assets",
]


def metric_coverage(
    dataframe: pd.DataFrame,
    metric: str,
) -> dict[str, Any]:
    """Calculate coverage for one financial metric."""

    column = f"{metric}_value"

    found = int(dataframe[column].notna().sum())
    documents = int(len(dataframe))

    return {
        "found": found,
        "documents": documents,
        "coverage_rate": round(
            found / max(documents, 1),
            4,
        ),
    }


def build_form_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Create extraction statistics for one filing type."""

    metric_results = {
        metric: metric_coverage(dataframe, metric)
        for metric in METRICS
    }

    core_columns = [
        f"{metric}_value"
        for metric in CORE_METRICS
    ]

    documents_with_any_core_metric = int(
        dataframe[core_columns]
        .notna()
        .any(axis=1)
        .sum()
    )

    documents_with_all_core_metrics = int(
        dataframe[core_columns]
        .notna()
        .all(axis=1)
        .sum()
    )

    return {
        "documents": int(len(dataframe)),
        "complete": int(
            (dataframe["status"] == "complete").sum()
        ),
        "partial": int(
            (dataframe["status"] == "partial").sum()
        ),
        "no_facts": int(
            (dataframe["status"] == "no_facts").sum()
        ),
        "failed": int(
            (dataframe["status"] == "failed").sum()
        ),
        "average_completeness": round(
            float(dataframe["completeness_rate"].mean()),
            4,
        ),
        "documents_with_any_core_metric": (
            documents_with_any_core_metric
        ),
        "documents_with_all_core_metrics": (
            documents_with_all_core_metrics
        ),
        "metric_coverage": metric_results,
    }


def run_validation() -> None:
    """Validate XBRL extraction coverage and create reports."""

    if not EXTRACTION_REPORT.exists():
        raise FileNotFoundError(
            f"Extraction report not found: "
            f"{EXTRACTION_REPORT}"
        )

    dataframe = pd.read_csv(EXTRACTION_REPORT)

    required_columns = {
        "ticker",
        "form",
        "status",
        "completeness_rate",
    }

    required_columns.update(
        f"{metric}_value"
        for metric in METRICS
    )

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Extraction report is missing columns: "
            f"{sorted(missing_columns)}"
        )

    detail_rows: list[dict[str, Any]] = []

    for _, row in dataframe.iterrows():
        available_metrics = [
            metric
            for metric in METRICS
            if pd.notna(row[f"{metric}_value"])
        ]

        missing_metrics = [
            metric
            for metric in METRICS
            if pd.isna(row[f"{metric}_value"])
        ]

        detail_rows.append(
            {
                "ticker": row["ticker"],
                "form": row["form"],
                "filing_date": row["filing_date"],
                "accession_number": row["accession_number"],
                "status": row["status"],
                "metrics_found": row["metrics_found"],
                "completeness_rate": row[
                    "completeness_rate"
                ],
                "available_metrics": ", ".join(
                    available_metrics
                ),
                "missing_metrics": ", ".join(
                    missing_metrics
                ),
                "needs_review": bool(
                    row["form"] in {"10-K", "10-Q"}
                    and row["status"] == "no_facts"
                ),
            }
        )

    detail_df = pd.DataFrame(detail_rows)

    DETAIL_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_df.to_csv(
        DETAIL_REPORT,
        index=False,
    )

    form_summary = {
        str(form): build_form_summary(form_df)
        for form, form_df in dataframe.groupby("form")
    }

    annual_quarterly_df = dataframe[
        dataframe["form"].isin(["10-K", "10-Q"])
    ]

    unexpected_no_fact_documents = int(
        (
            (annual_quarterly_df["status"] == "no_facts")
        ).sum()
    )

    failed_documents = int(
        (dataframe["status"] == "failed").sum()
    )

    summary = {
        "documents_checked": int(len(dataframe)),
        "failed_documents": failed_documents,
        "unexpected_10k_10q_no_facts": (
            unexpected_no_fact_documents
        ),
        "form_summary": form_summary,
        "validation_passed": bool(
            failed_documents == 0
            and unexpected_no_fact_documents <= 3
        ),
    }

    with SUMMARY_REPORT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nSEC XBRL quality validation completed")
    print(
        f"Documents checked: "
        f"{summary['documents_checked']}"
    )
    print(
        f"Failed documents: "
        f"{summary['failed_documents']}"
    )
    print(
        "10-K/10-Q documents without facts: "
        f"{summary['unexpected_10k_10q_no_facts']}"
    )

    for form, values in form_summary.items():
        print(f"\n{form}")
        print(f"  Documents: {values['documents']}")
        print(f"  Complete: {values['complete']}")
        print(f"  Partial: {values['partial']}")
        print(f"  No facts: {values['no_facts']}")
        print(
            "  Average completeness: "
            f"{values['average_completeness']:.2%}"
        )

    print(
        f"\nValidation passed: "
        f"{summary['validation_passed']}"
    )
    print(f"Detailed report: {DETAIL_REPORT}")
    print(f"Summary report: {SUMMARY_REPORT}")


if __name__ == "__main__":
    run_validation()