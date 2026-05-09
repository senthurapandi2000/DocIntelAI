from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MERGE_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_financial_merge_report.csv"
)

SUMMARY_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_final_financial_validation.json"
)


def run_validation() -> None:
    """Validate the final merged SEC financial dataset."""

    if not MERGE_REPORT.exists():
        raise FileNotFoundError(
            f"Merge report not found: {MERGE_REPORT}"
        )

    dataframe = pd.read_csv(MERGE_REPORT)

    annual_quarterly = dataframe[
        dataframe["form"].isin(["10-K", "10-Q"])
    ]

    event_filings = dataframe[
        dataframe["form"] == "8-K"
    ]

    failed_documents = int(
        (dataframe["final_status"] == "failed").sum()
    )

    annual_quarterly_without_facts = int(
        (
            annual_quarterly["final_status"]
            == "no_facts"
        ).sum()
    )

    missing_output_files = 0

    for output_path in dataframe["output_path"].dropna():
        path = PROJECT_ROOT / str(output_path)

        if not path.exists():
            missing_output_files += 1

    summary = {
        "documents_checked": int(len(dataframe)),
        "annual_quarterly_documents": int(
            len(annual_quarterly)
        ),
        "event_documents": int(len(event_filings)),
        "failed_documents": failed_documents,
        "missing_output_files": missing_output_files,
        "annual_quarterly_without_facts": (
            annual_quarterly_without_facts
        ),
        "documents_using_fallback": int(
            (
                dataframe["fallback_metrics_used"] > 0
            ).sum()
        ),
        "fallback_metrics_recovered": int(
            dataframe["fallback_metrics_used"].sum()
        ),
        "annual_quarterly_average_completeness": round(
            float(
                annual_quarterly[
                    "final_completeness_rate"
                ].mean()
            ),
            4,
        ),
        "validation_passed": bool(
            failed_documents == 0
            and missing_output_files == 0
            and annual_quarterly_without_facts == 0
        ),
    }

    SUMMARY_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_REPORT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nFinal SEC financial validation completed")
    print(
        f"Documents checked: "
        f"{summary['documents_checked']}"
    )
    print(
        f"10-K/10-Q documents: "
        f"{summary['annual_quarterly_documents']}"
    )
    print(
        f"8-K documents: "
        f"{summary['event_documents']}"
    )
    print(
        f"Failed documents: "
        f"{summary['failed_documents']}"
    )
    print(
        f"Missing output files: "
        f"{summary['missing_output_files']}"
    )
    print(
        f"10-K/10-Q without facts: "
        f"{summary['annual_quarterly_without_facts']}"
    )
    print(
        f"Fallback metrics recovered: "
        f"{summary['fallback_metrics_recovered']}"
    )
    print(
        "10-K/10-Q average completeness: "
        f"{summary['annual_quarterly_average_completeness']:.2%}"
    )
    print(
        f"Validation passed: "
        f"{summary['validation_passed']}"
    )
    print(f"Summary: {SUMMARY_REPORT}")


if __name__ == "__main__":
    run_validation()