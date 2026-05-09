from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.extraction.sec_xbrl_extractor import METRIC_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRIMARY_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_xbrl_extraction_report.csv"
)

PRIMARY_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_xbrl"
)

FALLBACK_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_inline_xbrl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_financials_final"
)

MERGE_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_financial_merge_report.csv"
)

MERGE_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "sec_financial_merge_summary.json"
)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_result_path(
    ticker: str,
    form: str,
    accession_number: str,
) -> Path:
    """Create a consistent output path."""

    return (
        OUTPUT_DIR
        / ticker
        / form.replace("/", "_")
        / f"{accession_number}.json"
    )


def get_primary_path(
    ticker: str,
    form: str,
    accession_number: str,
) -> Path:
    """Return the primary Company Facts result path."""

    return (
        PRIMARY_DIR
        / ticker
        / form.replace("/", "_")
        / f"{accession_number}.json"
    )


def get_fallback_path(
    ticker: str,
    form: str,
    accession_number: str,
) -> Path:
    """Return the Inline XBRL fallback result path."""

    return (
        FALLBACK_DIR
        / ticker
        / form.replace("/", "_")
        / f"{accession_number}.json"
    )


def merge_metrics(
    primary_metrics: dict[str, Any],
    fallback_metrics: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """
    Prefer Company Facts values and fill missing metrics using
    Inline XBRL.
    """

    merged_metrics: dict[str, Any] = {}
    fallback_metrics_used = 0

    for metric_name in METRIC_CONFIG:
        primary_metric = primary_metrics.get(metric_name)
        fallback_metric = fallback_metrics.get(metric_name)

        if primary_metric is not None:
            selected = dict(primary_metric)
            selected["selected_source"] = "companyfacts_api"

        elif fallback_metric is not None:
            selected = dict(fallback_metric)
            selected["selected_source"] = "inline_xbrl"
            fallback_metrics_used += 1

        else:
            selected = None

        merged_metrics[metric_name] = selected

    return merged_metrics, fallback_metrics_used


def merge_results() -> None:
    """Merge primary and fallback financial extraction results."""

    if not PRIMARY_REPORT.exists():
        raise FileNotFoundError(
            f"Primary extraction report not found: {PRIMARY_REPORT}"
        )

    report_df = pd.read_csv(PRIMARY_REPORT)

    merge_rows: list[dict[str, Any]] = []

    for index, row in report_df.iterrows():
        ticker = str(row["ticker"])
        form = str(row["form"])
        accession_number = str(row["accession_number"])

        print(
            f"[{index + 1}/{len(report_df)}] "
            f"Merging {ticker} {form}"
        )

        primary_path = get_primary_path(
            ticker,
            form,
            accession_number,
        )

        fallback_path = get_fallback_path(
            ticker,
            form,
            accession_number,
        )

        try:
            primary_result = load_json(primary_path)

            fallback_result: dict[str, Any] = {}

            if fallback_path.exists():
                fallback_result = load_json(fallback_path)

            primary_metrics = primary_result.get(
                "metrics",
                {},
            )

            fallback_metrics = fallback_result.get(
                "metrics",
                {},
            )

            merged_metrics, fallback_used = merge_metrics(
                primary_metrics,
                fallback_metrics,
            )

            metrics_found = sum(
                metric is not None
                for metric in merged_metrics.values()
            )

            metrics_requested = len(METRIC_CONFIG)

            if metrics_found == metrics_requested:
                status = "complete"
            elif metrics_found > 0:
                status = "partial"
            else:
                status = "no_facts"

            if fallback_used > 0:
                extraction_strategy = (
                    "companyfacts_plus_inline_xbrl"
                )
            else:
                extraction_strategy = "companyfacts_api"

            final_result = {
                "document_id": primary_result.get(
                    "document_id"
                ),
                "ticker": ticker,
                "company_name": primary_result.get(
                    "company_name"
                ),
                "sector": primary_result.get("sector"),
                "cik": primary_result.get("cik"),
                "form": form,
                "filing_date": primary_result.get(
                    "filing_date"
                ),
                "report_date": primary_result.get(
                    "report_date"
                ),
                "accession_number": accession_number,
                "source_url": primary_result.get(
                    "source_url"
                ),
                "extraction_strategy": extraction_strategy,
                "metrics": merged_metrics,
                "quality": {
                    "metrics_requested": metrics_requested,
                    "metrics_found": metrics_found,
                    "completeness_rate": round(
                        metrics_found / metrics_requested,
                        4,
                    ),
                    "status": status,
                    "fallback_metrics_used": fallback_used,
                    "requires_human_review": bool(
                        form in {"10-K", "10-Q"}
                        and status == "no_facts"
                    ),
                },
            }

            output_path = create_result_path(
                ticker,
                form,
                accession_number,
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with output_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    final_result,
                    file,
                    indent=2,
                )

            merge_rows.append(
                {
                    "ticker": ticker,
                    "form": form,
                    "filing_date": row["filing_date"],
                    "accession_number": accession_number,
                    "primary_status": row["status"],
                    "fallback_available": fallback_path.exists(),
                    "fallback_metrics_used": fallback_used,
                    "final_metrics_found": metrics_found,
                    "final_completeness_rate": (
                        final_result["quality"][
                            "completeness_rate"
                        ]
                    ),
                    "final_status": status,
                    "extraction_strategy": extraction_strategy,
                    "requires_human_review": (
                        final_result["quality"][
                            "requires_human_review"
                        ]
                    ),
                    "output_path": str(
                        output_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "error": "",
                }
            )

        except Exception as exc:
            merge_rows.append(
                {
                    "ticker": ticker,
                    "form": form,
                    "filing_date": row["filing_date"],
                    "accession_number": accession_number,
                    "primary_status": row["status"],
                    "fallback_available": fallback_path.exists(),
                    "fallback_metrics_used": 0,
                    "final_metrics_found": 0,
                    "final_completeness_rate": 0.0,
                    "final_status": "failed",
                    "extraction_strategy": "",
                    "requires_human_review": True,
                    "output_path": "",
                    "error": str(exc),
                }
            )

            print(
                f"Failed to merge {ticker} {form}: {exc}"
            )

    merge_df = pd.DataFrame(merge_rows)

    MERGE_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merge_df.to_csv(
        MERGE_REPORT,
        index=False,
    )

    annual_quarterly = merge_df[
        merge_df["form"].isin(["10-K", "10-Q"])
    ]

    summary = {
        "documents_processed": int(len(merge_df)),
        "complete": int(
            (merge_df["final_status"] == "complete").sum()
        ),
        "partial": int(
            (merge_df["final_status"] == "partial").sum()
        ),
        "no_facts": int(
            (merge_df["final_status"] == "no_facts").sum()
        ),
        "failed": int(
            (merge_df["final_status"] == "failed").sum()
        ),
        "documents_using_fallback": int(
            (
                merge_df["fallback_metrics_used"] > 0
            ).sum()
        ),
        "fallback_metrics_recovered": int(
            merge_df["fallback_metrics_used"].sum()
        ),
        "average_completeness": round(
            float(
                merge_df[
                    "final_completeness_rate"
                ].mean()
            ),
            4,
        ),
        "annual_quarterly_average_completeness": round(
            float(
                annual_quarterly[
                    "final_completeness_rate"
                ].mean()
            ),
            4,
        ),
        "annual_quarterly_without_facts": int(
            (
                annual_quarterly["final_status"]
                == "no_facts"
            ).sum()
        ),
        "documents_requiring_review": int(
            merge_df["requires_human_review"].sum()
        ),
    }

    with MERGE_SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nSEC financial results merged")
    print(
        f"Documents processed: "
        f"{summary['documents_processed']}"
    )
    print(f"Complete: {summary['complete']}")
    print(f"Partial: {summary['partial']}")
    print(f"No facts: {summary['no_facts']}")
    print(f"Failed: {summary['failed']}")
    print(
        f"Documents using fallback: "
        f"{summary['documents_using_fallback']}"
    )
    print(
        f"Fallback metrics recovered: "
        f"{summary['fallback_metrics_recovered']}"
    )
    print(
        f"Average completeness: "
        f"{summary['average_completeness']:.2%}"
    )
    print(
        "10-K/10-Q average completeness: "
        f"{summary['annual_quarterly_average_completeness']:.2%}"
    )
    print(
        "10-K/10-Q without facts: "
        f"{summary['annual_quarterly_without_facts']}"
    )
    print(f"Report: {MERGE_REPORT}")
    print(f"Summary: {MERGE_SUMMARY}")


if __name__ == "__main__":
    merge_results()