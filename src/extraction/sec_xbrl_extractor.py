from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sec_filings"
    / "dataset_manifest.csv"
)

SEC_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sec_filings"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_xbrl"
)

EXTRACTION_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_xbrl_extraction_report.csv"
)

SUMMARY_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_xbrl_extraction_summary.json"
)


# Candidate concepts are ordered from most preferred to least preferred.
METRIC_CONFIG: dict[str, dict[str, Any]] = {
    "revenue": {
        "period_type": "duration",
        "units": ["USD"],
        "concepts": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
            "SalesRevenueServicesNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ],
    },
    "net_income": {
        "period_type": "duration",
        "units": ["USD"],
        "concepts": [
            "NetIncomeLoss",
            "ProfitLoss",
        ],
    },
    "operating_income": {
        "period_type": "duration",
        "units": ["USD"],
        "concepts": [
            "OperatingIncomeLoss",
        ],
    },
    "total_assets": {
        "period_type": "instant",
        "units": ["USD"],
        "concepts": [
            "Assets",
        ],
    },
    "total_liabilities": {
        "period_type": "instant",
        "units": ["USD"],
        "concepts": [
            "Liabilities",
        ],
    },
    "stockholders_equity": {
        "period_type": "instant",
        "units": ["USD"],
        "concepts": [
            "StockholdersEquity",
            (
                "StockholdersEquityIncludingPortion"
                "AttributableToNoncontrollingInterest"
            ),
        ],
    },
    "cash_and_cash_equivalents": {
        "period_type": "instant",
        "units": ["USD"],
        "concepts": [
            "CashAndCashEquivalentsAtCarryingValue",
            (
                "CashCashEquivalentsRestrictedCash"
                "AndRestrictedCashEquivalents"
            ),
        ],
    },
}


def clean_string(value: Any) -> str:
    """Convert CSV values into safe strings."""

    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def normalize_accession_number(value: str) -> str:
    """Normalize accession numbers for reliable comparison."""

    return value.replace("-", "").strip()


def parse_iso_date(value: str) -> date | None:
    """Parse an ISO date safely."""

    if not value:
        return None

    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def duration_days(fact: dict[str, Any]) -> int | None:
    """Calculate the number of days represented by a duration fact."""

    start_date = parse_iso_date(
        clean_string(fact.get("start"))
    )
    end_date = parse_iso_date(
        clean_string(fact.get("end"))
    )

    if not start_date or not end_date:
        return None

    return (end_date - start_date).days


def target_duration_days(form: str) -> int:
    """Return the preferred reporting duration for each SEC form."""

    if form == "10-K":
        return 365

    if form == "10-Q":
        return 91

    # An 8-K containing earnings data commonly reports a quarter.
    return 91


def calculate_fact_score(
    fact: dict[str, Any],
    *,
    form: str,
    filing_date: str,
    report_date: str,
    period_type: str,
    concept_priority: int,
) -> float:
    """Score a candidate XBRL fact for the current filing."""

    score = 0.0

    fact_form = clean_string(fact.get("form"))
    fact_filed = clean_string(fact.get("filed"))
    fact_end = clean_string(fact.get("end"))
    fact_start = clean_string(fact.get("start"))

    if fact_form == form:
        score += 100.0

    if filing_date and fact_filed == filing_date:
        score += 100.0

    if report_date and fact_end == report_date:
        score += 1000.0

    # Prefer concepts appearing earlier in the configured list.
    score += max(0.0, 50.0 - concept_priority * 5.0)

    if period_type == "instant":
        if not fact_start:
            score += 200.0
        else:
            score -= 100.0

    elif period_type == "duration":
        days = duration_days(fact)

        if days is not None:
            target_days = target_duration_days(form)
            difference = abs(days - target_days)

            score += max(
                0.0,
                300.0 - float(difference),
            )
        else:
            score -= 100.0

    if fact.get("frame"):
        score += 5.0

    return score


def load_company_facts(ticker: str) -> dict[str, Any]:
    """Load one company's SEC Company Facts JSON."""

    path = (
        SEC_DATASET_DIR
        / ticker
        / "companyfacts.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Company Facts file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_metric_candidates(
    company_facts: dict[str, Any],
    *,
    accession_number: str,
    form: str,
    filing_date: str,
    report_date: str,
    metric_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect eligible XBRL candidates for one financial metric."""

    us_gaap_facts = (
        company_facts
        .get("facts", {})
        .get("us-gaap", {})
    )

    normalized_target_accession = (
        normalize_accession_number(accession_number)
    )

    candidates: list[dict[str, Any]] = []

    for concept_priority, concept in enumerate(
        metric_config["concepts"]
    ):
        concept_data = us_gaap_facts.get(concept)

        if not concept_data:
            continue

        units_data = concept_data.get("units", {})

        for unit in metric_config["units"]:
            unit_facts = units_data.get(unit, [])

            for fact in unit_facts:
                fact_accession = normalize_accession_number(
                    clean_string(fact.get("accn"))
                )

                if fact_accession != normalized_target_accession:
                    continue

                value = fact.get("val")

                if not isinstance(value, (int, float)):
                    continue

                score = calculate_fact_score(
                    fact,
                    form=form,
                    filing_date=filing_date,
                    report_date=report_date,
                    period_type=metric_config["period_type"],
                    concept_priority=concept_priority,
                )

                candidates.append(
                    {
                        "value": value,
                        "unit": unit,
                        "concept": concept,
                        "label": clean_string(
                            concept_data.get("label")
                        ),
                        "description": clean_string(
                            concept_data.get("description")
                        ),
                        "start_date": clean_string(
                            fact.get("start")
                        ),
                        "end_date": clean_string(
                            fact.get("end")
                        ),
                        "filed_date": clean_string(
                            fact.get("filed")
                        ),
                        "form": clean_string(
                            fact.get("form")
                        ),
                        "fiscal_year": fact.get("fy"),
                        "fiscal_period": clean_string(
                            fact.get("fp")
                        ),
                        "frame": clean_string(
                            fact.get("frame")
                        ),
                        "accession_number": clean_string(
                            fact.get("accn")
                        ),
                        "duration_days": duration_days(fact),
                        "selection_score": round(score, 2),
                    }
                )

    return candidates


def extract_metric(
    company_facts: dict[str, Any],
    *,
    accession_number: str,
    form: str,
    filing_date: str,
    report_date: str,
    metric_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Select the strongest candidate for one financial metric."""

    candidates = collect_metric_candidates(
        company_facts,
        accession_number=accession_number,
        form=form,
        filing_date=filing_date,
        report_date=report_date,
        metric_config=metric_config,
    )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item["selection_score"],
        reverse=True,
    )

    selected = candidates[0].copy()
    selected["candidate_count"] = len(candidates)

    return selected


def create_output_path(
    ticker: str,
    form: str,
    accession_number: str,
) -> Path:
    """Create the output location for one structured filing."""

    safe_form = form.replace("/", "_")
    safe_accession = accession_number.replace("/", "_")

    return (
        OUTPUT_DIR
        / ticker
        / safe_form
        / f"{safe_accession}.json"
    )


def extract_filing(
    row: pd.Series,
    company_facts: dict[str, Any],
) -> dict[str, Any]:
    """Extract structured Company Facts data for one SEC filing."""

    ticker = clean_string(row["ticker"])
    company_name = clean_string(row["company_name"])
    sector = clean_string(row.get("sector"))
    cik = clean_string(row["cik"])
    form = clean_string(row["form"])
    filing_date = clean_string(row["filing_date"])
    report_date = clean_string(row["report_date"])
    accession_number = clean_string(
        row["accession_number"]
    )

    metrics: dict[str, dict[str, Any] | None] = {}

    for metric_name, metric_config in METRIC_CONFIG.items():
        metrics[metric_name] = extract_metric(
            company_facts,
            accession_number=accession_number,
            form=form,
            filing_date=filing_date,
            report_date=report_date,
            metric_config=metric_config,
        )

    metrics_found = sum(
        value is not None
        for value in metrics.values()
    )

    metrics_requested = len(METRIC_CONFIG)

    if metrics_found == metrics_requested:
        extraction_status = "complete"
    elif metrics_found > 0:
        extraction_status = "partial"
    else:
        extraction_status = "no_facts"

    return {
        "document_id": (
            f"{ticker}_"
            f"{form.replace('-', '')}_"
            f"{accession_number}"
        ),
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "cik": cik,
        "form": form,
        "filing_date": filing_date,
        "report_date": report_date,
        "accession_number": accession_number,
        "source_url": clean_string(row["source_url"]),
        "local_path": clean_string(row["local_path"]),
        "entity_name_from_companyfacts": clean_string(
            company_facts.get("entityName")
        ),
        "extracted_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "metrics": metrics,
        "quality": {
            "metrics_requested": metrics_requested,
            "metrics_found": metrics_found,
            "completeness_rate": round(
                metrics_found / metrics_requested,
                4,
            ),
            "status": extraction_status,
        },
    }


def create_report_row(
    result: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Flatten an extraction result for CSV reporting."""

    report_row: dict[str, Any] = {
        "document_id": result["document_id"],
        "ticker": result["ticker"],
        "company_name": result["company_name"],
        "sector": result["sector"],
        "form": result["form"],
        "filing_date": result["filing_date"],
        "report_date": result["report_date"],
        "accession_number": result["accession_number"],
        "metrics_found": result["quality"]["metrics_found"],
        "metrics_requested": (
            result["quality"]["metrics_requested"]
        ),
        "completeness_rate": (
            result["quality"]["completeness_rate"]
        ),
        "status": result["quality"]["status"],
        "output_path": str(
            output_path.relative_to(PROJECT_ROOT)
        ),
        "error": "",
    }

    for metric_name in METRIC_CONFIG:
        metric = result["metrics"].get(metric_name)

        report_row[f"{metric_name}_value"] = (
            metric.get("value")
            if metric
            else None
        )

        report_row[f"{metric_name}_concept"] = (
            metric.get("concept")
            if metric
            else ""
        )

        report_row[f"{metric_name}_unit"] = (
            metric.get("unit")
            if metric
            else ""
        )

        report_row[f"{metric_name}_start_date"] = (
            metric.get("start_date")
            if metric
            else ""
        )

        report_row[f"{metric_name}_end_date"] = (
            metric.get("end_date")
            if metric
            else ""
        )

    return report_row


def build_summary(
    report_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build aggregate extraction-quality statistics."""

    successful_rows = report_df[
        report_df["status"] != "failed"
    ]

    metric_coverage: dict[str, dict[str, Any]] = {}

    for metric_name in METRIC_CONFIG:
        value_column = f"{metric_name}_value"

        found = int(
            successful_rows[value_column].notna().sum()
        )

        total = int(len(successful_rows))

        metric_coverage[metric_name] = {
            "documents_found": found,
            "documents_checked": total,
            "coverage_rate": round(
                found / max(total, 1),
                4,
            ),
        }

    coverage_by_form: dict[str, dict[str, Any]] = {}

    for form, form_df in successful_rows.groupby("form"):
        form_metrics: dict[str, Any] = {}

        for metric_name in METRIC_CONFIG:
            value_column = f"{metric_name}_value"

            found = int(
                form_df[value_column].notna().sum()
            )

            form_metrics[metric_name] = {
                "found": found,
                "documents": int(len(form_df)),
                "coverage_rate": round(
                    found / max(len(form_df), 1),
                    4,
                ),
            }

        coverage_by_form[str(form)] = form_metrics

    return {
        "documents_processed": int(len(report_df)),
        "complete_extractions": int(
            (report_df["status"] == "complete").sum()
        ),
        "partial_extractions": int(
            (report_df["status"] == "partial").sum()
        ),
        "documents_without_facts": int(
            (report_df["status"] == "no_facts").sum()
        ),
        "failed_extractions": int(
            (report_df["status"] == "failed").sum()
        ),
        "average_completeness_rate": round(
            float(
                successful_rows[
                    "completeness_rate"
                ].mean()
            )
            if not successful_rows.empty
            else 0.0,
            4,
        ),
        "metric_coverage": metric_coverage,
        "coverage_by_form": coverage_by_form,
    }


def run_extraction() -> None:
    """Extract structured XBRL facts for all collected filings."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"SEC manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype={"cik": str},
    )

    required_columns = {
        "ticker",
        "company_name",
        "cik",
        "form",
        "filing_date",
        "report_date",
        "accession_number",
        "source_url",
        "local_path",
    }

    missing_columns = required_columns - set(
        manifest.columns
    )

    if missing_columns:
        raise ValueError(
            "Manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    report_rows: list[dict[str, Any]] = []
    company_facts_cache: dict[str, dict[str, Any]] = {}

    for index, row in manifest.iterrows():
        ticker = clean_string(row["ticker"])
        form = clean_string(row["form"])

        print(
            f"[{index + 1}/{len(manifest)}] "
            f"Extracting {ticker} {form}"
        )

        try:
            if ticker not in company_facts_cache:
                company_facts_cache[ticker] = (
                    load_company_facts(ticker)
                )

            result = extract_filing(
                row,
                company_facts_cache[ticker],
            )

            output_path = create_output_path(
                ticker=ticker,
                form=form,
                accession_number=clean_string(
                    row["accession_number"]
                ),
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
                    result,
                    file,
                    indent=2,
                )

            report_rows.append(
                create_report_row(
                    result,
                    output_path,
                )
            )

        except Exception as exc:
            report_rows.append(
                {
                    "document_id": "",
                    "ticker": ticker,
                    "company_name": clean_string(
                        row["company_name"]
                    ),
                    "sector": clean_string(
                        row.get("sector")
                    ),
                    "form": form,
                    "filing_date": clean_string(
                        row["filing_date"]
                    ),
                    "report_date": clean_string(
                        row["report_date"]
                    ),
                    "accession_number": clean_string(
                        row["accession_number"]
                    ),
                    "metrics_found": 0,
                    "metrics_requested": len(
                        METRIC_CONFIG
                    ),
                    "completeness_rate": 0.0,
                    "status": "failed",
                    "output_path": "",
                    "error": str(exc),
                }
            )

            print(
                f"Failed to extract {ticker} {form}: "
                f"{exc}"
            )

    report_df = pd.DataFrame(report_rows)

    EXTRACTION_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_df.to_csv(
        EXTRACTION_REPORT,
        index=False,
    )

    summary = build_summary(report_df)

    with SUMMARY_REPORT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nSEC XBRL extraction completed")
    print(
        f"Documents processed: "
        f"{summary['documents_processed']}"
    )
    print(
        f"Complete extractions: "
        f"{summary['complete_extractions']}"
    )
    print(
        f"Partial extractions: "
        f"{summary['partial_extractions']}"
    )
    print(
        f"Documents without facts: "
        f"{summary['documents_without_facts']}"
    )
    print(
        f"Failed extractions: "
        f"{summary['failed_extractions']}"
    )
    print(
        f"Average completeness: "
        f"{summary['average_completeness_rate']:.2%}"
    )
    print(f"Structured output: {OUTPUT_DIR}")
    print(f"Extraction report: {EXTRACTION_REPORT}")
    print(f"Summary report: {SUMMARY_REPORT}")


if __name__ == "__main__":
    run_extraction()