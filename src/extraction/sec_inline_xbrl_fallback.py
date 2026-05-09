from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
from lxml import etree

from src.extraction.sec_xbrl_extractor import METRIC_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sec_filings"
    / "dataset_manifest.csv"
)

QUALITY_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_xbrl_quality_report.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_inline_xbrl"
)

FALLBACK_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_inline_xbrl_fallback_report.csv"
)

FALLBACK_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "sec_inline_xbrl_fallback_summary.json"
)


INLINE_METRIC_CONFIG = deepcopy(METRIC_CONFIG)

# Additional standardized concepts useful for financial institutions.
INLINE_METRIC_CONFIG["revenue"]["concepts"].extend(
    [
        "RevenuesNetOfInterestExpense",
        "InterestAndDividendIncomeOperating",
        "NoninterestIncome",
    ]
)


def local_name(value: Any) -> str:
    """Return an XML/HTML tag or attribute name without its namespace."""

    text = str(value)

    if "}" in text:
        text = text.rsplit("}", 1)[-1]

    if ":" in text:
        text = text.rsplit(":", 1)[-1]

    return text.lower()


def get_attribute(
    element: etree._Element,
    attribute_name: str,
) -> str:
    """Read an attribute regardless of namespace or letter casing."""

    target = attribute_name.lower()

    for key, value in element.attrib.items():
        if local_name(key) == target:
            return str(value).strip()

    return ""


def parse_iso_date(value: str) -> date | None:
    """Parse an ISO-formatted date safely."""

    if not value:
        return None

    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def calculate_duration_days(
    start_date: str,
    end_date: str,
) -> int | None:
    """Calculate the number of days represented by a context."""

    start = parse_iso_date(start_date)
    end = parse_iso_date(end_date)

    if not start or not end:
        return None

    return (end - start).days


def target_duration_days(form: str) -> int:
    """Return the preferred duration for annual and quarterly facts."""

    if form == "10-K":
        return 365

    return 91


def parse_numeric_value(
    element: etree._Element,
) -> int | float | None:
    """Convert an Inline XBRL numeric fact into a Python number."""

    nil_value = get_attribute(element, "nil").lower()

    if nil_value in {"true", "1"}:
        return None

    text = " ".join(element.itertext())
    text = text.replace("\xa0", " ").strip()

    if not text or text in {"-", "—", "–"}:
        return None

    parentheses_negative = (
        text.startswith("(")
        and text.endswith(")")
    )

    normalized = re.sub(
        r"[^0-9eE+\-.]",
        "",
        text,
    )

    if normalized in {"", "-", ".", "-."}:
        return None

    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return None

    scale_text = get_attribute(element, "scale")

    try:
        scale = int(scale_text) if scale_text else 0
    except ValueError:
        scale = 0

    value *= Decimal(10) ** scale

    sign = get_attribute(element, "sign")

    if sign == "-":
        value = -abs(value)
    elif parentheses_negative:
        value = -abs(value)

    if value == value.to_integral_value():
        return int(value)

    return float(value)


def parse_contexts(
    root: etree._Element,
) -> dict[str, dict[str, Any]]:
    """Extract reporting-period information from XBRL contexts."""

    contexts: dict[str, dict[str, Any]] = {}

    for element in root.iter():
        if local_name(element.tag) != "context":
            continue

        context_id = get_attribute(element, "id")

        if not context_id:
            continue

        instant = ""
        start_date = ""
        end_date = ""
        dimension_count = 0

        for descendant in element.iter():
            name = local_name(descendant.tag)
            value = (descendant.text or "").strip()

            if name == "instant":
                instant = value
            elif name == "startdate":
                start_date = value
            elif name == "enddate":
                end_date = value
            elif name in {"explicitmember", "typedmember"}:
                dimension_count += 1

        contexts[context_id] = {
            "instant": instant,
            "start_date": start_date,
            "end_date": end_date,
            "dimension_count": dimension_count,
        }

    return contexts


def parse_units(
    root: etree._Element,
) -> dict[str, str]:
    """Map XBRL unit identifiers to readable unit names."""

    units: dict[str, str] = {}

    for element in root.iter():
        if local_name(element.tag) != "unit":
            continue

        unit_id = get_attribute(element, "id")

        if not unit_id:
            continue

        measures: list[str] = []

        for descendant in element.iter():
            if local_name(descendant.tag) == "measure":
                value = (descendant.text or "").strip()

                if value:
                    measures.append(value)

        combined = " / ".join(measures)

        if any(
            measure.upper().endswith(":USD")
            or measure.upper() == "USD"
            for measure in measures
        ):
            units[unit_id] = "USD"
        else:
            units[unit_id] = combined or unit_id

    return units


def parse_inline_facts(
    source_path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Read numeric facts and contexts from one Inline XBRL filing."""

    parser = etree.HTMLParser(
        recover=True,
        huge_tree=True,
    )

    root = etree.parse(
        str(source_path),
        parser,
    ).getroot()

    contexts = parse_contexts(root)
    units = parse_units(root)

    facts: list[dict[str, Any]] = []

    for element in root.iter():
        if local_name(element.tag) != "nonfraction":
            continue

        concept_name = get_attribute(element, "name")
        context_ref = get_attribute(element, "contextref")
        unit_ref = get_attribute(element, "unitref")

        if not concept_name or not context_ref:
            continue

        value = parse_numeric_value(element)

        if value is None:
            continue

        concept = concept_name.rsplit(":", 1)[-1]
        context = contexts.get(context_ref, {})

        facts.append(
            {
                "concept": concept,
                "full_concept": concept_name,
                "value": value,
                "unit": units.get(unit_ref, unit_ref),
                "context_ref": context_ref,
                "instant": context.get("instant", ""),
                "start_date": context.get("start_date", ""),
                "end_date": context.get("end_date", ""),
                "dimension_count": context.get(
                    "dimension_count",
                    0,
                ),
            }
        )

    return facts, contexts


def score_candidate(
    fact: dict[str, Any],
    *,
    form: str,
    report_date: str,
    period_type: str,
    concept_priority: int,
) -> float:
    """Score an Inline XBRL fact for a requested financial metric."""

    score = max(
        0.0,
        100.0 - concept_priority * 5.0,
    )

    if fact["unit"] == "USD":
        score += 100.0

    dimension_count = int(
        fact.get("dimension_count", 0)
    )

    if dimension_count == 0:
        score += 200.0
    else:
        score -= dimension_count * 25.0

    if period_type == "instant":
        instant = fact.get("instant", "")

        if report_date and instant == report_date:
            score += 1000.0
        elif instant:
            score += 100.0
        else:
            score -= 200.0

    else:
        end_date = fact.get("end_date", "")
        start_date = fact.get("start_date", "")

        if report_date and end_date == report_date:
            score += 1000.0

        duration = calculate_duration_days(
            start_date,
            end_date,
        )

        if duration is not None:
            difference = abs(
                duration - target_duration_days(form)
            )

            score += max(
                0.0,
                400.0 - difference * 3.0,
            )
        else:
            score -= 200.0

    return score


def extract_metric(
    facts: list[dict[str, Any]],
    *,
    form: str,
    report_date: str,
    metric_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Select the strongest Inline XBRL candidate for one metric."""

    candidates: list[dict[str, Any]] = []

    for concept_priority, concept in enumerate(
        metric_config["concepts"]
    ):
        for fact in facts:
            if fact["concept"] != concept:
                continue

            candidate = fact.copy()

            candidate["selection_score"] = round(
                score_candidate(
                    fact,
                    form=form,
                    report_date=report_date,
                    period_type=metric_config["period_type"],
                    concept_priority=concept_priority,
                ),
                2,
            )

            candidates.append(candidate)

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item["selection_score"],
        reverse=True,
    )

    selected = candidates[0]
    selected["candidate_count"] = len(candidates)

    return selected


def run_fallback_extraction() -> None:
    """Extract Inline XBRL facts for unmatched 10-K and 10-Q filings."""

    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype={"cik": str},
    )

    quality = pd.read_csv(QUALITY_REPORT)

    targets = quality[
        quality["form"].isin(["10-K", "10-Q"])
        & (quality["status"] == "no_facts")
    ][
        [
            "ticker",
            "form",
            "filing_date",
            "accession_number",
        ]
    ]

    target_manifest = targets.merge(
        manifest,
        on=[
            "ticker",
            "form",
            "filing_date",
            "accession_number",
        ],
        how="left",
        validate="one_to_one",
    )

    report_rows: list[dict[str, Any]] = []

    for index, row in target_manifest.iterrows():
        ticker = str(row["ticker"])
        form = str(row["form"])
        report_date = (
            ""
            if pd.isna(row["report_date"])
            else str(row["report_date"])
        )

        source_path = (
            PROJECT_ROOT
            / str(row["local_path"])
        )

        print(
            f"[{index + 1}/{len(target_manifest)}] "
            f"Parsing Inline XBRL: {ticker} {form}"
        )

        try:
            facts, contexts = parse_inline_facts(
                source_path
            )

            metrics: dict[
                str,
                dict[str, Any] | None,
            ] = {}

            for metric_name, config in (
                INLINE_METRIC_CONFIG.items()
            ):
                metrics[metric_name] = extract_metric(
                    facts,
                    form=form,
                    report_date=report_date,
                    metric_config=config,
                )

            metrics_found = sum(
                metric is not None
                for metric in metrics.values()
            )

            if metrics_found == len(
                INLINE_METRIC_CONFIG
            ):
                status = "complete"
            elif metrics_found > 0:
                status = "partial"
            else:
                status = "no_facts"

            result = {
                "ticker": ticker,
                "company_name": str(
                    row["company_name"]
                ),
                "form": form,
                "filing_date": str(
                    row["filing_date"]
                ),
                "report_date": report_date,
                "accession_number": str(
                    row["accession_number"]
                ),
                "source": "inline_xbrl_fallback",
                "facts_detected": len(facts),
                "contexts_detected": len(contexts),
                "metrics": metrics,
                "quality": {
                    "metrics_requested": len(
                        INLINE_METRIC_CONFIG
                    ),
                    "metrics_found": metrics_found,
                    "completeness_rate": round(
                        metrics_found
                        / len(INLINE_METRIC_CONFIG),
                        4,
                    ),
                    "status": status,
                },
            }

            output_path = (
                OUTPUT_DIR
                / ticker
                / form.replace("/", "_")
                / (
                    str(row["accession_number"])
                    + ".json"
                )
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
                {
                    "ticker": ticker,
                    "form": form,
                    "filing_date": row["filing_date"],
                    "accession_number": (
                        row["accession_number"]
                    ),
                    "facts_detected": len(facts),
                    "contexts_detected": len(contexts),
                    "metrics_found": metrics_found,
                    "metrics_requested": len(
                        INLINE_METRIC_CONFIG
                    ),
                    "completeness_rate": (
                        result["quality"][
                            "completeness_rate"
                        ]
                    ),
                    "status": status,
                    "output_path": str(
                        output_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "error": "",
                }
            )

        except Exception as exc:
            report_rows.append(
                {
                    "ticker": ticker,
                    "form": form,
                    "filing_date": row["filing_date"],
                    "accession_number": (
                        row["accession_number"]
                    ),
                    "facts_detected": 0,
                    "contexts_detected": 0,
                    "metrics_found": 0,
                    "metrics_requested": len(
                        INLINE_METRIC_CONFIG
                    ),
                    "completeness_rate": 0.0,
                    "status": "failed",
                    "output_path": "",
                    "error": str(exc),
                }
            )

            print(
                f"Failed to process {ticker}: {exc}"
            )

    report_df = pd.DataFrame(report_rows)

    FALLBACK_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_df.to_csv(
        FALLBACK_REPORT,
        index=False,
    )

    summary = {
        "documents_attempted": int(
            len(report_df)
        ),
        "complete": int(
            (report_df["status"] == "complete").sum()
        ),
        "partial": int(
            (report_df["status"] == "partial").sum()
        ),
        "no_facts": int(
            (report_df["status"] == "no_facts").sum()
        ),
        "failed": int(
            (report_df["status"] == "failed").sum()
        ),
        "average_completeness": round(
            float(
                report_df["completeness_rate"].mean()
            ),
            4,
        ),
    }

    with FALLBACK_SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nInline XBRL fallback completed")
    print(
        f"Documents attempted: "
        f"{summary['documents_attempted']}"
    )
    print(f"Complete: {summary['complete']}")
    print(f"Partial: {summary['partial']}")
    print(f"No facts: {summary['no_facts']}")
    print(f"Failed: {summary['failed']}")
    print(
        f"Average completeness: "
        f"{summary['average_completeness']:.2%}"
    )
    print(f"Report: {FALLBACK_REPORT}")
    print(f"Summary: {FALLBACK_SUMMARY}")


if __name__ == "__main__":
    run_fallback_extraction()