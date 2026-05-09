from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "sec_filings"
MANIFEST_PATH = DATASET_DIR / "dataset_manifest.csv"

REPORT_DIR = PROJECT_ROOT / "reports"
DETAIL_REPORT = REPORT_DIR / "sec_dataset_validation.csv"
SUMMARY_REPORT = REPORT_DIR / "sec_dataset_summary.json"

REQUIRED_COLUMNS = {
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

EXPECTED_FORMS = {"10-K", "10-Q", "8-K"}


def validate_json_file(path: Path) -> bool:
    """Return True when a file contains valid JSON."""

    try:
        with path.open("r", encoding="utf-8") as file:
            json.load(file)
        return True
    except (OSError, json.JSONDecodeError):
        return False


def validate_filing_file(path: Path) -> dict[str, object]:
    """Validate one downloaded SEC filing."""

    exists = path.exists()
    size_bytes = path.stat().st_size if exists else 0

    readable = False
    looks_like_html = False

    if exists and size_bytes > 0:
        try:
            text_sample = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )[:5000]

            readable = bool(text_sample.strip())

            normalized = text_sample.lower()
            looks_like_html = any(
                marker in normalized
                for marker in (
                    "<html",
                    "<body",
                    "<document",
                    "<xbrl",
                    "<!doctype",
                )
            )
        except OSError:
            readable = False

    return {
        "file_exists": exists,
        "file_size_bytes": size_bytes,
        "file_readable": readable,
        "looks_like_sec_document": looks_like_html,
    }


def run_validation() -> None:
    """Validate the SEC dataset and generate reports."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    missing_columns = REQUIRED_COLUMNS - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_columns)}"
        )

    validation_rows: list[dict[str, object]] = []

    for _, row in manifest.iterrows():
        local_path = PROJECT_ROOT / str(row["local_path"])

        file_result = validate_filing_file(local_path)

        validation_rows.append(
            {
                "ticker": row["ticker"],
                "company_name": row["company_name"],
                "form": row["form"],
                "filing_date": row["filing_date"],
                "accession_number": row["accession_number"],
                "local_path": row["local_path"],
                **file_result,
            }
        )

    validation_df = pd.DataFrame(validation_rows)

    company_results: list[dict[str, object]] = []

    for ticker in sorted(manifest["ticker"].unique()):
        company_dir = DATASET_DIR / ticker

        submissions_path = company_dir / "submissions.json"
        companyfacts_path = company_dir / "companyfacts.json"

        company_results.append(
            {
                "ticker": ticker,
                "submissions_exists": submissions_path.exists(),
                "submissions_valid_json": validate_json_file(
                    submissions_path
                ),
                "companyfacts_exists": companyfacts_path.exists(),
                "companyfacts_valid_json": validate_json_file(
                    companyfacts_path
                ),
            }
        )

    company_df = pd.DataFrame(company_results)

    duplicate_accessions = int(
        manifest["accession_number"].duplicated().sum()
    )

    invalid_forms = sorted(
        set(manifest["form"]) - EXPECTED_FORMS
    )

    forms_by_type = {
        str(key): int(value)
        for key, value in manifest["form"].value_counts().items()
    }

    summary = {
        "companies": int(manifest["ticker"].nunique()),
        "manifest_rows": int(len(manifest)),
        "forms_by_type": forms_by_type,
        "duplicate_accession_numbers": duplicate_accessions,
        "invalid_forms": invalid_forms,
        "files_found": int(validation_df["file_exists"].sum()),
        "readable_files": int(
            validation_df["file_readable"].sum()
        ),
        "valid_sec_documents": int(
            validation_df["looks_like_sec_document"].sum()
        ),
        "valid_submission_json_files": int(
            company_df["submissions_valid_json"].sum()
        ),
        "valid_companyfacts_json_files": int(
            company_df["companyfacts_valid_json"].sum()
        ),
        "validation_passed": bool(
            len(manifest) > 0
            and validation_df["file_exists"].all()
            and validation_df["file_readable"].all()
            and company_df["submissions_valid_json"].all()
            and company_df["companyfacts_valid_json"].all()
            and duplicate_accessions == 0
            and not invalid_forms
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    validation_df.to_csv(DETAIL_REPORT, index=False)

    with SUMMARY_REPORT.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nSEC dataset validation completed")
    print(f"Companies: {summary['companies']}")
    print(f"Manifest rows: {summary['manifest_rows']}")
    print(f"Forms: {summary['forms_by_type']}")
    print(f"Files found: {summary['files_found']}")
    print(f"Readable files: {summary['readable_files']}")
    print(
        "Valid submissions JSON files: "
        f"{summary['valid_submission_json_files']}"
    )
    print(
        "Valid company-facts JSON files: "
        f"{summary['valid_companyfacts_json_files']}"
    )
    print(f"Validation passed: {summary['validation_passed']}")
    print(f"Detailed report: {DETAIL_REPORT}")
    print(f"Summary report: {SUMMARY_REPORT}")


if __name__ == "__main__":
    run_validation()
