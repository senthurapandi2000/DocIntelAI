from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TICKER_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "sec"
    / "company_tickers.json"
)

COMPANY_UNIVERSE_FILE = (
    PROJECT_ROOT
    / "config"
    / "company_universe.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sec_filings"
)

MANIFEST_FILE = OUTPUT_DIR / "dataset_manifest.csv"
FAILED_COMPANIES_FILE = OUTPUT_DIR / "failed_companies.csv"


# ============================================================
# Collection settings
# ============================================================

TARGET_FORMS = ("10-K", "10-Q", "8-K")

# Download one recent filing for each form type.
MAX_FILINGS_PER_FORM = 1

# Approximately four requests per second.
REQUEST_DELAY_SECONDS = 0.25

MAX_RETRIES = 4


class SECIngestionError(RuntimeError):
    """Raised when SEC data cannot be downloaded or processed."""


# ============================================================
# HTTP session
# ============================================================

def create_session() -> requests.Session:
    """Create an SEC-compliant HTTP session."""

    load_dotenv(PROJECT_ROOT / ".env")

    user_agent = os.getenv("SEC_USER_AGENT", "").strip()

    if not user_agent or "@" not in user_agent:
        raise SECIngestionError(
            "SEC_USER_AGENT is missing or invalid.\n"
            "Add a valid value to the root-level .env file.\n"
            "Example: SEC_USER_AGENT=\"DocIntelAI Your Name email@example.com\""
        )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json,text/html,*/*",
        }
    )

    return session


def request_with_retry(
    session: requests.Session,
    url: str,
) -> requests.Response:
    """Download a URL with retries and exponential backoff."""

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                url,
                timeout=60,
            )

            response.raise_for_status()

            time.sleep(REQUEST_DELAY_SECONDS)

            return response

        except requests.RequestException as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                wait_seconds = 2 ** attempt

                print(
                    f"Request failed on attempt "
                    f"{attempt}/{MAX_RETRIES}: {url}"
                )
                print(
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

    raise SECIngestionError(
        f"Failed to download after "
        f"{MAX_RETRIES} attempts: {url}"
    ) from last_error


# ============================================================
# File helpers
# ============================================================

def save_json(
    data: Any,
    output_path: Path,
) -> None:
    """Save JSON data using readable formatting."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def save_binary_file(
    content: bytes,
    output_path: Path,
) -> None:
    """Save downloaded document content."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(content)


# ============================================================
# Configuration loaders
# ============================================================

def load_company_directory() -> dict[str, dict[str, Any]]:
    """Load the official SEC ticker-to-CIK mapping."""

    if not TICKER_FILE.exists():
        raise FileNotFoundError(
            f"SEC company ticker file was not found:\n"
            f"{TICKER_FILE}"
        )

    with TICKER_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_data = json.load(file)

    company_directory: dict[str, dict[str, Any]] = {}

    for record in raw_data.values():
        ticker = str(record["ticker"]).strip().upper()

        company_directory[ticker] = record

    return company_directory


def load_target_companies() -> list[dict[str, str]]:
    """Load company tickers and sectors from the CSV configuration."""

    if not COMPANY_UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            f"Company universe file was not found:\n"
            f"{COMPANY_UNIVERSE_FILE}"
        )

    companies: list[dict[str, str]] = []
    seen_tickers: set[str] = set()

    with COMPANY_UNIVERSE_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {"ticker", "sector"}

        if not reader.fieldnames:
            raise SECIngestionError(
                "company_universe.csv has no column headers."
            )

        missing_columns = (
            required_columns - set(reader.fieldnames)
        )

        if missing_columns:
            raise SECIngestionError(
                "company_universe.csv is missing columns: "
                f"{sorted(missing_columns)}"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            ticker = str(
                row.get("ticker", "")
            ).strip().upper()

            sector = str(
                row.get("sector", "")
            ).strip()

            if not ticker:
                print(
                    f"Skipping row {row_number}: "
                    f"ticker is missing."
                )
                continue

            if ticker in seen_tickers:
                print(
                    f"Skipping duplicate ticker: {ticker}"
                )
                continue

            seen_tickers.add(ticker)

            companies.append(
                {
                    "ticker": ticker,
                    "sector": sector or "Unknown",
                }
            )

    if not companies:
        raise SECIngestionError(
            "No valid companies were found in "
            "config/company_universe.csv."
        )

    return companies


# ============================================================
# Filing collection
# ============================================================

def download_recent_filings(
    session: requests.Session,
    submissions: dict[str, Any],
    ticker: str,
    company_name: str,
    sector: str,
    cik: str,
    *,
    max_per_form: int = MAX_FILINGS_PER_FORM,
) -> list[dict[str, str]]:
    """
    Download recent 10-K, 10-Q, and 8-K filing documents.

    Existing files are reused so interrupted ingestion runs can resume.
    """

    recent = (
        submissions
        .get("filings", {})
        .get("recent", {})
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
    report_dates = recent.get(
        "reportDate",
        [],
    )
    primary_documents = recent.get(
        "primaryDocument",
        [],
    )

    record_count = min(
        len(forms),
        len(accession_numbers),
        len(filing_dates),
        len(report_dates),
        len(primary_documents),
    )

    form_counts = {
        form: 0
        for form in TARGET_FORMS
    }

    manifest_rows: list[dict[str, str]] = []

    cik_without_leading_zeros = str(int(cik))

    for index in range(record_count):
        form = forms[index]

        if form not in TARGET_FORMS:
            continue

        if form_counts[form] >= max_per_form:
            continue

        accession_number = accession_numbers[index]
        filing_date = filing_dates[index]
        report_date = report_dates[index]
        primary_document = primary_documents[index]

        if not primary_document:
            continue

        accession_without_dashes = (
            accession_number.replace("-", "")
        )

        filing_url = (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{cik_without_leading_zeros}/"
            f"{accession_without_dashes}/"
            f"{primary_document}"
        )

        form_folder = form.replace("/", "_")
        document_name = Path(primary_document).name

        local_path = (
            OUTPUT_DIR
            / ticker
            / "filings"
            / form_folder
            / (
                f"{filing_date}_"
                f"{accession_number}_"
                f"{document_name}"
            )
        )

        local_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            local_path.exists()
            and local_path.stat().st_size > 0
        ):
            print(
                f"Using existing {ticker} {form} "
                f"filed {filing_date}"
            )

        else:
            print(
                f"Downloading {ticker} {form} "
                f"filed {filing_date}"
            )

            response = request_with_retry(
                session,
                filing_url,
            )

            save_binary_file(
                response.content,
                local_path,
            )

        manifest_rows.append(
            {
                "ticker": ticker,
                "company_name": company_name,
                "sector": sector,
                "cik": cik,
                "form": form,
                "filing_date": filing_date,
                "report_date": report_date,
                "accession_number": accession_number,
                "source_url": filing_url,
                "local_path": str(
                    local_path.relative_to(PROJECT_ROOT)
                ),
            }
        )

        form_counts[form] += 1

        if all(
            count >= max_per_form
            for count in form_counts.values()
        ):
            break

    missing_forms = [
        form
        for form, count in form_counts.items()
        if count < max_per_form
    ]

    if missing_forms:
        print(
            f"Warning: {ticker} did not provide "
            f"all requested forms: {missing_forms}"
        )

    return manifest_rows


def download_company(
    session: requests.Session,
    ticker: str,
    sector: str,
    company_directory: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """Download metadata, XBRL facts, and filing documents."""

    normalized_ticker = ticker.upper()

    if normalized_ticker not in company_directory:
        raise SECIngestionError(
            f"Ticker was not found in the SEC directory: "
            f"{normalized_ticker}"
        )

    record = company_directory[normalized_ticker]

    cik = str(record["cik_str"]).zfill(10)
    company_name = str(record["title"]).strip()

    company_dir = OUTPUT_DIR / normalized_ticker

    company_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    submissions_url = (
        "https://data.sec.gov/submissions/"
        f"CIK{cik}.json"
    )

    company_facts_url = (
        "https://data.sec.gov/api/xbrl/"
        f"companyfacts/CIK{cik}.json"
    )

    print(
        f"\nProcessing {normalized_ticker}: "
        f"{company_name}"
    )
    print(f"Sector: {sector}")

    # Refresh submission metadata on every run.
    submissions_response = request_with_retry(
        session,
        submissions_url,
    )

    submissions = submissions_response.json()

    save_json(
        submissions,
        company_dir / "submissions.json",
    )

    # Refresh structured XBRL facts on every run.
    company_facts_response = request_with_retry(
        session,
        company_facts_url,
    )

    company_facts = (
        company_facts_response.json()
    )

    save_json(
        company_facts,
        company_dir / "companyfacts.json",
    )

    return download_recent_filings(
        session=session,
        submissions=submissions,
        ticker=normalized_ticker,
        company_name=company_name,
        sector=sector,
        cik=cik,
    )


# ============================================================
# Report generation
# ============================================================

def save_manifest(
    rows: list[dict[str, str]],
) -> None:
    """Save metadata for all collected filing documents."""

    MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "ticker",
        "company_name",
        "sector",
        "cik",
        "form",
        "filing_date",
        "report_date",
        "accession_number",
        "source_url",
        "local_path",
    ]

    with MANIFEST_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_failed_companies(
    failures: list[dict[str, str]],
) -> None:
    """Save information about companies that failed processing."""

    FAILED_COMPANIES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "ticker",
        "sector",
        "error",
    ]

    with FAILED_COMPANIES_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(failures)


# ============================================================
# Main ingestion workflow
# ============================================================

def run_ingestion() -> None:
    """Collect the SEC dataset for configured companies."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    company_directory = load_company_directory()
    target_companies = load_target_companies()
    session = create_session()

    manifest_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    for company in target_companies:
        ticker = company["ticker"]
        sector = company["sector"]

        try:
            rows = download_company(
                session=session,
                ticker=ticker,
                sector=sector,
                company_directory=company_directory,
            )

            manifest_rows.extend(rows)

        except Exception as exc:
            error_message = str(exc)

            failures.append(
                {
                    "ticker": ticker,
                    "sector": sector,
                    "error": error_message,
                }
            )

            print(
                f"Failed to process {ticker}: "
                f"{error_message}"
            )

    save_manifest(manifest_rows)
    save_failed_companies(failures)

    completed_tickers = {
        row["ticker"]
        for row in manifest_rows
    }

    form_counts: dict[str, int] = {}

    for row in manifest_rows:
        form = row["form"]

        form_counts[form] = (
            form_counts.get(form, 0) + 1
        )

    print("\nSEC ingestion completed")
    print(
        f"Companies requested: "
        f"{len(target_companies)}"
    )
    print(
        f"Companies with filings collected: "
        f"{len(completed_tickers)}"
    )
    print(
        f"Filing documents collected: "
        f"{len(manifest_rows)}"
    )
    print(f"Forms collected: {form_counts}")
    print(f"Failed companies: {len(failures)}")

    if failures:
        print(
            "Failed tickers: "
            f"{[item['ticker'] for item in failures]}"
        )

    print(f"Dataset location: {OUTPUT_DIR}")
    print(f"Manifest location: {MANIFEST_FILE}")
    print(
        f"Failure report: "
        f"{FAILED_COMPANIES_FILE}"
    )


if __name__ == "__main__":
    run_ingestion()