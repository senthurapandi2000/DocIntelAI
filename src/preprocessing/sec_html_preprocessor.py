from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import warnings

from bs4 import (
    BeautifulSoup,
    Comment,
    XMLParsedAsHTMLWarning,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sec_filings"
    / "dataset_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_text"
)

PREPROCESSING_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_preprocessing_report.csv"
)


def read_document(path: Path) -> str:
    """Read an SEC filing using safe encoding fallbacks."""

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    raw_bytes = path.read_bytes()

    for encoding in ("utf-8", "latin-1", "windows-1252"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw_bytes.decode("utf-8", errors="ignore")


def remove_invisible_content(soup: BeautifulSoup) -> None:
    """Remove scripts, styles, comments, and hidden XBRL content."""

    removable_tags = (
        "script",
        "style",
        "noscript",
        "svg",
        "iframe",
        "object",
        "canvas",
    )

    for tag in soup.find_all(removable_tags):
        tag.decompose()

    for comment in soup.find_all(
        string=lambda value: isinstance(value, Comment)
    ):
        comment.extract()

    for tag in soup.find_all(
        style=re.compile(
            r"display\s*:\s*none|visibility\s*:\s*hidden",
            flags=re.IGNORECASE,
        )
    ):
        tag.decompose()

    for tag in soup.find_all(
        attrs={"aria-hidden": "true"}
    ):
        tag.decompose()

    # Hidden inline-XBRL sections can contain large amounts of duplicate text.
    for tag_name in (
        "ix:hidden",
        "xbrli:hidden",
    ):
        for tag in soup.find_all(tag_name):
            tag.decompose()


def add_structure_markers(soup: BeautifulSoup) -> None:
    """Insert separators around content that should remain readable."""

    for break_tag in soup.find_all("br"):
        break_tag.replace_with("\n")

    block_tags = (
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    )

    for tag in soup.find_all(block_tags):
        tag.insert_before("\n")
        tag.insert_after("\n")

    # Preserve financial table cells in a readable sequence.
    for cell in soup.find_all(("td", "th")):
        cell.insert_after(" | ")


def normalize_text(text: str) -> str:
    """Normalize extracted filing text while preserving line structure."""

    text = html.unescape(text)

    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\| *", " | ", text)

    # Remove repeated table separators.
    text = re.sub(r"(?:\s*\|\s*){3,}", " | ", text)

    lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue

        # Drop lines made only of punctuation or separators.
        if re.fullmatch(r"[\W_]+", line):
            continue

        lines.append(line)

    cleaned_text = "\n".join(lines)

    cleaned_text = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned_text,
    )

    return cleaned_text.strip()


def extract_clean_text(raw_document: str) -> str:
    """Convert SEC HTML or XML content into clean readable text."""

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            XMLParsedAsHTMLWarning,
        )

        soup = BeautifulSoup(
            raw_document,
            "lxml",
    )

    remove_invisible_content(soup)
    add_structure_markers(soup)

    extracted_text = soup.get_text(
        separator=" ",
        strip=False,
    )

    return normalize_text(extracted_text)


def create_output_path(
    ticker: str,
    form: str,
    source_path: Path,
) -> Path:
    """Create a stable output path for a processed filing."""

    form_folder = form.replace("/", "_")

    output_name = f"{source_path.stem}.txt"

    return (
        OUTPUT_DIR
        / ticker
        / form_folder
        / output_name
    )


def preprocess_dataset() -> None:
    """Preprocess every SEC filing listed in the dataset manifest."""

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
        "sector",
        "form",
        "local_path",
    }

    missing_columns = (
        required_columns - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    report_rows: list[dict[str, object]] = []

    for index, row in manifest.iterrows():
        source_path = (
            PROJECT_ROOT
            / str(row["local_path"])
        )

        output_path = create_output_path(
            ticker=str(row["ticker"]),
            form=str(row["form"]),
            source_path=source_path,
        )

        print(
            f"[{index + 1}/{len(manifest)}] "
            f"Processing {row['ticker']} {row['form']}"
        )

        try:
            raw_document = read_document(source_path)
            cleaned_text = extract_clean_text(raw_document)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path.write_text(
                cleaned_text,
                encoding="utf-8",
            )

            raw_character_count = len(raw_document)
            clean_character_count = len(cleaned_text)
            word_count = len(cleaned_text.split())

            report_rows.append(
                {
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "sector": row["sector"],
                    "form": row["form"],
                    "source_path": str(
                        source_path.relative_to(PROJECT_ROOT)
                    ),
                    "output_path": str(
                        output_path.relative_to(PROJECT_ROOT)
                    ),
                    "raw_characters": raw_character_count,
                    "clean_characters": clean_character_count,
                    "word_count": word_count,
                    "reduction_percentage": round(
                        (
                            1
                            - clean_character_count
                            / max(raw_character_count, 1)
                        )
                        * 100,
                        2,
                    ),
                    "status": (
                        "success"
                        if cleaned_text
                        else "empty"
                    ),
                    "error": "",
                }
            )

        except Exception as exc:
            report_rows.append(
                {
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "sector": row["sector"],
                    "form": row["form"],
                    "source_path": str(row["local_path"]),
                    "output_path": "",
                    "raw_characters": 0,
                    "clean_characters": 0,
                    "word_count": 0,
                    "reduction_percentage": 0,
                    "status": "failed",
                    "error": str(exc),
                }
            )

            print(
                f"Failed to process "
                f"{row['ticker']} {row['form']}: {exc}"
            )

    report_df = pd.DataFrame(report_rows)

    PREPROCESSING_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_df.to_csv(
        PREPROCESSING_REPORT,
        index=False,
    )

    successful = int(
        (report_df["status"] == "success").sum()
    )

    failed = int(
        (report_df["status"] == "failed").sum()
    )

    empty = int(
        (report_df["status"] == "empty").sum()
    )

    print("\nSEC preprocessing completed")
    print(f"Documents processed: {len(report_df)}")
    print(f"Successful: {successful}")
    print(f"Empty outputs: {empty}")
    print(f"Failed: {failed}")
    print(f"Processed text directory: {OUTPUT_DIR}")
    print(f"Report: {PREPROCESSING_REPORT}")


if __name__ == "__main__":
    preprocess_dataset()