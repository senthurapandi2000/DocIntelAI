from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSING_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_preprocessing_report.csv"
)

QUALITY_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_text_quality_report.csv"
)

SUMMARY_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "sec_text_quality_summary.json"
)

MINIMUM_WORDS = {
    "10-K": 5000,
    "10-Q": 2000,
    "8-K": 200,
}

FORM_KEYWORDS = {
    "10-K": ["annual report", "form 10-k"],
    "10-Q": ["quarterly report", "form 10-q"],
    "8-K": ["current report", "form 8-k"],
}


def calculate_hash(text: str) -> str:
    """Create a hash used to identify duplicate documents."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def validate_text_file(
    path: Path,
    form: str,
) -> dict[str, object]:
    """Evaluate the quality of one processed filing."""

    if not path.exists():
        return {
            "file_exists": False,
            "character_count": 0,
            "word_count": 0,
            "line_count": 0,
            "alphabetic_ratio": 0.0,
            "expected_keyword_found": False,
            "content_hash": "",
            "quality_status": "missing",
        }

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    character_count = len(text)
    words = text.split()
    word_count = len(words)
    line_count = len(text.splitlines())

    alphabetic_count = sum(
        character.isalpha()
        for character in text
    )

    alphabetic_ratio = (
        alphabetic_count / max(character_count, 1)
    )

    normalized_text = text.lower()

    expected_keyword_found = any(
        keyword in normalized_text
        for keyword in FORM_KEYWORDS.get(form, [])
    )

    minimum_words = MINIMUM_WORDS.get(form, 200)

    quality_status = "pass"

    if not text.strip():
        quality_status = "empty"

    elif word_count < minimum_words:
        quality_status = "review_short"

    elif alphabetic_ratio < 0.35:
        quality_status = "review_noisy"

    return {
        "file_exists": True,
        "character_count": character_count,
        "word_count": word_count,
        "line_count": line_count,
        "alphabetic_ratio": round(
            alphabetic_ratio,
            4,
        ),
        "expected_keyword_found": (
            expected_keyword_found
        ),
        "content_hash": calculate_hash(text),
        "quality_status": quality_status,
    }


def run_quality_validation() -> None:
    """Validate all cleaned SEC filing text files."""

    if not PREPROCESSING_REPORT.exists():
        raise FileNotFoundError(
            f"Preprocessing report not found: "
            f"{PREPROCESSING_REPORT}"
        )

    preprocessing_df = pd.read_csv(
        PREPROCESSING_REPORT
    )

    successful_df = preprocessing_df[
        preprocessing_df["status"] == "success"
    ].copy()

    quality_rows: list[dict[str, object]] = []

    for _, row in successful_df.iterrows():
        output_path = (
            PROJECT_ROOT
            / str(row["output_path"])
        )

        result = validate_text_file(
            path=output_path,
            form=str(row["form"]),
        )

        quality_rows.append(
            {
                "ticker": row["ticker"],
                "company_name": row["company_name"],
                "sector": row["sector"],
                "form": row["form"],
                "output_path": row["output_path"],
                **result,
            }
        )

    quality_df = pd.DataFrame(quality_rows)

    if quality_df.empty:
        raise ValueError(
            "No successfully processed text files "
            "were found."
        )

    quality_df["is_duplicate"] = (
        quality_df["content_hash"]
        .duplicated(keep=False)
        & quality_df["content_hash"].ne("")
    )

    QUALITY_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_df.to_csv(
        QUALITY_REPORT,
        index=False,
    )

    average_words_by_form = {
        str(form): round(float(value), 2)
        for form, value in (
            quality_df
            .groupby("form")["word_count"]
            .mean()
            .items()
        )
    }

    summary = {
        "documents_checked": int(
            len(quality_df)
        ),
        "passed": int(
            (
                quality_df["quality_status"]
                == "pass"
            ).sum()
        ),
        "review_short": int(
            (
                quality_df["quality_status"]
                == "review_short"
            ).sum()
        ),
        "review_noisy": int(
            (
                quality_df["quality_status"]
                == "review_noisy"
            ).sum()
        ),
        "missing_or_empty": int(
            quality_df[
                "quality_status"
            ].isin(["missing", "empty"]).sum()
        ),
        "duplicate_documents": int(
            quality_df["is_duplicate"].sum()
        ),
        "average_words_by_form": (
            average_words_by_form
        ),
        "validation_passed": bool(
            quality_df["file_exists"].all()
            and not quality_df[
                "quality_status"
            ].isin(["missing", "empty"]).any()
            and not quality_df[
                "is_duplicate"
            ].any()
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

    print("\nSEC text-quality validation completed")
    print(
        f"Documents checked: "
        f"{summary['documents_checked']}"
    )
    print(f"Passed: {summary['passed']}")
    print(
        f"Short documents for review: "
        f"{summary['review_short']}"
    )
    print(
        f"Noisy documents for review: "
        f"{summary['review_noisy']}"
    )
    print(
        f"Missing or empty: "
        f"{summary['missing_or_empty']}"
    )
    print(
        f"Duplicate documents: "
        f"{summary['duplicate_documents']}"
    )
    print(
        f"Average words by form: "
        f"{summary['average_words_by_form']}"
    )
    print(
        f"Validation passed: "
        f"{summary['validation_passed']}"
    )
    print(f"Detailed report: {QUALITY_REPORT}")
    print(f"Summary report: {SUMMARY_REPORT}")


if __name__ == "__main__":
    run_quality_validation()