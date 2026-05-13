from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_ocr_final_test_predictions.csv"
)

FIELD_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_extraction_field_validation.csv"
)

DOCUMENT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_extraction_document_validation.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_extraction_validation_summary.json"
)


VALID_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "AS", "GU", "MP", "PR", "VI",
}

MONEY_FIELDS = {
    "wages",
    "federal_tax_withheld",
    "social_security_wages",
    "social_security_tax_withheld",
    "medicare_wages",
    "medicare_tax_withheld",
    "state_wages",
    "state_income_tax",
    "nonemployee_compensation",
    "federal_income_tax_withheld",
    "state_tax_withheld",
    "state_income",
}

SSN_FIELDS = {
    "employee_ssn",
    "recipient_tin",
}

EIN_FIELDS = {
    "employer_ein",
    "payer_tin",
}

ZIP_FIELDS = {
    "payer_zip",
    "recipient_zip",
}

STATE_FIELDS = {
    "state",
    "payer_state",
    "recipient_state",
}

IDENTIFIER_FIELDS = {
    "control_number",
    "employer_state_id",
    "account_number",
    "state_payer_number",
}

MULTILINE_FIELDS = {
    "employer_name_address",
    "employee_address",
}

W2_REQUIRED_FIELDS = {
    "employee_ssn",
    "employer_ein",
    "employer_name_address",
    "employee_first_name",
    "employee_last_name",
    "employee_address",
    "wages",
    "federal_tax_withheld",
    "social_security_wages",
    "social_security_tax_withheld",
    "medicare_wages",
    "medicare_tax_withheld",
    "state",
}

NEC_REQUIRED_FIELDS = {
    "payer_name",
    "payer_tin",
    "recipient_tin",
    "recipient_name",
    "recipient_street",
    "recipient_city",
    "recipient_state",
    "recipient_zip",
    "nonemployee_compensation",
    "federal_income_tax_withheld",
}


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return False

    return clean_text(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def parse_decimal(value: Any) -> Decimal | None:
    text = clean_text(value)

    if not text:
        return None

    text = (
        text.replace("$", "")
        .replace(",", "")
        .replace(" ", "")
    )

    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def normalized_alphanumeric(value: Any) -> str:
    return re.sub(
        r"[^A-Z0-9]",
        "",
        clean_text(value).upper(),
    )


def normalized_digits(value: Any) -> str:
    return re.sub(
        r"[^0-9]",
        "",
        clean_text(value),
    )


def field_format_validation(
    field_name: str,
    value: Any,
    *,
    required: bool,
) -> tuple[bool, float, str]:
    text = clean_text(value)

    if not text:
        if required:
            return False, 0.0, "required_value_missing"

        return True, 1.0, "optional_value_missing"

    if field_name in MONEY_FIELDS:
        amount = parse_decimal(text)

        if amount is None:
            return False, 0.0, "invalid_money"

        if amount < 0:
            return False, 0.0, "negative_money"

        return True, 1.0, "valid_money"

    if field_name in SSN_FIELDS:
        digits = normalized_digits(text)

        if len(digits) != 9:
            score = max(
                0.0,
                1.0 - abs(len(digits) - 9) * 0.18,
            )
            return False, score, "invalid_ssn_length"

        return True, 1.0, "valid_ssn_shape"

    if field_name in EIN_FIELDS:
        digits = normalized_digits(text)

        if len(digits) != 9:
            score = max(
                0.0,
                1.0 - abs(len(digits) - 9) * 0.18,
            )
            return False, score, "invalid_ein_length"

        return True, 1.0, "valid_ein_shape"

    if field_name in ZIP_FIELDS:
        digits = normalized_digits(text)

        if len(digits) not in {5, 9}:
            score = max(
                0.0,
                1.0 - min(
                    abs(len(digits) - 5),
                    abs(len(digits) - 9),
                )
                * 0.20,
            )
            return False, score, "invalid_zip_length"

        return True, 1.0, "valid_zip_shape"

    if field_name in STATE_FIELDS:
        state = re.sub(
            r"[^A-Z]",
            "",
            text.upper(),
        )

        if state not in VALID_STATE_CODES:
            return False, 0.20, "invalid_state_code"

        return True, 1.0, "valid_state_code"

    if field_name in IDENTIFIER_FIELDS:
        identifier = normalized_alphanumeric(
            text
        )

        if len(identifier) < 4:
            return False, 0.25, "identifier_too_short"

        if len(identifier) > 30:
            return False, 0.40, "identifier_too_long"

        return True, 1.0, "valid_identifier_shape"

    normalized = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    minimum_length = (
        8
        if field_name in MULTILINE_FIELDS
        else 2
    )

    if len(normalized) < minimum_length:
        return False, 0.35, "text_too_short"

    return True, 1.0, "valid_text"


def engine_confidence(row: pd.Series) -> float:
    engine = clean_text(
        row.get("final_engine", "tesseract")
    ).lower()

    raw_tesseract_confidence = pd.to_numeric(
        row.get("ocr_confidence", 0.0),
        errors="coerce",
    )

    tesseract_confidence = (
        0.0
        if pd.isna(raw_tesseract_confidence)
        else float(raw_tesseract_confidence)
    )

    tesseract_score = min(
        1.0,
        max(0.0, tesseract_confidence / 100.0),
    )

    if engine != "trocr":
        return tesseract_score

    raw_router_probability = pd.to_numeric(
        row.get("router_probability", 0.0),
        errors="coerce",
    )

    router_probability = (
        0.0
        if pd.isna(raw_router_probability)
        else float(raw_router_probability)
    )

    return min(
        1.0,
        max(
            0.0,
            0.65 * router_probability
            + 0.35 * tesseract_score,
        ),
    )


def required_fields_for(
    document_type: str,
) -> set[str]:
    if document_type == "w2":
        return W2_REQUIRED_FIELDS

    if document_type == "1099_nec":
        return NEC_REQUIRED_FIELDS

    return set()


def validate_cross_fields(
    document_type: str,
    field_values: dict[str, str],
) -> list[str]:
    issues: list[str] = []

    if document_type == "w2":
        social_security_wages = parse_decimal(
            field_values.get(
                "social_security_wages",
                "",
            )
        )

        social_security_tax = parse_decimal(
            field_values.get(
                "social_security_tax_withheld",
                "",
            )
        )

        medicare_wages = parse_decimal(
            field_values.get(
                "medicare_wages",
                "",
            )
        )

        medicare_tax = parse_decimal(
            field_values.get(
                "medicare_tax_withheld",
                "",
            )
        )

        if (
            social_security_wages is not None
            and social_security_tax is not None
        ):
            expected = (
                social_security_wages
                * Decimal("0.062")
            )

            tolerance = max(
                Decimal("1.00"),
                abs(expected) * Decimal("0.03"),
            )

            if abs(
                social_security_tax - expected
            ) > tolerance:
                issues.append(
                    "social_security_tax_inconsistent"
                )

        if (
            medicare_wages is not None
            and medicare_tax is not None
        ):
            expected = (
                medicare_wages
                * Decimal("0.0145")
            )

            tolerance = max(
                Decimal("1.00"),
                abs(expected) * Decimal("0.03"),
            )

            if abs(
                medicare_tax - expected
            ) > tolerance:
                issues.append(
                    "medicare_tax_inconsistent"
                )

        wages = parse_decimal(
            field_values.get("wages", "")
        )

        federal_tax = parse_decimal(
            field_values.get(
                "federal_tax_withheld",
                "",
            )
        )

        if (
            wages is not None
            and federal_tax is not None
            and federal_tax > wages
        ):
            issues.append(
                "federal_tax_exceeds_wages"
            )

    if document_type == "1099_nec":
        compensation = parse_decimal(
            field_values.get(
                "nonemployee_compensation",
                "",
            )
        )

        federal_tax = parse_decimal(
            field_values.get(
                "federal_income_tax_withheld",
                "",
            )
        )

        state_income = parse_decimal(
            field_values.get(
                "state_income",
                "",
            )
        )

        state_tax = parse_decimal(
            field_values.get(
                "state_tax_withheld",
                "",
            )
        )

        if (
            compensation is not None
            and federal_tax is not None
            and federal_tax > compensation
        ):
            issues.append(
                "federal_tax_exceeds_compensation"
            )

        if (
            state_income is not None
            and state_tax is not None
            and state_tax > state_income
        ):
            issues.append(
                "state_tax_exceeds_state_income"
            )

    return issues


def run_validation(
    *,
    predictions_path: Path,
    review_threshold: float,
) -> None:
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Final OCR predictions not found: "
            f"{predictions_path}"
        )

    predictions = pd.read_csv(
        predictions_path
    )

    required_columns = {
        "document_id",
        "document_type",
        "split",
        "quality",
        "field_name",
        "final_engine",
        "final_prediction",
        "final_normalized_prediction",
        "ocr_confidence",
        "router_probability",
    }

    missing_columns = (
        required_columns - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Final prediction file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    field_rows: list[dict[str, Any]] = []

    for _, row in predictions.iterrows():
        document_type = clean_text(
            row["document_type"]
        )

        field_name = clean_text(
            row["field_name"]
        )

        required = field_name in (
            required_fields_for(
                document_type
            )
        )

        final_value = clean_text(
            row[
                "final_normalized_prediction"
            ]
        )

        (
            format_valid,
            format_score,
            format_message,
        ) = field_format_validation(
            field_name,
            final_value,
            required=required,
        )

        model_confidence = engine_confidence(
            row
        )

        validation_score = (
            0.60 * format_score
            + 0.40 * model_confidence
        )

        field_review_required = bool(
            not format_valid
            or validation_score
            < review_threshold
        )

        field_rows.append(
            {
                "document_id": (
                    clean_text(
                        row["document_id"]
                    )
                ),
                "parent_document_id": (
                    clean_text(
                        row.get(
                            "parent_document_id",
                            "",
                        )
                    )
                ),
                "document_type": (
                    document_type
                ),
                "split": clean_text(
                    row["split"]
                ),
                "quality": clean_text(
                    row["quality"]
                ),
                "field_name": field_name,
                "required_field": required,
                "final_engine": clean_text(
                    row["final_engine"]
                ),
                "final_prediction": (
                    clean_text(
                        row[
                            "final_prediction"
                        ]
                    )
                ),
                "final_normalized_prediction": (
                    final_value
                ),
                "format_valid": (
                    format_valid
                ),
                "format_score": round(
                    format_score,
                    4,
                ),
                "format_message": (
                    format_message
                ),
                "model_confidence": round(
                    model_confidence,
                    4,
                ),
                "validation_score": round(
                    validation_score,
                    4,
                ),
                "field_review_required": (
                    field_review_required
                ),
                "ground_truth_exact_match": (
                    to_bool(
                        row.get(
                            "final_exact_match",
                            False,
                        )
                    )
                    if (
                        "final_exact_match"
                        in predictions.columns
                    )
                    else None
                ),
            }
        )

    field_report = pd.DataFrame(
        field_rows
    )

    document_rows: list[dict[str, Any]] = []

    for document_id, document_df in (
        field_report.groupby(
            "document_id",
            sort=False,
        )
    ):
        document_type = clean_text(
            document_df[
                "document_type"
            ].iloc[0]
        )

        field_values = {
            clean_text(row["field_name"]): (
                clean_text(
                    row[
                        "final_normalized_prediction"
                    ]
                )
            )
            for _, row in (
                document_df.iterrows()
            )
        }

        cross_field_issues = (
            validate_cross_fields(
                document_type,
                field_values,
            )
        )

        field_review_count = int(
            document_df[
                "field_review_required"
            ].sum()
        )

        required_invalid_count = int(
            (
                document_df[
                    "required_field"
                ]
                & ~document_df[
                    "format_valid"
                ]
            ).sum()
        )

        average_validation_score = float(
            document_df[
                "validation_score"
            ].mean()
        )

        minimum_validation_score = float(
            document_df[
                "validation_score"
            ].min()
        )

        document_review_required = bool(
            field_review_count > 0
            or required_invalid_count > 0
            or len(cross_field_issues) > 0
        )

        if cross_field_issues:
            document_status = (
                "review_cross_field_failure"
            )
        elif required_invalid_count > 0:
            document_status = (
                "review_required_field_failure"
            )
        elif field_review_count > 0:
            document_status = (
                "review_low_confidence"
            )
        else:
            document_status = (
                "auto_accept"
            )

        document_rows.append(
            {
                "document_id": document_id,
                "parent_document_id": (
                    clean_text(
                        document_df[
                            "parent_document_id"
                        ].iloc[0]
                    )
                ),
                "document_type": (
                    document_type
                ),
                "split": clean_text(
                    document_df[
                        "split"
                    ].iloc[0]
                ),
                "quality": clean_text(
                    document_df[
                        "quality"
                    ].iloc[0]
                ),
                "fields_checked": int(
                    len(document_df)
                ),
                "fields_flagged_for_review": (
                    field_review_count
                ),
                "required_invalid_fields": (
                    required_invalid_count
                ),
                "cross_field_issue_count": int(
                    len(cross_field_issues)
                ),
                "cross_field_issues": (
                    "; ".join(
                        cross_field_issues
                    )
                ),
                "average_validation_score": round(
                    average_validation_score,
                    4,
                ),
                "minimum_validation_score": round(
                    minimum_validation_score,
                    4,
                ),
                "document_review_required": (
                    document_review_required
                ),
                "document_status": (
                    document_status
                ),
            }
        )

    document_report = pd.DataFrame(
        document_rows
    )

    FIELD_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    field_report.to_csv(
        FIELD_REPORT_PATH,
        index=False,
    )

    document_report.to_csv(
        DOCUMENT_REPORT_PATH,
        index=False,
    )

    summary: dict[str, Any] = {
        "predictions_path": str(
            predictions_path
        ),
        "review_threshold": (
            review_threshold
        ),
        "documents_checked": int(
            len(document_report)
        ),
        "fields_checked": int(
            len(field_report)
        ),
        "documents_auto_accepted": int(
            (
                ~document_report[
                    "document_review_required"
                ]
            ).sum()
        ),
        "documents_flagged_for_review": int(
            document_report[
                "document_review_required"
            ].sum()
        ),
        "document_review_rate": round(
            float(
                document_report[
                    "document_review_required"
                ].mean()
            ),
            4,
        ),
        "fields_auto_accepted": int(
            (
                ~field_report[
                    "field_review_required"
                ]
            ).sum()
        ),
        "fields_flagged_for_review": int(
            field_report[
                "field_review_required"
            ].sum()
        ),
        "field_review_rate": round(
            float(
                field_report[
                    "field_review_required"
                ].mean()
            ),
            4,
        ),
        "invalid_format_fields": int(
            (
                ~field_report[
                    "format_valid"
                ]
            ).sum()
        ),
        "cross_field_failures": int(
            (
                document_report[
                    "cross_field_issue_count"
                ]
                > 0
            ).sum()
        ),
        "document_status_counts": {
            str(status): int(count)
            for status, count in (
                document_report[
                    "document_status"
                ]
                .value_counts()
                .items()
            )
        },
    }

    if (
        "final_exact_match"
        in predictions.columns
    ):
        correctness = field_report[
            "ground_truth_exact_match"
        ].astype(bool)

        review_flags = field_report[
            "field_review_required"
        ].astype(bool)

        incorrect = ~correctness

        summary[
            "evaluation_only"
        ] = {
            "overall_exact_match_rate": round(
                float(
                    correctness.mean()
                ),
                4,
            ),
            "auto_accepted_field_exact_match_rate": round(
                float(
                    correctness[
                        ~review_flags
                    ].mean()
                )
                if (
                    (~review_flags).any()
                )
                else 0.0,
                4,
            ),
            "reviewed_field_exact_match_rate": round(
                float(
                    correctness[
                        review_flags
                    ].mean()
                )
                if review_flags.any()
                else 0.0,
                4,
            ),
            "incorrect_field_capture_rate": round(
                float(
                    review_flags[
                        incorrect
                    ].mean()
                )
                if incorrect.any()
                else 0.0,
                4,
            ),
            "incorrect_fields_auto_accepted": int(
                (
                    incorrect
                    & ~review_flags
                ).sum()
            ),
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

    print(
        "\nIRS extraction validation completed"
    )
    print(
        f"Documents checked: "
        f"{summary['documents_checked']}"
    )
    print(
        f"Fields checked: "
        f"{summary['fields_checked']}"
    )
    print(
        f"Documents auto accepted: "
        f"{summary['documents_auto_accepted']}"
    )
    print(
        f"Documents flagged for review: "
        f"{summary['documents_flagged_for_review']}"
    )
    print(
        f"Document review rate: "
        f"{summary['document_review_rate']:.2%}"
    )
    print(
        f"Fields auto accepted: "
        f"{summary['fields_auto_accepted']}"
    )
    print(
        f"Fields flagged for review: "
        f"{summary['fields_flagged_for_review']}"
    )
    print(
        f"Field review rate: "
        f"{summary['field_review_rate']:.2%}"
    )
    print(
        f"Invalid-format fields: "
        f"{summary['invalid_format_fields']}"
    )
    print(
        f"Cross-field failures: "
        f"{summary['cross_field_failures']}"
    )

    if "evaluation_only" in summary:
        evaluation = summary[
            "evaluation_only"
        ]

        print(
            "\nReview-policy evaluation"
        )
        print(
            "Auto-accepted exact match: "
            f"{evaluation['auto_accepted_field_exact_match_rate']:.2%}"
        )
        print(
            "Incorrect-field capture rate: "
            f"{evaluation['incorrect_field_capture_rate']:.2%}"
        )
        print(
            "Incorrect fields auto accepted: "
            f"{evaluation['incorrect_fields_auto_accepted']}"
        )

    print(
        f"\nField report: "
        f"{FIELD_REPORT_PATH}"
    )
    print(
        f"Document report: "
        f"{DOCUMENT_REPORT_PATH}"
    )
    print(
        f"Summary: {SUMMARY_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate final IRS OCR extractions and "
            "route uncertain fields/documents to "
            "human review."
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=(
            DEFAULT_PREDICTIONS_PATH
        ),
    )

    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.72,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_validation(
        predictions_path=(
            arguments.predictions
        ),
        review_threshold=(
            arguments.review_threshold
        ),
    )
