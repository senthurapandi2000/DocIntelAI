from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import fitz
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "irs_synthetic_manifest.csv"
)

DETAIL_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "irs_synthetic_validation_report.csv"
)

SUMMARY_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "irs_synthetic_validation_summary.json"
)

EXPECTED_DOCUMENT_TYPES = {"w2", "1099_nec"}
EXPECTED_SPLITS = {"train", "validation", "test"}

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
    "calendar_year",
    "nonemployee_compensation",
    "federal_income_tax_withheld",
    "account_number",
    "state_tax_withheld",
    "state_income",
}


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def parse_money(value: Any) -> Decimal | None:
    text = clean_text(value)
    if not text:
        return None

    normalized = text.replace("$", "").replace(",", "").strip()

    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def amounts_match(
    actual: Decimal | None,
    expected: Decimal | None,
    tolerance: Decimal = Decimal("0.02"),
) -> bool:
    if actual is None or expected is None:
        return False
    return abs(actual - expected) <= tolerance


def validate_pdf(path: Path) -> tuple[bool, int]:
    if not path.exists() or path.stat().st_size == 0:
        return False, 0

    try:
        document = fitz.open(path)
        page_count = len(document)
        document.close()
        return page_count == 1, page_count
    except Exception:
        return False, 0


def validate_png(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        with path.open("rb") as file:
            signature = file.read(8)
        return signature == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def validate_required_fields(
    fields: dict[str, Any],
    required_fields: set[str],
) -> tuple[list[str], list[str]]:
    missing_fields = sorted(required_fields - set(fields))

    empty_fields = sorted(
        field
        for field in required_fields
        if field in fields and not clean_text(fields[field])
    )

    return missing_fields, empty_fields


def validate_w2_calculations(
    fields: dict[str, Any],
) -> dict[str, bool]:
    social_security_wages = parse_money(
        fields.get("social_security_wages")
    )
    social_security_tax = parse_money(
        fields.get("social_security_tax_withheld")
    )

    medicare_wages = parse_money(
        fields.get("medicare_wages")
    )
    medicare_tax = parse_money(
        fields.get("medicare_tax_withheld")
    )

    expected_social_security_tax = (
        social_security_wages * Decimal("0.062")
    ).quantize(Decimal("0.01")) if social_security_wages is not None else None

    expected_medicare_tax = (
        medicare_wages * Decimal("0.0145")
    ).quantize(Decimal("0.01")) if medicare_wages is not None else None

    return {
        "social_security_tax_valid": amounts_match(
            social_security_tax,
            expected_social_security_tax,
        ),
        "medicare_tax_valid": amounts_match(
            medicare_tax,
            expected_medicare_tax,
        ),
    }


def validate_identifier_shapes(
    document_type: str,
    fields: dict[str, Any],
) -> bool:
    if document_type == "w2":
        employee_ssn = clean_text(fields.get("employee_ssn"))
        employer_ein = clean_text(fields.get("employer_ein"))

        return (
            employee_ssn.startswith("000-")
            and employer_ein.startswith("00-")
        )

    payer_tin = clean_text(fields.get("payer_tin"))
    recipient_tin = clean_text(fields.get("recipient_tin"))

    return (
        payer_tin.startswith("00-")
        and recipient_tin.startswith("000-")
    )


def run_validation() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "document_id",
        "document_type",
        "split",
        "pdf_path",
        "image_path",
        "annotation_path",
        "template_name",
        "render_dpi",
        "synthetic",
    }

    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            "Manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    validation_rows: list[dict[str, Any]] = []

    for index, row in manifest.iterrows():
        document_id = clean_text(row["document_id"])
        document_type = clean_text(row["document_type"])
        split = clean_text(row["split"])

        pdf_path = PROJECT_ROOT / clean_text(row["pdf_path"])
        image_path = PROJECT_ROOT / clean_text(row["image_path"])
        annotation_path = (
            PROJECT_ROOT / clean_text(row["annotation_path"])
        )

        print(
            f"[{index + 1}/{len(manifest)}] "
            f"Validating {document_id}"
        )

        annotation_valid = False
        annotation: dict[str, Any] = {}
        annotation_error = ""

        if annotation_path.exists():
            try:
                with annotation_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    annotation = json.load(file)

                annotation_valid = True
            except (OSError, json.JSONDecodeError) as exc:
                annotation_error = str(exc)

        fields = annotation.get("fields", {})
        field_boxes = annotation.get("field_boxes_pixels", {})

        required_fields = (
            W2_REQUIRED_FIELDS
            if document_type == "w2"
            else NEC_REQUIRED_FIELDS
        )

        missing_fields, empty_fields = validate_required_fields(
            fields,
            required_fields,
        )

        pdf_valid, pdf_page_count = validate_pdf(pdf_path)
        png_valid = validate_png(image_path)

        w2_calculations = {
            "social_security_tax_valid": True,
            "medicare_tax_valid": True,
        }

        if document_type == "w2":
            w2_calculations = validate_w2_calculations(fields)

        identifiers_valid = (
            validate_identifier_shapes(
                document_type,
                fields,
            )
            if fields
            else False
        )

        metadata_matches = bool(
            annotation.get("document_id") == document_id
            and annotation.get("document_type") == document_type
            and annotation.get("split") == split
        )

        dimensions_valid = bool(
            int(annotation.get("image_width", 0)) > 0
            and int(annotation.get("image_height", 0)) > 0
        )

        boxes_valid = bool(
            field_boxes
            and all(
                isinstance(box, list)
                and len(box) == 4
                and box[2] > box[0]
                and box[3] > box[1]
                for box in field_boxes.values()
            )
        )

        document_passed = bool(
            document_type in EXPECTED_DOCUMENT_TYPES
            and split in EXPECTED_SPLITS
            and pdf_valid
            and png_valid
            and annotation_valid
            and metadata_matches
            and dimensions_valid
            and boxes_valid
            and not missing_fields
            and not empty_fields
            and identifiers_valid
            and w2_calculations["social_security_tax_valid"]
            and w2_calculations["medicare_tax_valid"]
        )

        validation_rows.append(
            {
                "document_id": document_id,
                "document_type": document_type,
                "split": split,
                "pdf_exists": pdf_path.exists(),
                "pdf_valid": pdf_valid,
                "pdf_page_count": pdf_page_count,
                "image_exists": image_path.exists(),
                "png_valid": png_valid,
                "annotation_exists": annotation_path.exists(),
                "annotation_valid": annotation_valid,
                "metadata_matches": metadata_matches,
                "dimensions_valid": dimensions_valid,
                "field_boxes_valid": boxes_valid,
                "missing_fields": ", ".join(missing_fields),
                "empty_required_fields": ", ".join(empty_fields),
                "synthetic_identifiers_valid": identifiers_valid,
                "social_security_tax_valid": (
                    w2_calculations["social_security_tax_valid"]
                ),
                "medicare_tax_valid": (
                    w2_calculations["medicare_tax_valid"]
                ),
                "validation_passed": document_passed,
                "error": annotation_error,
            }
        )

    validation_df = pd.DataFrame(validation_rows)

    DETAIL_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_df.to_csv(
        DETAIL_REPORT,
        index=False,
    )

    split_counts = {
        str(split): int(count)
        for split, count in manifest["split"].value_counts().items()
    }

    document_type_counts = {
        str(document_type): int(count)
        for document_type, count in (
            manifest["document_type"].value_counts().items()
        )
    }

    duplicate_document_ids = int(
        manifest["document_id"].duplicated().sum()
    )

    summary = {
        "documents_checked": int(len(validation_df)),
        "documents_passed": int(
            validation_df["validation_passed"].sum()
        ),
        "documents_failed": int(
            (~validation_df["validation_passed"]).sum()
        ),
        "document_type_counts": document_type_counts,
        "split_counts": split_counts,
        "duplicate_document_ids": duplicate_document_ids,
        "validation_passed": bool(
            validation_df["validation_passed"].all()
            and duplicate_document_ids == 0
        ),
    }

    with SUMMARY_REPORT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print(
        "\nIRS synthetic dataset validation completed"
    )
    print(
        f"Documents checked: {summary['documents_checked']}"
    )
    print(
        f"Documents passed: {summary['documents_passed']}"
    )
    print(
        f"Documents failed: {summary['documents_failed']}"
    )
    print(
        f"Document types: {summary['document_type_counts']}"
    )
    print(f"Split counts: {summary['split_counts']}")
    print(
        f"Duplicate IDs: {summary['duplicate_document_ids']}"
    )
    print(
        f"Validation passed: {summary['validation_passed']}"
    )
    print(f"Detailed report: {DETAIL_REPORT}")
    print(f"Summary report: {SUMMARY_REPORT}")


if __name__ == "__main__":
    run_validation()
