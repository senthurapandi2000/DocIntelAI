from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COORDINATE_FILE = (
    PROJECT_ROOT
    / "config"
    / "irs_field_coordinates.json"
)

TEMPLATE_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "irs_templates"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "coordinate_previews"
)


W2_SAMPLE = {
    "employee_ssn": "XXX-XX-4821",
    "employer_ein": "12-3456789",
    "employer_name_address": (
        "NORTHSTAR ANALYTICS LLC\n"
        "1200 MARKET STREET\n"
        "WILMINGTON, DE 19801"
    ),
    "control_number": "W2-2026-000184",
    "employee_first_name": "ALEX J.",
    "employee_last_name": "MORGAN",
    "employee_suffix": "",
    "employee_address": (
        "742 OAK AVENUE\n"
        "NEWARK, DE 19711"
    ),
    "wages": "84,750.00",
    "federal_tax_withheld": "12,910.00",
    "social_security_wages": "84,750.00",
    "social_security_tax_withheld": "5,254.50",
    "medicare_wages": "84,750.00",
    "medicare_tax_withheld": "1,228.88",
    "social_security_tips": "0.00",
    "allocated_tips": "0.00",
    "dependent_care_benefits": "0.00",
    "nonqualified_plans": "0.00",
    "box12a_code": "D",
    "box12a_amount": "4,500.00",
    "box12b_code": "DD",
    "box12b_amount": "7,200.00",
    "box12c_code": "",
    "box12c_amount": "",
    "box12d_code": "",
    "box12d_amount": "",
    "statutory_employee": False,
    "retirement_plan": True,
    "third_party_sick_pay": False,
    "other": "HEALTH SAVINGS ACCOUNT\n1,200.00",
    "tipped_occupation_codes": "",
    "state": "DE",
    "employer_state_id": "1234567",
    "state_wages": "84,750.00",
    "state_income_tax": "4,236.00",
    "local_wages": "",
    "local_income_tax": "",
    "locality_name": "",
}

FORM_1099_SAMPLE = {
    "payer_name": "NORTHSTAR CONSULTING LLC",
    "payer_street": "1200 MARKET STREET",
    "payer_suite": "SUITE 450",
    "payer_city": "WILMINGTON",
    "payer_phone": "302-555-0184",
    "payer_state": "DE",
    "payer_country": "USA",
    "payer_zip": "19801",
    "payer_tin": "12-3456789",
    "recipient_tin": "XXX-XX-4821",
    "recipient_name": "ALEX J. MORGAN",
    "recipient_street": "742 OAK AVENUE",
    "recipient_apt": "APT 3B",
    "recipient_city": "NEWARK",
    "recipient_state": "DE",
    "recipient_country": "USA",
    "recipient_zip": "19711",
    "calendar_year": "2026",
    "nonemployee_compensation": "38,500.00",
    "cash_tips": "0.00",
    "ttoc": "",
    "overtime_compensation": "0.00",
    "direct_sales": False,
    "excess_golden_parachute": "0.00",
    "federal_income_tax_withheld": "5,775.00",
    "account_number": "CONTRACTOR-2026-0184",
    "state_tax_withheld": "1,925.00",
    "state_payer_number": "DE-998877",
    "state_income": "38,500.00",
}


def load_coordinates() -> dict[str, Any]:
    """Load the coordinate configuration."""

    if not COORDINATE_FILE.exists():
        raise FileNotFoundError(
            f"Coordinate file not found: {COORDINATE_FILE}"
        )

    with COORDINATE_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def alignment_value(name: str) -> int:
    """Convert alignment text into a PyMuPDF alignment value."""

    return {
        "left": 0,
        "center": 1,
        "right": 2,
    }.get(name, 0)


def create_single_page_template(
    source_path: Path,
    source_page_index: int,
) -> fitz.Document:
    """Copy the selected form page into a new document."""

    source_document = fitz.open(source_path)
    output_document = fitz.open()

    output_document.insert_pdf(
        source_document,
        from_page=source_page_index,
        to_page=source_page_index,
    )

    source_document.close()

    return output_document


def place_values(
    document: fitz.Document,
    fields: dict[str, Any],
    values: dict[str, Any],
) -> None:
    """Place sample values inside configured field rectangles."""

    page = document[0]

    for field_name, field_config in fields.items():
        value = values.get(field_name)

        if field_config["kind"] == "checkbox":
            if bool(value):
                rectangle = fitz.Rect(field_config["rect"])
                inset = 0.8
                page.draw_line(
                    fitz.Point(
                        rectangle.x0 + inset,
                        rectangle.y0 + inset,
                    ),
                    fitz.Point(
                        rectangle.x1 - inset,
                        rectangle.y1 - inset,
                    ),
                    color=(0, 0, 0),
                    width=0.8,
                )
                page.draw_line(
                    fitz.Point(
                        rectangle.x0 + inset,
                        rectangle.y1 - inset,
                    ),
                    fitz.Point(
                        rectangle.x1 - inset,
                        rectangle.y0 + inset,
                    ),
                    color=(0, 0, 0),
                    width=0.8,
                )
            continue

        if value in (None, ""):
            continue

        rectangle = fitz.Rect(field_config["rect"])
        font_size = float(field_config["font_size"])
        inserted = -1.0

        while font_size >= 4.0:
            shape = page.new_shape()
            inserted = shape.insert_textbox(
                rectangle,
                str(value),
                fontsize=font_size,
                fontname="helv",
                align=alignment_value(
                    str(field_config["align"])
                ),
                lineheight=1.0,
                color=(0, 0, 0),
            )

            if inserted >= 0:
                shape.commit()
                break

            font_size -= 0.4

        if inserted < 0:
            raise ValueError(
                f"Text did not fit inside field: {field_name}"
            )

    # Place the warning below the form, outside the data fields.
    page.insert_text(
        fitz.Point(300, 405),
        "SYNTHETIC SAMPLE - NOT FOR FILING",
        fontsize=9,
        fontname="helv",
        color=(0.65, 0.65, 0.65),
    )


def generate_preview(
    template_name: str,
    values: dict[str, Any],
) -> Path:
    """Generate one coordinate-validation PDF."""

    configuration = load_coordinates()
    template = configuration["templates"][template_name]

    source_path = (
        TEMPLATE_DIR
        / template["source_filename"]
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"IRS template not found: {source_path}"
        )

    output_document = create_single_page_template(
        source_path=source_path,
        source_page_index=int(
            template["source_page_index"]
        ),
    )

    place_values(
        output_document,
        fields=template["fields"],
        values=values,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{template_name}_coordinate_preview.pdf"
    )

    output_document.save(
        output_path,
        garbage=4,
        deflate=True,
    )

    output_document.close()

    return output_path


def run_preview_generation() -> None:
    """Generate validated W-2 and 1099-NEC previews."""

    w2_path = generate_preview(
        "w2_2026",
        W2_SAMPLE,
    )

    form_1099_path = generate_preview(
        "1099_nec_2026",
        FORM_1099_SAMPLE,
    )

    print("\nIRS coordinate previews generated")
    print(f"W-2 preview: {w2_path}")
    print(f"1099-NEC preview: {form_1099_path}")


if __name__ == "__main__":
    run_preview_generation()
