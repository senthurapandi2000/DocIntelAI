from __future__ import annotations

import argparse
import csv
import json
import random
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import fitz
from faker import Faker

from src.preprocessing.preview_irs_coordinate_mapping import (
    create_single_page_template,
    load_coordinates,
    place_values,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEMPLATE_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "irs_templates"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "irs_synthetic"
)

ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "irs_synthetic"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "irs_synthetic_manifest.csv"
)

DEFAULT_DPI = 200
DEFAULT_SEED = 42

FAKE = Faker("en_US")


def money(value: Decimal | float | int) -> str:
    """Format a monetary value using two decimal places."""

    amount = Decimal(str(value)).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )
    return f"{amount:,.2f}"


def decimal_money(
    minimum: int,
    maximum: int,
) -> Decimal:
    """Generate a random monetary amount."""

    cents = random.randint(
        minimum * 100,
        maximum * 100,
    )
    return Decimal(cents) / Decimal(100)


def synthetic_ssn() -> str:
    """Generate a clearly synthetic SSN-shaped identifier."""

    return (
        f"000-{random.randint(10, 99):02d}-"
        f"{random.randint(1000, 9999):04d}"
    )


def synthetic_ein() -> str:
    """Generate a clearly synthetic EIN-shaped identifier."""

    return f"00-{random.randint(1000000, 9999999):07d}"


def synthetic_state_id(state: str) -> str:
    """Generate a fictional state employer identifier."""

    return (
        f"{state}-"
        f"{random.randint(1000000, 9999999):07d}"
    )


def split_name() -> str:
    """Assign a reproducible train, validation, or test split."""

    value = random.random()

    if value < 0.70:
        return "train"

    if value < 0.85:
        return "validation"

    return "test"


def format_multiline_address(
    company_or_street: str,
    street_or_city: str,
    city_state_zip: str,
) -> str:
    """Build a three-line address block."""

    return (
        f"{company_or_street}\n"
        f"{street_or_city}\n"
        f"{city_state_zip}"
    )


def generate_w2_values(index: int) -> dict[str, Any]:
    """Create one internally consistent fictional W-2 record."""

    first_name = FAKE.first_name().upper()
    middle_initial = f"{FAKE.random_uppercase_letter()}."
    last_name = FAKE.last_name().upper()

    employer_name = FAKE.company().upper()
    employer_street = FAKE.street_address().upper()
    employer_city = FAKE.city().upper()
    employer_state = FAKE.state_abbr()
    employer_zip = FAKE.zipcode()

    employee_street = FAKE.street_address().upper()
    employee_city = FAKE.city().upper()
    employee_state = FAKE.state_abbr()
    employee_zip = FAKE.zipcode()

    wages = decimal_money(28_000, 150_000)

    # Keep wages below the Social Security wage-base range so that
    # the synthetic calculations remain internally consistent.
    social_security_wages = wages
    medicare_wages = wages

    federal_rate = Decimal(
        str(random.uniform(0.08, 0.24))
    )
    state_rate = Decimal(
        str(random.uniform(0.00, 0.065))
    )

    federal_tax = (
        wages * federal_rate
    ).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )

    social_security_tax = (
        social_security_wages * Decimal("0.062")
    ).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )

    medicare_tax = (
        medicare_wages * Decimal("0.0145")
    ).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )

    state_tax = (
        wages * state_rate
    ).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )

    retirement_plan = random.random() < 0.70
    retirement_amount = (
        decimal_money(500, 12_000)
        if retirement_plan
        else Decimal("0.00")
    )

    health_cost = decimal_money(2_000, 15_000)

    return {
        "employee_ssn": synthetic_ssn(),
        "employer_ein": synthetic_ein(),
        "employer_name_address": format_multiline_address(
            employer_name,
            employer_street,
            (
                f"{employer_city}, "
                f"{employer_state} {employer_zip}"
            ),
        ),
        "control_number": f"W2-2026-{index:06d}",
        "employee_first_name": (
            f"{first_name} {middle_initial}"
        ),
        "employee_last_name": last_name,
        "employee_suffix": "",
        "employee_address": (
            f"{employee_street}\n"
            f"{employee_city}, "
            f"{employee_state} {employee_zip}"
        ),
        "wages": money(wages),
        "federal_tax_withheld": money(federal_tax),
        "social_security_wages": money(
            social_security_wages
        ),
        "social_security_tax_withheld": money(
            social_security_tax
        ),
        "medicare_wages": money(medicare_wages),
        "medicare_tax_withheld": money(
            medicare_tax
        ),
        "social_security_tips": "0.00",
        "allocated_tips": "0.00",
        "dependent_care_benefits": (
            money(decimal_money(0, 5_000))
            if random.random() < 0.20
            else "0.00"
        ),
        "nonqualified_plans": "0.00",
        "box12a_code": "D" if retirement_plan else "",
        "box12a_amount": (
            money(retirement_amount)
            if retirement_plan
            else ""
        ),
        "box12b_code": "DD",
        "box12b_amount": money(health_cost),
        "box12c_code": "",
        "box12c_amount": "",
        "box12d_code": "",
        "box12d_amount": "",
        "statutory_employee": random.random() < 0.05,
        "retirement_plan": retirement_plan,
        "third_party_sick_pay": random.random() < 0.03,
        "other": "",
        "tipped_occupation_codes": "",
        "state": employee_state,
        "employer_state_id": synthetic_state_id(
            employee_state
        ),
        "state_wages": money(wages),
        "state_income_tax": money(state_tax),
        "local_wages": "",
        "local_income_tax": "",
        "locality_name": "",
    }


def generate_1099_values(index: int) -> dict[str, Any]:
    """Create one internally consistent fictional 1099-NEC record."""

    payer_name = FAKE.company().upper()
    payer_street = FAKE.street_address().upper()
    payer_city = FAKE.city().upper()
    payer_state = FAKE.state_abbr()
    payer_zip = FAKE.zipcode()

    recipient_name = FAKE.name().upper()
    recipient_street = FAKE.street_address().upper()
    recipient_city = FAKE.city().upper()
    recipient_state = FAKE.state_abbr()
    recipient_zip = FAKE.zipcode()

    compensation = decimal_money(3_000, 160_000)

    federal_rate = (
        Decimal(str(random.uniform(0.00, 0.15)))
        if random.random() < 0.35
        else Decimal("0.00")
    )

    state_rate = (
        Decimal(str(random.uniform(0.00, 0.06)))
        if random.random() < 0.45
        else Decimal("0.00")
    )

    federal_tax = (
        compensation * federal_rate
    ).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )

    state_tax = (
        compensation * state_rate
    ).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )

    return {
        "payer_name": payer_name,
        "payer_street": payer_street,
        "payer_suite": (
            f"SUITE {random.randint(100, 999)}"
            if random.random() < 0.40
            else ""
        ),
        "payer_city": payer_city,
        "payer_phone": FAKE.numerify(
            text="###-###-####"
        ),
        "payer_state": payer_state,
        "payer_country": "USA",
        "payer_zip": payer_zip,
        "payer_tin": synthetic_ein(),
        "recipient_tin": synthetic_ssn(),
        "recipient_name": recipient_name,
        "recipient_street": recipient_street,
        "recipient_apt": (
            f"APT {random.randint(1, 50)}"
            if random.random() < 0.35
            else ""
        ),
        "recipient_city": recipient_city,
        "recipient_state": recipient_state,
        "recipient_country": "USA",
        "recipient_zip": recipient_zip,
        "calendar_year": "2026",
        "nonemployee_compensation": money(
            compensation
        ),
        "cash_tips": "0.00",
        "ttoc": "",
        "overtime_compensation": "0.00",
        "direct_sales": random.random() < 0.05,
        "excess_golden_parachute": "0.00",
        "federal_income_tax_withheld": money(
            federal_tax
        ),
        "account_number": f"NEC-2026-{index:06d}",
        "state_tax_withheld": money(state_tax),
        "state_payer_number": synthetic_state_id(
            recipient_state
        ),
        "state_income": money(compensation),
    }


def render_pdf_page(
    pdf_path: Path,
    image_path: Path,
    dpi: int,
) -> tuple[int, int]:
    """Render the first PDF page as a PNG image."""

    document = fitz.open(pdf_path)
    page = document[0]

    scale = dpi / 72.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False,
    )

    image_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pixmap.save(image_path)

    width = pixmap.width
    height = pixmap.height

    document.close()

    return width, height


def field_boxes_in_pixels(
    fields: dict[str, Any],
    dpi: int,
) -> dict[str, list[int]]:
    """Convert configured PDF-point rectangles into pixel boxes."""

    scale = dpi / 72.0

    return {
        field_name: [
            round(float(rect[0]) * scale),
            round(float(rect[1]) * scale),
            round(float(rect[2]) * scale),
            round(float(rect[3]) * scale),
        ]
        for field_name, config in fields.items()
        for rect in [config["rect"]]
    }


def write_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """Write one JSON annotation file."""

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


def generate_document(
    *,
    template_name: str,
    document_type: str,
    values: dict[str, Any],
    document_id: str,
    split: str,
    dpi: int,
    coordinates: dict[str, Any],
) -> dict[str, Any]:
    """Generate one synthetic PDF, image, and annotation."""

    template = coordinates["templates"][template_name]

    source_path = (
        TEMPLATE_DIR
        / template["source_filename"]
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"IRS template not found: {source_path}"
        )

    pdf_path = (
        OUTPUT_ROOT
        / document_type
        / split
        / "pdf"
        / f"{document_id}.pdf"
    )

    image_path = (
        OUTPUT_ROOT
        / document_type
        / split
        / "images"
        / f"{document_id}.png"
    )

    annotation_path = (
        ANNOTATION_DIR
        / document_type
        / split
        / f"{document_id}.json"
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

    pdf_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_document.save(
        pdf_path,
        garbage=4,
        deflate=True,
    )
    output_document.close()

    image_width, image_height = render_pdf_page(
        pdf_path=pdf_path,
        image_path=image_path,
        dpi=dpi,
    )

    annotation = {
        "document_id": document_id,
        "document_type": document_type,
        "split": split,
        "synthetic": True,
        "not_for_filing": True,
        "template_name": template_name,
        "source_template": str(
            source_path.relative_to(PROJECT_ROOT)
        ),
        "source_page_number": int(
            template["source_page_number"]
        ),
        "pdf_path": str(
            pdf_path.relative_to(PROJECT_ROOT)
        ),
        "image_path": str(
            image_path.relative_to(PROJECT_ROOT)
        ),
        "render_dpi": dpi,
        "image_width": image_width,
        "image_height": image_height,
        "fields": values,
        "field_boxes_pdf_points": {
            name: config["rect"]
            for name, config in (
                template["fields"].items()
            )
        },
        "field_boxes_pixels": field_boxes_in_pixels(
            template["fields"],
            dpi,
        ),
    }

    write_json(
        annotation,
        annotation_path,
    )

    return {
        "document_id": document_id,
        "document_type": document_type,
        "split": split,
        "pdf_path": annotation["pdf_path"],
        "image_path": annotation["image_path"],
        "annotation_path": str(
            annotation_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "template_name": template_name,
        "render_dpi": dpi,
        "synthetic": True,
    }


def generate_dataset(
    *,
    w2_count: int,
    nec_count: int,
    seed: int,
    dpi: int,
) -> None:
    """Generate the complete clean synthetic IRS dataset."""

    random.seed(seed)
    Faker.seed(seed)

    coordinates = load_coordinates()
    manifest_rows: list[dict[str, Any]] = []

    total = w2_count + nec_count
    current = 0

    for index in range(1, w2_count + 1):
        current += 1
        document_id = f"w2_2026_{index:06d}"
        split = split_name()

        print(
            f"[{current}/{total}] "
            f"Generating {document_id} ({split})"
        )

        manifest_rows.append(
            generate_document(
                template_name="w2_2026",
                document_type="w2",
                values=generate_w2_values(index),
                document_id=document_id,
                split=split,
                dpi=dpi,
                coordinates=coordinates,
            )
        )

    for index in range(1, nec_count + 1):
        current += 1
        document_id = f"1099_nec_2026_{index:06d}"
        split = split_name()

        print(
            f"[{current}/{total}] "
            f"Generating {document_id} ({split})"
        )

        manifest_rows.append(
            generate_document(
                template_name="1099_nec_2026",
                document_type="1099_nec",
                values=generate_1099_values(index),
                document_id=document_id,
                split=split,
                dpi=dpi,
                coordinates=coordinates,
            )
        )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "document_id",
        "document_type",
        "split",
        "pdf_path",
        "image_path",
        "annotation_path",
        "template_name",
        "render_dpi",
        "synthetic",
    ]

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    split_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}

    for row in manifest_rows:
        split = str(row["split"])
        document_type = str(row["document_type"])

        split_counts[split] = (
            split_counts.get(split, 0) + 1
        )
        type_counts[document_type] = (
            type_counts.get(document_type, 0) + 1
        )

    print("\nSynthetic IRS dataset generated")
    print(f"Documents generated: {len(manifest_rows)}")
    print(f"Document types: {type_counts}")
    print(f"Split counts: {split_counts}")
    print(f"Dataset directory: {OUTPUT_ROOT}")
    print(f"Annotations: {ANNOTATION_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate clean synthetic W-2 and "
            "1099-NEC documents."
        )
    )

    parser.add_argument(
        "--w2-count",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--nec-count",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    generate_dataset(
        w2_count=arguments.w2_count,
        nec_count=arguments.nec_count,
        seed=arguments.seed,
        dpi=arguments.dpi,
    )
