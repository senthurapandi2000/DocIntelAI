from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "irs_synthetic_manifest.csv"
)

VARIANT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "irs_scan_variants_manifest.csv"
)

DETAIL_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_scan_variants_validation_report.csv"
)

SUMMARY_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_scan_variants_validation_summary.json"
)

EXPECTED_PROFILES = {"mild", "hard"}
EXPECTED_SPLITS = {"train", "validation", "test"}
EXPECTED_DOCUMENT_TYPES = {"w2", "1099_nec"}
EXPECTED_VARIANTS_PER_PARENT = 2


def clean_text(value: Any) -> str:
    """Convert a value into a safe string."""

    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON annotation file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_image(
    path: Path,
) -> tuple[bool, int, int, str]:
    """Validate that a JPEG image opens and return its dimensions."""

    if not path.exists() or path.stat().st_size == 0:
        return False, 0, 0, "missing_or_empty"

    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            width, height = image.size
            image_format = str(image.format or "")

        return (
            bool(width > 0 and height > 0),
            int(width),
            int(height),
            image_format,
        )

    except Exception as exc:
        return False, 0, 0, str(exc)


def boxes_are_valid(
    boxes: dict[str, Any],
    *,
    width: int,
    height: int,
) -> bool:
    """Validate all field boxes against image boundaries."""

    if not boxes or width <= 0 or height <= 0:
        return False

    for box in boxes.values():
        if not isinstance(box, list) or len(box) != 4:
            return False

        try:
            x1, y1, x2, y2 = [
                int(round(float(value)))
                for value in box
            ]
        except (TypeError, ValueError):
            return False

        if not (
            0 <= x1 < x2 <= width
            and 0 <= y1 < y2 <= height
        ):
            return False

    return True


def run_validation() -> None:
    """Validate all generated IRS scan variants."""

    if not BASE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Base manifest not found: {BASE_MANIFEST_PATH}"
        )

    if not VARIANT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Variant manifest not found: {VARIANT_MANIFEST_PATH}"
        )

    base_manifest = pd.read_csv(BASE_MANIFEST_PATH)
    variant_manifest = pd.read_csv(VARIANT_MANIFEST_PATH)

    required_columns = {
        "document_id",
        "parent_document_id",
        "document_type",
        "split",
        "variant_profile",
        "image_path",
        "annotation_path",
        "rotation_degrees",
        "blur_radius",
        "noise_sigma",
        "jpeg_quality",
    }

    missing_columns = (
        required_columns - set(variant_manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Variant manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    base_lookup = (
        base_manifest[
            [
                "document_id",
                "document_type",
                "split",
            ]
        ]
        .drop_duplicates("document_id")
        .set_index("document_id")
        .to_dict(orient="index")
    )

    validation_rows: list[dict[str, Any]] = []

    for index, row in variant_manifest.iterrows():
        document_id = clean_text(row["document_id"])
        parent_document_id = clean_text(
            row["parent_document_id"]
        )
        document_type = clean_text(
            row["document_type"]
        )
        split = clean_text(row["split"])
        profile = clean_text(row["variant_profile"])

        image_path = (
            PROJECT_ROOT
            / clean_text(row["image_path"])
        )
        annotation_path = (
            PROJECT_ROOT
            / clean_text(row["annotation_path"])
        )

        print(
            f"[{index + 1}/{len(variant_manifest)}] "
            f"Validating {document_id}"
        )

        image_valid, width, height, image_result = (
            validate_image(image_path)
        )

        annotation_valid = False
        annotation: dict[str, Any] = {}
        annotation_error = ""

        if annotation_path.exists():
            try:
                annotation = load_json(annotation_path)
                annotation_valid = True
            except Exception as exc:
                annotation_error = str(exc)

        parent_record = base_lookup.get(
            parent_document_id
        )

        parent_exists = parent_record is not None

        split_matches_parent = bool(
            parent_record
            and clean_text(parent_record["split"]) == split
        )

        type_matches_parent = bool(
            parent_record
            and clean_text(
                parent_record["document_type"]
            ) == document_type
        )

        metadata_matches = bool(
            annotation.get("document_id")
            == document_id
            and annotation.get("parent_document_id")
            == parent_document_id
            and annotation.get("document_type")
            == document_type
            and annotation.get("split") == split
            and annotation.get("variant_profile")
            == profile
        )

        annotation_dimensions_match = bool(
            int(annotation.get("image_width", 0))
            == width
            and int(annotation.get("image_height", 0))
            == height
        )

        field_boxes_valid = boxes_are_valid(
            annotation.get(
                "field_boxes_pixels",
                {},
            ),
            width=width,
            height=height,
        )

        augmentation = annotation.get(
            "augmentation",
            {},
        )

        augmentation_present = bool(
            augmentation
            and "rotation_degrees" in augmentation
            and "blur_radius" in augmentation
            and "noise_sigma" in augmentation
            and "jpeg_quality" in augmentation
        )

        document_passed = bool(
            document_type in EXPECTED_DOCUMENT_TYPES
            and split in EXPECTED_SPLITS
            and profile in EXPECTED_PROFILES
            and image_valid
            and annotation_valid
            and parent_exists
            and split_matches_parent
            and type_matches_parent
            and metadata_matches
            and annotation_dimensions_match
            and field_boxes_valid
            and augmentation_present
        )

        validation_rows.append(
            {
                "document_id": document_id,
                "parent_document_id": parent_document_id,
                "document_type": document_type,
                "split": split,
                "variant_profile": profile,
                "image_exists": image_path.exists(),
                "image_valid": image_valid,
                "image_width": width,
                "image_height": height,
                "image_result": image_result,
                "annotation_exists": (
                    annotation_path.exists()
                ),
                "annotation_valid": annotation_valid,
                "parent_exists": parent_exists,
                "split_matches_parent": (
                    split_matches_parent
                ),
                "type_matches_parent": (
                    type_matches_parent
                ),
                "metadata_matches": metadata_matches,
                "annotation_dimensions_match": (
                    annotation_dimensions_match
                ),
                "field_boxes_valid": field_boxes_valid,
                "augmentation_present": (
                    augmentation_present
                ),
                "validation_passed": (
                    document_passed
                ),
                "error": annotation_error,
            }
        )

    validation_df = pd.DataFrame(
        validation_rows
    )

    DETAIL_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_df.to_csv(
        DETAIL_REPORT_PATH,
        index=False,
    )

    duplicate_variant_ids = int(
        variant_manifest["document_id"]
        .duplicated()
        .sum()
    )

    parent_variant_counts = (
        variant_manifest
        .groupby("parent_document_id")
        .size()
    )

    parents_with_wrong_variant_count = int(
        (
            parent_variant_counts
            != EXPECTED_VARIANTS_PER_PARENT
        ).sum()
    )

    parent_profile_counts = (
        variant_manifest
        .groupby(
            [
                "parent_document_id",
                "variant_profile",
            ]
        )
        .size()
        .unstack(fill_value=0)
    )

    for profile in EXPECTED_PROFILES:
        if profile not in parent_profile_counts.columns:
            parent_profile_counts[profile] = 0

    parents_missing_profile = int(
        (
            (
                parent_profile_counts["mild"]
                != 1
            )
            |
            (
                parent_profile_counts["hard"]
                != 1
            )
        ).sum()
    )

    profile_counts = {
        str(profile): int(count)
        for profile, count in (
            variant_manifest[
                "variant_profile"
            ]
            .value_counts()
            .items()
        )
    }

    split_counts = {
        str(split): int(count)
        for split, count in (
            variant_manifest["split"]
            .value_counts()
            .items()
        )
    }

    type_counts = {
        str(document_type): int(count)
        for document_type, count in (
            variant_manifest[
                "document_type"
            ]
            .value_counts()
            .items()
        )
    }

    expected_total_variants = (
        len(base_manifest)
        * EXPECTED_VARIANTS_PER_PARENT
    )

    summary = {
        "base_documents": int(
            len(base_manifest)
        ),
        "variants_checked": int(
            len(validation_df)
        ),
        "expected_variants": int(
            expected_total_variants
        ),
        "variants_passed": int(
            validation_df[
                "validation_passed"
            ].sum()
        ),
        "variants_failed": int(
            (
                ~validation_df[
                    "validation_passed"
                ]
            ).sum()
        ),
        "profile_counts": profile_counts,
        "split_counts": split_counts,
        "document_type_counts": type_counts,
        "duplicate_variant_ids": (
            duplicate_variant_ids
        ),
        "parents_with_wrong_variant_count": (
            parents_with_wrong_variant_count
        ),
        "parents_missing_mild_or_hard_profile": (
            parents_missing_profile
        ),
        "validation_passed": bool(
            len(validation_df)
            == expected_total_variants
            and validation_df[
                "validation_passed"
            ].all()
            and duplicate_variant_ids == 0
            and parents_with_wrong_variant_count == 0
            and parents_missing_profile == 0
        ),
    }

    with SUMMARY_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\nIRS scan-variant validation completed"
    )
    print(
        f"Base documents: "
        f"{summary['base_documents']}"
    )
    print(
        f"Variants checked: "
        f"{summary['variants_checked']}"
    )
    print(
        f"Expected variants: "
        f"{summary['expected_variants']}"
    )
    print(
        f"Variants passed: "
        f"{summary['variants_passed']}"
    )
    print(
        f"Variants failed: "
        f"{summary['variants_failed']}"
    )
    print(
        f"Profiles: "
        f"{summary['profile_counts']}"
    )
    print(
        f"Split counts: "
        f"{summary['split_counts']}"
    )
    print(
        f"Document types: "
        f"{summary['document_type_counts']}"
    )
    print(
        f"Duplicate variant IDs: "
        f"{summary['duplicate_variant_ids']}"
    )
    print(
        "Parents with incorrect variant count: "
        f"{summary['parents_with_wrong_variant_count']}"
    )
    print(
        "Parents missing mild or hard profile: "
        f"{summary['parents_missing_mild_or_hard_profile']}"
    )
    print(
        f"Validation passed: "
        f"{summary['validation_passed']}"
    )
    print(
        f"Detailed report: {DETAIL_REPORT_PATH}"
    )
    print(
        f"Summary report: {SUMMARY_REPORT_PATH}"
    )


if __name__ == "__main__":
    run_validation()
