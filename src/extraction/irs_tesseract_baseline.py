from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pytesseract import Output
from rapidfuzz import fuzz


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_MANIFEST_PATH = (
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

REPORT_DIR = PROJECT_ROOT / "reports"

PREDICTIONS_PATH = (
    REPORT_DIR / "irs_tesseract_baseline_predictions.csv"
)

FIELD_METRICS_PATH = (
    REPORT_DIR / "irs_tesseract_baseline_field_metrics.csv"
)

SUMMARY_PATH = (
    REPORT_DIR / "irs_tesseract_baseline_summary.json"
)

RANDOM_SEED = 42

W2_EVALUATION_FIELDS = [
    "employee_ssn",
    "employer_ein",
    "employer_name_address",
    "control_number",
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
    "employer_state_id",
    "state_wages",
    "state_income_tax",
]

NEC_EVALUATION_FIELDS = [
    "payer_name",
    "payer_street",
    "payer_city",
    "payer_state",
    "payer_zip",
    "payer_tin",
    "recipient_tin",
    "recipient_name",
    "recipient_street",
    "recipient_city",
    "recipient_state",
    "recipient_zip",
    "nonemployee_compensation",
    "federal_income_tax_withheld",
    "account_number",
    "state_tax_withheld",
    "state_payer_number",
    "state_income",
]

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
    "state_tax_withheld",
    "state_income",
}

IDENTIFIER_FIELDS = {
    "employee_ssn",
    "employer_ein",
    "control_number",
    "employer_state_id",
    "payer_tin",
    "recipient_tin",
    "account_number",
    "state_payer_number",
}

STATE_FIELDS = {
    "state",
    "payer_state",
    "recipient_state",
}

MULTILINE_FIELDS = {
    "employer_name_address",
    "employee_address",
}

NUMERIC_CHARACTER_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "|": "1",
        "S": "5",
        "B": "8",
    }
)


def configure_tesseract() -> Path:
    """Locate Tesseract on PATH, via env, or in the default Windows folder."""

    candidates: list[Path] = []

    configured = os.getenv("TESSERACT_CMD", "").strip()

    if configured:
        candidates.append(Path(configured))

    path_match = shutil.which("tesseract")

    if path_match:
        candidates.append(Path(path_match))

    candidates.append(
        Path(
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
    )

    for candidate in candidates:
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(
                candidate
            )
            return candidate

    raise FileNotFoundError(
        "Tesseract executable was not found. Set TESSERACT_CMD "
        "or install it under C:\\Program Files\\Tesseract-OCR."
    )


def clean_text(value: Any) -> str:
    """Convert a value into a safe string."""

    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def load_json(path: Path) -> dict[str, Any]:
    """Load one annotation JSON file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_evaluation_manifest(split: str) -> pd.DataFrame:
    """Combine clean, mild, and hard images into one evaluation manifest."""

    if not CLEAN_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Clean manifest not found: {CLEAN_MANIFEST_PATH}"
        )

    if not VARIANT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Variant manifest not found: {VARIANT_MANIFEST_PATH}"
        )

    clean_df = pd.read_csv(CLEAN_MANIFEST_PATH)

    clean_df = clean_df[
        clean_df["split"] == split
    ].copy()

    clean_df["quality"] = "clean"
    clean_df["parent_document_id"] = (
        clean_df["document_id"]
    )

    clean_df = clean_df[
        [
            "document_id",
            "parent_document_id",
            "document_type",
            "split",
            "quality",
            "image_path",
            "annotation_path",
        ]
    ]

    variant_df = pd.read_csv(
        VARIANT_MANIFEST_PATH
    )

    variant_df = variant_df[
        variant_df["split"] == split
    ].copy()

    variant_df["quality"] = (
        variant_df["variant_profile"]
    )

    variant_df = variant_df[
        [
            "document_id",
            "parent_document_id",
            "document_type",
            "split",
            "quality",
            "image_path",
            "annotation_path",
        ]
    ]

    combined = pd.concat(
        [clean_df, variant_df],
        ignore_index=True,
    )

    return combined


def balanced_sample(
    dataframe: pd.DataFrame,
    max_documents_per_quality: int,
) -> pd.DataFrame:
    """Take a deterministic document-type-balanced sample per quality."""

    if max_documents_per_quality <= 0:
        return dataframe.copy()

    sampled_groups: list[pd.DataFrame] = []

    for quality, quality_df in dataframe.groupby(
        "quality",
        sort=True,
    ):
        document_types = sorted(
            quality_df["document_type"].unique()
        )

        base_quota = (
            max_documents_per_quality
            // max(len(document_types), 1)
        )
        remainder = (
            max_documents_per_quality
            % max(len(document_types), 1)
        )

        selected_parts: list[pd.DataFrame] = []

        for index, document_type in enumerate(
            document_types
        ):
            type_df = quality_df[
                quality_df["document_type"]
                == document_type
            ]

            quota = base_quota + (
                1 if index < remainder else 0
            )

            quota = min(quota, len(type_df))

            if quota > 0:
                selected_parts.append(
                    type_df.sample(
                        n=quota,
                        random_state=(
                            RANDOM_SEED
                            + index
                            + len(quality)
                        ),
                    )
                )

        sampled_quality = pd.concat(
            selected_parts,
            ignore_index=True,
        )

        if (
            len(sampled_quality)
            < max_documents_per_quality
        ):
            remaining = quality_df[
                ~quality_df["document_id"].isin(
                    sampled_quality["document_id"]
                )
            ]

            additional = min(
                max_documents_per_quality
                - len(sampled_quality),
                len(remaining),
            )

            if additional > 0:
                sampled_quality = pd.concat(
                    [
                        sampled_quality,
                        remaining.sample(
                            n=additional,
                            random_state=(
                                RANDOM_SEED + 100
                            ),
                        ),
                    ],
                    ignore_index=True,
                )

        sampled_groups.append(sampled_quality)

    return pd.concat(
        sampled_groups,
        ignore_index=True,
    )


def evaluation_fields(
    document_type: str,
) -> list[str]:
    """Return the selected benchmark fields for a document type."""

    if document_type == "w2":
        return W2_EVALUATION_FIELDS

    if document_type == "1099_nec":
        return NEC_EVALUATION_FIELDS

    return []


def field_kind(field_name: str) -> str:
    """Categorize a field for OCR configuration and normalization."""

    if field_name in MONEY_FIELDS:
        return "money"

    if field_name in IDENTIFIER_FIELDS:
        return "identifier"

    if field_name in STATE_FIELDS:
        return "state"

    if field_name in MULTILINE_FIELDS:
        return "multiline"

    return "text"


def crop_field(
    image: Image.Image,
    box: list[int | float],
    *,
    padding: int = 5,
) -> Image.Image:
    """Crop a field safely with a small amount of context padding."""

    width, height = image.size

    x1, y1, x2, y2 = [
        int(round(float(value)))
        for value in box
    ]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid crop rectangle: {box}"
        )

    return image.crop((x1, y1, x2, y2))


def preprocess_crop(
    crop: Image.Image,
) -> Image.Image:
    """Upscale and enhance a field crop for the Tesseract baseline."""

    grayscale = ImageOps.grayscale(crop)
    grayscale = ImageOps.autocontrast(
        grayscale,
        cutoff=1,
    )

    enlarged = grayscale.resize(
        (
            max(1, grayscale.width * 3),
            max(1, grayscale.height * 3),
        ),
        resample=Image.Resampling.LANCZOS,
    )

    enlarged = ImageEnhance.Contrast(
        enlarged
    ).enhance(1.25)

    enlarged = enlarged.filter(
        ImageFilter.UnsharpMask(
            radius=1.0,
            percent=140,
            threshold=3,
        )
    )

    return enlarged


def tesseract_config(
    field_name: str,
) -> str:
    """Choose a Tesseract page-segmentation mode and whitelist."""

    kind = field_kind(field_name)

    if kind == "money":
        return (
            "--oem 3 --psm 7 "
            "-c tessedit_char_whitelist="
            "0123456789,.$()-"
        )

    if kind == "identifier":
        return (
            "--oem 3 --psm 7 "
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        )

    if kind == "state":
        return (
            "--oem 3 --psm 7 "
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

    if kind == "multiline":
        return "--oem 3 --psm 6"

    return "--oem 3 --psm 7"


def run_ocr(
    crop: Image.Image,
    *,
    field_name: str,
) -> tuple[str, float]:
    """Run Tesseract once and return joined text and mean confidence."""

    data = pytesseract.image_to_data(
        crop,
        lang="eng",
        config=tesseract_config(field_name),
        output_type=Output.DICT,
    )

    tokens: list[str] = []
    confidences: list[float] = []

    for text, confidence in zip(
        data.get("text", []),
        data.get("conf", []),
    ):
        token = clean_text(text)

        try:
            numeric_confidence = float(confidence)
        except (TypeError, ValueError):
            numeric_confidence = -1.0

        if token:
            tokens.append(token)

        if numeric_confidence >= 0:
            confidences.append(
                numeric_confidence
            )

    joined_text = " ".join(tokens).strip()

    mean_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )

    return joined_text, round(
        mean_confidence,
        2,
    )


def normalize_money(value: str) -> str:
    """Normalize OCR money text into a two-decimal representation."""

    text = value.upper().translate(
        NUMERIC_CHARACTER_TRANSLATION
    )

    text = text.replace(" ", "")
    text = text.replace("$", "")
    text = text.replace(",", "")

    negative = (
        text.startswith("(")
        and text.endswith(")")
    )

    text = re.sub(
        r"[^0-9.\-]",
        "",
        text,
    )

    if not text:
        return ""

    if text.count(".") > 1:
        first_dot = text.find(".")
        text = (
            text[: first_dot + 1]
            + text[first_dot + 1 :]
            .replace(".", "")
        )

    try:
        amount = Decimal(text)

        if negative:
            amount = -abs(amount)

        return f"{amount.quantize(Decimal('0.01')):.2f}"

    except InvalidOperation:
        return text


def normalize_identifier(value: str) -> str:
    """Normalize identifiers while ignoring separators."""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        value.upper(),
    )


def normalize_text_value(value: str) -> str:
    """Normalize names and addresses for comparison."""

    text = value.upper()
    text = text.replace("&", " AND ")

    text = re.sub(
        r"[^A-Z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def normalize_value(
    value: Any,
    *,
    field_name: str,
) -> str:
    """Normalize expected and OCR values using field-aware rules."""

    text = clean_text(value)
    kind = field_kind(field_name)

    if kind == "money":
        return normalize_money(text)

    if kind in {"identifier", "state"}:
        return normalize_identifier(text)

    return normalize_text_value(text)


def evaluate_field(
    *,
    expected_value: Any,
    predicted_value: str,
    field_name: str,
) -> tuple[str, str, bool, float]:
    """Compare one OCR prediction with ground truth."""

    normalized_expected = normalize_value(
        expected_value,
        field_name=field_name,
    )

    normalized_prediction = normalize_value(
        predicted_value,
        field_name=field_name,
    )

    exact_match = bool(
        normalized_expected
        and normalized_expected
        == normalized_prediction
    )

    similarity = float(
        fuzz.ratio(
            normalized_expected,
            normalized_prediction,
        )
    )

    return (
        normalized_expected,
        normalized_prediction,
        exact_match,
        round(similarity, 2),
    )


def process_document(
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    """OCR all selected fields in one annotated document."""

    document_id = clean_text(
        row["document_id"]
    )
    document_type = clean_text(
        row["document_type"]
    )
    quality = clean_text(row["quality"])
    split = clean_text(row["split"])

    image_path = (
        PROJECT_ROOT
        / clean_text(row["image_path"])
    )

    annotation_path = (
        PROJECT_ROOT
        / clean_text(row["annotation_path"])
    )

    annotation = load_json(
        annotation_path
    )

    fields = annotation.get("fields", {})
    boxes = annotation.get(
        "field_boxes_pixels",
        {},
    )

    results: list[dict[str, Any]] = []

    with Image.open(image_path) as source:
        image = source.convert("RGB")

        for field_name in evaluation_fields(
            document_type
        ):
            expected_value = fields.get(
                field_name,
                "",
            )

            box = boxes.get(field_name)

            if box is None:
                results.append(
                    {
                        "document_id": document_id,
                        "parent_document_id": clean_text(
                            row[
                                "parent_document_id"
                            ]
                        ),
                        "document_type": document_type,
                        "split": split,
                        "quality": quality,
                        "field_name": field_name,
                        "field_kind": field_kind(
                            field_name
                        ),
                        "expected_value": clean_text(
                            expected_value
                        ),
                        "predicted_value": "",
                        "normalized_expected": (
                            normalize_value(
                                expected_value,
                                field_name=field_name,
                            )
                        ),
                        "normalized_prediction": "",
                        "exact_match": False,
                        "similarity": 0.0,
                        "ocr_confidence": 0.0,
                        "status": "missing_box",
                        "error": "",
                    }
                )
                continue

            try:
                crop = crop_field(
                    image,
                    box,
                )

                prepared_crop = preprocess_crop(
                    crop
                )

                predicted_value, confidence = (
                    run_ocr(
                        prepared_crop,
                        field_name=field_name,
                    )
                )

                (
                    normalized_expected,
                    normalized_prediction,
                    exact_match,
                    similarity,
                ) = evaluate_field(
                    expected_value=expected_value,
                    predicted_value=predicted_value,
                    field_name=field_name,
                )

                results.append(
                    {
                        "document_id": document_id,
                        "parent_document_id": clean_text(
                            row[
                                "parent_document_id"
                            ]
                        ),
                        "document_type": document_type,
                        "split": split,
                        "quality": quality,
                        "field_name": field_name,
                        "field_kind": field_kind(
                            field_name
                        ),
                        "expected_value": clean_text(
                            expected_value
                        ),
                        "predicted_value": (
                            predicted_value
                        ),
                        "normalized_expected": (
                            normalized_expected
                        ),
                        "normalized_prediction": (
                            normalized_prediction
                        ),
                        "exact_match": exact_match,
                        "similarity": similarity,
                        "ocr_confidence": confidence,
                        "status": "success",
                        "error": "",
                    }
                )

            except Exception as exc:
                results.append(
                    {
                        "document_id": document_id,
                        "parent_document_id": clean_text(
                            row[
                                "parent_document_id"
                            ]
                        ),
                        "document_type": document_type,
                        "split": split,
                        "quality": quality,
                        "field_name": field_name,
                        "field_kind": field_kind(
                            field_name
                        ),
                        "expected_value": clean_text(
                            expected_value
                        ),
                        "predicted_value": "",
                        "normalized_expected": (
                            normalize_value(
                                expected_value,
                                field_name=field_name,
                            )
                        ),
                        "normalized_prediction": "",
                        "exact_match": False,
                        "similarity": 0.0,
                        "ocr_confidence": 0.0,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    return results


def metric_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate aggregate OCR metrics for one dataframe."""

    if dataframe.empty:
        return {
            "field_predictions": 0,
            "exact_match_rate": 0.0,
            "mean_similarity": 0.0,
            "mean_ocr_confidence": 0.0,
            "failed_fields": 0,
        }

    return {
        "field_predictions": int(
            len(dataframe)
        ),
        "exact_match_rate": round(
            float(
                dataframe[
                    "exact_match"
                ].mean()
            ),
            4,
        ),
        "mean_similarity": round(
            float(
                dataframe[
                    "similarity"
                ].mean()
            ),
            2,
        ),
        "mean_ocr_confidence": round(
            float(
                dataframe[
                    "ocr_confidence"
                ].mean()
            ),
            2,
        ),
        "failed_fields": int(
            (
                dataframe["status"]
                != "success"
            ).sum()
        ),
    }


def train_baseline(
    *,
    split: str,
    max_documents_per_quality: int,
    workers: int,
) -> None:
    """Run the field-level Tesseract OCR benchmark."""

    tesseract_path = configure_tesseract()

    evaluation_df = load_evaluation_manifest(
        split
    )

    evaluation_df = balanced_sample(
        evaluation_df,
        max_documents_per_quality,
    )

    if evaluation_df.empty:
        raise ValueError(
            f"No documents found for split: {split}"
        )

    quality_counts = {
        str(quality): int(count)
        for quality, count in (
            evaluation_df["quality"]
            .value_counts()
            .items()
        )
    }

    print("\nTesseract OCR baseline")
    print(f"Tesseract: {tesseract_path}")
    print(f"Split: {split}")
    print(
        f"Documents selected: "
        f"{len(evaluation_df)}"
    )
    print(
        f"Documents by quality: "
        f"{quality_counts}"
    )
    print(f"Workers: {workers}")

    records = evaluation_df.to_dict(
        orient="records"
    )

    prediction_rows: list[
        dict[str, Any]
    ] = []

    with ThreadPoolExecutor(
        max_workers=max(1, workers)
    ) as executor:
        futures = {
            executor.submit(
                process_document,
                record,
            ): record["document_id"]
            for record in records
        }

        completed = 0

        for future in as_completed(futures):
            completed += 1
            document_id = futures[future]

            try:
                prediction_rows.extend(
                    future.result()
                )

                print(
                    f"[{completed}/{len(records)}] "
                    f"OCR completed: {document_id}"
                )

            except Exception as exc:
                print(
                    f"[{completed}/{len(records)}] "
                    f"OCR failed: {document_id}: {exc}"
                )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    if predictions_df.empty:
        raise ValueError(
            "No field-level OCR predictions were produced."
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    field_metrics_df = (
        predictions_df
        .groupby(
            [
                "quality",
                "document_type",
                "field_name",
            ],
            dropna=False,
        )
        .agg(
            field_predictions=(
                "field_name",
                "size",
            ),
            exact_match_rate=(
                "exact_match",
                "mean",
            ),
            mean_similarity=(
                "similarity",
                "mean",
            ),
            mean_ocr_confidence=(
                "ocr_confidence",
                "mean",
            ),
        )
        .reset_index()
    )

    field_metrics_df[
        "exact_match_rate"
    ] = (
        field_metrics_df[
            "exact_match_rate"
        ].round(4)
    )

    field_metrics_df[
        "mean_similarity"
    ] = (
        field_metrics_df[
            "mean_similarity"
        ].round(2)
    )

    field_metrics_df[
        "mean_ocr_confidence"
    ] = (
        field_metrics_df[
            "mean_ocr_confidence"
        ].round(2)
    )

    field_metrics_df.to_csv(
        FIELD_METRICS_PATH,
        index=False,
    )

    by_quality = {
        str(quality): metric_summary(
            quality_df
        )
        for quality, quality_df in (
            predictions_df.groupby(
                "quality"
            )
        )
    }

    by_document_type = {
        str(document_type): metric_summary(
            document_df
        )
        for document_type, document_df in (
            predictions_df.groupby(
                "document_type"
            )
        )
    }

    summary = {
        "split": split,
        "documents_evaluated": int(
            predictions_df[
                "document_id"
            ].nunique()
        ),
        "field_predictions": int(
            len(predictions_df)
        ),
        "max_documents_per_quality": int(
            max_documents_per_quality
        ),
        "workers": int(workers),
        "overall": metric_summary(
            predictions_df
        ),
        "by_quality": by_quality,
        "by_document_type": (
            by_document_type
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

    print("\nTesseract baseline completed")
    print(
        f"Documents evaluated: "
        f"{summary['documents_evaluated']}"
    )
    print(
        f"Field predictions: "
        f"{summary['field_predictions']}"
    )
    print(
        "Overall exact-match rate: "
        f"{summary['overall']['exact_match_rate']:.2%}"
    )
    print(
        "Overall mean similarity: "
        f"{summary['overall']['mean_similarity']:.2f}"
    )

    for quality, metrics in (
        summary["by_quality"].items()
    ):
        print(
            f"{quality}: exact match "
            f"{metrics['exact_match_rate']:.2%}, "
            f"similarity "
            f"{metrics['mean_similarity']:.2f}"
        )

    print(
        f"Predictions: {PREDICTIONS_PATH}"
    )
    print(
        f"Field metrics: {FIELD_METRICS_PATH}"
    )
    print(f"Summary: {SUMMARY_PATH}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Tesseract OCR on clean, mild, "
            "and hard IRS document fields."
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "test",
        ],
        default="test",
    )

    parser.add_argument(
        "--max-documents-per-quality",
        type=int,
        default=6,
        help=(
            "Use 0 to evaluate every document "
            "in the selected split."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    train_baseline(
        split=arguments.split,
        max_documents_per_quality=(
            arguments.max_documents_per_quality
        ),
        workers=arguments.workers,
    )
