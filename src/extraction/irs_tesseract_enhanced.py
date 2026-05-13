from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pytesseract import Output
from rapidfuzz import fuzz

from src.extraction.irs_tesseract_baseline import (
    PROJECT_ROOT,
    REPORT_DIR,
    balanced_sample,
    clean_text,
    configure_tesseract,
    evaluation_fields,
    field_kind,
    load_evaluation_manifest,
    load_json,
)


PREDICTIONS_PATH = (
    REPORT_DIR / "irs_tesseract_enhanced_predictions.csv"
)

FIELD_METRICS_PATH = (
    REPORT_DIR / "irs_tesseract_enhanced_field_metrics.csv"
)

SUMMARY_PATH = (
    REPORT_DIR / "irs_tesseract_enhanced_summary.json"
)

ZIP_FIELDS = {
    "payer_zip",
    "recipient_zip",
}

NUMERIC_IDENTIFIER_FIELDS = {
    "employee_ssn",
    "employer_ein",
    "payer_tin",
    "recipient_tin",
}

ALPHANUMERIC_IDENTIFIER_FIELDS = {
    "control_number",
    "employer_state_id",
    "account_number",
    "state_payer_number",
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

STATE_FIELDS = {
    "state",
    "payer_state",
    "recipient_state",
}

MULTILINE_FIELDS = {
    "employer_name_address",
    "employee_address",
}

NUMERIC_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "|": "1",
        "S": "5",
        "B": "8",
        "G": "6",
        "Z": "2",
    }
)


def normalize_money(value: Any) -> str:
    text = clean_text(value).upper().translate(
        NUMERIC_TRANSLATION
    )

    negative = (
        text.startswith("(")
        and text.endswith(")")
    )

    text = (
        text.replace("$", "")
        .replace(",", "")
        .replace(" ", "")
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
            + text[first_dot + 1 :].replace(".", "")
        )

    try:
        amount = Decimal(text)

        if negative:
            amount = -abs(amount)

        return f"{amount.quantize(Decimal('0.01')):.2f}"

    except InvalidOperation:
        return text


def normalize_numeric_identifier(value: Any) -> str:
    text = clean_text(value).upper().translate(
        NUMERIC_TRANSLATION
    )

    return re.sub(
        r"[^0-9]",
        "",
        text,
    )


def normalize_alphanumeric_identifier(value: Any) -> str:
    return re.sub(
        r"[^A-Z0-9]",
        "",
        clean_text(value).upper(),
    )


def normalize_text(value: Any) -> str:
    text = clean_text(value).upper()
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
    field_name: str,
) -> str:
    if field_name in MONEY_FIELDS:
        return normalize_money(value)

    if (
        field_name in NUMERIC_IDENTIFIER_FIELDS
        or field_name in ZIP_FIELDS
    ):
        return normalize_numeric_identifier(value)

    if field_name in ALPHANUMERIC_IDENTIFIER_FIELDS:
        return normalize_alphanumeric_identifier(value)

    if field_name in STATE_FIELDS:
        return re.sub(
            r"[^A-Z]",
            "",
            clean_text(value).upper(),
        )

    return normalize_text(value)


def evaluate_prediction(
    expected_value: Any,
    predicted_value: Any,
    field_name: str,
) -> tuple[str, str, bool, float]:
    expected = normalize_value(
        expected_value,
        field_name,
    )

    prediction = normalize_value(
        predicted_value,
        field_name,
    )

    exact = bool(
        expected
        and expected == prediction
    )

    similarity = float(
        fuzz.ratio(
            expected,
            prediction,
        )
    )

    return (
        expected,
        prediction,
        exact,
        round(similarity, 2),
    )


def crop_padding(field_name: str) -> int:
    if field_name in STATE_FIELDS:
        return 12

    if field_name in ZIP_FIELDS:
        return 12

    if (
        field_name in NUMERIC_IDENTIFIER_FIELDS
        or field_name in ALPHANUMERIC_IDENTIFIER_FIELDS
    ):
        return 9

    if field_name in MONEY_FIELDS:
        return 8

    if field_name in MULTILINE_FIELDS:
        return 10

    return 7


def crop_field(
    image: Image.Image,
    box: list[int | float],
    field_name: str,
) -> Image.Image:
    width, height = image.size
    padding = crop_padding(field_name)

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
            f"Invalid crop for {field_name}: {box}"
        )

    return image.crop(
        (x1, y1, x2, y2)
    )


def deskew_crop(
    crop: Image.Image,
    angle_degrees: float,
) -> Image.Image:
    if abs(angle_degrees) < 0.05:
        return crop

    return crop.rotate(
        -angle_degrees,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(255, 255, 255),
    )


def otsu_threshold(image: Image.Image) -> Image.Image:
    array = np.asarray(
        ImageOps.grayscale(image),
        dtype=np.uint8,
    )

    histogram = np.bincount(
        array.ravel(),
        minlength=256,
    ).astype(np.float64)

    total = array.size
    sum_total = np.dot(
        np.arange(256),
        histogram,
    )

    sum_background = 0.0
    weight_background = 0.0
    maximum_variance = -1.0
    threshold = 127

    for candidate in range(256):
        weight_background += histogram[candidate]

        if weight_background == 0:
            continue

        weight_foreground = (
            total - weight_background
        )

        if weight_foreground == 0:
            break

        sum_background += (
            candidate
            * histogram[candidate]
        )

        mean_background = (
            sum_background
            / weight_background
        )

        mean_foreground = (
            sum_total - sum_background
        ) / weight_foreground

        between_class_variance = (
            weight_background
            * weight_foreground
            * (
                mean_background
                - mean_foreground
            )
            ** 2
        )

        if (
            between_class_variance
            > maximum_variance
        ):
            maximum_variance = (
                between_class_variance
            )
            threshold = candidate

    binary = np.where(
        array > threshold,
        255,
        0,
    ).astype(np.uint8)

    return Image.fromarray(
        binary,
        mode="L",
    )


def prepare_candidates(
    crop: Image.Image,
    field_name: str,
) -> list[tuple[str, Image.Image]]:
    grayscale = ImageOps.grayscale(crop)
    grayscale = ImageOps.autocontrast(
        grayscale,
        cutoff=1,
    )

    scale = 4 if (
        field_name in STATE_FIELDS
        or field_name in ZIP_FIELDS
        or field_name in NUMERIC_IDENTIFIER_FIELDS
    ) else 3

    resized = grayscale.resize(
        (
            max(1, grayscale.width * scale),
            max(1, grayscale.height * scale),
        ),
        resample=Image.Resampling.LANCZOS,
    )

    sharpened = ImageEnhance.Contrast(
        resized
    ).enhance(1.35)

    sharpened = sharpened.filter(
        ImageFilter.UnsharpMask(
            radius=1.2,
            percent=170,
            threshold=2,
        )
    )

    thresholded = otsu_threshold(
        resized
    )

    denoised = resized.filter(
        ImageFilter.MedianFilter(size=3)
    )

    candidates = [
        ("sharpened", sharpened),
        ("thresholded", thresholded),
    ]

    if (
        field_name in MULTILINE_FIELDS
        or field_name not in (
            MONEY_FIELDS
            | STATE_FIELDS
            | ZIP_FIELDS
            | NUMERIC_IDENTIFIER_FIELDS
            | ALPHANUMERIC_IDENTIFIER_FIELDS
        )
    ):
        candidates.append(
            ("denoised", denoised)
        )

    return candidates


def configs_for_field(
    field_name: str,
) -> list[str]:
    if field_name in MONEY_FIELDS:
        return [
            (
                "--oem 3 --psm 7 "
                "-c tessedit_char_whitelist="
                "0123456789,.$()-"
            ),
            (
                "--oem 3 --psm 8 "
                "-c tessedit_char_whitelist="
                "0123456789,.$()-"
            ),
        ]

    if field_name in ZIP_FIELDS:
        return [
            (
                "--oem 3 --psm 8 "
                "-c tessedit_char_whitelist="
                "0123456789-"
            ),
            (
                "--oem 3 --psm 10 "
                "-c tessedit_char_whitelist="
                "0123456789-"
            ),
        ]

    if field_name in STATE_FIELDS:
        return [
            (
                "--oem 3 --psm 8 "
                "-c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ),
            (
                "--oem 3 --psm 10 "
                "-c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ),
        ]

    if field_name in NUMERIC_IDENTIFIER_FIELDS:
        return [
            (
                "--oem 3 --psm 7 "
                "-c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            ),
            (
                "--oem 3 --psm 8 "
                "-c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            ),
        ]

    if field_name in ALPHANUMERIC_IDENTIFIER_FIELDS:
        return [
            (
                "--oem 3 --psm 7 "
                "-c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            ),
            (
                "--oem 3 --psm 8 "
                "-c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            ),
        ]

    if field_name in MULTILINE_FIELDS:
        return [
            "--oem 3 --psm 6",
            "--oem 3 --psm 11",
        ]

    return [
        "--oem 3 --psm 7",
        "--oem 3 --psm 6",
    ]


def format_score(
    text: str,
    field_name: str,
) -> float:
    normalized = normalize_value(
        text,
        field_name,
    )

    if not normalized:
        return 0.0

    if field_name in ZIP_FIELDS:
        return (
            1.0
            if len(normalized) in {5, 9}
            else max(
                0.1,
                1.0
                - abs(len(normalized) - 5)
                * 0.18,
            )
        )

    if field_name in STATE_FIELDS:
        return (
            1.0
            if len(normalized) == 2
            else 0.15
        )

    if field_name in NUMERIC_IDENTIFIER_FIELDS:
        return (
            1.0
            if len(normalized) == 9
            else max(
                0.1,
                1.0
                - abs(len(normalized) - 9)
                * 0.12,
            )
        )

    if field_name in ALPHANUMERIC_IDENTIFIER_FIELDS:
        return (
            1.0
            if 5 <= len(normalized) <= 24
            else 0.35
        )

    if field_name in MONEY_FIELDS:
        try:
            Decimal(normalized)
            return 1.0
        except InvalidOperation:
            return 0.2

    token_count = len(
        normalized.split()
    )

    if field_name in MULTILINE_FIELDS:
        return min(
            1.0,
            token_count / 4.0,
        )

    return min(
        1.0,
        max(0.25, token_count / 2.0),
    )


def ocr_candidate(
    image: Image.Image,
    config: str,
) -> tuple[str, float]:
    data = pytesseract.image_to_data(
        image,
        lang="eng",
        config=config,
        output_type=Output.DICT,
    )

    tokens: list[str] = []
    confidences: list[float] = []

    for token, confidence in zip(
        data.get("text", []),
        data.get("conf", []),
    ):
        token_text = clean_text(token)

        try:
            confidence_value = float(
                confidence
            )
        except (TypeError, ValueError):
            confidence_value = -1.0

        if token_text:
            tokens.append(token_text)

        if confidence_value >= 0:
            confidences.append(
                confidence_value
            )

    text = " ".join(tokens).strip()

    mean_confidence = (
        sum(confidences)
        / len(confidences)
        if confidences
        else 0.0
    )

    return text, round(
        mean_confidence,
        2,
    )


def select_best_ocr(
    crop: Image.Image,
    field_name: str,
) -> tuple[str, float, str, str]:
    best_text = ""
    best_confidence = 0.0
    best_candidate_name = ""
    best_config = ""
    best_score = -math.inf

    for candidate_name, candidate_image in (
        prepare_candidates(
            crop,
            field_name,
        )
    ):
        for config in configs_for_field(
            field_name
        ):
            text, confidence = ocr_candidate(
                candidate_image,
                config,
            )

            score = (
                confidence * 0.65
                + format_score(
                    text,
                    field_name,
                )
                * 35.0
            )

            if not text:
                score -= 30.0

            if score > best_score:
                best_score = score
                best_text = text
                best_confidence = confidence
                best_candidate_name = (
                    candidate_name
                )
                best_config = config

    return (
        best_text,
        best_confidence,
        best_candidate_name,
        best_config,
    )


def process_document(
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    document_id = clean_text(
        row["document_id"]
    )
    document_type = clean_text(
        row["document_type"]
    )
    quality = clean_text(
        row["quality"]
    )
    split = clean_text(
        row["split"]
    )

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

    fields = annotation.get(
        "fields",
        {},
    )

    boxes = annotation.get(
        "field_boxes_pixels",
        {},
    )

    augmentation = annotation.get(
        "augmentation",
        {},
    )

    rotation_degrees = float(
        augmentation.get(
            "rotation_degrees",
            0.0,
        )
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

            box = boxes.get(
                field_name
            )

            if box is None:
                results.append(
                    {
                        "document_id": document_id,
                        "parent_document_id": clean_text(
                            row["parent_document_id"]
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
                        "normalized_expected": normalize_value(
                            expected_value,
                            field_name,
                        ),
                        "normalized_prediction": "",
                        "exact_match": False,
                        "similarity": 0.0,
                        "ocr_confidence": 0.0,
                        "selected_preprocessing": "",
                        "selected_config": "",
                        "rotation_corrected": rotation_degrees,
                        "status": "missing_box",
                        "error": "",
                    }
                )
                continue

            try:
                crop = crop_field(
                    image,
                    box,
                    field_name,
                )

                crop = deskew_crop(
                    crop,
                    rotation_degrees,
                )

                (
                    prediction,
                    confidence,
                    selected_preprocessing,
                    selected_config,
                ) = select_best_ocr(
                    crop,
                    field_name,
                )

                (
                    normalized_expected,
                    normalized_prediction,
                    exact_match,
                    similarity,
                ) = evaluate_prediction(
                    expected_value,
                    prediction,
                    field_name,
                )

                results.append(
                    {
                        "document_id": document_id,
                        "parent_document_id": clean_text(
                            row["parent_document_id"]
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
                        "predicted_value": prediction,
                        "normalized_expected": normalized_expected,
                        "normalized_prediction": normalized_prediction,
                        "exact_match": exact_match,
                        "similarity": similarity,
                        "ocr_confidence": confidence,
                        "selected_preprocessing": selected_preprocessing,
                        "selected_config": selected_config,
                        "rotation_corrected": rotation_degrees,
                        "status": "success",
                        "error": "",
                    }
                )

            except Exception as exc:
                results.append(
                    {
                        "document_id": document_id,
                        "parent_document_id": clean_text(
                            row["parent_document_id"]
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
                        "normalized_expected": normalize_value(
                            expected_value,
                            field_name,
                        ),
                        "normalized_prediction": "",
                        "exact_match": False,
                        "similarity": 0.0,
                        "ocr_confidence": 0.0,
                        "selected_preprocessing": "",
                        "selected_config": "",
                        "rotation_corrected": rotation_degrees,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    return results


def metric_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    if dataframe.empty:
        return {
            "field_predictions": 0,
            "exact_match_rate": 0.0,
            "mean_similarity": 0.0,
            "mean_ocr_confidence": 0.0,
            "empty_prediction_rate": 0.0,
            "failed_fields": 0,
        }

    empty_prediction_rate = (
        dataframe[
            "normalized_prediction"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .mean()
    )

    return {
        "field_predictions": int(
            len(dataframe)
        ),
        "exact_match_rate": round(
            float(
                dataframe["exact_match"].mean()
            ),
            4,
        ),
        "mean_similarity": round(
            float(
                dataframe["similarity"].mean()
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
        "empty_prediction_rate": round(
            float(empty_prediction_rate),
            4,
        ),
        "failed_fields": int(
            (
                dataframe["status"]
                != "success"
            ).sum()
        ),
    }


def run_enhanced_baseline(
    split: str,
    max_documents_per_quality: int,
    workers: int,
) -> None:
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
            evaluation_df[
                "quality"
            ].value_counts().items()
        )
    }

    print(
        "\nEnhanced Tesseract OCR"
    )
    print(
        f"Tesseract: {tesseract_path}"
    )
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

        for future in as_completed(
            futures
        ):
            completed += 1
            document_id = futures[
                future
            ]

            try:
                prediction_rows.extend(
                    future.result()
                )

                print(
                    f"[{completed}/{len(records)}] "
                    f"OCR completed: "
                    f"{document_id}"
                )

            except Exception as exc:
                print(
                    f"[{completed}/{len(records)}] "
                    f"OCR failed: "
                    f"{document_id}: {exc}"
                )

    predictions = pd.DataFrame(
        prediction_rows
    )

    if predictions.empty:
        raise ValueError(
            "No OCR predictions were produced."
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    field_metrics = (
        predictions
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

    field_metrics[
        "exact_match_rate"
    ] = (
        field_metrics[
            "exact_match_rate"
        ].round(4)
    )

    field_metrics[
        "mean_similarity"
    ] = (
        field_metrics[
            "mean_similarity"
        ].round(2)
    )

    field_metrics[
        "mean_ocr_confidence"
    ] = (
        field_metrics[
            "mean_ocr_confidence"
        ].round(2)
    )

    field_metrics.to_csv(
        FIELD_METRICS_PATH,
        index=False,
    )

    summary = {
        "split": split,
        "documents_evaluated": int(
            predictions[
                "document_id"
            ].nunique()
        ),
        "field_predictions": int(
            len(predictions)
        ),
        "overall": metric_summary(
            predictions
        ),
        "by_quality": {
            str(quality): metric_summary(
                quality_df
            )
            for quality, quality_df in (
                predictions.groupby(
                    "quality"
                )
            )
        },
        "by_document_type": {
            str(document_type): (
                metric_summary(
                    document_df
                )
            )
            for document_type, document_df in (
                predictions.groupby(
                    "document_type"
                )
            )
        },
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
        "\nEnhanced Tesseract completed"
    )
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
    print(
        "Overall empty-prediction rate: "
        f"{summary['overall']['empty_prediction_rate']:.2%}"
    )

    for quality, metrics in (
        summary["by_quality"].items()
    ):
        print(
            f"{quality}: exact match "
            f"{metrics['exact_match_rate']:.2%}, "
            f"similarity "
            f"{metrics['mean_similarity']:.2f}, "
            f"empty "
            f"{metrics['empty_prediction_rate']:.2%}"
        )

    print(
        f"Predictions: {PREDICTIONS_PATH}"
    )
    print(
        f"Field metrics: {FIELD_METRICS_PATH}"
    )
    print(
        f"Summary: {SUMMARY_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run enhanced, field-aware Tesseract OCR "
            "on IRS synthetic document crops."
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "test",
        ],
        default="validation",
    )

    parser.add_argument(
        "--max-documents-per-quality",
        type=int,
        default=6,
        help=(
            "Use 0 to process every document "
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

    run_enhanced_baseline(
        split=arguments.split,
        max_documents_per_quality=(
            arguments.max_documents_per_quality
        ),
        workers=arguments.workers,
    )
