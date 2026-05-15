from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

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
    crop_field as baseline_crop_field,
    preprocess_crop as baseline_preprocess_crop,
    run_ocr as baseline_run_ocr,
)

from src.extraction.irs_tesseract_enhanced import (
    ALPHANUMERIC_IDENTIFIER_FIELDS,
    MONEY_FIELDS,
    MULTILINE_FIELDS,
    NUMERIC_IDENTIFIER_FIELDS,
    STATE_FIELDS,
    ZIP_FIELDS,
    crop_field as enhanced_crop_field,
    evaluate_prediction,
    format_score,
    select_best_ocr,
)


PREDICTIONS_PATH = (
    REPORT_DIR / "irs_tesseract_hybrid_predictions.csv"
)

FIELD_METRICS_PATH = (
    REPORT_DIR / "irs_tesseract_hybrid_field_metrics.csv"
)

SUMMARY_PATH = (
    REPORT_DIR / "irs_tesseract_hybrid_summary.json"
)


def estimate_skew_correction(
    image: Image.Image,
    *,
    max_angle: float = 3.0,
    step: float = 0.5,
) -> float:
    """
    Estimate page-level skew without using synthetic augmentation metadata.

    The selected value is the correction angle that maximizes the variance of
    horizontal ink projections. IRS forms contain many horizontal text lines
    and rules, making this a useful lightweight baseline.
    """

    working = image.convert("L")

    target_width = 900

    if working.width > target_width:
        scale = target_width / working.width

        working = working.resize(
            (
                target_width,
                max(1, round(working.height * scale)),
            ),
            resample=Image.Resampling.BILINEAR,
        )

    working = ImageOps.autocontrast(
        working,
        cutoff=1,
    )

    binary = working.point(
        lambda pixel: 0 if pixel < 205 else 255
    )

    angles = np.arange(
        -max_angle,
        max_angle + step / 2.0,
        step,
    )

    scores: dict[float, float] = {}

    for angle in angles:
        rotated = binary.rotate(
            float(angle),
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=255,
        )

        ink = (
            np.asarray(rotated, dtype=np.uint8)
            < 128
        ).astype(np.float32)

        horizontal_projection = ink.sum(axis=1)

        score = float(
            np.var(horizontal_projection)
        )

        scores[float(angle)] = score

    zero_score = scores.get(0.0, 0.0)

    best_angle = max(
        scores,
        key=scores.get,
    )

    best_score = scores[best_angle]

    # Avoid unnecessary correction when the score improvement is negligible.
    if zero_score > 0 and best_score < zero_score * 1.015:
        return 0.0

    return round(
        float(best_angle),
        2,
    )


def baseline_candidate(
    image: Image.Image,
    box: list[int | float],
    field_name: str,
    skew_correction: float,
) -> tuple[str, float]:
    crop = baseline_crop_field(
        image,
        box,
        padding=5,
    )

    if abs(skew_correction) >= 0.05:
        crop = crop.rotate(
            skew_correction,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(255, 255, 255),
        )

    prepared = baseline_preprocess_crop(
        crop
    )

    return baseline_run_ocr(
        prepared,
        field_name=field_name,
    )


def enhanced_candidate(
    image: Image.Image,
    box: list[int | float],
    field_name: str,
    skew_correction: float,
) -> tuple[str, float, str, str]:
    crop = enhanced_crop_field(
        image,
        box,
        field_name,
    )

    if abs(skew_correction) >= 0.05:
        crop = crop.rotate(
            skew_correction,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(255, 255, 255),
        )

    return select_best_ocr(
        crop,
        field_name,
    )


def fallback_threshold(
    field_name: str,
) -> float:
    if (
        field_name in STATE_FIELDS
        or field_name in ZIP_FIELDS
        or field_name in MONEY_FIELDS
        or field_name in NUMERIC_IDENTIFIER_FIELDS
        or field_name in ALPHANUMERIC_IDENTIFIER_FIELDS
    ):
        return 72.0

    if field_name in MULTILINE_FIELDS:
        return 55.0

    return 62.0


def should_run_enhanced(
    text: str,
    confidence: float,
    field_name: str,
) -> bool:
    if not clean_text(text):
        return True

    if confidence < fallback_threshold(
        field_name
    ):
        return True

    field_format_score = format_score(
        text,
        field_name,
    )

    if (
        field_name in STATE_FIELDS
        or field_name in ZIP_FIELDS
        or field_name in MONEY_FIELDS
        or field_name in NUMERIC_IDENTIFIER_FIELDS
    ):
        return field_format_score < 1.0

    return field_format_score < 0.70


def candidate_score(
    text: str,
    confidence: float,
    field_name: str,
    *,
    baseline_preference: float = 0.0,
) -> float:
    if not clean_text(text):
        return -50.0

    return (
        confidence * 0.65
        + format_score(
            text,
            field_name,
        )
        * 35.0
        + baseline_preference
    )


def choose_candidate(
    *,
    baseline_text: str,
    baseline_confidence: float,
    enhanced_text: str,
    enhanced_confidence: float,
    field_name: str,
) -> tuple[str, float, str]:
    if not clean_text(baseline_text) and clean_text(
        enhanced_text
    ):
        return (
            enhanced_text,
            enhanced_confidence,
            "enhanced",
        )

    if not clean_text(enhanced_text):
        return (
            baseline_text,
            baseline_confidence,
            "baseline",
        )

    baseline_format = format_score(
        baseline_text,
        field_name,
    )

    enhanced_format = format_score(
        enhanced_text,
        field_name,
    )

    # Prefer a structurally valid candidate over an invalid one.
    if enhanced_format > baseline_format + 0.25:
        return (
            enhanced_text,
            enhanced_confidence,
            "enhanced",
        )

    if baseline_format > enhanced_format + 0.25:
        return (
            baseline_text,
            baseline_confidence,
            "baseline",
        )

    baseline_score = candidate_score(
        baseline_text,
        baseline_confidence,
        field_name,
        baseline_preference=3.0,
    )

    enhanced_score = candidate_score(
        enhanced_text,
        enhanced_confidence,
        field_name,
    )

    if enhanced_score > baseline_score + 1.0:
        return (
            enhanced_text,
            enhanced_confidence,
            "enhanced",
        )

    return (
        baseline_text,
        baseline_confidence,
        "baseline",
    )


def process_document(
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    document_id = clean_text(
        row["document_id"]
    )

    parent_document_id = clean_text(
        row["parent_document_id"]
    )

    document_type = clean_text(
        row["document_type"]
    )

    split = clean_text(
        row["split"]
    )

    quality = clean_text(
        row["quality"]
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

    results: list[dict[str, Any]] = []

    with Image.open(image_path) as source:
        image = source.convert("RGB")

        skew_correction = (
            estimate_skew_correction(image)
        )

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
                        "parent_document_id": parent_document_id,
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
                        "normalized_expected": "",
                        "normalized_prediction": "",
                        "exact_match": False,
                        "similarity": 0.0,
                        "ocr_confidence": 0.0,
                        "chosen_engine": "",
                        "baseline_prediction": "",
                        "baseline_confidence": 0.0,
                        "enhanced_prediction": "",
                        "enhanced_confidence": 0.0,
                        "estimated_skew_correction": (
                            skew_correction
                        ),
                        "status": "missing_box",
                        "error": "",
                    }
                )
                continue

            try:
                (
                    baseline_text,
                    baseline_confidence,
                ) = baseline_candidate(
                    image,
                    box,
                    field_name,
                    skew_correction,
                )

                enhanced_text = ""
                enhanced_confidence = 0.0

                if should_run_enhanced(
                    baseline_text,
                    baseline_confidence,
                    field_name,
                ):
                    (
                        enhanced_text,
                        enhanced_confidence,
                        _,
                        _,
                    ) = enhanced_candidate(
                        image,
                        box,
                        field_name,
                        skew_correction,
                    )

                    (
                        prediction,
                        confidence,
                        chosen_engine,
                    ) = choose_candidate(
                        baseline_text=baseline_text,
                        baseline_confidence=(
                            baseline_confidence
                        ),
                        enhanced_text=enhanced_text,
                        enhanced_confidence=(
                            enhanced_confidence
                        ),
                        field_name=field_name,
                    )

                else:
                    prediction = baseline_text
                    confidence = (
                        baseline_confidence
                    )
                    chosen_engine = "baseline"

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
                        "parent_document_id": parent_document_id,
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
                        "chosen_engine": chosen_engine,
                        "baseline_prediction": baseline_text,
                        "baseline_confidence": baseline_confidence,
                        "enhanced_prediction": enhanced_text,
                        "enhanced_confidence": enhanced_confidence,
                        "estimated_skew_correction": (
                            skew_correction
                        ),
                        "status": "success",
                        "error": "",
                    }
                )

            except Exception as exc:
                results.append(
                    {
                        "document_id": document_id,
                        "parent_document_id": parent_document_id,
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
                        "normalized_expected": "",
                        "normalized_prediction": "",
                        "exact_match": False,
                        "similarity": 0.0,
                        "ocr_confidence": 0.0,
                        "chosen_engine": "",
                        "baseline_prediction": "",
                        "baseline_confidence": 0.0,
                        "enhanced_prediction": "",
                        "enhanced_confidence": 0.0,
                        "estimated_skew_correction": (
                            skew_correction
                        ),
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

    empty_rate = (
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
        "empty_prediction_rate": round(
            float(empty_rate),
            4,
        ),
        "failed_fields": int(
            (
                dataframe["status"]
                != "success"
            ).sum()
        ),
    }


def run_hybrid(
    *,
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

    print("\nHybrid Tesseract OCR")
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
            enhanced_selection_rate=(
                "chosen_engine",
                lambda values: (
                    values.eq(
                        "enhanced"
                    ).mean()
                ),
            ),
        )
        .reset_index()
    )

    for column in [
        "exact_match_rate",
        "enhanced_selection_rate",
    ]:
        field_metrics[column] = (
            field_metrics[column].round(4)
        )

    for column in [
        "mean_similarity",
        "mean_ocr_confidence",
    ]:
        field_metrics[column] = (
            field_metrics[column].round(2)
        )

    field_metrics.to_csv(
        FIELD_METRICS_PATH,
        index=False,
    )

    engine_counts = {
        str(engine): int(count)
        for engine, count in (
            predictions[
                "chosen_engine"
            ]
            .value_counts()
            .items()
        )
    }

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
            str(document_type): metric_summary(
                document_df
            )
            for document_type, document_df in (
                predictions.groupby(
                    "document_type"
                )
            )
        },
        "chosen_engine_counts": (
            engine_counts
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
        "\nHybrid Tesseract completed"
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
        "Chosen engines: "
        f"{summary['chosen_engine_counts']}"
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
            "Run an automatic baseline-plus-fallback "
            "Tesseract OCR pipeline."
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
            "Use 0 to evaluate all documents "
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

    run_hybrid(
        split=arguments.split,
        max_documents_per_quality=(
            arguments.max_documents_per_quality
        ),
        workers=arguments.workers,
    )
