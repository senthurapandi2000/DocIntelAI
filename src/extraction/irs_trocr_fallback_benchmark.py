from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.extraction.irs_tesseract_baseline import (
    PROJECT_ROOT,
    clean_text,
    load_json,
)
from src.extraction.irs_tesseract_enhanced import (
    MULTILINE_FIELDS,
    crop_field,
    evaluate_prediction,
    format_score,
)


MODEL_NAME = "microsoft/trocr-base-printed"

DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_tesseract_hybrid_predictions.csv"
)

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

OUTPUT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_trocr_fallback_benchmark_predictions.csv"
)

OUTPUT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_trocr_fallback_benchmark_summary.json"
)

RANDOM_SEED = 42


def load_document_lookup() -> dict[str, dict[str, Any]]:
    if not CLEAN_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Clean manifest not found: {CLEAN_MANIFEST_PATH}"
        )

    if not VARIANT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Variant manifest not found: {VARIANT_MANIFEST_PATH}"
        )

    clean_df = pd.read_csv(CLEAN_MANIFEST_PATH).copy()
    clean_df["quality"] = "clean"

    variant_df = pd.read_csv(VARIANT_MANIFEST_PATH).copy()
    variant_df["quality"] = variant_df["variant_profile"]

    columns = [
        "document_id",
        "document_type",
        "split",
        "quality",
        "image_path",
        "annotation_path",
    ]

    combined = pd.concat(
        [
            clean_df[columns],
            variant_df[columns],
        ],
        ignore_index=True,
    )

    return (
        combined
        .drop_duplicates("document_id")
        .set_index("document_id")
        .to_dict(orient="index")
    )


def should_send_to_trocr(
    row: pd.Series,
    *,
    confidence_threshold: float,
) -> bool:
    field_name = clean_text(row["field_name"])

    if field_name in MULTILINE_FIELDS:
        return False

    prediction = clean_text(
        row.get("predicted_value", "")
    )

    confidence = float(
        row.get("ocr_confidence", 0.0)
    )

    if not prediction:
        return True

    if confidence < confidence_threshold:
        return True

    if format_score(
        prediction,
        field_name,
    ) < 1.0:
        return True

    return False


def balanced_limit(
    dataframe: pd.DataFrame,
    *,
    max_fields: int,
) -> pd.DataFrame:
    if max_fields <= 0 or len(dataframe) <= max_fields:
        return dataframe.copy()

    quality_names = sorted(
        dataframe["quality"].dropna().unique()
    )

    base_quota = max_fields // max(
        1,
        len(quality_names),
    )

    remainder = max_fields % max(
        1,
        len(quality_names),
    )

    selected_parts: list[pd.DataFrame] = []

    for index, quality in enumerate(quality_names):
        quality_df = dataframe[
            dataframe["quality"] == quality
        ]

        quota = base_quota + (
            1 if index < remainder else 0
        )

        quota = min(quota, len(quality_df))

        if quota > 0:
            selected_parts.append(
                quality_df.sample(
                    n=quota,
                    random_state=RANDOM_SEED + index,
                )
            )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    return selected.head(max_fields).copy()


def prepare_trocr_crop(
    image: Image.Image,
    *,
    box: list[int | float],
    field_name: str,
    skew_correction: float,
) -> Image.Image:
    crop = crop_field(
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

    crop = crop.convert("RGB")

    return ImageOps.expand(
        crop,
        border=16,
        fill=(255, 255, 255),
    )


def load_selected_fields(
    *,
    predictions_path: Path,
    confidence_threshold: float,
    max_fields: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Hybrid predictions not found: {predictions_path}"
        )

    predictions = pd.read_csv(predictions_path)

    required_columns = {
        "document_id",
        "document_type",
        "split",
        "quality",
        "field_name",
        "expected_value",
        "predicted_value",
        "normalized_expected",
        "normalized_prediction",
        "exact_match",
        "similarity",
        "ocr_confidence",
        "estimated_skew_correction",
        "status",
    }

    missing_columns = required_columns - set(
        predictions.columns
    )

    if missing_columns:
        raise ValueError(
            "Hybrid prediction file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    predictions = predictions[
        predictions["split"] == "validation"
    ].copy()

    predictions["ocr_confidence"] = pd.to_numeric(
        predictions["ocr_confidence"],
        errors="coerce",
    ).fillna(0.0)

    predictions["similarity"] = pd.to_numeric(
        predictions["similarity"],
        errors="coerce",
    ).fillna(0.0)

    predictions["estimated_skew_correction"] = pd.to_numeric(
        predictions["estimated_skew_correction"],
        errors="coerce",
    ).fillna(0.0)

    predictions["send_to_trocr"] = predictions.apply(
        should_send_to_trocr,
        axis=1,
        confidence_threshold=confidence_threshold,
    )

    selected = predictions[
        predictions["send_to_trocr"]
    ].copy()

    selected = balanced_limit(
        selected,
        max_fields=max_fields,
    )

    return selected, load_document_lookup()


def metric_summary(
    dataframe: pd.DataFrame,
    *,
    exact_column: str,
    similarity_column: str,
) -> dict[str, Any]:
    if dataframe.empty:
        return {
            "field_predictions": 0,
            "exact_match_rate": 0.0,
            "mean_similarity": 0.0,
        }

    return {
        "field_predictions": int(len(dataframe)),
        "exact_match_rate": round(
            float(dataframe[exact_column].mean()),
            4,
        ),
        "mean_similarity": round(
            float(dataframe[similarity_column].mean()),
            2,
        ),
    }


def run_benchmark(
    *,
    predictions_path: Path,
    max_fields: int,
    confidence_threshold: float,
    batch_size: int,
    num_beams: int,
) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    selected, document_lookup = load_selected_fields(
        predictions_path=predictions_path,
        confidence_threshold=confidence_threshold,
        max_fields=max_fields,
    )

    if selected.empty:
        raise ValueError(
            "No validation fields met the TrOCR fallback criteria."
        )

    print("\nTrOCR fallback benchmark")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(
        f"Selected fallback fields: {len(selected)}"
    )
    print(
        "By quality: "
        f"{selected['quality'].value_counts().to_dict()}"
    )
    print(f"Batch size: {batch_size}")
    print(f"Beams: {num_beams}")

    print("\nLoading processor...")
    processor = TrOCRProcessor.from_pretrained(
        MODEL_NAME,
        use_fast=False,
    )

    print("Loading model...")
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
    ).to(device)

    model.eval()

    prepared_rows: list[dict[str, Any]] = []
    prepared_images: list[Image.Image] = []

    for _, row in selected.iterrows():
        document_id = clean_text(row["document_id"])
        document_record = document_lookup.get(
            document_id
        )

        if document_record is None:
            continue

        image_path = (
            PROJECT_ROOT
            / clean_text(
                document_record["image_path"]
            )
        )

        annotation_path = (
            PROJECT_ROOT
            / clean_text(
                document_record["annotation_path"]
            )
        )

        annotation = load_json(
            annotation_path
        )

        field_name = clean_text(
            row["field_name"]
        )

        field_box = annotation.get(
            "field_boxes_pixels",
            {},
        ).get(field_name)

        if field_box is None:
            continue

        with Image.open(image_path) as source:
            image = source.convert("RGB")

            crop = prepare_trocr_crop(
                image,
                box=field_box,
                field_name=field_name,
                skew_correction=float(
                    row[
                        "estimated_skew_correction"
                    ]
                ),
            )

        prepared_images.append(crop)
        prepared_rows.append(row.to_dict())

    if not prepared_rows:
        raise ValueError(
            "No field crops were prepared for TrOCR."
        )

    output_rows: list[dict[str, Any]] = []

    total_batches = math.ceil(
        len(prepared_images) / batch_size
    )

    for batch_index in range(total_batches):
        start = batch_index * batch_size
        end = start + batch_size

        image_batch = prepared_images[start:end]
        row_batch = prepared_rows[start:end]

        pixel_values = processor(
            images=image_batch,
            return_tensors="pt",
        ).pixel_values.to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=32,
                num_beams=num_beams,
                early_stopping=True,
            )

        decoded_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        for row, trocr_text in zip(
            row_batch,
            decoded_texts,
        ):
            field_name = clean_text(
                row["field_name"]
            )

            (
                normalized_expected,
                normalized_trocr,
                trocr_exact,
                trocr_similarity,
            ) = evaluate_prediction(
                row["expected_value"],
                trocr_text,
                field_name,
            )

            tesseract_exact = (
                str(
                    row.get(
                        "exact_match",
                        False,
                    )
                ).lower()
                == "true"
            )

            tesseract_similarity = float(
                row.get("similarity", 0.0)
            )

            output_rows.append(
                {
                    "document_id": row["document_id"],
                    "document_type": row["document_type"],
                    "quality": row["quality"],
                    "field_name": field_name,
                    "expected_value": row["expected_value"],
                    "normalized_expected": normalized_expected,
                    "tesseract_prediction": row[
                        "predicted_value"
                    ],
                    "tesseract_normalized": row[
                        "normalized_prediction"
                    ],
                    "tesseract_confidence": row[
                        "ocr_confidence"
                    ],
                    "tesseract_exact_match": tesseract_exact,
                    "tesseract_similarity": tesseract_similarity,
                    "trocr_prediction": trocr_text,
                    "trocr_normalized": normalized_trocr,
                    "trocr_exact_match": trocr_exact,
                    "trocr_similarity": trocr_similarity,
                    "trocr_better_similarity": bool(
                        trocr_similarity
                        > tesseract_similarity
                    ),
                    "oracle_exact_match": bool(
                        tesseract_exact
                        or trocr_exact
                    ),
                    "oracle_similarity": max(
                        tesseract_similarity,
                        trocr_similarity,
                    ),
                }
            )

        print(
            f"[{batch_index + 1}/{total_batches}] "
            f"Processed {len(output_rows)}/"
            f"{len(prepared_rows)} fields"
        )

    results = pd.DataFrame(output_rows)

    OUTPUT_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PREDICTIONS_PATH,
        index=False,
    )

    by_quality: dict[str, Any] = {}

    for quality, quality_df in results.groupby(
        "quality"
    ):
        by_quality[str(quality)] = {
            "tesseract": metric_summary(
                quality_df,
                exact_column=(
                    "tesseract_exact_match"
                ),
                similarity_column=(
                    "tesseract_similarity"
                ),
            ),
            "trocr": metric_summary(
                quality_df,
                exact_column=(
                    "trocr_exact_match"
                ),
                similarity_column=(
                    "trocr_similarity"
                ),
            ),
            "oracle": metric_summary(
                quality_df,
                exact_column=(
                    "oracle_exact_match"
                ),
                similarity_column=(
                    "oracle_similarity"
                ),
            ),
        }

    summary = {
        "model_name": MODEL_NAME,
        "device": str(device),
        "validation_fields_selected": int(
            len(results)
        ),
        "confidence_threshold": float(
            confidence_threshold
        ),
        "batch_size": int(batch_size),
        "num_beams": int(num_beams),
        "tesseract": metric_summary(
            results,
            exact_column=(
                "tesseract_exact_match"
            ),
            similarity_column=(
                "tesseract_similarity"
            ),
        ),
        "trocr": metric_summary(
            results,
            exact_column="trocr_exact_match",
            similarity_column="trocr_similarity",
        ),
        "oracle": metric_summary(
            results,
            exact_column="oracle_exact_match",
            similarity_column="oracle_similarity",
        ),
        "trocr_better_similarity_rate": round(
            float(
                results[
                    "trocr_better_similarity"
                ].mean()
            ),
            4,
        ),
        "by_quality": by_quality,
    }

    with OUTPUT_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nTrOCR fallback benchmark completed")
    print(
        f"Fields evaluated: "
        f"{summary['validation_fields_selected']}"
    )
    print(
        "Tesseract fallback-set exact match: "
        f"{summary['tesseract']['exact_match_rate']:.2%}"
    )
    print(
        "Tesseract fallback-set similarity: "
        f"{summary['tesseract']['mean_similarity']:.2f}"
    )
    print(
        "TrOCR exact match: "
        f"{summary['trocr']['exact_match_rate']:.2%}"
    )
    print(
        "TrOCR similarity: "
        f"{summary['trocr']['mean_similarity']:.2f}"
    )
    print(
        "Oracle exact match: "
        f"{summary['oracle']['exact_match_rate']:.2%}"
    )
    print(
        "Oracle similarity: "
        f"{summary['oracle']['mean_similarity']:.2f}"
    )
    print(
        "TrOCR better-similarity rate: "
        f"{summary['trocr_better_similarity_rate']:.2%}"
    )
    print(
        f"Predictions: {OUTPUT_PREDICTIONS_PATH}"
    )
    print(f"Summary: {OUTPUT_SUMMARY_PATH}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark TrOCR on low-confidence "
            "validation fields selected from the "
            "hybrid Tesseract pipeline."
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
    )

    parser.add_argument(
        "--max-fields",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=70.0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-beams",
        type=int,
        default=2,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_benchmark(
        predictions_path=arguments.predictions,
        max_fields=arguments.max_fields,
        confidence_threshold=(
            arguments.confidence_threshold
        ),
        batch_size=arguments.batch_size,
        num_beams=arguments.num_beams,
    )
