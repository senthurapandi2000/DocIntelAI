from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rapidfuzz import fuzz
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.classification.train_irs_ocr_router import (
    ALL_FEATURES,
    character_ratios,
)
from src.extraction.irs_tesseract_baseline import (
    PROJECT_ROOT,
    clean_text,
    load_json,
)
from src.extraction.irs_tesseract_enhanced import (
    MULTILINE_FIELDS,
    evaluate_prediction,
    format_score,
)
from src.extraction.irs_trocr_fallback_benchmark import (
    MODEL_NAME,
    load_document_lookup,
    prepare_trocr_crop,
)


DEFAULT_HYBRID_PREDICTIONS = (
    PROJECT_ROOT
    / "reports"
    / "irs_tesseract_hybrid_predictions.csv"
)

DEFAULT_ROUTER_MODEL = (
    PROJECT_ROOT
    / "models"
    / "irs_ocr_router.joblib"
)

FINAL_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_ocr_final_test_predictions.csv"
)

FINAL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_ocr_final_test_summary.json"
)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def should_send_to_trocr(
    row: pd.Series,
    *,
    confidence_threshold: float,
) -> bool:
    """
    Use only inference-time signals.

    Ground-truth correctness and similarity are not used for routing.
    """

    field_name = clean_text(
        row["field_name"]
    )

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

    return bool(
        format_score(
            prediction,
            field_name,
        )
        < 1.0
    )


def add_router_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the router's inference-time feature engineering.

    This deliberately excludes ground truth, exact-match labels,
    ground-truth similarity, and synthetic quality labels.
    """

    result = dataframe.copy()

    for column in [
        "document_type",
        "field_name",
        "tesseract_prediction",
        "tesseract_normalized",
        "trocr_prediction",
        "trocr_normalized",
    ]:
        result[column] = (
            result[column]
            .fillna("")
            .astype(str)
        )

    result["tesseract_confidence"] = (
        pd.to_numeric(
            result[
                "tesseract_confidence"
            ],
            errors="coerce",
        ).fillna(0.0)
    )

    result["tesseract_length"] = (
        result["tesseract_normalized"]
        .str.len()
        .astype(float)
    )

    result["trocr_length"] = (
        result["trocr_normalized"]
        .str.len()
        .astype(float)
    )

    result["length_difference"] = (
        result["trocr_length"]
        - result["tesseract_length"]
    ).abs()

    result["tesseract_format_score"] = (
        result.apply(
            lambda row: format_score(
                row[
                    "tesseract_prediction"
                ],
                row["field_name"],
            ),
            axis=1,
        )
    )

    result["trocr_format_score"] = (
        result.apply(
            lambda row: format_score(
                row["trocr_prediction"],
                row["field_name"],
            ),
            axis=1,
        )
    )

    result[
        "model_agreement_similarity"
    ] = result.apply(
        lambda row: float(
            fuzz.ratio(
                row[
                    "tesseract_normalized"
                ],
                row[
                    "trocr_normalized"
                ],
            )
        ),
        axis=1,
    )

    tesseract_ratios = result[
        "tesseract_prediction"
    ].map(character_ratios)

    trocr_ratios = result[
        "trocr_prediction"
    ].map(character_ratios)

    result[
        [
            "tesseract_digit_ratio",
            "tesseract_alpha_ratio",
            "tesseract_punctuation_ratio",
        ]
    ] = pd.DataFrame(
        tesseract_ratios.tolist(),
        index=result.index,
    )

    result[
        [
            "trocr_digit_ratio",
            "trocr_alpha_ratio",
            "trocr_punctuation_ratio",
        ]
    ] = pd.DataFrame(
        trocr_ratios.tolist(),
        index=result.index,
    )

    result["tesseract_empty"] = (
        result["tesseract_normalized"]
        .str.strip()
        .eq("")
        .astype(float)
    )

    result["trocr_empty"] = (
        result["trocr_normalized"]
        .str.strip()
        .eq("")
        .astype(float)
    )

    result[
        "same_normalized_output"
    ] = (
        result["tesseract_normalized"]
        .eq(
            result["trocr_normalized"]
        )
        .astype(float)
    )

    return result


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
        "field_predictions": int(
            len(dataframe)
        ),
        "exact_match_rate": round(
            float(
                dataframe[
                    exact_column
                ].mean()
            ),
            4,
        ),
        "mean_similarity": round(
            float(
                dataframe[
                    similarity_column
                ].mean()
            ),
            2,
        ),
    }


def run_final_test(
    *,
    hybrid_predictions_path: Path,
    router_model_path: Path,
    confidence_threshold: float,
    batch_size: int,
    num_beams: int,
) -> None:
    if not hybrid_predictions_path.exists():
        raise FileNotFoundError(
            "Hybrid test predictions not found: "
            f"{hybrid_predictions_path}"
        )

    if not router_model_path.exists():
        raise FileNotFoundError(
            f"OCR router model not found: "
            f"{router_model_path}"
        )

    predictions = pd.read_csv(
        hybrid_predictions_path
    )

    required_columns = {
        "document_id",
        "parent_document_id",
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

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Hybrid prediction file is "
            "missing columns: "
            f"{sorted(missing_columns)}"
        )

    test_data = predictions[
        predictions["split"] == "test"
    ].copy()

    if test_data.empty:
        raise ValueError(
            "The hybrid predictions file does "
            "not contain test-split rows. Run "
            "irs_tesseract_hybrid with "
            "--split test first."
        )

    test_data["exact_match"] = (
        test_data[
            "exact_match"
        ].map(to_bool)
    )

    for column in [
        "similarity",
        "ocr_confidence",
        "estimated_skew_correction",
    ]:
        test_data[column] = (
            pd.to_numeric(
                test_data[column],
                errors="coerce",
            ).fillna(0.0)
        )

    test_data[
        "fallback_selected"
    ] = test_data.apply(
        should_send_to_trocr,
        axis=1,
        confidence_threshold=(
            confidence_threshold
        ),
    )

    fallback_data = test_data[
        test_data["fallback_selected"]
    ].copy()

    print(
        "\nFinal IRS OCR test evaluation"
    )
    print(
        f"Test documents: "
        f"{test_data['document_id'].nunique()}"
    )
    print(
        f"Test fields: {len(test_data)}"
    )
    print(
        f"TrOCR fallback fields: "
        f"{len(fallback_data)}"
    )
    print(
        "Fallback by quality: "
        f"{fallback_data['quality'].value_counts().to_dict()}"
    )

    router_bundle = joblib.load(
        router_model_path
    )

    router_pipeline = router_bundle[
        "pipeline"
    ]

    router_threshold = float(
        router_bundle["threshold"]
    )

    document_lookup = (
        load_document_lookup()
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Router threshold: "
        f"{router_threshold:.2f}"
    )
    print(f"TrOCR device: {device}")

    processor = None
    trocr_model = None

    trocr_rows: list[
        dict[str, Any]
    ] = []

    prepared_images: list[
        Image.Image
    ] = []

    prepared_rows: list[
        dict[str, Any]
    ] = []

    if not fallback_data.empty:
        print("\nLoading TrOCR processor...")
        processor = (
            TrOCRProcessor.from_pretrained(
                MODEL_NAME,
                use_fast=False,
            )
        )

        print("Loading TrOCR model...")
        trocr_model = (
            VisionEncoderDecoderModel
            .from_pretrained(
                MODEL_NAME,
                use_safetensors=True,
            )
            .to(device)
        )

        trocr_model.eval()

        for _, row in (
            fallback_data.iterrows()
        ):
            document_id = clean_text(
                row["document_id"]
            )

            document_record = (
                document_lookup.get(
                    document_id
                )
            )

            if document_record is None:
                continue

            image_path = (
                PROJECT_ROOT
                / clean_text(
                    document_record[
                        "image_path"
                    ]
                )
            )

            annotation_path = (
                PROJECT_ROOT
                / clean_text(
                    document_record[
                        "annotation_path"
                    ]
                )
            )

            annotation = load_json(
                annotation_path
            )

            field_name = clean_text(
                row["field_name"]
            )

            field_box = (
                annotation.get(
                    "field_boxes_pixels",
                    {},
                ).get(field_name)
            )

            if field_box is None:
                continue

            with Image.open(
                image_path
            ) as source:
                image = source.convert(
                    "RGB"
                )

                crop = prepare_trocr_crop(
                    image,
                    box=field_box,
                    field_name=(
                        field_name
                    ),
                    skew_correction=float(
                        row[
                            "estimated_skew_correction"
                        ]
                    ),
                )

            prepared_images.append(
                crop
            )

            prepared_rows.append(
                row.to_dict()
            )

        total_batches = math.ceil(
            len(prepared_images)
            / batch_size
        )

        for batch_index in range(
            total_batches
        ):
            start = (
                batch_index
                * batch_size
            )
            end = (
                start
                + batch_size
            )

            image_batch = (
                prepared_images[
                    start:end
                ]
            )

            row_batch = (
                prepared_rows[
                    start:end
                ]
            )

            pixel_values = processor(
                images=image_batch,
                return_tensors="pt",
            ).pixel_values.to(device)

            with torch.inference_mode():
                generated_ids = (
                    trocr_model.generate(
                        pixel_values,
                        max_new_tokens=32,
                        num_beams=num_beams,
                        early_stopping=True,
                    )
                )

            decoded_texts = (
                processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            )

            for row, trocr_text in zip(
                row_batch,
                decoded_texts,
            ):
                field_name = clean_text(
                    row["field_name"]
                )

                (
                    _,
                    normalized_trocr,
                    trocr_exact,
                    trocr_similarity,
                ) = evaluate_prediction(
                    row[
                        "expected_value"
                    ],
                    trocr_text,
                    field_name,
                )

                trocr_rows.append(
                    {
                        "document_id": (
                            row[
                                "document_id"
                            ]
                        ),
                        "field_name": (
                            field_name
                        ),
                        "trocr_prediction": (
                            trocr_text
                        ),
                        "trocr_normalized": (
                            normalized_trocr
                        ),
                        "trocr_exact_match": (
                            trocr_exact
                        ),
                        "trocr_similarity": (
                            trocr_similarity
                        ),
                    }
                )

            print(
                f"[{batch_index + 1}/"
                f"{total_batches}] "
                f"TrOCR processed "
                f"{len(trocr_rows)}/"
                f"{len(prepared_rows)}"
            )

    trocr_results = pd.DataFrame(
        trocr_rows
    )

    if not trocr_results.empty:
        fallback_candidates = (
            fallback_data.merge(
                trocr_results,
                on=[
                    "document_id",
                    "field_name",
                ],
                how="left",
                validate="one_to_one",
            )
        )

        router_input = pd.DataFrame(
            {
                "document_type": (
                    fallback_candidates[
                        "document_type"
                    ]
                ),
                "field_name": (
                    fallback_candidates[
                        "field_name"
                    ]
                ),
                "tesseract_prediction": (
                    fallback_candidates[
                        "predicted_value"
                    ]
                ),
                "tesseract_normalized": (
                    fallback_candidates[
                        "normalized_prediction"
                    ]
                ),
                "tesseract_confidence": (
                    fallback_candidates[
                        "ocr_confidence"
                    ]
                ),
                "trocr_prediction": (
                    fallback_candidates[
                        "trocr_prediction"
                    ]
                ),
                "trocr_normalized": (
                    fallback_candidates[
                        "trocr_normalized"
                    ]
                ),
            }
        )

        router_features = (
            add_router_features(
                router_input
            )
        )

        router_probabilities = (
            router_pipeline.predict_proba(
                router_features[
                    ALL_FEATURES
                ]
            )[:, 1]
        )

        fallback_candidates[
            "router_probability"
        ] = router_probabilities

        fallback_candidates[
            "router_choose_trocr"
        ] = (
            router_probabilities
            >= router_threshold
        )

        routed_subset = (
            fallback_candidates[
                [
                    "document_id",
                    "field_name",
                    "trocr_prediction",
                    "trocr_normalized",
                    "trocr_exact_match",
                    "trocr_similarity",
                    "router_probability",
                    "router_choose_trocr",
                ]
            ]
        )

        final_data = test_data.merge(
            routed_subset,
            on=[
                "document_id",
                "field_name",
            ],
            how="left",
            validate="one_to_one",
        )

    else:
        final_data = test_data.copy()

        final_data[
            "trocr_prediction"
        ] = ""

        final_data[
            "trocr_normalized"
        ] = ""

        final_data[
            "trocr_exact_match"
        ] = False

        final_data[
            "trocr_similarity"
        ] = 0.0

        final_data[
            "router_probability"
        ] = 0.0

        final_data[
            "router_choose_trocr"
        ] = False

    final_data[
        "router_choose_trocr"
    ] = (
        final_data[
            "router_choose_trocr"
        ]
        .fillna(False)
        .map(to_bool)
    )

    final_data[
        "final_engine"
    ] = np.where(
        final_data[
            "router_choose_trocr"
        ],
        "trocr",
        "tesseract",
    )

    final_data[
        "final_prediction"
    ] = np.where(
        final_data[
            "router_choose_trocr"
        ],
        final_data[
            "trocr_prediction"
        ],
        final_data[
            "predicted_value"
        ],
    )

    final_data[
        "final_normalized_prediction"
    ] = np.where(
        final_data[
            "router_choose_trocr"
        ],
        final_data[
            "trocr_normalized"
        ],
        final_data[
            "normalized_prediction"
        ],
    )

    final_data[
        "final_exact_match"
    ] = np.where(
        final_data[
            "router_choose_trocr"
        ],
        final_data[
            "trocr_exact_match"
        ].fillna(False).map(
            to_bool
        ),
        final_data[
            "exact_match"
        ],
    ).astype(bool)

    final_data[
        "final_similarity"
    ] = np.where(
        final_data[
            "router_choose_trocr"
        ],
        pd.to_numeric(
            final_data[
                "trocr_similarity"
            ],
            errors="coerce",
        ).fillna(0.0),
        final_data[
            "similarity"
        ],
    )

    FINAL_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_data.to_csv(
        FINAL_PREDICTIONS_PATH,
        index=False,
    )

    tesseract_metrics = (
        metric_summary(
            final_data,
            exact_column="exact_match",
            similarity_column=(
                "similarity"
            ),
        )
    )

    final_metrics = (
        metric_summary(
            final_data,
            exact_column=(
                "final_exact_match"
            ),
            similarity_column=(
                "final_similarity"
            ),
        )
    )

    fallback_only = final_data[
        final_data[
            "fallback_selected"
        ]
    ].copy()

    fallback_tesseract = (
        metric_summary(
            fallback_only,
            exact_column="exact_match",
            similarity_column=(
                "similarity"
            ),
        )
    )

    fallback_router = (
        metric_summary(
            fallback_only,
            exact_column=(
                "final_exact_match"
            ),
            similarity_column=(
                "final_similarity"
            ),
        )
    )

    by_quality: dict[
        str,
        Any,
    ] = {}

    for quality, quality_df in (
        final_data.groupby(
            "quality"
        )
    ):
        by_quality[str(quality)] = {
            "tesseract": (
                metric_summary(
                    quality_df,
                    exact_column=(
                        "exact_match"
                    ),
                    similarity_column=(
                        "similarity"
                    ),
                )
            ),
            "final_router": (
                metric_summary(
                    quality_df,
                    exact_column=(
                        "final_exact_match"
                    ),
                    similarity_column=(
                        "final_similarity"
                    ),
                )
            ),
        }

    by_document_type: dict[
        str,
        Any,
    ] = {}

    for document_type, type_df in (
        final_data.groupby(
            "document_type"
        )
    ):
        by_document_type[
            str(document_type)
        ] = {
            "tesseract": (
                metric_summary(
                    type_df,
                    exact_column=(
                        "exact_match"
                    ),
                    similarity_column=(
                        "similarity"
                    ),
                )
            ),
            "final_router": (
                metric_summary(
                    type_df,
                    exact_column=(
                        "final_exact_match"
                    ),
                    similarity_column=(
                        "final_similarity"
                    ),
                )
            ),
        }

    summary = {
        "split": "test",
        "documents_evaluated": int(
            final_data[
                "document_id"
            ].nunique()
        ),
        "field_predictions": int(
            len(final_data)
        ),
        "fallback_fields": int(
            final_data[
                "fallback_selected"
            ].sum()
        ),
        "trocr_selected_fields": int(
            final_data[
                "router_choose_trocr"
            ].sum()
        ),
        "trocr_selection_rate_overall": round(
            float(
                final_data[
                    "router_choose_trocr"
                ].mean()
            ),
            4,
        ),
        "trocr_selection_rate_on_fallback": round(
            float(
                fallback_only[
                    "router_choose_trocr"
                ].mean()
            )
            if not fallback_only.empty
            else 0.0,
            4,
        ),
        "router_threshold": (
            router_threshold
        ),
        "confidence_threshold": (
            confidence_threshold
        ),
        "tesseract_hybrid": (
            tesseract_metrics
        ),
        "final_router": (
            final_metrics
        ),
        "fallback_only": {
            "tesseract": (
                fallback_tesseract
            ),
            "final_router": (
                fallback_router
            ),
        },
        "by_quality": by_quality,
        "by_document_type": (
            by_document_type
        ),
    }

    with FINAL_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\nFinal test evaluation completed"
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
        f"Fallback fields: "
        f"{summary['fallback_fields']}"
    )
    print(
        f"TrOCR selected fields: "
        f"{summary['trocr_selected_fields']}"
    )

    print("\nOverall test metrics")
    print(
        "Hybrid Tesseract: exact match "
        f"{tesseract_metrics['exact_match_rate']:.2%}, "
        f"similarity "
        f"{tesseract_metrics['mean_similarity']:.2f}"
    )
    print(
        "Final router: exact match "
        f"{final_metrics['exact_match_rate']:.2%}, "
        f"similarity "
        f"{final_metrics['mean_similarity']:.2f}"
    )

    print("\nBy quality")

    for quality, metrics in (
        by_quality.items()
    ):
        tesseract_quality = (
            metrics["tesseract"]
        )

        final_quality = (
            metrics["final_router"]
        )

        print(
            f"{quality}: Tesseract "
            f"{tesseract_quality['exact_match_rate']:.2%}"
            f"/{tesseract_quality['mean_similarity']:.2f}, "
            f"Final "
            f"{final_quality['exact_match_rate']:.2%}"
            f"/{final_quality['mean_similarity']:.2f}"
        )

    print(
        f"\nPredictions: "
        f"{FINAL_PREDICTIONS_PATH}"
    )
    print(
        f"Summary: {FINAL_SUMMARY_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Tesseract + TrOCR router "
            "on the untouched IRS test split."
        )
    )

    parser.add_argument(
        "--hybrid-predictions",
        type=Path,
        default=(
            DEFAULT_HYBRID_PREDICTIONS
        ),
    )

    parser.add_argument(
        "--router-model",
        type=Path,
        default=(
            DEFAULT_ROUTER_MODEL
        ),
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=70.0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-beams",
        type=int,
        default=2,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_final_test(
        hybrid_predictions_path=(
            arguments.hybrid_predictions
        ),
        router_model_path=(
            arguments.router_model
        ),
        confidence_threshold=(
            arguments.confidence_threshold
        ),
        batch_size=(
            arguments.batch_size
        ),
        num_beams=(
            arguments.num_beams
        ),
    )
