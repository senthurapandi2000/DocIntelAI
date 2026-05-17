from __future__ import annotations

import gc
import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import joblib
import numpy as np
import pandas as pd
import pytesseract
import torch
from PIL import Image, ImageOps
from rapidfuzz import fuzz
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.classification.train_irs_field_confidence_model import (
    ALL_FEATURES as CONFIDENCE_FEATURES,
)
from src.classification.train_irs_ocr_router import (
    ALL_FEATURES as ROUTER_FEATURES,
)
from src.extraction.irs_tesseract_baseline import (
    PROJECT_ROOT,
    clean_text,
    configure_tesseract,
    evaluation_fields,
    field_kind,
)
from src.extraction.irs_tesseract_enhanced import (
    MULTILINE_FIELDS,
    evaluate_prediction,
    format_score,
)
from src.extraction.irs_tesseract_hybrid import (
    baseline_candidate,
    choose_candidate,
    enhanced_candidate,
    estimate_skew_correction,
    should_run_enhanced,
)
from src.extraction.irs_trocr_fallback_benchmark import (
    MODEL_NAME,
    prepare_trocr_crop,
)
from src.validation.evaluate_irs_field_confidence_final_test import (
    engineer_features as engineer_confidence_features,
)
from src.validation.evaluate_irs_ocr_final_test import (
    add_router_features,
    should_send_to_trocr,
)
from src.validation.irs_extraction_validator import (
    clean_text as validator_clean_text,
)


COORDINATE_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "irs_field_coordinates.json"
)

OCR_ROUTER_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "irs_ocr_router.joblib"
)

FIELD_CONFIDENCE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "irs_field_confidence_model.joblib"
)

PROCESSED_UPLOAD_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "irs_uploads"
)

SUPPORTED_DOCUMENT_TYPES = {
    "w2",
    "1099_nec",
}

TEMPLATE_BY_DOCUMENT_TYPE = {
    "w2": "w2_2026",
    "1099_nec": "1099_nec_2026",
}

DOCUMENT_TYPE_KEYWORDS = {
    "w2": {
        "wage and tax statement": 5.0,
        "form w-2": 4.0,
        "employee's social security number": 3.0,
        "employer identification number": 2.5,
        "social security wages": 2.5,
        "medicare wages": 2.5,
        "federal income tax withheld": 1.5,
    },
    "1099_nec": {
        "nonemployee compensation": 5.0,
        "form 1099-nec": 4.0,
        "payer's tin": 3.0,
        "recipient's tin": 3.0,
        "recipient's name": 2.0,
        "federal income tax withheld": 1.5,
    },
}


@dataclass(frozen=True)
class PreparedDocument:
    document_id: str
    document_type: str
    source_image_path: Path
    annotation_path: Path
    source_page_index: int
    image: Image.Image
    field_boxes_pixels: dict[str, list[int]]
    classification_scores: dict[str, float]


@dataclass(frozen=True)
class ProcessingResult:
    document_id: str
    document_type: str
    source_image_path: Path
    annotation_path: Path
    field_results: pd.DataFrame
    document_status: str
    priority: int
    classification_scores: dict[str, float]


def relative_project_path(path: Path) -> str:
    return path.resolve().relative_to(
        PROJECT_ROOT.resolve()
    ).as_posix()


def normalize_document_type(
    value: str | None,
) -> str:
    normalized = clean_text(value).lower()

    aliases = {
        "1099-nec": "1099_nec",
        "1099nec": "1099_nec",
        "w-2": "w2",
        "auto": "auto",
        "": "auto",
    }

    normalized = aliases.get(
        normalized,
        normalized,
    )

    if (
        normalized != "auto"
        and normalized
        not in SUPPORTED_DOCUMENT_TYPES
    ):
        raise ValueError(
            "Document type must be auto, w2, "
            "or 1099_nec."
        )

    return normalized


def load_coordinate_config() -> dict[str, Any]:
    if not COORDINATE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "IRS coordinate config not found: "
            f"{COORDINATE_CONFIG_PATH}"
        )

    with COORDINATE_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def render_pdf_pages(
    pdf_path: Path,
    *,
    dpi: int = 200,
) -> list[Image.Image]:
    pages: list[Image.Image] = []

    with fitz.open(pdf_path) as document:
        if document.page_count == 0:
            raise ValueError(
                "The uploaded PDF has no pages."
            )

        zoom = dpi / 72.0
        matrix = fitz.Matrix(
            zoom,
            zoom,
        )

        for page in document:
            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            image = Image.frombytes(
                "RGB",
                (
                    pixmap.width,
                    pixmap.height,
                ),
                pixmap.samples,
            )

            pages.append(image)

    return pages


def load_uploaded_pages(
    source_path: Path,
) -> list[Image.Image]:
    extension = source_path.suffix.lower()

    if extension == ".pdf":
        return render_pdf_pages(
            source_path
        )

    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(
            source
        ).convert("RGB")

    return [image]


def page_ocr_text(
    image: Image.Image,
) -> str:
    working = image.convert("L")

    max_width = 1400

    if working.width > max_width:
        ratio = (
            max_width / working.width
        )

        working = working.resize(
            (
                max_width,
                max(
                    1,
                    round(
                        working.height
                        * ratio
                    ),
                ),
            ),
            Image.Resampling.BILINEAR,
        )

    working = ImageOps.autocontrast(
        working,
        cutoff=1,
    )

    text = pytesseract.image_to_string(
        working,
        config="--oem 3 --psm 6",
    )

    return re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()


def keyword_score(
    text: str,
    document_type: str,
) -> float:
    score = 0.0

    for keyword, weight in (
        DOCUMENT_TYPE_KEYWORDS[
            document_type
        ].items()
    ):
        if keyword in text:
            score += weight

    return round(
        score,
        4,
    )


def choose_document_page(
    pages: list[Image.Image],
    *,
    expected_document_type: str,
) -> tuple[
    int,
    str,
    dict[str, float],
]:
    if not pages:
        raise ValueError(
            "No uploaded pages are available."
        )

    page_records: list[
        dict[str, Any]
    ] = []

    for page_index, image in enumerate(
        pages
    ):
        text = page_ocr_text(image)

        scores = {
            document_type: keyword_score(
                text,
                document_type,
            )
            for document_type in (
                "w2",
                "1099_nec",
            )
        }

        page_records.append(
            {
                "page_index": page_index,
                "text": text,
                "scores": scores,
            }
        )

    if expected_document_type != "auto":
        best = max(
            page_records,
            key=lambda record: (
                record["scores"][
                    expected_document_type
                ],
                -record["page_index"],
            ),
        )

        return (
            int(best["page_index"]),
            expected_document_type,
            {
                key: float(value)
                for key, value in (
                    best["scores"].items()
                )
            },
        )

    candidates: list[
        tuple[float, int, str]
    ] = []

    for record in page_records:
        for document_type, score in (
            record["scores"].items()
        ):
            candidates.append(
                (
                    float(score),
                    int(
                        record[
                            "page_index"
                        ]
                    ),
                    document_type,
                )
            )

    best_score, best_page, best_type = max(
        candidates,
        key=lambda item: (
            item[0],
            -item[1],
        ),
    )

    if best_score <= 0:
        if len(pages) == 1:
            raise ValueError(
                "Could not automatically identify "
                "the uploaded form. Upload again "
                "with expected_document_type set "
                "to w2 or 1099_nec."
            )

        raise ValueError(
            "No W-2 or 1099-NEC page was "
            "identified in the uploaded PDF."
        )

    selected_record = page_records[
        best_page
    ]

    return (
        best_page,
        best_type,
        {
            key: float(value)
            for key, value in (
                selected_record[
                    "scores"
                ].items()
            )
        },
    )


def coordinate_rect_to_pixels(
    rect: list[int | float],
    *,
    image_width: int,
    image_height: int,
    page_width_points: float,
    page_height_points: float,
) -> list[int]:
    x_scale = (
        image_width / page_width_points
    )

    y_scale = (
        image_height / page_height_points
    )

    x1, y1, x2, y2 = [
        float(value)
        for value in rect
    ]

    pixel_box = [
        max(
            0,
            round(x1 * x_scale),
        ),
        max(
            0,
            round(y1 * y_scale),
        ),
        min(
            image_width,
            round(x2 * x_scale),
        ),
        min(
            image_height,
            round(y2 * y_scale),
        ),
    ]

    if (
        pixel_box[2]
        <= pixel_box[0]
        or pixel_box[3]
        <= pixel_box[1]
    ):
        raise ValueError(
            "A configured field coordinate "
            "produced an invalid pixel box."
        )

    return pixel_box


def build_field_boxes(
    *,
    image: Image.Image,
    document_type: str,
    coordinate_config: dict[str, Any],
) -> dict[str, list[int]]:
    template_name = (
        TEMPLATE_BY_DOCUMENT_TYPE[
            document_type
        ]
    )

    template = (
        coordinate_config[
            "templates"
        ][template_name]
    )

    page_size = coordinate_config[
        "page_size_points"
    ]

    page_width_points = float(
        page_size[0]
    )

    page_height_points = float(
        page_size[1]
    )

    template_fields = template[
        "fields"
    ]

    boxes: dict[
        str,
        list[int]
    ] = {}

    for field_name in evaluation_fields(
        document_type
    ):
        field_config = (
            template_fields.get(
                field_name
            )
        )

        if field_config is None:
            continue

        boxes[field_name] = (
            coordinate_rect_to_pixels(
                field_config["rect"],
                image_width=image.width,
                image_height=image.height,
                page_width_points=(
                    page_width_points
                ),
                page_height_points=(
                    page_height_points
                ),
            )
        )

    if not boxes:
        raise ValueError(
            "No OCR field coordinates were "
            f"found for {document_type}."
        )

    return boxes


def prepare_document(
    *,
    upload_id: str,
    source_path: Path,
    expected_document_type: str,
) -> PreparedDocument:
    configure_tesseract()

    coordinate_config = (
        load_coordinate_config()
    )

    pages = load_uploaded_pages(
        source_path
    )

    normalized_expected = (
        normalize_document_type(
            expected_document_type
        )
    )

    (
        source_page_index,
        document_type,
        classification_scores,
    ) = choose_document_page(
        pages,
        expected_document_type=(
            normalized_expected
        ),
    )

    image = pages[
        source_page_index
    ].convert("RGB")

    field_boxes = build_field_boxes(
        image=image,
        document_type=document_type,
        coordinate_config=(
            coordinate_config
        ),
    )

    document_id = (
        "upload_"
        + upload_id.replace(
            "-",
            "",
        )
    )

    output_directory = (
        PROCESSED_UPLOAD_DIRECTORY
        / upload_id
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_image_path = (
        output_directory
        / "source.png"
    )

    annotation_path = (
        output_directory
        / "annotation.json"
    )

    image.save(
        source_image_path,
        format="PNG",
    )

    annotation = {
        "document_id": document_id,
        "parent_document_id": (
            document_id
        ),
        "document_type": (
            document_type
        ),
        "split": "production",
        "quality": "uploaded",
        "source_page_index": (
            source_page_index
        ),
        "source_upload_path": (
            relative_project_path(
                source_path
            )
            if source_path.resolve()
            .is_relative_to(
                PROJECT_ROOT.resolve()
            )
            else str(
                source_path.resolve()
            )
        ),
        "image_path": (
            relative_project_path(
                source_image_path
            )
        ),
        "annotation_path": (
            relative_project_path(
                annotation_path
            )
        ),
        "fields": {},
        "field_boxes_pixels": (
            field_boxes
        ),
        "classification_scores": (
            classification_scores
        ),
    }

    with annotation_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            annotation,
            file,
            indent=2,
        )

    return PreparedDocument(
        document_id=document_id,
        document_type=document_type,
        source_image_path=(
            source_image_path
        ),
        annotation_path=(
            annotation_path
        ),
        source_page_index=(
            source_page_index
        ),
        image=image,
        field_boxes_pixels=(
            field_boxes
        ),
        classification_scores=(
            classification_scores
        ),
    )


def normalize_prediction(
    prediction: str,
    field_name: str,
) -> str:
    (
        _,
        normalized_prediction,
        _,
        _,
    ) = evaluate_prediction(
        "",
        prediction,
        field_name,
    )

    return clean_text(
        normalized_prediction
    )


def run_hybrid_tesseract(
    prepared: PreparedDocument,
) -> pd.DataFrame:
    skew_correction = (
        estimate_skew_correction(
            prepared.image
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    for field_name in evaluation_fields(
        prepared.document_type
    ):
        box = (
            prepared
            .field_boxes_pixels
            .get(field_name)
        )

        if box is None:
            rows.append(
                {
                    "document_id": (
                        prepared.document_id
                    ),
                    "parent_document_id": (
                        prepared.document_id
                    ),
                    "document_type": (
                        prepared.document_type
                    ),
                    "split": "production",
                    "quality": "uploaded",
                    "field_name": (
                        field_name
                    ),
                    "field_kind": field_kind(
                        field_name
                    ),
                    "predicted_value": "",
                    "normalized_prediction": "",
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
                prepared.image,
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
                    prepared.image,
                    box,
                    field_name,
                    skew_correction,
                )

                (
                    prediction,
                    confidence,
                    chosen_engine,
                ) = choose_candidate(
                    baseline_text=(
                        baseline_text
                    ),
                    baseline_confidence=(
                        baseline_confidence
                    ),
                    enhanced_text=(
                        enhanced_text
                    ),
                    enhanced_confidence=(
                        enhanced_confidence
                    ),
                    field_name=(
                        field_name
                    ),
                )
            else:
                prediction = (
                    baseline_text
                )
                confidence = (
                    baseline_confidence
                )
                chosen_engine = (
                    "baseline"
                )

            rows.append(
                {
                    "document_id": (
                        prepared.document_id
                    ),
                    "parent_document_id": (
                        prepared.document_id
                    ),
                    "document_type": (
                        prepared.document_type
                    ),
                    "split": "production",
                    "quality": "uploaded",
                    "field_name": (
                        field_name
                    ),
                    "field_kind": field_kind(
                        field_name
                    ),
                    "predicted_value": (
                        prediction
                    ),
                    "normalized_prediction": (
                        normalize_prediction(
                            prediction,
                            field_name,
                        )
                    ),
                    "ocr_confidence": float(
                        confidence
                    ),
                    "chosen_engine": (
                        chosen_engine
                    ),
                    "baseline_prediction": (
                        baseline_text
                    ),
                    "baseline_confidence": float(
                        baseline_confidence
                    ),
                    "enhanced_prediction": (
                        enhanced_text
                    ),
                    "enhanced_confidence": float(
                        enhanced_confidence
                    ),
                    "estimated_skew_correction": (
                        skew_correction
                    ),
                    "status": "success",
                    "error": "",
                }
            )

        except Exception as error:
            rows.append(
                {
                    "document_id": (
                        prepared.document_id
                    ),
                    "parent_document_id": (
                        prepared.document_id
                    ),
                    "document_type": (
                        prepared.document_type
                    ),
                    "split": "production",
                    "quality": "uploaded",
                    "field_name": (
                        field_name
                    ),
                    "field_kind": field_kind(
                        field_name
                    ),
                    "predicted_value": "",
                    "normalized_prediction": "",
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
                    "error": str(error),
                }
            )

    return pd.DataFrame(rows)


class ProductionModels:
    def __init__(
        self,
        *,
        ocr_router_path: Path = (
            OCR_ROUTER_MODEL_PATH
        ),
        field_confidence_path: Path = (
            FIELD_CONFIDENCE_MODEL_PATH
        ),
        batch_size: int = 8,
        num_beams: int = 2,
        confidence_threshold: float = 70.0,
        disable_trocr: bool = False,
    ) -> None:
        if not ocr_router_path.exists():
            raise FileNotFoundError(
                "OCR router model not found: "
                f"{ocr_router_path}"
            )

        if not field_confidence_path.exists():
            raise FileNotFoundError(
                "Field-confidence model not found: "
                f"{field_confidence_path}"
            )

        self.ocr_router_bundle = (
            joblib.load(
                ocr_router_path
            )
        )

        self.field_confidence_bundle = (
            joblib.load(
                field_confidence_path
            )
        )

        self.batch_size = max(
            1,
            int(batch_size),
        )

        self.num_beams = max(
            1,
            int(num_beams),
        )

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.disable_trocr = bool(
            disable_trocr
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.trocr_dtype = (
            torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )

        self.trocr_processor: (
            TrOCRProcessor | None
        ) = None

        self.trocr_model: (
            VisionEncoderDecoderModel
            | None
        ) = None

    def ensure_trocr_loaded(
        self,
    ) -> None:
        if self.disable_trocr:
            return

        if (
            self.trocr_processor
            is not None
            and self.trocr_model
            is not None
        ):
            return

        self.trocr_processor = (
            TrOCRProcessor
            .from_pretrained(
                MODEL_NAME,
                use_fast=False,
            )
        )

        try:
            self.trocr_model = (
                VisionEncoderDecoderModel
                .from_pretrained(
                    MODEL_NAME,
                    use_safetensors=True,
                )
                .to(
                    device=self.device,
                    dtype=self.trocr_dtype,
                )
            )

        except Exception as error:
            message = str(error).lower()

            out_of_memory = (
                "out of memory" in message
                or "cudaerrormemoryallocation"
                in message
                or "cuda error: out of memory"
                in message
            )

            if (
                not out_of_memory
                or self.device.type != "cuda"
            ):
                raise

            print(
                "CUDA memory exhausted while "
                "loading TrOCR. Loading the model "
                "on CPU instead."
            )

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            gc.collect()

            self.device = torch.device(
                "cpu"
            )
            self.trocr_dtype = (
                torch.float32
            )
            self.batch_size = 1
            self.num_beams = 1

            self.trocr_model = (
                VisionEncoderDecoderModel
                .from_pretrained(
                    MODEL_NAME,
                    use_safetensors=True,
                )
                .to(
                    device=self.device,
                    dtype=self.trocr_dtype,
                )
            )

        self.trocr_model.eval()

    def generate_trocr_batch(
        self,
        image_batch: list[Image.Image],
    ) -> list[str]:
        self.ensure_trocr_loaded()

        assert (
            self.trocr_processor
            is not None
        )

        assert (
            self.trocr_model
            is not None
        )

        try:
            pixel_values = (
                self.trocr_processor(
                    images=image_batch,
                    return_tensors="pt",
                )
                .pixel_values
                .to(
                    device=self.device,
                    dtype=self.trocr_dtype,
                )
            )

            with torch.inference_mode():
                generated_ids = (
                    self.trocr_model.generate(
                        pixel_values,
                        max_new_tokens=32,
                        num_beams=(
                            self.num_beams
                        ),
                        early_stopping=True,
                    )
                )

            decoded = (
                self.trocr_processor
                .batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            )

            del pixel_values
            del generated_ids

            return decoded

        except Exception as error:
            message = str(error).lower()

            out_of_memory = (
                "out of memory" in message
                or "cudaerrormemoryallocation"
                in message
                or "cuda error: out of memory"
                in message
            )

            if (
                not out_of_memory
                or self.device.type != "cuda"
            ):
                raise

            print(
                "CUDA memory exhausted during "
                "TrOCR inference. Retrying on CPU "
                "with batch size 1 and one beam."
            )

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            gc.collect()

            self.device = torch.device(
                "cpu"
            )
            self.trocr_dtype = (
                torch.float32
            )
            self.batch_size = 1
            self.num_beams = 1

            self.trocr_model = (
                self.trocr_model
                .to(
                    device=self.device,
                    dtype=self.trocr_dtype,
                )
            )

            self.trocr_model.eval()

            decoded: list[str] = []

            for image in image_batch:
                pixel_values = (
                    self.trocr_processor(
                        images=[image],
                        return_tensors="pt",
                    )
                    .pixel_values
                    .to(
                        device=self.device,
                        dtype=self.trocr_dtype,
                    )
                )

                with torch.inference_mode():
                    generated_ids = (
                        self.trocr_model.generate(
                            pixel_values,
                            max_new_tokens=32,
                            num_beams=1,
                            early_stopping=True,
                        )
                    )

                decoded.extend(
                    self.trocr_processor
                    .batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )
                )

                del pixel_values
                del generated_ids
                gc.collect()

            return decoded


    def run_trocr_fallback(
        self,
        prepared: PreparedDocument,
        hybrid: pd.DataFrame,
    ) -> pd.DataFrame:
        hybrid = hybrid.copy()

        hybrid[
            "fallback_selected"
        ] = hybrid.apply(
            should_send_to_trocr,
            axis=1,
            confidence_threshold=(
                self.confidence_threshold
            ),
        )

        fallback = hybrid[
            hybrid[
                "fallback_selected"
            ]
        ].copy()

        if (
            fallback.empty
            or self.disable_trocr
        ):
            hybrid[
                "trocr_prediction"
            ] = ""

            hybrid[
                "trocr_normalized"
            ] = ""

            hybrid[
                "router_probability"
            ] = 0.0

            hybrid[
                "router_choose_trocr"
            ] = False

            hybrid["final_engine"] = (
                "tesseract"
            )

            hybrid[
                "final_prediction"
            ] = hybrid[
                "predicted_value"
            ]

            hybrid[
                "final_normalized_prediction"
            ] = hybrid[
                "normalized_prediction"
            ]

            return hybrid

        self.ensure_trocr_loaded()

        assert (
            self.trocr_processor
            is not None
        )

        assert (
            self.trocr_model
            is not None
        )

        prepared_images: list[
            Image.Image
        ] = []

        prepared_rows: list[
            dict[str, Any]
        ] = []

        for _, row in (
            fallback.iterrows()
        ):
            field_name = clean_text(
                row["field_name"]
            )

            if field_name in MULTILINE_FIELDS:
                continue

            box = (
                prepared
                .field_boxes_pixels
                .get(field_name)
            )

            if box is None:
                continue

            crop = prepare_trocr_crop(
                prepared.image,
                box=box,
                field_name=field_name,
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

        trocr_rows: list[
            dict[str, Any]
        ] = []

        total_batches = math.ceil(
            len(prepared_images)
            / self.batch_size
        )

        for batch_index in range(
            total_batches
        ):
            start = (
                batch_index
                * self.batch_size
            )

            end = (
                start
                + self.batch_size
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

            decoded_texts = (
                self.generate_trocr_batch(
                    image_batch
                )
            )

            for row, text in zip(
                row_batch,
                decoded_texts,
            ):
                field_name = clean_text(
                    row["field_name"]
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
                            text
                        ),
                        "trocr_normalized": (
                            normalize_prediction(
                                text,
                                field_name,
                            )
                        ),
                    }
                )

        trocr_frame = pd.DataFrame(
            trocr_rows
        )

        if trocr_frame.empty:
            hybrid[
                "trocr_prediction"
            ] = ""

            hybrid[
                "trocr_normalized"
            ] = ""

            hybrid[
                "router_probability"
            ] = 0.0

            hybrid[
                "router_choose_trocr"
            ] = False

            hybrid["final_engine"] = (
                "tesseract"
            )

            hybrid[
                "final_prediction"
            ] = hybrid[
                "predicted_value"
            ]

            hybrid[
                "final_normalized_prediction"
            ] = hybrid[
                "normalized_prediction"
            ]

            return hybrid

        routed_candidates = (
            fallback.merge(
                trocr_frame,
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
                    routed_candidates[
                        "document_type"
                    ]
                ),
                "field_name": (
                    routed_candidates[
                        "field_name"
                    ]
                ),
                "tesseract_prediction": (
                    routed_candidates[
                        "predicted_value"
                    ]
                ),
                "tesseract_normalized": (
                    routed_candidates[
                        "normalized_prediction"
                    ]
                ),
                "tesseract_confidence": (
                    routed_candidates[
                        "ocr_confidence"
                    ]
                ),
                "trocr_prediction": (
                    routed_candidates[
                        "trocr_prediction"
                    ]
                ),
                "trocr_normalized": (
                    routed_candidates[
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

        router_pipeline = (
            self.ocr_router_bundle[
                "pipeline"
            ]
        )

        router_threshold = float(
            self.ocr_router_bundle[
                "threshold"
            ]
        )

        probabilities = (
            router_pipeline.predict_proba(
                router_features[
                    ROUTER_FEATURES
                ]
            )[:, 1]
        )

        routed_candidates[
            "router_probability"
        ] = probabilities

        routed_candidates[
            "router_choose_trocr"
        ] = (
            probabilities
            >= router_threshold
        )

        routed_subset = (
            routed_candidates[
                [
                    "document_id",
                    "field_name",
                    "trocr_prediction",
                    "trocr_normalized",
                    "router_probability",
                    "router_choose_trocr",
                ]
            ]
        )

        final = hybrid.merge(
            routed_subset,
            on=[
                "document_id",
                "field_name",
            ],
            how="left",
            validate="one_to_one",
        )

        final[
            "trocr_prediction"
        ] = (
            final[
                "trocr_prediction"
            ]
            .fillna("")
            .astype(str)
        )

        final[
            "trocr_normalized"
        ] = (
            final[
                "trocr_normalized"
            ]
            .fillna("")
            .astype(str)
        )

        final[
            "router_probability"
        ] = pd.to_numeric(
            final[
                "router_probability"
            ],
            errors="coerce",
        ).fillna(0.0)

        final[
            "router_choose_trocr"
        ] = (
            final[
                "router_choose_trocr"
            ]
            .fillna(False)
            .astype(bool)
        )

        final["final_engine"] = np.where(
            final[
                "router_choose_trocr"
            ],
            "trocr",
            "tesseract",
        )

        final[
            "final_prediction"
        ] = np.where(
            final[
                "router_choose_trocr"
            ],
            final[
                "trocr_prediction"
            ],
            final[
                "predicted_value"
            ],
        )

        final[
            "final_normalized_prediction"
        ] = np.where(
            final[
                "router_choose_trocr"
            ],
            final[
                "trocr_normalized"
            ],
            final[
                "normalized_prediction"
            ],
        )

        return final

    def score_field_confidence(
        self,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        data = predictions.copy()

        data["final_exact_match"] = False
        data["final_similarity"] = 0.0

        engineered = (
            engineer_confidence_features(
                data
            )
        )

        pipeline = (
            self.field_confidence_bundle[
                "pipeline"
            ]
        )

        thresholds = (
            self.field_confidence_bundle[
                "thresholds"
            ]
        )

        probabilities = (
            pipeline.predict_proba(
                engineered[
                    CONFIDENCE_FEATURES
                ]
            )[:, 1]
        )

        engineered[
            "correctness_probability"
        ] = probabilities

        engineered[
            "risk_threshold"
        ] = (
            engineered[
                "risk_tier"
            ].map(thresholds)
        )

        engineered[
            "field_review_required"
        ] = (
            ~engineered[
                "format_valid"
            ].astype(bool)
            | (
                engineered[
                    "correctness_probability"
                ]
                < engineered[
                    "risk_threshold"
                ]
            )
            | (
                engineered[
                    "cross_field_flag"
                ]
                > 0
            )
        )

        return engineered


def review_reason(
    row: pd.Series,
) -> str:
    reasons: list[str] = []

    if not bool(
        row["format_valid"]
    ):
        reasons.append(
            clean_text(
                row[
                    "format_message"
                ]
            )
            or "format_validation_failed"
        )

    if float(
        row[
            "cross_field_flag"
        ]
    ) > 0:
        reasons.append(
            "cross_field_consistency_failure"
        )

    if float(
        row[
            "correctness_probability"
        ]
    ) < float(
        row[
            "risk_threshold"
        ]
    ):
        reasons.append(
            "correctness_probability_below_threshold"
        )

    if not reasons:
        reasons.append(
            "manual_verification_required"
        )

    return "; ".join(
        dict.fromkeys(reasons)
    )


def document_policy(
    fields: pd.DataFrame,
) -> tuple[str, int]:
    review = fields[
        fields[
            "field_review_required"
        ].astype(bool)
    ]

    if review.empty:
        return "auto_accept", 4

    required_invalid = (
        review[
            "required_field"
        ].astype(bool)
        & ~review[
            "format_valid"
        ].astype(bool)
    )

    cross_field_failure = (
        review[
            "cross_field_flag"
        ]
        > 0
    )

    if (
        required_invalid.any()
        or cross_field_failure.any()
    ):
        return "exception_review", 1

    critical = (
        review[
            "risk_tier"
        ]
        == "critical"
    )

    if critical.any():
        return (
            "critical_field_verification",
            2,
        )

    return "targeted_field_review", 3


def process_uploaded_document(
    *,
    upload_id: str,
    source_path: Path,
    expected_document_type: str,
    models: ProductionModels,
) -> ProcessingResult:
    prepared = prepare_document(
        upload_id=upload_id,
        source_path=source_path,
        expected_document_type=(
            expected_document_type
        ),
    )

    hybrid = run_hybrid_tesseract(
        prepared
    )

    routed = models.run_trocr_fallback(
        prepared,
        hybrid,
    )

    scored = (
        models.score_field_confidence(
            routed
        )
    )

    scored[
        "review_reason"
    ] = scored.apply(
        review_reason,
        axis=1,
    )

    status, priority = (
        document_policy(scored)
    )

    return ProcessingResult(
        document_id=(
            prepared.document_id
        ),
        document_type=(
            prepared.document_type
        ),
        source_image_path=(
            prepared.source_image_path
        ),
        annotation_path=(
            prepared.annotation_path
        ),
        field_results=scored,
        document_status=status,
        priority=priority,
        classification_scores=(
            prepared
            .classification_scores
        ),
    )
