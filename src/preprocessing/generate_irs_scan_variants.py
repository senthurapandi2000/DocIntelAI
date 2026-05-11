from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "irs_synthetic_manifest.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "irs_scan_variants"
)

ANNOTATION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "irs_scan_variants"
)

OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "irs_scan_variants_manifest.csv"
)

DEFAULT_SEED = 42
DEFAULT_VARIANTS_PER_DOCUMENT = 2


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def rotate_point(
    x: float,
    y: float,
    *,
    center_x: float,
    center_y: float,
    angle_degrees: float,
) -> tuple[float, float]:
    """
    Rotate a point using image coordinates where y increases downward.

    Positive angles match Pillow's visual counterclockwise rotation.
    """

    radians = math.radians(angle_degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)

    translated_x = x - center_x
    translated_y = y - center_y

    rotated_x = (
        center_x
        + cosine * translated_x
        + sine * translated_y
    )

    rotated_y = (
        center_y
        - sine * translated_x
        + cosine * translated_y
    )

    return rotated_x, rotated_y


def rotate_box(
    box: list[int | float],
    *,
    image_width: int,
    image_height: int,
    angle_degrees: float,
) -> list[int]:
    """Rotate an axis-aligned box and return its new enclosing box."""

    x1, y1, x2, y2 = map(float, box)

    center_x = image_width / 2.0
    center_y = image_height / 2.0

    corners = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]

    rotated = [
        rotate_point(
            x,
            y,
            center_x=center_x,
            center_y=center_y,
            angle_degrees=angle_degrees,
        )
        for x, y in corners
    ]

    xs = [point[0] for point in rotated]
    ys = [point[1] for point in rotated]

    new_x1 = int(round(clip(min(xs), 0, image_width - 1)))
    new_y1 = int(round(clip(min(ys), 0, image_height - 1)))
    new_x2 = int(round(clip(max(xs), 1, image_width)))
    new_y2 = int(round(clip(max(ys), 1, image_height)))

    if new_x2 <= new_x1:
        new_x2 = min(image_width, new_x1 + 1)

    if new_y2 <= new_y1:
        new_y2 = min(image_height, new_y1 + 1)

    return [new_x1, new_y1, new_x2, new_y2]


def add_gaussian_noise(
    image: Image.Image,
    *,
    sigma: float,
    rng: np.random.Generator,
) -> Image.Image:
    """Add low-amplitude Gaussian sensor or scan noise."""

    array = np.asarray(image).astype(np.float32)

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=array.shape,
    )

    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy, mode="RGB")


def add_soft_shadow(
    image: Image.Image,
    *,
    strength: float,
    direction: str,
) -> Image.Image:
    """Apply a soft directional shadow while preserving document content."""

    array = np.asarray(image).astype(np.float32)
    height, width, _ = array.shape

    if direction in {"left", "right"}:
        gradient = np.linspace(
            1.0 - strength,
            1.0,
            width,
            dtype=np.float32,
        )

        if direction == "right":
            gradient = gradient[::-1]

        mask = gradient.reshape(1, width, 1)

    else:
        gradient = np.linspace(
            1.0 - strength,
            1.0,
            height,
            dtype=np.float32,
        )

        if direction == "bottom":
            gradient = gradient[::-1]

        mask = gradient.reshape(height, 1, 1)

    shadowed = np.clip(array * mask, 0, 255).astype(np.uint8)

    return Image.fromarray(shadowed, mode="RGB")


def apply_jpeg_roundtrip(
    image: Image.Image,
    *,
    quality: int,
) -> Image.Image:
    """Introduce realistic JPEG compression artifacts in memory."""

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
    )

    buffer.seek(0)

    compressed = Image.open(buffer).convert("RGB")
    compressed.load()

    return compressed


def augmentation_parameters(
    profile: str,
    rng: random.Random,
) -> dict[str, Any]:
    """Sample parameters for a mild or hard scan profile."""

    if profile == "mild":
        return {
            "rotation_degrees": rng.uniform(-1.25, 1.25),
            "blur_radius": rng.uniform(0.15, 0.65),
            "brightness_factor": rng.uniform(0.96, 1.04),
            "contrast_factor": rng.uniform(0.90, 1.06),
            "noise_sigma": rng.uniform(2.0, 5.0),
            "jpeg_quality": rng.randint(72, 90),
            "low_resolution_scale": rng.uniform(0.88, 1.0),
            "shadow_strength": (
                rng.uniform(0.05, 0.12)
                if rng.random() < 0.40
                else 0.0
            ),
            "shadow_direction": rng.choice(
                ["left", "right", "top", "bottom"]
            ),
        }

    return {
        "rotation_degrees": rng.uniform(-2.5, 2.5),
        "blur_radius": rng.uniform(0.65, 1.35),
        "brightness_factor": rng.uniform(0.86, 1.02),
        "contrast_factor": rng.uniform(0.76, 1.02),
        "noise_sigma": rng.uniform(5.0, 10.0),
        "jpeg_quality": rng.randint(48, 70),
        "low_resolution_scale": rng.uniform(0.66, 0.84),
        "shadow_strength": rng.uniform(0.10, 0.24),
        "shadow_direction": rng.choice(
            ["left", "right", "top", "bottom"]
        ),
    }


def augment_image(
    source: Image.Image,
    *,
    parameters: dict[str, Any],
    numpy_rng: np.random.Generator,
) -> Image.Image:
    """Apply geometric and photometric scan degradation."""

    image = source.convert("RGB")
    width, height = image.size

    angle = float(parameters["rotation_degrees"])

    image = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )

    low_resolution_scale = float(
        parameters["low_resolution_scale"]
    )

    if low_resolution_scale < 0.999:
        reduced_width = max(
            1,
            round(width * low_resolution_scale),
        )
        reduced_height = max(
            1,
            round(height * low_resolution_scale),
        )

        image = image.resize(
            (reduced_width, reduced_height),
            resample=Image.Resampling.BILINEAR,
        )

        image = image.resize(
            (width, height),
            resample=Image.Resampling.BICUBIC,
        )

    image = ImageEnhance.Brightness(image).enhance(
        float(parameters["brightness_factor"])
    )

    image = ImageEnhance.Contrast(image).enhance(
        float(parameters["contrast_factor"])
    )

    shadow_strength = float(
        parameters["shadow_strength"]
    )

    if shadow_strength > 0:
        image = add_soft_shadow(
            image,
            strength=shadow_strength,
            direction=str(
                parameters["shadow_direction"]
            ),
        )

    blur_radius = float(parameters["blur_radius"])

    if blur_radius > 0:
        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=blur_radius
            )
        )

    image = add_gaussian_noise(
        image,
        sigma=float(parameters["noise_sigma"]),
        rng=numpy_rng,
    )

    image = apply_jpeg_roundtrip(
        image,
        quality=int(parameters["jpeg_quality"]),
    )

    return image


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def create_variants(
    *,
    variants_per_document: int,
    seed: int,
) -> None:
    """Generate scan variants while preserving original dataset splits."""

    if not BASE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Base manifest not found: {BASE_MANIFEST}"
        )

    base_manifest = pd.read_csv(BASE_MANIFEST)

    required_columns = {
        "document_id",
        "document_type",
        "split",
        "image_path",
        "annotation_path",
    }

    missing_columns = (
        required_columns - set(base_manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Base manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    manifest_rows: list[dict[str, Any]] = []

    total_variants = (
        len(base_manifest)
        * variants_per_document
    )
    completed = 0

    for document_index, row in base_manifest.iterrows():
        document_id = str(row["document_id"])
        document_type = str(row["document_type"])
        split = str(row["split"])

        source_image_path = (
            PROJECT_ROOT
            / str(row["image_path"])
        )

        source_annotation_path = (
            PROJECT_ROOT
            / str(row["annotation_path"])
        )

        if not source_image_path.exists():
            raise FileNotFoundError(
                f"Base image not found: {source_image_path}"
            )

        if not source_annotation_path.exists():
            raise FileNotFoundError(
                "Base annotation not found: "
                f"{source_annotation_path}"
            )

        source_image = Image.open(
            source_image_path
        ).convert("RGB")

        base_annotation = load_json(
            source_annotation_path
        )

        source_boxes = base_annotation.get(
            "field_boxes_pixels",
            {},
        )

        width, height = source_image.size

        for variant_index in range(
            1,
            variants_per_document + 1,
        ):
            completed += 1

            profile = (
                "mild"
                if variant_index % 2 == 1
                else "hard"
            )

            variant_seed = (
                seed
                + document_index * 10_000
                + variant_index
            )

            python_rng = random.Random(
                variant_seed
            )
            numpy_rng = np.random.default_rng(
                variant_seed
            )

            parameters = augmentation_parameters(
                profile,
                python_rng,
            )

            variant_image = augment_image(
                source_image,
                parameters=parameters,
                numpy_rng=numpy_rng,
            )

            variant_id = (
                f"{document_id}_scan_"
                f"{profile}_{variant_index:02d}"
            )

            output_image_path = (
                OUTPUT_ROOT
                / document_type
                / split
                / f"{variant_id}.jpg"
            )

            annotation_path = (
                ANNOTATION_ROOT
                / document_type
                / split
                / f"{variant_id}.json"
            )

            output_image_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            variant_image.save(
                output_image_path,
                format="JPEG",
                quality=95,
                optimize=True,
            )

            angle = float(
                parameters["rotation_degrees"]
            )

            transformed_boxes = {
                field_name: rotate_box(
                    box,
                    image_width=width,
                    image_height=height,
                    angle_degrees=angle,
                )
                for field_name, box in (
                    source_boxes.items()
                )
            }

            annotation = {
                **base_annotation,
                "document_id": variant_id,
                "parent_document_id": document_id,
                "document_type": document_type,
                "split": split,
                "variant": True,
                "variant_profile": profile,
                "image_path": str(
                    output_image_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "annotation_path": str(
                    annotation_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "source_image_path": str(
                    source_image_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "image_width": width,
                "image_height": height,
                "field_boxes_pixels": (
                    transformed_boxes
                ),
                "augmentation": {
                    key: (
                        round(value, 4)
                        if isinstance(value, float)
                        else value
                    )
                    for key, value in (
                        parameters.items()
                    )
                },
            }

            save_json(
                annotation,
                annotation_path,
            )

            manifest_rows.append(
                {
                    "document_id": variant_id,
                    "parent_document_id": (
                        document_id
                    ),
                    "document_type": (
                        document_type
                    ),
                    "split": split,
                    "variant_profile": profile,
                    "image_path": str(
                        output_image_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "annotation_path": str(
                        annotation_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "rotation_degrees": round(
                        angle,
                        4,
                    ),
                    "blur_radius": round(
                        float(
                            parameters[
                                "blur_radius"
                            ]
                        ),
                        4,
                    ),
                    "noise_sigma": round(
                        float(
                            parameters[
                                "noise_sigma"
                            ]
                        ),
                        4,
                    ),
                    "jpeg_quality": int(
                        parameters[
                            "jpeg_quality"
                        ]
                    ),
                }
            )

            print(
                f"[{completed}/{total_variants}] "
                f"Generated {variant_id}"
            )

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
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
    ]

    with OUTPUT_MANIFEST.open(
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

    profile_counts = {
        str(profile): int(count)
        for profile, count in (
            pd.DataFrame(manifest_rows)[
                "variant_profile"
            ]
            .value_counts()
            .items()
        )
    }

    split_counts = {
        str(split): int(count)
        for split, count in (
            pd.DataFrame(manifest_rows)[
                "split"
            ]
            .value_counts()
            .items()
        )
    }

    print("\nIRS scan variants generated")
    print(
        f"Base documents: {len(base_manifest)}"
    )
    print(
        f"Variants per document: "
        f"{variants_per_document}"
    )
    print(
        f"Total variants: {len(manifest_rows)}"
    )
    print(f"Profiles: {profile_counts}")
    print(f"Split counts: {split_counts}")
    print(f"Images: {OUTPUT_ROOT}")
    print(f"Annotations: {ANNOTATION_ROOT}")
    print(f"Manifest: {OUTPUT_MANIFEST}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate realistic scan variants for "
            "synthetic IRS documents."
        )
    )

    parser.add_argument(
        "--variants-per-document",
        type=int,
        default=DEFAULT_VARIANTS_PER_DOCUMENT,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    create_variants(
        variants_per_document=(
            arguments.variants_per_document
        ),
        seed=arguments.seed,
    )
