from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEMPLATE_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "irs_templates"
)

PREVIEW_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "irs_template_previews"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_template_inspection.json"
)

TARGET_FILES = [
    "form_w2_2026.pdf",
    "form_1099_nec_2026.pdf",
]


def inspect_template(pdf_path: Path) -> dict[str, Any]:
    """Inspect one IRS PDF and generate page previews."""

    document = fitz.open(pdf_path)

    template_preview_dir = (
        PREVIEW_DIR / pdf_path.stem
    )

    template_preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pages: list[dict[str, Any]] = []

    for page_index in range(len(document)):
        page = document.load_page(page_index)
        rectangle = page.rect
        extracted_text = page.get_text("text").strip()

        preview_path = (
            template_preview_dir
            / f"page_{page_index + 1}.png"
        )

        # 1.5× scale gives a readable preview without huge files.
        matrix = fitz.Matrix(1.5, 1.5)
        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        pixmap.save(preview_path)

        pages.append(
            {
                "page_number": page_index + 1,
                "width_points": round(
                    rectangle.width,
                    2,
                ),
                "height_points": round(
                    rectangle.height,
                    2,
                ),
                "rotation": page.rotation,
                "text_characters": len(
                    extracted_text
                ),
                "text_preview": (
                    extracted_text[:500]
                ),
                "preview_path": str(
                    preview_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
            }
        )

    result = {
        "filename": pdf_path.name,
        "file_size_bytes": pdf_path.stat().st_size,
        "page_count": len(document),
        "pages": pages,
    }

    document.close()

    return result


def run_inspection() -> None:
    """Inspect all downloaded IRS templates."""

    results: dict[str, Any] = {
        "templates": {},
    }

    for filename in TARGET_FILES:
        pdf_path = TEMPLATE_DIR / filename

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"Template not found: {pdf_path}"
            )

        print(f"Inspecting {filename}...")

        results["templates"][filename] = (
            inspect_template(pdf_path)
        )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\nIRS template inspection completed")

    for filename, details in (
        results["templates"].items()
    ):
        print(
            f"{filename}: "
            f"{details['page_count']} pages"
        )

    print(f"Preview directory: {PREVIEW_DIR}")
    print(f"Inspection report: {REPORT_PATH}")


if __name__ == "__main__":
    run_inspection()