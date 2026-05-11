from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "irs_templates"
)

MANIFEST_PATH = OUTPUT_DIR / "template_manifest.json"

TEMPLATES = {
    "w2_2026": {
        "url": "https://www.irs.gov/pub/irs-pdf/fw2.pdf",
        "filename": "form_w2_2026.pdf",
    },
    "1099_nec_2026": {
        "url": "https://www.irs.gov/pub/irs-pdf/f1099nec.pdf",
        "filename": "form_1099_nec_2026.pdf",
    },
}


class IRSTemplateDownloadError(RuntimeError):
    """Raised when an IRS template cannot be downloaded safely."""


def calculate_sha256(path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_pdf(
    session: requests.Session,
    url: str,
    output_path: Path,
) -> dict[str, object]:
    """Download and validate one PDF."""

    response = session.get(url, timeout=60)
    response.raise_for_status()

    content = response.content

    if not content.startswith(b"%PDF"):
        raise IRSTemplateDownloadError(
            f"Downloaded content is not a PDF: {url}"
        )

    if len(content) < 10_000:
        raise IRSTemplateDownloadError(
            f"Downloaded PDF is unexpectedly small: {url}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(content)

    return {
        "source_url": url,
        "local_path": str(
            output_path.relative_to(PROJECT_ROOT)
        ),
        "size_bytes": output_path.stat().st_size,
        "sha256": calculate_sha256(output_path),
        "downloaded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def run_download() -> None:
    """Download official IRS document templates."""

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "DocIntelAI document-research project"
            ),
            "Accept": "application/pdf",
        }
    )

    manifest: dict[str, object] = {
        "source": "Internal Revenue Service",
        "templates": {},
    }

    for template_name, template in TEMPLATES.items():
        output_path = (
            OUTPUT_DIR
            / str(template["filename"])
        )

        print(f"Downloading {template_name}...")

        result = download_pdf(
            session=session,
            url=str(template["url"]),
            output_path=output_path,
        )

        manifest["templates"][template_name] = result

        print(
            f"Saved: {result['local_path']} "
            f"({result['size_bytes']} bytes)"
        )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print("\nIRS template download completed")
    print(f"Templates downloaded: {len(TEMPLATES)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    run_download()