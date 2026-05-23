from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import src.api.irs_review_api as review_api


def create_review_schema(
    database_path: Path,
) -> None:
    connection = sqlite3.connect(
        database_path
    )

    connection.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE review_documents (
            document_id TEXT PRIMARY KEY,
            parent_document_id TEXT NOT NULL,
            document_type TEXT NOT NULL,
            quality TEXT NOT NULL,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL,
            total_field_count INTEGER NOT NULL,
            accepted_field_count INTEGER NOT NULL,
            review_field_count INTEGER NOT NULL,
            critical_review_count INTEGER NOT NULL,
            structured_review_count INTEGER NOT NULL,
            descriptive_review_count INTEGER NOT NULL,
            exception_count INTEGER NOT NULL,
            review_state TEXT NOT NULL,
            reviewed_field_count INTEGER NOT NULL DEFAULT 0,
            assigned_to TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE review_fields (
            field_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            field_name TEXT NOT NULL,
            value TEXT,
            raw_value TEXT,
            engine TEXT,
            risk_tier TEXT NOT NULL,
            required INTEGER NOT NULL,
            correctness_probability REAL NOT NULL,
            acceptance_threshold REAL NOT NULL,
            format_valid INTEGER NOT NULL,
            cross_field_flag INTEGER NOT NULL,
            review_required INTEGER NOT NULL,
            review_reason TEXT,
            review_state TEXT NOT NULL,
            corrected_value TEXT,
            reviewer_note TEXT,
            reviewed_by TEXT,
            reviewed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (
                document_id,
                field_name
            ),
            FOREIGN KEY (
                document_id
            )
            REFERENCES review_documents (
                document_id
            )
            ON DELETE CASCADE
        );

        CREATE TABLE review_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            field_name TEXT,
            event_type TEXT NOT NULL,
            actor TEXT,
            details_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (
                document_id
            )
            REFERENCES review_documents (
                document_id
            )
            ON DELETE CASCADE
        );
        """
    )

    now = "2026-07-30T20:00:00+00:00"

    connection.execute(
        """
        INSERT INTO review_documents (
            document_id,
            parent_document_id,
            document_type,
            quality,
            status,
            priority,
            total_field_count,
            accepted_field_count,
            review_field_count,
            critical_review_count,
            structured_review_count,
            descriptive_review_count,
            exception_count,
            review_state,
            reviewed_field_count,
            assigned_to,
            created_at,
            updated_at
        )
        VALUES (
            'test_w2_001',
            'test_w2_001',
            'w2',
            'uploaded',
            'critical_field_verification',
            2,
            2,
            1,
            1,
            1,
            0,
            0,
            0,
            'pending',
            0,
            NULL,
            ?,
            ?
        )
        """,
        (
            now,
            now,
        ),
    )

    connection.execute(
        """
        INSERT INTO review_fields (
            document_id,
            field_name,
            value,
            raw_value,
            engine,
            risk_tier,
            required,
            correctness_probability,
            acceptance_threshold,
            format_valid,
            cross_field_flag,
            review_required,
            review_reason,
            review_state,
            created_at,
            updated_at
        )
        VALUES
        (
            'test_w2_001',
            'employee_ssn',
            '000123456',
            '000-12-3456',
            'tesseract',
            'critical',
            1,
            0.80,
            0.98,
            1,
            0,
            1,
            'correctness_probability_below_threshold',
            'pending',
            ?,
            ?
        ),
        (
            'test_w2_001',
            'wages',
            '1500.00',
            '1500.00',
            'tesseract',
            'critical',
            1,
            0.995,
            0.98,
            1,
            0,
            0,
            '',
            'accepted',
            ?,
            ?
        )
        """,
        (
            now,
            now,
            now,
            now,
        ),
    )

    connection.execute(
        """
        INSERT INTO review_events (
            document_id,
            field_name,
            event_type,
            actor,
            details_json,
            created_at
        )
        VALUES (
            'test_w2_001',
            NULL,
            'queue_imported',
            'system',
            ?,
            ?
        )
        """,
        (
            json.dumps(
                {
                    "source": "pytest",
                }
            ),
            now,
        ),
    )

    connection.commit()
    connection.close()


def create_media_fixture(
    project_root: Path,
    database_path: Path,
) -> None:
    media_directory = (
        project_root
        / "data"
        / "processed"
        / "test_media"
    )

    media_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_path = (
        media_directory
        / "source.png"
    )

    annotation_path = (
        media_directory
        / "annotation.json"
    )

    image = Image.new(
        "RGB",
        (
            800,
            1000,
        ),
        "white",
    )

    image.save(
        image_path
    )

    annotation = {
        "document_id": "test_w2_001",
        "field_boxes_pixels": {
            "employee_ssn": [
                100,
                120,
                300,
                180,
            ],
            "wages": [
                100,
                220,
                300,
                280,
            ],
        },
    }

    annotation_path.write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    connection = sqlite3.connect(
        database_path
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS document_media (
            document_id TEXT PRIMARY KEY,
            upload_id TEXT,
            source_image_path TEXT NOT NULL,
            annotation_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (
                document_id
            )
            REFERENCES review_documents (
                document_id
            )
            ON DELETE CASCADE
        )
        """
    )

    connection.execute(
        """
        INSERT INTO document_media (
            document_id,
            upload_id,
            source_image_path,
            annotation_path,
            created_at,
            updated_at
        )
        VALUES (
            'test_w2_001',
            NULL,
            ?,
            ?,
            '2026-07-30T20:00:00+00:00',
            '2026-07-30T20:00:00+00:00'
        )
        """,
        (
            image_path.relative_to(
                project_root
            ).as_posix(),
            annotation_path.relative_to(
                project_root
            ).as_posix(),
        ),
    )

    connection.commit()
    connection.close()


@pytest.fixture()
def api_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, Path | TestClient]]:
    project_root = (
        tmp_path
        / "DocIntelAI"
    )

    project_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    database_path = (
        project_root
        / "data"
        / "processed"
        / "irs_review_queue.db"
    )

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    upload_directory = (
        project_root
        / "data"
        / "uploads"
    )

    create_review_schema(
        database_path
    )

    create_media_fixture(
        project_root,
        database_path,
    )

    monkeypatch.setattr(
        review_api,
        "PROJECT_ROOT",
        project_root,
    )

    monkeypatch.setattr(
        review_api,
        "DATABASE_PATH",
        database_path,
    )

    monkeypatch.setattr(
        review_api,
        "UPLOAD_DIRECTORY",
        upload_directory,
    )

    review_api.load_document_media_lookup.cache_clear()

    with TestClient(
        review_api.app
    ) as client:
        yield {
            "client": client,
            "database_path": (
                database_path
            ),
            "project_root": (
                project_root
            ),
            "upload_directory": (
                upload_directory
            ),
        }
