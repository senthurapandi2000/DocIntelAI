from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "irs_review_queue.db"
)

DATABASE_PATH = Path(
    os.getenv(
        "IRS_REVIEW_DB",
        str(DEFAULT_DATABASE_PATH),
    )
).resolve()


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


UPLOAD_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "uploads"
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

ALLOWED_UPLOAD_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


@contextmanager
def database_connection() -> Iterator[
    sqlite3.Connection
]:
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30.0,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    connection.execute(
        "PRAGMA busy_timeout = 30000"
    )

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def row_to_dict(
    row: sqlite3.Row | None,
) -> dict[str, Any] | None:
    if row is None:
        return None

    return dict(row)


def require_database() -> None:
    if not DATABASE_PATH.exists():
        raise RuntimeError(
            "IRS review database does not exist: "
            f"{DATABASE_PATH}"
        )


def initialize_upload_schema() -> None:
    UPLOAD_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with database_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS document_uploads (
                upload_id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                file_extension TEXT NOT NULL,
                content_type TEXT,
                file_size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                expected_document_type TEXT,
                uploaded_by TEXT,
                upload_status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS processing_jobs (
                job_id TEXT PRIMARY KEY,
                upload_id TEXT NOT NULL,
                job_status TEXT NOT NULL,
                processing_stage TEXT NOT NULL,
                progress_percent REAL NOT NULL DEFAULT 0,
                detected_document_type TEXT,
                result_document_id TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (
                    upload_id
                )
                REFERENCES document_uploads (
                    upload_id
                )
                ON DELETE CASCADE
            );


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
                ON DELETE CASCADE,
                FOREIGN KEY (
                    upload_id
                )
                REFERENCES document_uploads (
                    upload_id
                )
                ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_document_uploads_status
            ON document_uploads (
                upload_status,
                created_at
            );

            CREATE INDEX IF NOT EXISTS idx_processing_jobs_queue
            ON processing_jobs (
                job_status,
                created_at
            );


            CREATE INDEX IF NOT EXISTS idx_document_media_upload
            ON document_media (
                upload_id
            );
            """
        )


def sanitize_filename(
    filename: str,
) -> str:
    name = Path(
        filename
    ).name.strip()

    if not name:
        raise HTTPException(
            status_code=422,
            detail=(
                "The uploaded file must have "
                "a filename."
            ),
        )

    safe_characters = []

    for character in name:
        if (
            character.isalnum()
            or character
            in {
                ".",
                "-",
                "_",
            }
        ):
            safe_characters.append(
                character
            )
        else:
            safe_characters.append(
                "_"
            )

    sanitized = "".join(
        safe_characters
    ).strip("._")

    if not sanitized:
        raise HTTPException(
            status_code=422,
            detail=(
                "The filename does not contain "
                "usable characters."
            ),
        )

    return sanitized


def serialize_upload_row(
    row: sqlite3.Row,
) -> dict[str, Any]:
    return {
        key: row[key]
        for key in row.keys()
    }


async def save_uploaded_document(
    upload: UploadFile,
) -> dict[str, Any]:
    original_filename = (
        upload.filename
        or ""
    )

    safe_filename = (
        sanitize_filename(
            original_filename
        )
    )

    extension = Path(
        safe_filename
    ).suffix.lower()

    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported file type. "
                "Allowed extensions: "
                ".png, .jpg, .jpeg, .pdf."
            ),
        )

    upload_id = str(
        uuid.uuid4()
    )

    stored_filename = (
        f"{upload_id}{extension}"
    )

    stored_path = (
        UPLOAD_DIRECTORY
        / stored_filename
    ).resolve()

    hasher = hashlib.sha256()
    total_bytes = 0

    try:
        with stored_path.open(
            "wb"
        ) as destination:
            while True:
                chunk = await upload.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_bytes += len(
                    chunk
                )

                if (
                    total_bytes
                    > MAX_UPLOAD_BYTES
                ):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded file exceeds "
                            "the 20 MB limit."
                        ),
                    )

                hasher.update(
                    chunk
                )

                destination.write(
                    chunk
                )
    except Exception:
        stored_path.unlink(
            missing_ok=True
        )
        raise
    finally:
        await upload.close()

    if total_bytes == 0:
        stored_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=422,
            detail=(
                "The uploaded file is empty."
            ),
        )

    return {
        "upload_id": upload_id,
        "original_filename": (
            original_filename
        ),
        "stored_filename": (
            stored_filename
        ),
        "stored_path": str(
            stored_path
        ),
        "file_extension": extension,
        "content_type": (
            upload.content_type
        ),
        "file_size_bytes": (
            total_bytes
        ),
        "sha256": (
            hasher.hexdigest()
        ),
    }


@lru_cache(maxsize=1)
def load_document_media_lookup() -> dict[
    str,
    dict[str, Any],
]:
    if not CLEAN_MANIFEST_PATH.exists():
        raise RuntimeError(
            "Clean IRS manifest does not exist: "
            f"{CLEAN_MANIFEST_PATH}"
        )

    if not VARIANT_MANIFEST_PATH.exists():
        raise RuntimeError(
            "IRS scan-variant manifest does not exist: "
            f"{VARIANT_MANIFEST_PATH}"
        )

    clean = pd.read_csv(
        CLEAN_MANIFEST_PATH
    ).copy()

    clean["quality"] = "clean"

    variants = pd.read_csv(
        VARIANT_MANIFEST_PATH
    ).copy()

    variants["quality"] = (
        variants["variant_profile"]
    )

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
            clean[columns],
            variants[columns],
        ],
        ignore_index=True,
    )

    return (
        combined
        .drop_duplicates(
            "document_id"
        )
        .set_index(
            "document_id"
        )
        .to_dict(
            orient="index"
        )
    )


def safe_project_path(
    relative_path: str,
) -> Path:
    candidate = (
        PROJECT_ROOT
        / relative_path
    ).resolve()

    try:
        candidate.relative_to(
            PROJECT_ROOT.resolve()
        )
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Document media path is outside "
                "the project directory."
            ),
        ) from error

    return candidate


def load_document_media(
    document_id: str,
) -> tuple[
    Image.Image,
    dict[str, Any],
]:
    database_record = None

    with database_connection() as connection:
        table_exists = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE
                type = 'table'
                AND name = 'document_media'
            """
        ).fetchone()

        if table_exists is not None:
            database_record = connection.execute(
                """
                SELECT
                    source_image_path,
                    annotation_path
                FROM document_media
                WHERE document_id = ?
                """,
                (
                    document_id,
                ),
            ).fetchone()

    if database_record is not None:
        image_path = safe_project_path(
            str(
                database_record[
                    "source_image_path"
                ]
            )
        )

        annotation_path = safe_project_path(
            str(
                database_record[
                    "annotation_path"
                ]
            )
        )

    else:
        try:
            lookup = (
                load_document_media_lookup()
            )
        except RuntimeError as error:
            raise HTTPException(
                status_code=500,
                detail=str(error),
            ) from error

        record = lookup.get(
            document_id
        )

        if record is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Document media was not found "
                    "in the review database or "
                    "IRS manifests."
                ),
            )

        image_path = safe_project_path(
            str(
                record["image_path"]
            )
        )

        annotation_path = safe_project_path(
            str(
                record[
                    "annotation_path"
                ]
            )
        )

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Source image file was not found: "
                f"{image_path}"
            ),
        )

    if not annotation_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Annotation file was not found: "
                f"{annotation_path}"
            ),
        )

    try:
        with annotation_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            annotation = json.load(
                file
            )
    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not read the annotation "
                f"file: {error}"
            ),
        ) from error

    try:
        with Image.open(
            image_path
        ) as source:
            image = source.convert(
                "RGB"
            )
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not open the source image: "
                f"{error}"
            ),
        ) from error

    return image, annotation


def get_field_box(
    annotation: dict[str, Any],
    field_name: str,
) -> tuple[int, int, int, int]:
    raw_box = (
        annotation
        .get(
            "field_boxes_pixels",
            {},
        )
        .get(field_name)
    )

    if (
        not isinstance(
            raw_box,
            list,
        )
        or len(raw_box) != 4
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                "The selected field does not "
                "have a source-image bounding box."
            ),
        )

    try:
        x1, y1, x2, y2 = [
            int(round(float(value)))
            for value in raw_box
        ]
    except (
        TypeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "The field bounding box is "
                "invalid."
            ),
        ) from error

    return x1, y1, x2, y2


def image_response(
    image: Image.Image,
    *,
    filename: str,
) -> Response:
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "Cache-Control": (
                "no-store"
            ),
            "Content-Disposition": (
                "inline; "
                f'filename="{filename}"'
            ),
        },
    )


def insert_event(
    connection: sqlite3.Connection,
    *,
    document_id: str,
    event_type: str,
    actor: str,
    details: dict[str, Any],
    field_name: str | None = None,
) -> None:
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
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            field_name,
            event_type,
            actor,
            json.dumps(
                details,
                ensure_ascii=False,
            ),
            utc_now(),
        ),
    )


def recalculate_document_state(
    connection: sqlite3.Connection,
    document_id: str,
) -> dict[str, int | str]:
    counts = connection.execute(
        """
        SELECT
            SUM(
                CASE
                    WHEN review_required = 1
                    THEN 1
                    ELSE 0
                END
            ) AS review_fields,
            SUM(
                CASE
                    WHEN review_required = 1
                         AND review_state = 'pending'
                    THEN 1
                    ELSE 0
                END
            ) AS pending_fields,
            SUM(
                CASE
                    WHEN review_required = 1
                         AND review_state IN (
                             'approved',
                             'corrected'
                         )
                    THEN 1
                    ELSE 0
                END
            ) AS reviewed_fields
        FROM review_fields
        WHERE document_id = ?
        """,
        (document_id,),
    ).fetchone()

    if counts is None:
        raise HTTPException(
            status_code=404,
            detail="Document fields not found.",
        )

    review_fields = int(
        counts["review_fields"] or 0
    )

    pending_fields = int(
        counts["pending_fields"] or 0
    )

    reviewed_fields = int(
        counts["reviewed_fields"] or 0
    )

    current = connection.execute(
        """
        SELECT review_state
        FROM review_documents
        WHERE document_id = ?
        """,
        (document_id,),
    ).fetchone()

    if current is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    current_state = str(
        current["review_state"]
    )

    if current_state == "completed":
        next_state = "completed"
    elif pending_fields == 0:
        next_state = (
            "ready_for_completion"
        )
    elif reviewed_fields > 0:
        next_state = "in_progress"
    else:
        next_state = "pending"

    connection.execute(
        """
        UPDATE review_documents
        SET
            reviewed_field_count = ?,
            review_state = ?,
            updated_at = ?
        WHERE document_id = ?
        """,
        (
            reviewed_fields,
            next_state,
            utc_now(),
            document_id,
        ),
    )

    return {
        "review_fields": review_fields,
        "pending_fields": pending_fields,
        "reviewed_fields": reviewed_fields,
        "review_state": next_state,
    }


class AssignmentRequest(BaseModel):
    reviewer: str = Field(
        min_length=1,
        max_length=100,
    )

    force: bool = False


class FieldDecisionRequest(BaseModel):
    action: Literal[
        "approve",
        "correct",
    ]

    reviewer: str = Field(
        min_length=1,
        max_length=100,
    )

    corrected_value: str | None = Field(
        default=None,
        max_length=2000,
    )

    note: str | None = Field(
        default=None,
        max_length=4000,
    )


class CompletionRequest(BaseModel):
    reviewer: str = Field(
        min_length=1,
        max_length=100,
    )

    note: str | None = Field(
        default=None,
        max_length=4000,
    )



class JobActionRequest(BaseModel):
    actor: str = Field(
        min_length=1,
        max_length=100,
    )

    note: str | None = Field(
        default=None,
        max_length=4000,
    )


@asynccontextmanager
async def lifespan(
    _: FastAPI,
):
    require_database()
    initialize_upload_schema()
    yield


app = FastAPI(
    title="DocIntelAI IRS Review API",
    version="1.0.0",
    description=(
        "Field-level human-review API for "
        "W-2 and 1099-NEC OCR results."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    require_database()

    with database_connection() as connection:
        connection.execute(
            "SELECT 1"
        ).fetchone()

    return {
        "status": "healthy",
        "database": str(
            DATABASE_PATH
        ),
    }


@app.get(
    "/api/v1/review/statistics"
)
def review_statistics() -> dict[str, Any]:
    with database_connection() as connection:
        document_counts = connection.execute(
            """
            SELECT
                COUNT(*) AS total_documents,
                SUM(
                    CASE
                        WHEN review_state = 'pending'
                        THEN 1
                        ELSE 0
                    END
                ) AS pending_documents,
                SUM(
                    CASE
                        WHEN review_state = 'in_progress'
                        THEN 1
                        ELSE 0
                    END
                ) AS in_progress_documents,
                SUM(
                    CASE
                        WHEN review_state = 'ready_for_completion'
                        THEN 1
                        ELSE 0
                    END
                ) AS ready_documents,
                SUM(
                    CASE
                        WHEN review_state = 'completed'
                        THEN 1
                        ELSE 0
                    END
                ) AS completed_documents
            FROM review_documents
            """
        ).fetchone()

        field_counts = connection.execute(
            """
            SELECT
                COUNT(*) AS total_fields,
                SUM(
                    CASE
                        WHEN review_required = 0
                        THEN 1
                        ELSE 0
                    END
                ) AS auto_accepted_fields,
                SUM(
                    CASE
                        WHEN review_required = 1
                             AND review_state = 'pending'
                        THEN 1
                        ELSE 0
                    END
                ) AS pending_review_fields,
                SUM(
                    CASE
                        WHEN review_state = 'approved'
                        THEN 1
                        ELSE 0
                    END
                ) AS approved_fields,
                SUM(
                    CASE
                        WHEN review_state = 'corrected'
                        THEN 1
                        ELSE 0
                    END
                ) AS corrected_fields
            FROM review_fields
            """
        ).fetchone()

    return {
        "documents": {
            key: int(
                document_counts[key] or 0
            )
            for key in (
                "total_documents",
                "pending_documents",
                "in_progress_documents",
                "ready_documents",
                "completed_documents",
            )
        },
        "fields": {
            key: int(
                field_counts[key] or 0
            )
            for key in (
                "total_fields",
                "auto_accepted_fields",
                "pending_review_fields",
                "approved_fields",
                "corrected_fields",
            )
        },
    }


@app.get(
    "/api/v1/review/documents"
)
def list_review_documents(
    review_state: str | None = None,
    status: str | None = None,
    assigned_to: str | None = None,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> dict[str, Any]:
    conditions: list[str] = []
    parameters: list[Any] = []

    if review_state:
        conditions.append(
            "review_state = ?"
        )
        parameters.append(
            review_state
        )

    if status:
        conditions.append(
            "status = ?"
        )
        parameters.append(status)

    if assigned_to:
        conditions.append(
            "assigned_to = ?"
        )
        parameters.append(
            assigned_to
        )

    where_clause = (
        "WHERE "
        + " AND ".join(conditions)
        if conditions
        else ""
    )

    with database_connection() as connection:
        total = connection.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM review_documents
            {where_clause}
            """,
            parameters,
        ).fetchone()

        rows = connection.execute(
            f"""
            SELECT
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
            FROM review_documents
            {where_clause}
            ORDER BY
                priority ASC,
                exception_count DESC,
                critical_review_count DESC,
                review_field_count DESC,
                document_id ASC
            LIMIT ? OFFSET ?
            """,
            [
                *parameters,
                limit,
                offset,
            ],
        ).fetchall()

    return {
        "total": int(
            total["count"] or 0
        ),
        "limit": limit,
        "offset": offset,
        "items": [
            dict(row)
            for row in rows
        ],
    }


@app.get(
    "/api/v1/review/documents/{document_id}"
)
def get_review_document(
    document_id: str,
) -> dict[str, Any]:
    with database_connection() as connection:
        document = connection.execute(
            """
            SELECT *
            FROM review_documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        fields = connection.execute(
            """
            SELECT
                field_id,
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
                corrected_value,
                reviewer_note,
                reviewed_by,
                reviewed_at,
                created_at,
                updated_at
            FROM review_fields
            WHERE document_id = ?
            ORDER BY
                review_required DESC,
                CASE risk_tier
                    WHEN 'critical' THEN 1
                    WHEN 'structured' THEN 2
                    ELSE 3
                END,
                field_id ASC
            """,
            (document_id,),
        ).fetchall()

    return {
        "document": dict(document),
        "fields": [
            dict(field)
            for field in fields
        ],
    }


@app.post(
    "/api/v1/review/documents/{document_id}/assign"
)
def assign_review_document(
    document_id: str,
    request: AssignmentRequest,
) -> dict[str, Any]:
    reviewer = request.reviewer.strip()

    with database_connection() as connection:
        document = connection.execute(
            """
            SELECT
                assigned_to,
                review_state
            FROM review_documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        if (
            document["review_state"]
            == "completed"
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Completed documents cannot "
                    "be reassigned."
                ),
            )

        current_assignee = (
            document["assigned_to"]
        )

        if (
            current_assignee
            and current_assignee != reviewer
            and not request.force
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Document is already assigned "
                    f"to {current_assignee}."
                ),
            )

        next_state = (
            "in_progress"
            if document["review_state"]
            == "pending"
            else document["review_state"]
        )

        connection.execute(
            """
            UPDATE review_documents
            SET
                assigned_to = ?,
                review_state = ?,
                updated_at = ?
            WHERE document_id = ?
            """,
            (
                reviewer,
                next_state,
                utc_now(),
                document_id,
            ),
        )

        insert_event(
            connection,
            document_id=document_id,
            event_type="document_assigned",
            actor=reviewer,
            details={
                "previous_assignee": (
                    current_assignee
                ),
                "forced": request.force,
            },
        )

    return {
        "document_id": document_id,
        "assigned_to": reviewer,
        "review_state": next_state,
    }


@app.post(
    "/api/v1/review/documents/"
    "{document_id}/fields/"
    "{field_name}/decision"
)
def submit_field_decision(
    document_id: str,
    field_name: str,
    request: FieldDecisionRequest,
) -> dict[str, Any]:
    reviewer = request.reviewer.strip()

    corrected_value = (
        request.corrected_value.strip()
        if request.corrected_value
        is not None
        else None
    )

    if (
        request.action == "correct"
        and not corrected_value
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "corrected_value is required "
                "when action is 'correct'."
            ),
        )

    with database_connection() as connection:
        field = connection.execute(
            """
            SELECT
                review_required,
                review_state,
                value
            FROM review_fields
            WHERE
                document_id = ?
                AND field_name = ?
            """,
            (
                document_id,
                field_name,
            ),
        ).fetchone()

        if field is None:
            raise HTTPException(
                status_code=404,
                detail="Review field not found.",
            )

        if not int(
            field["review_required"]
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "This field was already "
                    "auto accepted."
                ),
            )

        next_field_state = (
            "approved"
            if request.action
            == "approve"
            else "corrected"
        )

        connection.execute(
            """
            UPDATE review_fields
            SET
                review_state = ?,
                corrected_value = ?,
                reviewer_note = ?,
                reviewed_by = ?,
                reviewed_at = ?,
                updated_at = ?
            WHERE
                document_id = ?
                AND field_name = ?
            """,
            (
                next_field_state,
                corrected_value,
                request.note,
                reviewer,
                utc_now(),
                utc_now(),
                document_id,
                field_name,
            ),
        )

        insert_event(
            connection,
            document_id=document_id,
            field_name=field_name,
            event_type=(
                "field_approved"
                if request.action
                == "approve"
                else "field_corrected"
            ),
            actor=reviewer,
            details={
                "original_value": (
                    field["value"]
                ),
                "corrected_value": (
                    corrected_value
                ),
                "note": request.note,
                "previous_state": (
                    field["review_state"]
                ),
            },
        )

        state = (
            recalculate_document_state(
                connection,
                document_id,
            )
        )

    return {
        "document_id": document_id,
        "field_name": field_name,
        "field_review_state": (
            next_field_state
        ),
        "document_review_state": (
            state["review_state"]
        ),
        "pending_fields": (
            state["pending_fields"]
        ),
        "reviewed_fields": (
            state["reviewed_fields"]
        ),
    }


@app.post(
    "/api/v1/review/documents/"
    "{document_id}/complete"
)
def complete_review_document(
    document_id: str,
    request: CompletionRequest,
) -> dict[str, Any]:
    reviewer = request.reviewer.strip()

    with database_connection() as connection:
        document = connection.execute(
            """
            SELECT review_state
            FROM review_documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        pending = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM review_fields
            WHERE
                document_id = ?
                AND review_required = 1
                AND review_state = 'pending'
            """,
            (document_id,),
        ).fetchone()

        pending_count = int(
            pending["count"] or 0
        )

        if pending_count > 0:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Document cannot be completed "
                    f"while {pending_count} review "
                    "fields remain pending."
                ),
            )

        connection.execute(
            """
            UPDATE review_documents
            SET
                review_state = 'completed',
                assigned_to = COALESCE(
                    assigned_to,
                    ?
                ),
                updated_at = ?
            WHERE document_id = ?
            """,
            (
                reviewer,
                utc_now(),
                document_id,
            ),
        )

        insert_event(
            connection,
            document_id=document_id,
            event_type=(
                "document_completed"
            ),
            actor=reviewer,
            details={
                "note": request.note,
            },
        )

    return {
        "document_id": document_id,
        "review_state": "completed",
        "completed_by": reviewer,
    }


@app.get(
    "/api/v1/review/documents/"
    "{document_id}/events"
)
def get_review_events(
    document_id: str,
) -> dict[str, Any]:
    with database_connection() as connection:
        exists = connection.execute(
            """
            SELECT 1
            FROM review_documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

        if exists is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        events = connection.execute(
            """
            SELECT
                event_id,
                document_id,
                field_name,
                event_type,
                actor,
                details_json,
                created_at
            FROM review_events
            WHERE document_id = ?
            ORDER BY
                created_at ASC,
                event_id ASC
            """,
            (document_id,),
        ).fetchall()

    items: list[dict[str, Any]] = []

    for event in events:
        item = dict(event)

        try:
            item["details"] = json.loads(
                item.pop("details_json")
                or "{}"
            )
        except json.JSONDecodeError:
            item["details"] = {
                "raw": item.pop(
                    "details_json",
                    "",
                )
            }

        items.append(item)

    return {
        "document_id": document_id,
        "events": items,
    }


@app.get(
    "/api/v1/review/documents/"
    "{document_id}/source-image"
)
def get_review_source_image(
    document_id: str,
    field_name: str | None = None,
    max_width: int = Query(
        default=1400,
        ge=400,
        le=3000,
    ),
) -> Response:
    image, annotation = (
        load_document_media(
            document_id
        )
    )

    original_width = image.width

    if image.width > max_width:
        ratio = (
            max_width
            / image.width
        )

        new_height = max(
            1,
            round(
                image.height
                * ratio
            ),
        )

        image = image.resize(
            (
                max_width,
                new_height,
            ),
            Image.Resampling.LANCZOS,
        )
    else:
        ratio = 1.0

    if field_name:
        x1, y1, x2, y2 = (
            get_field_box(
                annotation,
                field_name,
            )
        )

        scaled_box = (
            round(x1 * ratio),
            round(y1 * ratio),
            round(x2 * ratio),
            round(y2 * ratio),
        )

        draw = ImageDraw.Draw(
            image
        )

        line_width = max(
            4,
            round(
                image.width
                / 300
            ),
        )

        draw.rectangle(
            scaled_box,
            outline=(
                255,
                0,
                0,
            ),
            width=line_width,
        )

        outer_padding = (
            line_width * 2
        )

        draw.rectangle(
            (
                max(
                    0,
                    scaled_box[0]
                    - outer_padding,
                ),
                max(
                    0,
                    scaled_box[1]
                    - outer_padding,
                ),
                min(
                    image.width - 1,
                    scaled_box[2]
                    + outer_padding,
                ),
                min(
                    image.height - 1,
                    scaled_box[3]
                    + outer_padding,
                ),
            ),
            outline=(
                255,
                215,
                0,
            ),
            width=max(
                2,
                line_width // 2,
            ),
        )

    return image_response(
        image,
        filename=(
            f"{document_id}_source.png"
        ),
    )


@app.get(
    "/api/v1/review/documents/"
    "{document_id}/fields/"
    "{field_name}/crop"
)
def get_review_field_crop(
    document_id: str,
    field_name: str,
    padding: int = Query(
        default=30,
        ge=0,
        le=200,
    ),
    scale: float = Query(
        default=2.0,
        ge=1.0,
        le=5.0,
    ),
) -> Response:
    image, annotation = (
        load_document_media(
            document_id
        )
    )

    x1, y1, x2, y2 = get_field_box(
        annotation,
        field_name,
    )

    crop_box = (
        max(
            0,
            x1 - padding,
        ),
        max(
            0,
            y1 - padding,
        ),
        min(
            image.width,
            x2 + padding,
        ),
        min(
            image.height,
            y2 + padding,
        ),
    )

    if (
        crop_box[2]
        <= crop_box[0]
        or crop_box[3]
        <= crop_box[1]
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "The field crop box has "
                "invalid dimensions."
            ),
        )

    crop = image.crop(
        crop_box
    )

    if scale > 1.0:
        crop = crop.resize(
            (
                max(
                    1,
                    round(
                        crop.width
                        * scale
                    ),
                ),
                max(
                    1,
                    round(
                        crop.height
                        * scale
                    ),
                ),
            ),
            Image.Resampling.LANCZOS,
        )

    draw = ImageDraw.Draw(
        crop
    )

    border_width = max(
        3,
        round(
            crop.width
            / 180
        ),
    )

    draw.rectangle(
        (
            0,
            0,
            crop.width - 1,
            crop.height - 1,
        ),
        outline=(
            255,
            215,
            0,
        ),
        width=border_width,
    )

    return image_response(
        crop,
        filename=(
            f"{document_id}_"
            f"{field_name}_crop.png"
        ),
    )


@app.post(
    "/api/v1/documents/upload",
    status_code=201,
)
async def upload_document(
    file: UploadFile = File(...),
    expected_document_type: str | None = Form(
        default=None,
    ),
    uploaded_by: str | None = Form(
        default=None,
    ),
    allow_duplicate: bool = Form(
        default=False,
    ),
) -> dict[str, Any]:
    allowed_document_types = {
        "w2",
        "1099_nec",
        "auto",
    }

    normalized_expected_type = (
        expected_document_type.strip().lower()
        if expected_document_type
        else "auto"
    )

    if (
        normalized_expected_type
        not in allowed_document_types
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "expected_document_type must be "
                "'auto', 'w2', or '1099_nec'."
            ),
        )

    saved = await save_uploaded_document(
        file
    )

    job_id = str(
        uuid.uuid4()
    )

    now = utc_now()

    with database_connection() as connection:
        duplicate = connection.execute(
            """
            SELECT
                u.upload_id,
                u.original_filename,
                u.created_at,
                j.job_id,
                j.job_status,
                j.result_document_id
            FROM document_uploads u
            LEFT JOIN processing_jobs j
                ON j.upload_id = u.upload_id
            WHERE u.sha256 = ?
            ORDER BY
                u.created_at DESC,
                j.created_at DESC
            LIMIT 1
            """,
            (
                saved["sha256"],
            ),
        ).fetchone()

        if (
            duplicate is not None
            and not allow_duplicate
        ):
            Path(
                saved["stored_path"]
            ).unlink(
                missing_ok=True
            )

            raise HTTPException(
                status_code=409,
                detail=(
                    "Duplicate file detected. "
                    "A previous upload already "
                    "exists with upload_id "
                    f"{duplicate['upload_id']} "
                    "and latest job status "
                    f"{duplicate['job_status'] or 'unknown'}. "
                    "Enable 'Process duplicate anyway' "
                    "only when a deliberate reprocessing "
                    "run is required."
                ),
            )

        connection.execute(
            """
            INSERT INTO document_uploads (
                upload_id,
                original_filename,
                stored_filename,
                stored_path,
                file_extension,
                content_type,
                file_size_bytes,
                sha256,
                expected_document_type,
                uploaded_by,
                upload_status,
                created_at,
                updated_at
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, 'queued', ?, ?
            )
            """,
            (
                saved["upload_id"],
                saved[
                    "original_filename"
                ],
                saved[
                    "stored_filename"
                ],
                saved["stored_path"],
                saved[
                    "file_extension"
                ],
                saved["content_type"],
                saved[
                    "file_size_bytes"
                ],
                saved["sha256"],
                normalized_expected_type,
                (
                    uploaded_by.strip()
                    if uploaded_by
                    else None
                ),
                now,
                now,
            ),
        )

        connection.execute(
            """
            INSERT INTO processing_jobs (
                job_id,
                upload_id,
                job_status,
                processing_stage,
                progress_percent,
                created_at,
                updated_at
            )
            VALUES (
                ?, ?, 'queued', 'awaiting_worker',
                0, ?, ?
            )
            """,
            (
                job_id,
                saved["upload_id"],
                now,
                now,
            ),
        )

    return {
        "upload_id": (
            saved["upload_id"]
        ),
        "job_id": job_id,
        "status": "queued",
        "processing_stage": (
            "awaiting_worker"
        ),
        "original_filename": (
            saved[
                "original_filename"
            ]
        ),
        "file_size_bytes": (
            saved[
                "file_size_bytes"
            ]
        ),
        "sha256": saved["sha256"],
        "expected_document_type": (
            normalized_expected_type
        ),
        "duplicate_of": (
            {
                "upload_id": (
                    duplicate[
                        "upload_id"
                    ]
                ),
                "original_filename": (
                    duplicate[
                        "original_filename"
                    ]
                ),
                "created_at": (
                    duplicate[
                        "created_at"
                    ]
                ),
                "job_id": (
                    duplicate[
                        "job_id"
                    ]
                ),
                "job_status": (
                    duplicate[
                        "job_status"
                    ]
                ),
                "result_document_id": (
                    duplicate[
                        "result_document_id"
                    ]
                ),
            }
            if duplicate
            else None
        ),
        "allow_duplicate": (
            allow_duplicate
        ),
    }


@app.get(
    "/api/v1/documents/jobs"
)
def list_processing_jobs(
    job_status: str | None = None,
    include_archived: bool = Query(
        default=False,
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> dict[str, Any]:
    conditions: list[str] = []
    parameters: list[Any] = []

    if job_status:
        conditions.append(
            "j.job_status = ?"
        )
        parameters.append(
            job_status
        )

    if (
        not include_archived
        and job_status != "archived"
    ):
        conditions.append(
            "j.job_status != 'archived'"
        )

    where_clause = (
        "WHERE "
        + " AND ".join(
            conditions
        )
        if conditions
        else ""
    )

    with database_connection() as connection:
        total = connection.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM processing_jobs j
            {where_clause}
            """,
            parameters,
        ).fetchone()

        rows = connection.execute(
            f"""
            SELECT
                j.job_id,
                j.upload_id,
                j.job_status,
                j.processing_stage,
                j.progress_percent,
                j.detected_document_type,
                j.result_document_id,
                j.error_message,
                j.created_at,
                j.updated_at,
                j.started_at,
                j.completed_at,
                u.original_filename,
                u.file_extension,
                u.file_size_bytes,
                u.expected_document_type,
                u.uploaded_by
            FROM processing_jobs j
            JOIN document_uploads u
                ON u.upload_id = j.upload_id
            {where_clause}
            ORDER BY
                CASE j.job_status
                    WHEN 'processing' THEN 1
                    WHEN 'queued' THEN 2
                    WHEN 'failed' THEN 3
                    WHEN 'completed' THEN 4
                    WHEN 'archived' THEN 5
                    ELSE 6
                END,
                j.created_at DESC
            LIMIT ? OFFSET ?
            """,
            [
                *parameters,
                limit,
                offset,
            ],
        ).fetchall()

    return {
        "total": int(
            total["count"]
            or 0
        ),
        "limit": limit,
        "offset": offset,
        "items": [
            serialize_upload_row(
                row
            )
            for row in rows
        ],
    }


@app.get(
    "/api/v1/documents/jobs/{job_id}"
)
def get_processing_job(
    job_id: str,
) -> dict[str, Any]:
    with database_connection() as connection:
        row = connection.execute(
            """
            SELECT
                j.job_id,
                j.upload_id,
                j.job_status,
                j.processing_stage,
                j.progress_percent,
                j.detected_document_type,
                j.result_document_id,
                j.error_message,
                j.created_at,
                j.updated_at,
                j.started_at,
                j.completed_at,
                u.original_filename,
                u.stored_filename,
                u.stored_path,
                u.file_extension,
                u.content_type,
                u.file_size_bytes,
                u.sha256,
                u.expected_document_type,
                u.uploaded_by,
                u.upload_status
            FROM processing_jobs j
            JOIN document_uploads u
                ON u.upload_id = j.upload_id
            WHERE j.job_id = ?
            """,
            (
                job_id,
            ),
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Processing job not found."
            ),
        )

    return serialize_upload_row(
        row
    )



@app.post(
    "/api/v1/documents/jobs/{job_id}/retry"
)
def retry_processing_job(
    job_id: str,
    request: JobActionRequest,
) -> dict[str, Any]:
    actor = request.actor.strip()
    now = utc_now()

    with database_connection() as connection:
        job = connection.execute(
            """
            SELECT
                j.job_id,
                j.upload_id,
                j.job_status,
                u.original_filename
            FROM processing_jobs j
            JOIN document_uploads u
                ON u.upload_id = j.upload_id
            WHERE j.job_id = ?
            """,
            (
                job_id,
            ),
        ).fetchone()

        if job is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Processing job not found."
                ),
            )

        if job["job_status"] not in {
            "failed",
            "archived",
        }:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Only failed or archived jobs "
                    "can be retried."
                ),
            )

        connection.execute(
            """
            UPDATE processing_jobs
            SET
                job_status = 'queued',
                processing_stage = 'awaiting_worker',
                progress_percent = 0,
                detected_document_type = NULL,
                result_document_id = NULL,
                error_message = NULL,
                started_at = NULL,
                completed_at = NULL,
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                now,
                job_id,
            ),
        )

        connection.execute(
            """
            UPDATE document_uploads
            SET
                upload_status = 'queued',
                updated_at = ?
            WHERE upload_id = ?
            """,
            (
                now,
                job["upload_id"],
            ),
        )

    return {
        "job_id": job_id,
        "upload_id": job["upload_id"],
        "original_filename": (
            job["original_filename"]
        ),
        "job_status": "queued",
        "processing_stage": (
            "awaiting_worker"
        ),
        "progress_percent": 0,
        "retried_by": actor,
        "note": request.note,
    }


@app.post(
    "/api/v1/documents/jobs/{job_id}/archive"
)
def archive_processing_job(
    job_id: str,
    request: JobActionRequest,
) -> dict[str, Any]:
    actor = request.actor.strip()
    now = utc_now()

    with database_connection() as connection:
        job = connection.execute(
            """
            SELECT
                j.job_id,
                j.upload_id,
                j.job_status,
                u.original_filename
            FROM processing_jobs j
            JOIN document_uploads u
                ON u.upload_id = j.upload_id
            WHERE j.job_id = ?
            """,
            (
                job_id,
            ),
        ).fetchone()

        if job is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Processing job not found."
                ),
            )

        if job["job_status"] in {
            "queued",
            "processing",
        }:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Queued or processing jobs "
                    "cannot be archived."
                ),
            )

        if job["job_status"] == "archived":
            return {
                "job_id": job_id,
                "upload_id": job["upload_id"],
                "original_filename": (
                    job["original_filename"]
                ),
                "job_status": "archived",
                "processing_stage": "archived",
                "archived_by": actor,
                "note": request.note,
            }

        connection.execute(
            """
            UPDATE processing_jobs
            SET
                job_status = 'archived',
                processing_stage = 'archived',
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                now,
                job_id,
            ),
        )

        connection.execute(
            """
            UPDATE document_uploads
            SET
                upload_status = 'archived',
                updated_at = ?
            WHERE upload_id = ?
            """,
            (
                now,
                job["upload_id"],
            ),
        )

    return {
        "job_id": job_id,
        "upload_id": job["upload_id"],
        "original_filename": (
            job["original_filename"]
        ),
        "job_status": "archived",
        "processing_stage": "archived",
        "archived_by": actor,
        "note": request.note,
    }
