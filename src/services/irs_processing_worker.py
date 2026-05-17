from __future__ import annotations

import argparse
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.extraction.irs_tesseract_baseline import (
    PROJECT_ROOT,
)
from src.services.irs_document_processor import (
    ProductionModels,
    ProcessingResult,
    process_uploaded_document,
    relative_project_path,
)


DEFAULT_DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "irs_review_queue.db"
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


@contextmanager
def connect(
    database_path: Path,
) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(
        database_path,
        timeout=30.0,
    )

    connection.row_factory = (
        sqlite3.Row
    )

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


def initialize_worker_schema(
    database_path: Path,
) -> None:
    with connect(database_path) as connection:
        connection.executescript(
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
                ON DELETE CASCADE,
                FOREIGN KEY (
                    upload_id
                )
                REFERENCES document_uploads (
                    upload_id
                )
                ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_document_media_upload
            ON document_media (
                upload_id
            );
            """
        )


def claim_next_job(
    database_path: Path,
) -> dict[str, Any] | None:
    connection = sqlite3.connect(
        database_path,
        timeout=30.0,
        isolation_level=None,
    )

    connection.row_factory = (
        sqlite3.Row
    )

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    connection.execute(
        "PRAGMA busy_timeout = 30000"
    )

    try:
        connection.execute(
            "BEGIN IMMEDIATE"
        )

        row = connection.execute(
            """
            SELECT
                j.job_id,
                j.upload_id,
                u.original_filename,
                u.stored_path,
                u.expected_document_type,
                u.uploaded_by
            FROM processing_jobs j
            JOIN document_uploads u
                ON u.upload_id = j.upload_id
            WHERE j.job_status = 'queued'
            ORDER BY j.created_at ASC
            LIMIT 1
            """
        ).fetchone()

        if row is None:
            connection.execute(
                "COMMIT"
            )
            return None

        now = utc_now()

        updated = connection.execute(
            """
            UPDATE processing_jobs
            SET
                job_status = 'processing',
                processing_stage = 'claimed',
                progress_percent = 2,
                started_at = COALESCE(
                    started_at,
                    ?
                ),
                updated_at = ?
            WHERE
                job_id = ?
                AND job_status = 'queued'
            """,
            (
                now,
                now,
                row["job_id"],
            ),
        )

        if updated.rowcount != 1:
            connection.execute(
                "ROLLBACK"
            )
            return None

        connection.execute(
            """
            UPDATE document_uploads
            SET
                upload_status = 'processing',
                updated_at = ?
            WHERE upload_id = ?
            """,
            (
                now,
                row["upload_id"],
            ),
        )

        connection.execute(
            "COMMIT"
        )

        return dict(row)

    except Exception:
        connection.execute(
            "ROLLBACK"
        )
        raise
    finally:
        connection.close()


def update_job_stage(
    database_path: Path,
    *,
    job_id: str,
    stage: str,
    progress_percent: float,
    detected_document_type: str | None = None,
) -> None:
    with connect(database_path) as connection:
        connection.execute(
            """
            UPDATE processing_jobs
            SET
                processing_stage = ?,
                progress_percent = ?,
                detected_document_type = COALESCE(
                    ?,
                    detected_document_type
                ),
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                stage,
                float(progress_percent),
                detected_document_type,
                utc_now(),
                job_id,
            ),
        )


def clean_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def bool_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)

    if value is None or pd.isna(value):
        return 0

    if isinstance(
        value,
        (int, float),
    ):
        return int(
            bool(value)
        )

    return int(
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def result_counts(
    result: ProcessingResult,
) -> dict[str, int]:
    fields = (
        result.field_results
    )

    review_mask = fields[
        "field_review_required"
    ].astype(bool)

    accepted_mask = ~review_mask

    critical_review = (
        review_mask
        & (
            fields["risk_tier"]
            == "critical"
        )
    )

    structured_review = (
        review_mask
        & (
            fields["risk_tier"]
            == "structured"
        )
    )

    descriptive_review = (
        review_mask
        & (
            fields["risk_tier"]
            == "descriptive"
        )
    )

    required_invalid = (
        review_mask
        & fields[
            "required_field"
        ].astype(bool)
        & ~fields[
            "format_valid"
        ].astype(bool)
    )

    cross_field = (
        review_mask
        & (
            fields[
                "cross_field_flag"
            ]
            > 0
        )
    )

    exception_mask = (
        required_invalid
        | cross_field
    )

    return {
        "total": int(
            len(fields)
        ),
        "accepted": int(
            accepted_mask.sum()
        ),
        "review": int(
            review_mask.sum()
        ),
        "critical": int(
            critical_review.sum()
        ),
        "structured": int(
            structured_review.sum()
        ),
        "descriptive": int(
            descriptive_review.sum()
        ),
        "exceptions": int(
            exception_mask.sum()
        ),
    }


def persist_result(
    database_path: Path,
    *,
    job: dict[str, Any],
    result: ProcessingResult,
) -> None:
    counts = result_counts(
        result
    )

    now = utc_now()

    review_state = (
        "pending"
        if counts["review"] > 0
        else "completed"
    )

    assigned_to = None

    with connect(database_path) as connection:
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
                ?, ?, ?, 'uploaded', ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, 0, ?, ?, ?
            )
            ON CONFLICT (
                document_id
            )
            DO UPDATE SET
                parent_document_id = excluded.parent_document_id,
                document_type = excluded.document_type,
                quality = excluded.quality,
                status = excluded.status,
                priority = excluded.priority,
                total_field_count = excluded.total_field_count,
                accepted_field_count = excluded.accepted_field_count,
                review_field_count = excluded.review_field_count,
                critical_review_count = excluded.critical_review_count,
                structured_review_count = excluded.structured_review_count,
                descriptive_review_count = excluded.descriptive_review_count,
                exception_count = excluded.exception_count,
                review_state = excluded.review_state,
                reviewed_field_count = 0,
                assigned_to = excluded.assigned_to,
                updated_at = excluded.updated_at
            """,
            (
                result.document_id,
                result.document_id,
                result.document_type,
                result.document_status,
                result.priority,
                counts["total"],
                counts["accepted"],
                counts["review"],
                counts["critical"],
                counts["structured"],
                counts["descriptive"],
                counts["exceptions"],
                review_state,
                assigned_to,
                now,
                now,
            ),
        )

        connection.execute(
            """
            DELETE FROM review_fields
            WHERE document_id = ?
            """,
            (
                result.document_id,
            ),
        )

        for _, row in (
            result.field_results
            .iterrows()
        ):
            review_required = bool(
                row[
                    "field_review_required"
                ]
            )

            field_state = (
                "pending"
                if review_required
                else "accepted"
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
                    corrected_value,
                    reviewer_note,
                    reviewed_by,
                    reviewed_at,
                    created_at,
                    updated_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, NULL, NULL, NULL,
                    NULL, ?, ?
                )
                """,
                (
                    result.document_id,
                    clean_value(
                        row[
                            "field_name"
                        ]
                    ),
                    clean_value(
                        row[
                            "final_normalized_prediction"
                        ]
                    ),
                    clean_value(
                        row[
                            "final_prediction"
                        ]
                    ),
                    clean_value(
                        row[
                            "final_engine"
                        ]
                    ),
                    clean_value(
                        row[
                            "risk_tier"
                        ]
                    ),
                    bool_int(
                        row[
                            "required_field"
                        ]
                    ),
                    float(
                        row[
                            "correctness_probability"
                        ]
                    ),
                    float(
                        row[
                            "risk_threshold"
                        ]
                    ),
                    bool_int(
                        row[
                            "format_valid"
                        ]
                    ),
                    bool_int(
                        float(
                            row[
                                "cross_field_flag"
                            ]
                        )
                        > 0
                    ),
                    int(
                        review_required
                    ),
                    clean_value(
                        row[
                            "review_reason"
                        ]
                    ),
                    field_state,
                    now,
                    now,
                ),
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
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (
                document_id
            )
            DO UPDATE SET
                upload_id = excluded.upload_id,
                source_image_path = excluded.source_image_path,
                annotation_path = excluded.annotation_path,
                updated_at = excluded.updated_at
            """,
            (
                result.document_id,
                job["upload_id"],
                relative_project_path(
                    result.source_image_path
                ),
                relative_project_path(
                    result.annotation_path
                ),
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
                ?, NULL, 'document_processed',
                'system', ?, ?
            )
            """,
            (
                result.document_id,
                json.dumps(
                    {
                        "job_id": (
                            job["job_id"]
                        ),
                        "upload_id": (
                            job[
                                "upload_id"
                            ]
                        ),
                        "original_filename": (
                            job[
                                "original_filename"
                            ]
                        ),
                        "document_type": (
                            result
                            .document_type
                        ),
                        "classification_scores": (
                            result
                            .classification_scores
                        ),
                        "document_status": (
                            result
                            .document_status
                        ),
                        "field_counts": (
                            counts
                        ),
                    },
                    ensure_ascii=False,
                ),
                now,
            ),
        )

        connection.execute(
            """
            UPDATE processing_jobs
            SET
                job_status = 'completed',
                processing_stage = 'completed',
                progress_percent = 100,
                detected_document_type = ?,
                result_document_id = ?,
                error_message = NULL,
                completed_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                result.document_type,
                result.document_id,
                now,
                now,
                job["job_id"],
            ),
        )

        connection.execute(
            """
            UPDATE document_uploads
            SET
                upload_status = 'completed',
                updated_at = ?
            WHERE upload_id = ?
            """,
            (
                now,
                job["upload_id"],
            ),
        )


def mark_job_failed(
    database_path: Path,
    *,
    job: dict[str, Any],
    error: Exception,
) -> None:
    now = utc_now()

    message = (
        f"{type(error).__name__}: "
        f"{error}"
    )

    if len(message) > 4000:
        message = message[:4000]

    with connect(database_path) as connection:
        connection.execute(
            """
            UPDATE processing_jobs
            SET
                job_status = 'failed',
                processing_stage = 'failed',
                error_message = ?,
                completed_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                message,
                now,
                now,
                job["job_id"],
            ),
        )

        connection.execute(
            """
            UPDATE document_uploads
            SET
                upload_status = 'failed',
                updated_at = ?
            WHERE upload_id = ?
            """,
            (
                now,
                job["upload_id"],
            ),
        )


def process_job(
    database_path: Path,
    *,
    job: dict[str, Any],
    models: ProductionModels,
) -> ProcessingResult:
    source_path = Path(
        job["stored_path"]
    ).resolve()

    if not source_path.exists():
        raise FileNotFoundError(
            "Uploaded source file not found: "
            f"{source_path}"
        )

    update_job_stage(
        database_path,
        job_id=job["job_id"],
        stage="loading_document",
        progress_percent=8,
    )

    update_job_stage(
        database_path,
        job_id=job["job_id"],
        stage="classifying_document",
        progress_percent=15,
    )

    result = process_uploaded_document(
        upload_id=job["upload_id"],
        source_path=source_path,
        expected_document_type=(
            job[
                "expected_document_type"
            ]
            or "auto"
        ),
        models=models,
    )

    update_job_stage(
        database_path,
        job_id=job["job_id"],
        stage="persisting_review_queue",
        progress_percent=92,
        detected_document_type=(
            result.document_type
        ),
    )

    persist_result(
        database_path,
        job=job,
        result=result,
    )

    return result


def run_worker(
    *,
    database_path: Path,
    once: bool,
    poll_interval: float,
    max_jobs: int,
    batch_size: int,
    num_beams: int,
    confidence_threshold: float,
    disable_trocr: bool,
) -> None:
    database_path = (
        database_path.resolve()
    )

    if not database_path.exists():
        raise FileNotFoundError(
            "Review database not found: "
            f"{database_path}"
        )

    initialize_worker_schema(
        database_path
    )

    print(
        "\nDocIntelAI IRS processing worker"
    )
    print(
        f"Database: {database_path}"
    )
    print(
        f"TrOCR disabled: "
        f"{disable_trocr}"
    )

    models = ProductionModels(
        batch_size=batch_size,
        num_beams=num_beams,
        confidence_threshold=(
            confidence_threshold
        ),
        disable_trocr=(
            disable_trocr
        ),
    )

    processed_jobs = 0

    while True:
        if (
            max_jobs > 0
            and processed_jobs
            >= max_jobs
        ):
            print(
                "\nMaximum job count reached."
            )
            break

        job = claim_next_job(
            database_path
        )

        if job is None:
            if once:
                print(
                    "\nNo queued jobs found."
                )
                break

            print(
                "No queued jobs. Waiting..."
            )

            time.sleep(
                max(
                    1.0,
                    poll_interval,
                )
            )

            continue

        print(
            "\nProcessing job "
            f"{job['job_id']}"
        )
        print(
            "File: "
            f"{job['original_filename']}"
        )

        try:
            result = process_job(
                database_path,
                job=job,
                models=models,
            )
        except Exception as error:
            mark_job_failed(
                database_path,
                job=job,
                error=error,
            )

            print(
                "Job failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )
        else:
            print(
                "Job completed"
            )
            print(
                "Document ID: "
                f"{result.document_id}"
            )
            print(
                "Document type: "
                f"{result.document_type}"
            )
            print(
                "Review status: "
                f"{result.document_status}"
            )
            print(
                "Fields: "
                f"{len(result.field_results)}"
            )

        processed_jobs += 1

        if once:
            break


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process queued IRS document uploads "
            "and create persistent field-level "
            "review tasks."
        )
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=(
            DEFAULT_DATABASE_PATH
        ),
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Process at most one queued job "
            "and then exit."
        ),
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        help=(
            "Maximum jobs before exit. "
            "Use 0 for no limit."
        ),
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

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=70.0,
    )

    parser.add_argument(
        "--disable-trocr",
        action="store_true",
        help=(
            "Use only hybrid Tesseract. "
            "Useful for quick CPU debugging."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_worker(
        database_path=(
            arguments.database
        ),
        once=arguments.once,
        poll_interval=(
            arguments.poll_interval
        ),
        max_jobs=arguments.max_jobs,
        batch_size=arguments.batch_size,
        num_beams=arguments.num_beams,
        confidence_threshold=(
            arguments.confidence_threshold
        ),
        disable_trocr=(
            arguments.disable_trocr
        ),
    )
