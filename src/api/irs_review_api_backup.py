from __future__ import annotations

import json
import os
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from fastapi import FastAPI, HTTPException, Query
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


@asynccontextmanager
async def lifespan(
    _: FastAPI,
):
    require_database()
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
