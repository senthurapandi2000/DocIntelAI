from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_QUEUE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_review_queue.jsonl"
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


class IRSReviewStore:
    def __init__(
        self,
        database_path: Path,
    ) -> None:
        self.database_path = (
            database_path.resolve()
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    @contextmanager
    def connect(
        self,
    ) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = (
            sqlite3.Row
        )

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize_schema(
        self,
    ) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_documents (
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

                CREATE TABLE IF NOT EXISTS review_fields (
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

                CREATE TABLE IF NOT EXISTS review_events (
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

                CREATE INDEX IF NOT EXISTS idx_review_documents_queue
                ON review_documents (
                    review_state,
                    priority,
                    exception_count DESC,
                    critical_review_count DESC
                );

                CREATE INDEX IF NOT EXISTS idx_review_fields_pending
                ON review_fields (
                    document_id,
                    review_required,
                    review_state,
                    risk_tier
                );

                CREATE INDEX IF NOT EXISTS idx_review_events_document
                ON review_events (
                    document_id,
                    created_at
                );
                """
            )

    def import_queue(
        self,
        queue_path: Path,
        *,
        replace_existing: bool,
    ) -> dict[str, int]:
        if not queue_path.exists():
            raise FileNotFoundError(
                "IRS review queue not found: "
                f"{queue_path}"
            )

        imported_documents = 0
        imported_fields = 0
        review_fields = 0
        accepted_fields = 0

        with queue_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            queue_items = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        with self.connect() as connection:
            if replace_existing:
                connection.execute(
                    "DELETE FROM review_events"
                )

                connection.execute(
                    "DELETE FROM review_fields"
                )

                connection.execute(
                    "DELETE FROM review_documents"
                )

            for item in queue_items:
                now = utc_now()

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
                        created_at,
                        updated_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?
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
                        reviewed_field_count = excluded.reviewed_field_count,
                        updated_at = excluded.updated_at
                    """,
                    (
                        item["document_id"],
                        item.get(
                            "parent_document_id",
                            "",
                        ),
                        item.get(
                            "document_type",
                            "",
                        ),
                        item.get(
                            "quality",
                            "",
                        ),
                        item.get(
                            "status",
                            "targeted_field_review",
                        ),
                        int(
                            item.get(
                                "priority",
                                3,
                            )
                        ),
                        int(
                            item.get(
                                "total_field_count",
                                0,
                            )
                        ),
                        int(
                            item.get(
                                "accepted_field_count",
                                0,
                            )
                        ),
                        int(
                            item.get(
                                "review_field_count",
                                0,
                            )
                        ),
                        int(
                            item.get(
                                "critical_review_count",
                                0,
                            )
                        ),
                        int(
                            item.get(
                                "structured_review_count",
                                0,
                            )
                        ),
                        int(
                            item.get(
                                "descriptive_review_count",
                                0,
                            )
                        ),
                        int(
                            item.get(
                                "exception_count",
                                0,
                            )
                        ),
                        item.get(
                            "review_state",
                            "pending",
                        ),
                        int(
                            item.get(
                                "reviewed_field_count",
                                0,
                            )
                        ),
                        now,
                        now,
                    ),
                )

                imported_documents += 1

                all_fields = []

                for field in item.get(
                    "accepted_fields",
                    [],
                ):
                    all_fields.append(
                        (
                            field,
                            False,
                        )
                    )

                for field in item.get(
                    "fields_for_review",
                    [],
                ):
                    all_fields.append(
                        (
                            field,
                            True,
                        )
                    )

                for field, needs_review in all_fields:
                    field_state = (
                        "pending"
                        if needs_review
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
                            created_at,
                            updated_at
                        )
                        VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?
                        )
                        ON CONFLICT (
                            document_id,
                            field_name
                        )
                        DO UPDATE SET
                            value = excluded.value,
                            raw_value = excluded.raw_value,
                            engine = excluded.engine,
                            risk_tier = excluded.risk_tier,
                            required = excluded.required,
                            correctness_probability = excluded.correctness_probability,
                            acceptance_threshold = excluded.acceptance_threshold,
                            format_valid = excluded.format_valid,
                            cross_field_flag = excluded.cross_field_flag,
                            review_required = excluded.review_required,
                            review_reason = excluded.review_reason,
                            review_state = excluded.review_state,
                            updated_at = excluded.updated_at
                        """,
                        (
                            item["document_id"],
                            field.get(
                                "field_name",
                                "",
                            ),
                            field.get(
                                "value",
                                "",
                            ),
                            field.get(
                                "raw_value",
                                "",
                            ),
                            field.get(
                                "engine",
                                "",
                            ),
                            field.get(
                                "risk_tier",
                                "",
                            ),
                            int(
                                bool(
                                    field.get(
                                        "required",
                                        False,
                                    )
                                )
                            ),
                            float(
                                field.get(
                                    "correctness_probability",
                                    0.0,
                                )
                            ),
                            float(
                                field.get(
                                    "acceptance_threshold",
                                    1.0,
                                )
                            ),
                            int(
                                bool(
                                    field.get(
                                        "format_valid",
                                        True,
                                    )
                                )
                            ),
                            int(
                                bool(
                                    field.get(
                                        "cross_field_flag",
                                        False,
                                    )
                                )
                            ),
                            int(needs_review),
                            field.get(
                                "review_reason",
                                "",
                            ),
                            field_state,
                            now,
                            now,
                        ),
                    )

                    imported_fields += 1

                    if needs_review:
                        review_fields += 1
                    else:
                        accepted_fields += 1

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
                        ?, NULL, ?, ?, ?, ?
                    )
                    """,
                    (
                        item["document_id"],
                        "queue_imported",
                        "system",
                        json.dumps(
                            {
                                "status": item.get(
                                    "status"
                                ),
                                "priority": item.get(
                                    "priority"
                                ),
                                "review_field_count": item.get(
                                    "review_field_count"
                                ),
                            }
                        ),
                        now,
                    ),
                )

        return {
            "documents": (
                imported_documents
            ),
            "fields": imported_fields,
            "accepted_fields": (
                accepted_fields
            ),
            "review_fields": (
                review_fields
            ),
        }

    def database_summary(
        self,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            document_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS documents,
                    SUM(
                        CASE
                            WHEN review_state = 'pending'
                            THEN 1
                            ELSE 0
                        END
                    ) AS pending_documents,
                    SUM(
                        CASE
                            WHEN status = 'exception_review'
                            THEN 1
                            ELSE 0
                        END
                    ) AS exception_documents,
                    SUM(
                        CASE
                            WHEN status = 'critical_field_verification'
                            THEN 1
                            ELSE 0
                        END
                    ) AS critical_documents
                FROM review_documents
                """
            ).fetchone()

            field_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS fields,
                    SUM(
                        CASE
                            WHEN review_required = 1
                            THEN 1
                            ELSE 0
                        END
                    ) AS review_fields,
                    SUM(
                        CASE
                            WHEN review_required = 0
                            THEN 1
                            ELSE 0
                        END
                    ) AS accepted_fields,
                    SUM(
                        CASE
                            WHEN review_state = 'pending'
                            THEN 1
                            ELSE 0
                        END
                    ) AS pending_fields
                FROM review_fields
                """
            ).fetchone()

        return {
            "documents": int(
                document_row["documents"]
                or 0
            ),
            "pending_documents": int(
                document_row[
                    "pending_documents"
                ]
                or 0
            ),
            "exception_documents": int(
                document_row[
                    "exception_documents"
                ]
                or 0
            ),
            "critical_documents": int(
                document_row[
                    "critical_documents"
                ]
                or 0
            ),
            "fields": int(
                field_row["fields"]
                or 0
            ),
            "review_fields": int(
                field_row[
                    "review_fields"
                ]
                or 0
            ),
            "accepted_fields": int(
                field_row[
                    "accepted_fields"
                ]
                or 0
            ),
            "pending_fields": int(
                field_row["pending_fields"]
                or 0
            ),
        }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a persistent SQLite store "
            "for the IRS human-review queue."
        )
    )

    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE_PATH,
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
    )

    parser.add_argument(
        "--replace-existing",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    store = IRSReviewStore(
        arguments.database
    )

    store.initialize_schema()

    imported = store.import_queue(
        arguments.queue,
        replace_existing=(
            arguments.replace_existing
        ),
    )

    summary = store.database_summary()

    print(
        "\nIRS review database created"
    )
    print(
        f"Database: "
        f"{store.database_path}"
    )
    print(
        f"Imported documents: "
        f"{imported['documents']}"
    )
    print(
        f"Imported fields: "
        f"{imported['fields']}"
    )
    print(
        f"Accepted fields: "
        f"{summary['accepted_fields']}"
    )
    print(
        f"Review fields: "
        f"{summary['review_fields']}"
    )
    print(
        f"Pending documents: "
        f"{summary['pending_documents']}"
    )
    print(
        f"Pending fields: "
        f"{summary['pending_fields']}"
    )
    print(
        "Exception-review documents: "
        f"{summary['exception_documents']}"
    )
    print(
        "Critical-verification documents: "
        f"{summary['critical_documents']}"
    )


if __name__ == "__main__":
    main()
