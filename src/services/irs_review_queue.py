from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FIELDS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_field_confidence_final_test_fields.csv"
)

QUEUE_JSONL_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_review_queue.jsonl"
)

QUEUE_CSV_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_review_queue_summary.csv"
)

QUEUE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "irs_review_queue_summary.json"
)


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return False

    return clean_text(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def to_float(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    numeric = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(numeric):
        return default

    return float(numeric)


def review_reason(row: pd.Series) -> str:
    reasons: list[str] = []

    if not to_bool(
        row.get("format_valid", True)
    ):
        message = clean_text(
            row.get(
                "format_message",
                "format_validation_failed",
            )
        )
        reasons.append(message)

    if to_float(
        row.get("cross_field_flag", 0.0)
    ) > 0:
        reasons.append(
            "cross_field_consistency_failure"
        )

    probability = to_float(
        row.get(
            "correctness_probability",
            0.0,
        )
    )

    threshold = to_float(
        row.get(
            "risk_threshold",
            1.0,
        ),
        default=1.0,
    )

    if probability < threshold:
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


def field_payload(
    row: pd.Series,
    *,
    include_reason: bool,
) -> dict[str, Any]:
    payload = {
        "field_name": clean_text(
            row.get("field_name", "")
        ),
        "value": clean_text(
            row.get(
                "final_normalized_prediction",
                "",
            )
        ),
        "raw_value": clean_text(
            row.get(
                "final_prediction",
                "",
            )
        ),
        "engine": clean_text(
            row.get(
                "final_engine",
                "",
            )
        ),
        "risk_tier": clean_text(
            row.get(
                "risk_tier",
                "",
            )
        ),
        "required": to_bool(
            row.get(
                "required_field",
                False,
            )
        ),
        "correctness_probability": round(
            to_float(
                row.get(
                    "correctness_probability",
                    0.0,
                )
            ),
            4,
        ),
        "acceptance_threshold": round(
            to_float(
                row.get(
                    "risk_threshold",
                    1.0,
                ),
                default=1.0,
            ),
            4,
        ),
        "format_valid": to_bool(
            row.get(
                "format_valid",
                True,
            )
        ),
        "cross_field_flag": bool(
            to_float(
                row.get(
                    "cross_field_flag",
                    0.0,
                )
            )
            > 0
        ),
    }

    if include_reason:
        payload["review_reason"] = (
            review_reason(row)
        )

    return payload


def choose_document_status(
    document_df: pd.DataFrame,
) -> str:
    review_df = document_df[
        document_df[
            "field_review_required"
        ].map(to_bool)
    ]

    if review_df.empty:
        return "auto_accept"

    has_exception = bool(
        (
            review_df[
                "cross_field_flag"
            ].map(
                lambda value: (
                    to_float(value) > 0
                )
            )
        ).any()
        or (
            ~review_df[
                "format_valid"
            ].map(to_bool)
            & review_df[
                "required_field"
            ].map(to_bool)
        ).any()
    )

    if has_exception:
        return "exception_review"

    has_critical = bool(
        (
            review_df["risk_tier"]
            .fillna("")
            .astype(str)
            == "critical"
        ).any()
    )

    if has_critical:
        return (
            "critical_field_verification"
        )

    return "targeted_field_review"


def priority_for_status(
    status: str,
) -> int:
    priorities = {
        "exception_review": 1,
        "critical_field_verification": 2,
        "targeted_field_review": 3,
        "auto_accept": 4,
    }

    return priorities.get(status, 4)


def build_review_queue(
    fields: pd.DataFrame,
) -> tuple[
    list[dict[str, Any]],
    pd.DataFrame,
]:
    queue_items: list[
        dict[str, Any]
    ] = []

    summary_rows: list[
        dict[str, Any]
    ] = []

    for document_id, document_df in (
        fields.groupby(
            "document_id",
            sort=False,
        )
    ):
        document_df = (
            document_df.copy()
        )

        document_df[
            "field_review_required"
        ] = document_df[
            "field_review_required"
        ].map(to_bool)

        status = choose_document_status(
            document_df
        )

        accepted_df = document_df[
            ~document_df[
                "field_review_required"
            ]
        ]

        review_df = document_df[
            document_df[
                "field_review_required"
            ]
        ]

        accepted_fields = [
            field_payload(
                row,
                include_reason=False,
            )
            for _, row in (
                accepted_df.iterrows()
            )
        ]

        fields_for_review = [
            field_payload(
                row,
                include_reason=True,
            )
            for _, row in (
                review_df.iterrows()
            )
        ]

        critical_review_count = int(
            (
                review_df[
                    "risk_tier"
                ]
                .fillna("")
                .astype(str)
                == "critical"
            ).sum()
        )

        structured_review_count = int(
            (
                review_df[
                    "risk_tier"
                ]
                .fillna("")
                .astype(str)
                == "structured"
            ).sum()
        )

        descriptive_review_count = int(
            (
                review_df[
                    "risk_tier"
                ]
                .fillna("")
                .astype(str)
                == "descriptive"
            ).sum()
        )

        exception_count = int(
            (
                review_df[
                    "cross_field_flag"
                ].map(
                    lambda value: (
                        to_float(value) > 0
                    )
                )
                | (
                    ~review_df[
                        "format_valid"
                    ].map(to_bool)
                    & review_df[
                        "required_field"
                    ].map(to_bool)
                )
            ).sum()
        )

        priority = priority_for_status(
            status
        )

        item = {
            "document_id": (
                clean_text(document_id)
            ),
            "parent_document_id": (
                clean_text(
                    document_df[
                        "parent_document_id"
                    ].iloc[0]
                )
            ),
            "document_type": (
                clean_text(
                    document_df[
                        "document_type"
                    ].iloc[0]
                )
            ),
            "quality": clean_text(
                document_df[
                    "quality"
                ].iloc[0]
            ),
            "status": status,
            "priority": priority,
            "total_field_count": int(
                len(document_df)
            ),
            "accepted_field_count": int(
                len(accepted_df)
            ),
            "review_field_count": int(
                len(review_df)
            ),
            "critical_review_count": (
                critical_review_count
            ),
            "structured_review_count": (
                structured_review_count
            ),
            "descriptive_review_count": (
                descriptive_review_count
            ),
            "exception_count": (
                exception_count
            ),
            "accepted_fields": (
                accepted_fields
            ),
            "fields_for_review": (
                fields_for_review
            ),
            "review_state": (
                "pending"
                if status != "auto_accept"
                else "not_required"
            ),
            "reviewed_field_count": 0,
        }

        queue_items.append(item)

        summary_rows.append(
            {
                "document_id": (
                    item["document_id"]
                ),
                "parent_document_id": (
                    item[
                        "parent_document_id"
                    ]
                ),
                "document_type": (
                    item["document_type"]
                ),
                "quality": (
                    item["quality"]
                ),
                "status": (
                    item["status"]
                ),
                "priority": (
                    item["priority"]
                ),
                "total_field_count": (
                    item[
                        "total_field_count"
                    ]
                ),
                "accepted_field_count": (
                    item[
                        "accepted_field_count"
                    ]
                ),
                "review_field_count": (
                    item[
                        "review_field_count"
                    ]
                ),
                "critical_review_count": (
                    item[
                        "critical_review_count"
                    ]
                ),
                "structured_review_count": (
                    item[
                        "structured_review_count"
                    ]
                ),
                "descriptive_review_count": (
                    item[
                        "descriptive_review_count"
                    ]
                ),
                "exception_count": (
                    item[
                        "exception_count"
                    ]
                ),
                "review_state": (
                    item["review_state"]
                ),
            }
        )

    queue_items.sort(
        key=lambda item: (
            item["priority"],
            -item[
                "exception_count"
            ],
            -item[
                "critical_review_count"
            ],
            -item[
                "review_field_count"
            ],
            item["document_id"],
        )
    )

    summary_frame = (
        pd.DataFrame(summary_rows)
        .sort_values(
            [
                "priority",
                "exception_count",
                "critical_review_count",
                "review_field_count",
                "document_id",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    return (
        queue_items,
        summary_frame,
    )


def run_review_queue(
    *,
    fields_path: Path,
) -> None:
    if not fields_path.exists():
        raise FileNotFoundError(
            "Field-confidence report not found: "
            f"{fields_path}"
        )

    fields = pd.read_csv(
        fields_path
    )

    required_columns = {
        "document_id",
        "parent_document_id",
        "document_type",
        "quality",
        "field_name",
        "final_engine",
        "final_prediction",
        "final_normalized_prediction",
        "required_field",
        "format_valid",
        "format_message",
        "risk_tier",
        "correctness_probability",
        "risk_threshold",
        "cross_field_flag",
        "field_review_required",
    }

    missing_columns = (
        required_columns
        - set(fields.columns)
    )

    if missing_columns:
        raise ValueError(
            "Field-confidence report is "
            "missing columns: "
            f"{sorted(missing_columns)}"
        )

    queue_items, summary_frame = (
        build_review_queue(fields)
    )

    QUEUE_JSONL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with QUEUE_JSONL_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for item in queue_items:
            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary_frame.to_csv(
        QUEUE_CSV_PATH,
        index=False,
    )

    status_counts = {
        str(status): int(count)
        for status, count in (
            summary_frame[
                "status"
            ]
            .value_counts()
            .items()
        )
    }

    review_documents = (
        summary_frame[
            summary_frame[
                "status"
            ]
            != "auto_accept"
        ]
    )

    summary = {
        "source_fields_path": str(
            fields_path
        ),
        "documents": int(
            len(summary_frame)
        ),
        "documents_requiring_review": int(
            len(review_documents)
        ),
        "documents_auto_accepted": int(
            (
                summary_frame[
                    "status"
                ]
                == "auto_accept"
            ).sum()
        ),
        "total_fields": int(
            summary_frame[
                "total_field_count"
            ].sum()
        ),
        "accepted_fields": int(
            summary_frame[
                "accepted_field_count"
            ].sum()
        ),
        "review_fields": int(
            summary_frame[
                "review_field_count"
            ].sum()
        ),
        "critical_review_fields": int(
            summary_frame[
                "critical_review_count"
            ].sum()
        ),
        "structured_review_fields": int(
            summary_frame[
                "structured_review_count"
            ].sum()
        ),
        "descriptive_review_fields": int(
            summary_frame[
                "descriptive_review_count"
            ].sum()
        ),
        "exception_fields": int(
            summary_frame[
                "exception_count"
            ].sum()
        ),
        "status_counts": (
            status_counts
        ),
        "queue_jsonl_path": str(
            QUEUE_JSONL_PATH
        ),
        "queue_csv_path": str(
            QUEUE_CSV_PATH
        ),
    }

    with QUEUE_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\nIRS review queue created"
    )
    print(
        f"Documents: "
        f"{summary['documents']}"
    )
    print(
        "Documents requiring review: "
        f"{summary['documents_requiring_review']}"
    )
    print(
        "Documents auto accepted: "
        f"{summary['documents_auto_accepted']}"
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
        "Critical review fields: "
        f"{summary['critical_review_fields']}"
    )
    print(
        "Structured review fields: "
        f"{summary['structured_review_fields']}"
    )
    print(
        "Descriptive review fields: "
        f"{summary['descriptive_review_fields']}"
    )
    print(
        f"Exception fields: "
        f"{summary['exception_fields']}"
    )
    print(
        f"Status counts: "
        f"{summary['status_counts']}"
    )
    print(
        f"\nQueue JSONL: "
        f"{QUEUE_JSONL_PATH}"
    )
    print(
        f"Queue summary CSV: "
        f"{QUEUE_CSV_PATH}"
    )
    print(
        f"Queue summary JSON: "
        f"{QUEUE_SUMMARY_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a field-level IRS human-review "
            "queue from confidence-model output."
        )
    )

    parser.add_argument(
        "--fields",
        type=Path,
        default=DEFAULT_FIELDS_PATH,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_review_queue(
        fields_path=arguments.fields,
    )
