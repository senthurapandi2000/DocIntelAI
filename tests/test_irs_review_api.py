from __future__ import annotations

import io
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image


def get_client(
    api_environment: dict[
        str,
        Path | TestClient,
    ],
) -> TestClient:
    client = api_environment[
        "client"
    ]

    assert isinstance(
        client,
        TestClient,
    )

    return client


def create_png_bytes() -> bytes:
    buffer = io.BytesIO()

    Image.new(
        "RGB",
        (
            200,
            100,
        ),
        "white",
    ).save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def test_health_endpoint(
    api_environment: dict[
        str,
        Path | TestClient,
    ],
) -> None:
    client = get_client(
        api_environment
    )

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "healthy"
    assert payload["database"].endswith(
        "irs_review_queue.db"
    )


def test_statistics_endpoint(
    api_environment: dict[
        str,
        Path | TestClient,
    ],
) -> None:
    client = get_client(
        api_environment
    )

    response = client.get(
        "/api/v1/review/statistics"
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["documents"][
            "total_documents"
        ]
        == 1
    )

    assert (
        payload["documents"][
            "pending_documents"
        ]
        == 1
    )

    assert (
        payload["fields"][
            "total_fields"
        ]
        == 2
    )

    assert (
        payload["fields"][
            "auto_accepted_fields"
        ]
        == 1
    )

    assert (
        payload["fields"][
            "pending_review_fields"
        ]
        == 1
    )


def test_list_and_get_document(
    api_environment: dict[
        str,
        Path | TestClient,
    ],
) -> None:
    client = get_client(
        api_environment
    )

    list_response = client.get(
        "/api/v1/review/documents",
        params={
            "limit": 10,
            "offset": 0,
        },
    )

    assert (
        list_response.status_code
        == 200
    )

    list_payload = (
        list_response.json()
    )

    assert list_payload["total"] == 1

    document_id = (
        list_payload["items"][0][
            "document_id"
        ]
    )

    assert document_id == "test_w2_001"

    detail_response = client.get(
        (
            "/api/v1/review/documents/"
            f"{document_id}"
        )
    )

    assert (
        detail_response.status_code
        == 200
    )

    detail_payload = (
        detail_response.json()
    )

    assert (
        detail_payload["document"][
            "document_type"
        ]
        == "w2"
    )

    field_names = {
        field["field_name"]
        for field in (
            detail_payload["fields"]
        )
    }

    assert field_names == {
        "employee_ssn",
        "wages",
    }


def test_assign_approve_and_complete(
    api_environment: dict[
        str,
        Path | TestClient,
    ],
) -> None:
    client = get_client(
        api_environment
    )

    assign_response = client.post(
        (
            "/api/v1/review/documents/"
            "test_w2_001/assign"
        ),
        json={
            "reviewer": "Raj",
            "force": False,
        },
    )

    assert (
        assign_response.status_code
        == 200
    )

    assert (
        assign_response.json()[
            "review_state"
        ]
        == "in_progress"
    )

    early_complete = client.post(
        (
            "/api/v1/review/documents/"
            "test_w2_001/complete"
        ),
        json={
            "reviewer": "Raj",
            "note": "Too early",
        },
    )

    assert (
        early_complete.status_code
        == 409
    )

    decision_response = client.post(
        (
            "/api/v1/review/documents/"
            "test_w2_001/fields/"
            "employee_ssn/decision"
        ),
        json={
            "action": "approve",
            "reviewer": "Raj",
            "corrected_value": None,
            "note": (
                "Verified against source."
            ),
        },
    )

    assert (
        decision_response.status_code
        == 200
    )

    decision_payload = (
        decision_response.json()
    )

    assert (
        decision_payload[
            "field_review_state"
        ]
        == "approved"
    )

    assert (
        decision_payload[
            "document_review_state"
        ]
        == "ready_for_completion"
    )

    complete_response = client.post(
        (
            "/api/v1/review/documents/"
            "test_w2_001/complete"
        ),
        json={
            "reviewer": "Raj",
            "note": (
                "All required fields verified."
            ),
        },
    )

    assert (
        complete_response.status_code
        == 200
    )

    assert (
        complete_response.json()[
            "review_state"
        ]
        == "completed"
    )


def test_correct_field_and_audit_event(
    api_environment: dict[
        str,
        Path | TestClient,
    ],
) -> None:
    client = get_client(
        api_environment
    )

    response = client.post(
        (
            "/api/v1/review/documents/"
            "test_w2_001/fields/"
            "employee_ssn/decision"
        ),
        json={
            "action": "correct",
            "reviewer": "Raj",
            "corrected_value": (
                "000987654"
            ),
            "note": (
                "Corrected after visual review."
            ),
        },
    )

    assert response.status_code == 200

    detail = client.get(
        (
            "/api/v1/review/documents/"
            "test_w2_001"
        )
    ).json()

    employee_ssn = next(
        field
        for field in detail["fields"]
        if field["field_name"]
        == "employee_ssn"
    )

    assert (
        employee_ssn[
            "review_state"
        ]
        == "corrected"
    )

    assert (
        employee_ssn[
            "corrected_value"
        ]
        == "000987654"
    )

    assert (
        employee_ssn[
            "reviewed_by"
        ]
        == "Raj"
    )

    events_response = client.get(
        (
            "/api/v1/review/documents/"
            "test_w2_001/events"
        )
    )

    assert (
        events_response.status_code
        == 200
    )

    event_types = {
        event["event_type"]
        for event in (
            events_response.json()[
                "events"
            ]
        )
    }

    assert "field_corrected" in (
        event_types
    )


def test_upload_and_job_tracking(
    api_environment: dict[
        str,
        Path | TestClient,
    ],
) -> None:
    client = get_client(
        api_environment
    )

    png_bytes = create_png_bytes()

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "sample_w2.png",
                png_bytes,
                "image/png",
            )
        },
        data={
            "expected_document_type": (
                "w2"
            ),
            "uploaded_by": "Raj",
        },
    )

    assert response.status_code == 201

    payload = response.json()

    assert payload["status"] == "queued"

    assert (
        payload[
            "processing_stage"
        ]
        == "awaiting_worker"
    )

    job_id = payload["job_id"]

    job_response = client.get(
        (
            "/api/v1/documents/jobs/"
            f"{job_id}"
        )
    )

    assert (
        job_response.status_code
        == 200
    )

    job_payload = job_response.json()

    assert (
        job_payload["job_status"]
        == "queued"
    )

    assert (
        job_payload[
            "expected_document_type"
        ]
        == "w2"
    )

    list_response = client.get(
        "/api/v1/documents/jobs",
        params={
            "job_status": "queued",
            "limit": 10,
            "offset": 0,
        },
    )

    assert (
        list_response.status_code
        == 200
    )

    assert (
        list_response.json()[
            "total"
        ]
        == 1
    )


def test_duplicate_upload_detection(
    api_environment: dict[
        str,
        Path | TestClient,
    ],
) -> None:
    client = get_client(
        api_environment
    )

    png_bytes = create_png_bytes()

    first = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "first.png",
                png_bytes,
                "image/png",
            )
        },
        data={
            "expected_document_type": (
                "auto"
            ),
            "uploaded_by": "Raj",
        },
    )

    assert first.status_code == 201

    blocked = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "second.png",
                png_bytes,
                "image/png",
            )
        },
        data={
            "expected_document_type": (
                "auto"
            ),
            "uploaded_by": "Raj",
        },
    )

    assert blocked.status_code == 409
    assert "Duplicate file detected" in (
        blocked.json()["detail"]
    )

    allowed = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "second.png",
                png_bytes,
                "image/png",
            )
        },
        data={
            "expected_document_type": (
                "auto"
            ),
            "uploaded_by": "Raj",
            "allow_duplicate": "true",
        },
    )

    assert allowed.status_code == 201

    duplicate = allowed.json()[
        "duplicate_of"
    ]

    assert duplicate is not None

    assert (
        duplicate["upload_id"]
        == first.json()["upload_id"]
    )


def test_reject_unsupported_upload(
    api_environment: dict[
        str,
        Path | TestClient,
    ],
) -> None:
    client = get_client(
        api_environment
    )

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "malware.exe",
                b"not-an-image",
                "application/octet-stream",
            )
        },
        data={
            "expected_document_type": (
                "auto"
            ),
            "uploaded_by": "Raj",
        },
    )

    assert response.status_code == 415


def test_source_image_and_field_crop(
    api_environment: dict[
        str,
        Path | TestClient,
    ],
) -> None:
    client = get_client(
        api_environment
    )

    source_response = client.get(
        (
            "/api/v1/review/documents/"
            "test_w2_001/source-image"
        ),
        params={
            "field_name": (
                "employee_ssn"
            ),
        },
    )

    assert (
        source_response.status_code
        == 200
    )

    assert (
        source_response.headers[
            "content-type"
        ]
        == "image/png"
    )

    crop_response = client.get(
        (
            "/api/v1/review/documents/"
            "test_w2_001/fields/"
            "employee_ssn/crop"
        )
    )

    assert (
        crop_response.status_code
        == 200
    )

    assert (
        crop_response.headers[
            "content-type"
        ]
        == "image/png"
    )

    assert len(
        crop_response.content
    ) > 0


def test_database_contains_uploaded_job(
    api_environment: dict[
        str,
        Path | TestClient,
    ],
) -> None:
    client = get_client(
        api_environment
    )

    database_path = api_environment[
        "database_path"
    ]

    assert isinstance(
        database_path,
        Path,
    )

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "database_test.png",
                create_png_bytes(),
                "image/png",
            )
        },
        data={
            "expected_document_type": (
                "1099_nec"
            ),
            "uploaded_by": "Raj",
        },
    )

    upload_id = response.json()[
        "upload_id"
    ]

    connection = sqlite3.connect(
        database_path
    )

    row = connection.execute(
        """
        SELECT
            upload_status,
            expected_document_type
        FROM document_uploads
        WHERE upload_id = ?
        """,
        (
            upload_id,
        ),
    ).fetchone()

    connection.close()

    assert row == (
        "queued",
        "1099_nec",
    )



def test_retry_and_archive_failed_job(
    api_environment: dict[
        str,
        Path | TestClient,
    ],
) -> None:
    client = get_client(
        api_environment
    )

    upload = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "retry.png",
                create_png_bytes(),
                "image/png",
            )
        },
        data={
            "expected_document_type": (
                "w2"
            ),
            "uploaded_by": "Raj",
        },
    )

    assert upload.status_code == 201

    job_id = upload.json()["job_id"]
    upload_id = upload.json()[
        "upload_id"
    ]

    database_path = api_environment[
        "database_path"
    ]

    assert isinstance(
        database_path,
        Path,
    )

    connection = sqlite3.connect(
        database_path
    )

    connection.execute(
        """
        UPDATE processing_jobs
        SET
            job_status = 'failed',
            processing_stage = 'failed',
            progress_percent = 15,
            error_message = 'Synthetic failure'
        WHERE job_id = ?
        """,
        (
            job_id,
        ),
    )

    connection.execute(
        """
        UPDATE document_uploads
        SET upload_status = 'failed'
        WHERE upload_id = ?
        """,
        (
            upload_id,
        ),
    )

    connection.commit()
    connection.close()

    retry = client.post(
        (
            "/api/v1/documents/jobs/"
            f"{job_id}/retry"
        ),
        json={
            "actor": "Raj",
            "note": "Retry test",
        },
    )

    assert retry.status_code == 200
    assert (
        retry.json()["job_status"]
        == "queued"
    )

    connection = sqlite3.connect(
        database_path
    )

    connection.execute(
        """
        UPDATE processing_jobs
        SET
            job_status = 'failed',
            processing_stage = 'failed'
        WHERE job_id = ?
        """,
        (
            job_id,
        ),
    )

    connection.commit()
    connection.close()

    archive = client.post(
        (
            "/api/v1/documents/jobs/"
            f"{job_id}/archive"
        ),
        json={
            "actor": "Raj",
            "note": "Archive test",
        },
    )

    assert archive.status_code == 200
    assert (
        archive.json()["job_status"]
        == "archived"
    )

    default_list = client.get(
        "/api/v1/documents/jobs"
    ).json()

    assert default_list["total"] == 0

    archived_list = client.get(
        "/api/v1/documents/jobs",
        params={
            "job_status": "archived",
            "include_archived": True,
        },
    ).json()

    assert archived_list["total"] == 1
