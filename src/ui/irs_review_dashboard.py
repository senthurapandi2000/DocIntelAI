from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = os.getenv(
    "DOCINTELAI_REVIEW_API",
    "http://127.0.0.1:8000",
).rstrip("/")

REQUEST_TIMEOUT_SECONDS = 15


def api_request(
    method: str,
    path: str,
    *,
    api_url: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    url = f"{api_url}{path}"

    try:
        response = requests.request(
            method=method,
            url=url,
            params=params,
            json=json_body,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            "Could not reach the FastAPI service. "
            "Confirm that Uvicorn is running on "
            f"{api_url}. Details: {error}"
        ) from error

    if response.status_code >= 400:
        try:
            detail = response.json().get(
                "detail",
                response.text,
            )
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"API request failed "
            f"({response.status_code}): {detail}"
        )

    if not response.content:
        return None

    try:
        return response.json()
    except ValueError as error:
        raise RuntimeError(
            "API returned a non-JSON response."
        ) from error




def api_image_request(
    path: str,
    *,
    api_url: str,
    params: dict[str, Any] | None = None,
) -> bytes:
    url = f"{api_url}{path}"

    try:
        response = requests.get(
            url,
            params=params,
            timeout=(
                REQUEST_TIMEOUT_SECONDS
            ),
        )
    except requests.RequestException as error:
        raise RuntimeError(
            "Could not load source-document "
            f"media from {api_url}. "
            f"Details: {error}"
        ) from error

    if response.status_code >= 400:
        try:
            detail = response.json().get(
                "detail",
                response.text,
            )
        except ValueError:
            detail = response.text

        raise RuntimeError(
            "Media request failed "
            f"({response.status_code}): "
            f"{detail}"
        )

    return response.content


def load_source_image(
    api_url: str,
    *,
    document_id: str,
    field_name: str,
) -> bytes:
    return api_image_request(
        (
            "/api/v1/review/documents/"
            f"{quote(document_id, safe='')}"
            "/source-image"
        ),
        api_url=api_url,
        params={
            "field_name": field_name,
            "max_width": 1400,
        },
    )


def load_field_crop(
    api_url: str,
    *,
    document_id: str,
    field_name: str,
) -> bytes:
    return api_image_request(
        (
            "/api/v1/review/documents/"
            f"{quote(document_id, safe='')}"
            "/fields/"
            f"{quote(field_name, safe='')}"
            "/crop"
        ),
        api_url=api_url,
        params={
            "padding": 35,
            "scale": 2.5,
        },
    )


def format_probability(
    value: Any,
) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "—"


def status_label(
    item: dict[str, Any],
) -> str:
    return (
        f"{item['document_id']} | "
        f"{item['document_type']} | "
        f"{item['status']} | "
        f"{item['reviewed_field_count']}/"
        f"{item['review_field_count']} reviewed"
    )


def load_statistics(
    api_url: str,
) -> dict[str, Any]:
    return api_request(
        "GET",
        "/api/v1/review/statistics",
        api_url=api_url,
    )


def load_documents(
    api_url: str,
    *,
    review_state: str | None,
    status: str | None,
    assigned_to: str | None,
    limit: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "limit": limit,
        "offset": 0,
    }

    if review_state:
        params["review_state"] = review_state

    if status:
        params["status"] = status

    if assigned_to:
        params["assigned_to"] = assigned_to

    return api_request(
        "GET",
        "/api/v1/review/documents",
        api_url=api_url,
        params=params,
    )


def load_document(
    api_url: str,
    document_id: str,
) -> dict[str, Any]:
    return api_request(
        "GET",
        (
            "/api/v1/review/documents/"
            f"{quote(document_id, safe='')}"
        ),
        api_url=api_url,
    )


def load_events(
    api_url: str,
    document_id: str,
) -> dict[str, Any]:
    return api_request(
        "GET",
        (
            "/api/v1/review/documents/"
            f"{quote(document_id, safe='')}"
            "/events"
        ),
        api_url=api_url,
    )


def assign_document(
    api_url: str,
    document_id: str,
    reviewer: str,
) -> dict[str, Any]:
    return api_request(
        "POST",
        (
            "/api/v1/review/documents/"
            f"{quote(document_id, safe='')}"
            "/assign"
        ),
        api_url=api_url,
        json_body={
            "reviewer": reviewer,
            "force": False,
        },
    )


def submit_field_decision(
    api_url: str,
    *,
    document_id: str,
    field_name: str,
    reviewer: str,
    action: str,
    corrected_value: str | None,
    note: str | None,
) -> dict[str, Any]:
    return api_request(
        "POST",
        (
            "/api/v1/review/documents/"
            f"{quote(document_id, safe='')}"
            "/fields/"
            f"{quote(field_name, safe='')}"
            "/decision"
        ),
        api_url=api_url,
        json_body={
            "action": action,
            "reviewer": reviewer,
            "corrected_value": corrected_value,
            "note": note,
        },
    )


def complete_document(
    api_url: str,
    *,
    document_id: str,
    reviewer: str,
    note: str | None,
) -> dict[str, Any]:
    return api_request(
        "POST",
        (
            "/api/v1/review/documents/"
            f"{quote(document_id, safe='')}"
            "/complete"
        ),
        api_url=api_url,
        json_body={
            "reviewer": reviewer,
            "note": note,
        },
    )



def upload_document_file(
    api_url: str,
    *,
    uploaded_file: Any,
    expected_document_type: str,
    uploaded_by: str,
    allow_duplicate: bool,
) -> dict[str, Any]:
    url = (
        f"{api_url}"
        "/api/v1/documents/upload"
    )

    file_bytes = (
        uploaded_file.getvalue()
    )

    files = {
        "file": (
            uploaded_file.name,
            file_bytes,
            uploaded_file.type
            or "application/octet-stream",
        )
    }

    data = {
        "expected_document_type": (
            expected_document_type
        ),
        "uploaded_by": uploaded_by,
        "allow_duplicate": str(
            allow_duplicate
        ).lower(),
    }

    try:
        response = requests.post(
            url,
            files=files,
            data=data,
            timeout=60,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            "Could not upload the document. "
            f"Details: {error}"
        ) from error

    if response.status_code >= 400:
        try:
            detail = response.json().get(
                "detail",
                response.text,
            )
        except ValueError:
            detail = response.text

        raise RuntimeError(
            "Upload failed "
            f"({response.status_code}): "
            f"{detail}"
        )

    return response.json()


def load_processing_jobs(
    api_url: str,
    *,
    job_status: str | None,
    include_archived: bool,
    limit: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "limit": limit,
        "offset": 0,
        "include_archived": (
            include_archived
        ),
    }

    if job_status:
        params["job_status"] = (
            job_status
        )

    return api_request(
        "GET",
        "/api/v1/documents/jobs",
        api_url=api_url,
        params=params,
    )


def load_processing_job(
    api_url: str,
    job_id: str,
) -> dict[str, Any]:
    return api_request(
        "GET",
        (
            "/api/v1/documents/jobs/"
            f"{quote(job_id, safe='')}"
        ),
        api_url=api_url,
    )


def retry_processing_job(
    api_url: str,
    *,
    job_id: str,
    actor: str,
    note: str | None = None,
) -> dict[str, Any]:
    return api_request(
        "POST",
        (
            "/api/v1/documents/jobs/"
            f"{quote(job_id, safe='')}"
            "/retry"
        ),
        api_url=api_url,
        json_body={
            "actor": actor,
            "note": note,
        },
    )


def archive_processing_job(
    api_url: str,
    *,
    job_id: str,
    actor: str,
    note: str | None = None,
) -> dict[str, Any]:
    return api_request(
        "POST",
        (
            "/api/v1/documents/jobs/"
            f"{quote(job_id, safe='')}"
            "/archive"
        ),
        api_url=api_url,
        json_body={
            "actor": actor,
            "note": note,
        },
    )


def job_label(
    item: dict[str, Any],
) -> str:
    return (
        f"{item['original_filename']} | "
        f"{item['job_status']} | "
        f"{item['processing_stage']} | "
        f"{float(item['progress_percent']):.0f}%"
    )


def render_upload_workspace(
    *,
    api_url: str,
    reviewer: str,
) -> None:
    st.subheader("Upload a tax document")
    st.caption(
        "Upload a template-aligned 2026 W-2 "
        "or 1099-NEC image/PDF. The worker "
        "will classify, extract, score and add "
        "the result to the review queue."
    )

    with st.form(
        "document_upload_form",
        clear_on_submit=False,
    ):
        uploaded_file = st.file_uploader(
            "Document",
            type=[
                "png",
                "jpg",
                "jpeg",
                "pdf",
            ],
            help=(
                "Maximum file size accepted by "
                "the API is 20 MB."
            ),
        )

        expected_type = st.selectbox(
            "Expected document type",
            options=[
                "auto",
                "w2",
                "1099_nec",
            ],
            help=(
                "Use auto for classification. "
                "Choose a type when the form is "
                "known or auto-detection fails."
            ),
        )

        allow_duplicate = st.checkbox(
            "Process duplicate anyway",
            value=False,
            help=(
                "Leave this disabled to prevent "
                "unnecessary repeated OCR jobs."
            ),
        )

        submitted = st.form_submit_button(
            "Upload and queue",
            type="primary",
            disabled=not reviewer,
        )

    if submitted:
        if uploaded_file is None:
            st.error(
                "Choose a document before uploading."
            )
            return

        try:
            result = upload_document_file(
                api_url,
                uploaded_file=uploaded_file,
                expected_document_type=(
                    expected_type
                ),
                uploaded_by=reviewer,
                allow_duplicate=(
                    allow_duplicate
                ),
            )
        except RuntimeError as error:
            st.error(str(error))
        else:
            st.session_state[
                "last_uploaded_job_id"
            ] = result["job_id"]

            st.success(
                "Document uploaded and queued."
            )

            result_columns = st.columns(4)

            result_columns[0].metric(
                "Status",
                result["status"],
            )
            result_columns[1].metric(
                "Stage",
                result[
                    "processing_stage"
                ],
            )
            result_columns[2].metric(
                "File size",
                (
                    f"{result['file_size_bytes'] / 1024:.1f} KB"
                ),
            )
            result_columns[3].metric(
                "Expected type",
                result[
                    "expected_document_type"
                ],
            )

            st.code(
                result["job_id"],
                language=None,
            )

            if result.get(
                "duplicate_of"
            ):
                duplicate = result[
                    "duplicate_of"
                ]

                st.warning(
                    "This file matches a previous "
                    "upload. It was still queued as "
                    "a new processing job. Previous "
                    f"upload: {duplicate['upload_id']}."
                )

    last_job_id = st.session_state.get(
        "last_uploaded_job_id"
    )

    if last_job_id:
        st.divider()
        st.markdown(
            "### Latest uploaded job"
        )

        try:
            job = load_processing_job(
                api_url,
                last_job_id,
            )
        except RuntimeError as error:
            st.error(str(error))
        else:
            st.progress(
                min(
                    float(
                        job[
                            "progress_percent"
                        ]
                    )
                    / 100.0,
                    1.0,
                ),
                text=(
                    f"{job['job_status']} · "
                    f"{job['processing_stage']} · "
                    f"{float(job['progress_percent']):.0f}%"
                ),
            )

            st.json(
                {
                    "job_id": job[
                        "job_id"
                    ],
                    "filename": job[
                        "original_filename"
                    ],
                    "job_status": job[
                        "job_status"
                    ],
                    "processing_stage": job[
                        "processing_stage"
                    ],
                    "detected_document_type": (
                        job[
                            "detected_document_type"
                        ]
                    ),
                    "result_document_id": (
                        job[
                            "result_document_id"
                        ]
                    ),
                    "error_message": (
                        job[
                            "error_message"
                        ]
                    ),
                }
            )

            if job["job_status"] == "queued":
                st.info(
                    "The job is waiting for the "
                    "processing worker. Keep the "
                    "worker terminal running."
                )

            elif job["job_status"] == "completed":
                st.success(
                    "Processing finished. Open the "
                    "Review Queue workspace and "
                    "refresh the dashboard."
                )

            elif job["job_status"] == "failed":
                st.error(
                    job["error_message"]
                    or "Processing failed."
                )


def render_jobs_workspace(
    *,
    api_url: str,
    reviewer: str,
) -> None:
    st.subheader("Processing jobs")
    st.caption(
        "Track uploaded files from queueing "
        "through classification, OCR, confidence "
        "scoring and review-queue creation."
    )

    control_columns = st.columns(
        [2, 1, 1]
    )

    status_choice = (
        control_columns[0].selectbox(
            "Job status",
            options=[
                "Active + completed",
                "queued",
                "processing",
                "completed",
                "failed",
                "archived",
                "All",
            ],
            key="job_status_filter",
        )
    )

    limit = control_columns[1].selectbox(
        "Jobs to load",
        options=[
            10,
            25,
            50,
            100,
        ],
        index=1,
    )

    if control_columns[2].button(
        "Refresh jobs",
        use_container_width=True,
    ):
        st.rerun()

    try:
        requested_status = (
            status_choice
            if status_choice
            in {
                "queued",
                "processing",
                "completed",
                "failed",
                "archived",
            }
            else None
        )

        payload = load_processing_jobs(
            api_url,
            job_status=requested_status,
            include_archived=(
                status_choice
                in {
                    "archived",
                    "All",
                }
            ),
            limit=int(limit),
        )
    except RuntimeError as error:
        st.error(str(error))
        return

    jobs = payload["items"]

    if (
        status_choice
        == "Active + completed"
    ):
        jobs = [
            job
            for job in jobs
            if job["job_status"]
            in {
                "queued",
                "processing",
                "completed",
            }
        ]

    if not jobs:
        st.info(
            "No processing jobs match the "
            "selected filter."
        )
        return

    summary = pd.DataFrame(
        [
            {
                "Filename": job[
                    "original_filename"
                ],
                "Status": job[
                    "job_status"
                ],
                "Stage": job[
                    "processing_stage"
                ],
                "Progress": (
                    f"{float(job['progress_percent']):.0f}%"
                ),
                "Expected type": (
                    job[
                        "expected_document_type"
                    ]
                ),
                "Detected type": (
                    job[
                        "detected_document_type"
                    ]
                    or ""
                ),
                "Result document": (
                    job[
                        "result_document_id"
                    ]
                    or ""
                ),
                "Uploaded by": (
                    job["uploaded_by"]
                    or ""
                ),
                "Created": job[
                    "created_at"
                ],
            }
            for job in jobs
        ]
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    selected_job = st.selectbox(
        "Choose a job",
        options=jobs,
        format_func=job_label,
    )

    try:
        details = load_processing_job(
            api_url,
            selected_job["job_id"],
        )
    except RuntimeError as error:
        st.error(str(error))
        return

    st.progress(
        min(
            float(
                details[
                    "progress_percent"
                ]
            )
            / 100.0,
            1.0,
        ),
        text=(
            f"{details['job_status']} · "
            f"{details['processing_stage']} · "
            f"{float(details['progress_percent']):.0f}%"
        ),
    )

    metrics = st.columns(4)

    metrics[0].metric(
        "Status",
        details["job_status"],
    )
    metrics[1].metric(
        "Stage",
        details[
            "processing_stage"
        ],
    )
    metrics[2].metric(
        "Detected type",
        details[
            "detected_document_type"
        ]
        or "—",
    )
    metrics[3].metric(
        "File size",
        (
            f"{details['file_size_bytes'] / 1024:.1f} KB"
        ),
    )

    detail_payload = {
        "job_id": details["job_id"],
        "upload_id": details[
            "upload_id"
        ],
        "original_filename": (
            details[
                "original_filename"
            ]
        ),
        "expected_document_type": (
            details[
                "expected_document_type"
            ]
        ),
        "detected_document_type": (
            details[
                "detected_document_type"
            ]
        ),
        "result_document_id": (
            details[
                "result_document_id"
            ]
        ),
        "error_message": details[
            "error_message"
        ],
        "started_at": details[
            "started_at"
        ],
        "completed_at": details[
            "completed_at"
        ],
    }

    st.json(detail_payload)

    if details["job_status"] == "queued":
        st.info(
            "This job is waiting for the worker. "
            "Run the worker continuously or once "
            "from a separate terminal."
        )

    elif details["job_status"] == "completed":
        st.success(
            "The document was processed and added "
            "to the review queue."
        )

    elif details["job_status"] == "failed":
        st.error(
            details["error_message"]
            or "The processing job failed."
        )

    st.divider()
    action_columns = st.columns(
        [1, 1, 3]
    )

    if details["job_status"] in {
        "failed",
        "archived",
    }:
        if action_columns[0].button(
            "Retry job",
            type="primary",
            use_container_width=True,
            disabled=not reviewer,
        ):
            try:
                retry_processing_job(
                    api_url,
                    job_id=details["job_id"],
                    actor=reviewer,
                    note=(
                        "Retried from Streamlit."
                    ),
                )
            except RuntimeError as error:
                st.error(str(error))
            else:
                st.success(
                    "Job requeued. Keep the "
                    "processing worker running."
                )
                st.rerun()

    if details["job_status"] in {
        "failed",
        "completed",
    }:
        if action_columns[1].button(
            "Archive job",
            use_container_width=True,
            disabled=not reviewer,
        ):
            try:
                archive_processing_job(
                    api_url,
                    job_id=details["job_id"],
                    actor=reviewer,
                    note=(
                        "Archived from Streamlit."
                    ),
                )
            except RuntimeError as error:
                st.error(str(error))
            else:
                st.success(
                    "Job archived."
                )
                st.rerun()


def render_statistics(
    statistics: dict[str, Any],
) -> None:
    documents = statistics["documents"]
    fields = statistics["fields"]

    row_one = st.columns(5)

    row_one[0].metric(
        "Documents",
        documents["total_documents"],
    )
    row_one[1].metric(
        "Pending",
        documents["pending_documents"],
    )
    row_one[2].metric(
        "In progress",
        documents["in_progress_documents"],
    )
    row_one[3].metric(
        "Ready",
        documents["ready_documents"],
    )
    row_one[4].metric(
        "Completed",
        documents["completed_documents"],
    )

    row_two = st.columns(5)

    row_two[0].metric(
        "Fields",
        fields["total_fields"],
    )
    row_two[1].metric(
        "Auto accepted",
        fields["auto_accepted_fields"],
    )
    row_two[2].metric(
        "Pending review",
        fields["pending_review_fields"],
    )
    row_two[3].metric(
        "Approved",
        fields["approved_fields"],
    )
    row_two[4].metric(
        "Corrected",
        fields["corrected_fields"],
    )


def render_document_summary(
    document: dict[str, Any],
) -> None:
    columns = st.columns(5)

    columns[0].metric(
        "Priority",
        document["priority"],
    )
    columns[1].metric(
        "Review fields",
        document["review_field_count"],
    )
    columns[2].metric(
        "Reviewed",
        document["reviewed_field_count"],
    )
    columns[3].metric(
        "Exceptions",
        document["exception_count"],
    )
    columns[4].metric(
        "Accepted fields",
        document["accepted_field_count"],
    )

    review_total = max(
        int(document["review_field_count"]),
        1,
    )

    reviewed = int(
        document["reviewed_field_count"]
    )

    st.progress(
        min(
            reviewed / review_total,
            1.0,
        ),
        text=(
            f"Review progress: {reviewed}/"
            f"{document['review_field_count']}"
        ),
    )

    st.caption(
        f"Type: {document['document_type']} · "
        f"Quality: {document['quality']} · "
        f"Status: {document['status']} · "
        f"Workflow state: {document['review_state']} · "
        f"Assigned to: "
        f"{document['assigned_to'] or 'Unassigned'}"
    )


def build_field_table(
    fields: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for field in fields:
        rows.append(
            {
                "Field": field["field_name"],
                "Value": (
                    field["corrected_value"]
                    or field["value"]
                    or ""
                ),
                "Risk": field["risk_tier"],
                "Required": bool(
                    field["required"]
                ),
                "Engine": field["engine"],
                "Confidence": format_probability(
                    field[
                        "correctness_probability"
                    ]
                ),
                "Threshold": format_probability(
                    field[
                        "acceptance_threshold"
                    ]
                ),
                "Format valid": bool(
                    field["format_valid"]
                ),
                "Cross-field flag": bool(
                    field["cross_field_flag"]
                ),
                "Review state": field[
                    "review_state"
                ],
                "Reason": (
                    field["review_reason"]
                    or ""
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(
        page_title=(
            "DocIntelAI IRS Review"
        ),
        page_icon="📄",
        layout="wide",
    )

    st.title(
        "DocIntelAI IRS Review Dashboard"
    )
    st.caption(
        "Human-in-the-loop verification for "
        "W-2 and 1099-NEC OCR extractions."
    )

    review_state_choice = "All"
    status_choice = "All"
    assigned_filter = ""
    queue_limit = 25

    with st.sidebar:
        st.header("Connection")

        api_url = st.text_input(
            "FastAPI URL",
            value=DEFAULT_API_URL,
        ).rstrip("/")

        reviewer = st.text_input(
            "Reviewer name",
            value="Raj",
        ).strip()

        st.divider()
        st.header("Workspace")

        workspace = st.radio(
            "Choose workspace",
            options=[
                "Review Queue",
                "Upload Document",
                "Processing Jobs",
            ],
        )

        if workspace == "Review Queue":
            st.divider()
            st.header("Queue filters")

            review_state_choice = st.selectbox(
                "Workflow state",
                options=[
                    "All",
                    "pending",
                    "in_progress",
                    "ready_for_completion",
                    "completed",
                ],
            )

            status_choice = st.selectbox(
                "Document status",
                options=[
                    "All",
                    "exception_review",
                    "critical_field_verification",
                    "targeted_field_review",
                    "auto_accept",
                ],
            )

            assigned_filter = st.text_input(
                "Assigned reviewer",
                value="",
            ).strip()

            queue_limit = st.slider(
                "Documents to load",
                min_value=5,
                max_value=100,
                value=25,
                step=5,
            )

        refresh = st.button(
            "Refresh dashboard",
            use_container_width=True,
        )

    if refresh:
        st.rerun()

    if not api_url:
        st.error(
            "Enter the FastAPI URL."
        )
        st.stop()

    if not reviewer:
        st.warning(
            "Enter a reviewer name before "
            "submitting decisions."
        )

    if workspace == "Upload Document":
        render_upload_workspace(
            api_url=api_url,
            reviewer=reviewer,
        )
        return

    if workspace == "Processing Jobs":
        render_jobs_workspace(
            api_url=api_url,
            reviewer=reviewer,
        )
        return

    try:
        statistics = load_statistics(
            api_url
        )
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    render_statistics(statistics)

    st.divider()

    try:
        queue = load_documents(
            api_url,
            review_state=(
                None
                if review_state_choice
                == "All"
                else review_state_choice
            ),
            status=(
                None
                if status_choice == "All"
                else status_choice
            ),
            assigned_to=(
                assigned_filter or None
            ),
            limit=queue_limit,
        )
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    documents = queue["items"]

    st.subheader(
        f"Review queue ({queue['total']} matching)"
    )

    if not documents:
        st.info(
            "No documents match the selected "
            "filters."
        )
        st.stop()

    selected_document = st.selectbox(
        "Choose a document",
        options=documents,
        format_func=status_label,
    )

    document_id = selected_document[
        "document_id"
    ]

    try:
        details = load_document(
            api_url,
            document_id,
        )
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    document = details["document"]
    fields = details["fields"]

    render_document_summary(document)

    action_columns = st.columns(
        [1, 1, 3]
    )

    if action_columns[0].button(
        "Assign to me",
        disabled=(
            not reviewer
            or document["review_state"]
            == "completed"
        ),
        use_container_width=True,
    ):
        try:
            assign_document(
                api_url,
                document_id,
                reviewer,
            )
            st.success(
                f"Assigned to {reviewer}."
            )
            st.rerun()
        except RuntimeError as error:
            st.error(str(error))

    if action_columns[1].button(
        "Reload document",
        use_container_width=True,
    ):
        st.rerun()

    review_fields = [
        field
        for field in fields
        if bool(field["review_required"])
    ]

    accepted_fields = [
        field
        for field in fields
        if not bool(
            field["review_required"]
        )
    ]

    pending_fields = [
        field
        for field in review_fields
        if field["review_state"]
        == "pending"
    ]

    st.divider()

    review_tab, accepted_tab, audit_tab = (
        st.tabs(
            [
                f"Fields for review "
                f"({len(review_fields)})",
                f"Auto-accepted fields "
                f"({len(accepted_fields)})",
                "Audit history",
            ]
        )
    )

    with review_tab:
        if not review_fields:
            st.success(
                "No fields require review."
            )
        else:
            st.dataframe(
                build_field_table(
                    review_fields
                ),
                use_container_width=True,
                hide_index=True,
            )

            selectable_fields = (
                pending_fields
                if pending_fields
                else review_fields
            )

            selected_field = st.selectbox(
                "Select a field",
                options=selectable_fields,
                format_func=lambda field: (
                    f"{field['field_name']} | "
                    f"{field['risk_tier']} | "
                    f"{field['review_state']}"
                ),
            )

            st.markdown(
                f"### {selected_field['field_name']}"
            )

            st.markdown(
                "#### Source verification"
            )

            media_columns = st.columns(
                [3, 2]
            )

            try:
                source_image = (
                    load_source_image(
                        api_url,
                        document_id=document_id,
                        field_name=(
                            selected_field[
                                "field_name"
                            ]
                        ),
                    )
                )

                field_crop = (
                    load_field_crop(
                        api_url,
                        document_id=document_id,
                        field_name=(
                            selected_field[
                                "field_name"
                            ]
                        ),
                    )
                )
            except RuntimeError as error:
                st.error(str(error))
            else:
                media_columns[0].image(
                    source_image,
                    caption=(
                        "Source document — "
                        "selected field highlighted"
                    ),
                    use_container_width=True,
                )

                media_columns[1].image(
                    field_crop,
                    caption=(
                        "Selected field crop"
                    ),
                    use_container_width=True,
                )

            field_columns = st.columns(
                [2, 1, 1, 1]
            )

            field_columns[0].write(
                "**Extracted value**"
            )
            field_columns[0].code(
                selected_field["value"]
                or "(empty)"
            )

            field_columns[1].metric(
                "Correctness",
                format_probability(
                    selected_field[
                        "correctness_probability"
                    ]
                ),
            )

            field_columns[2].metric(
                "Threshold",
                format_probability(
                    selected_field[
                        "acceptance_threshold"
                    ]
                ),
            )

            field_columns[3].metric(
                "OCR engine",
                selected_field["engine"],
            )

            st.warning(
                selected_field[
                    "review_reason"
                ]
                or "Manual verification required."
            )

            with st.form(
                key=(
                    "field_decision_"
                    f"{document_id}_"
                    f"{selected_field['field_name']}"
                )
            ):
                action = st.radio(
                    "Decision",
                    options=[
                        "approve",
                        "correct",
                    ],
                    horizontal=True,
                )

                corrected_value = st.text_input(
                    "Corrected value",
                    value=(
                        selected_field[
                            "corrected_value"
                        ]
                        or selected_field[
                            "value"
                        ]
                        or ""
                    ),
                    disabled=(
                        action == "approve"
                    ),
                )

                note = st.text_area(
                    "Reviewer note",
                    value=(
                        selected_field[
                            "reviewer_note"
                        ]
                        or ""
                    ),
                    placeholder=(
                        "Example: Verified against "
                        "the source document."
                    ),
                )

                submitted = st.form_submit_button(
                    "Submit decision",
                    type="primary",
                    disabled=(
                        not reviewer
                        or selected_field[
                            "review_state"
                        ]
                        != "pending"
                    ),
                )

                if submitted:
                    try:
                        submit_field_decision(
                            api_url,
                            document_id=document_id,
                            field_name=(
                                selected_field[
                                    "field_name"
                                ]
                            ),
                            reviewer=reviewer,
                            action=action,
                            corrected_value=(
                                corrected_value
                                if action
                                == "correct"
                                else None
                            ),
                            note=(
                                note or None
                            ),
                        )
                        st.success(
                            "Field decision saved."
                        )
                        st.rerun()
                    except RuntimeError as error:
                        st.error(str(error))

            if not pending_fields:
                st.success(
                    "All review fields have decisions."
                )

                completion_note = st.text_area(
                    "Completion note",
                    key=(
                        "completion_note_"
                        f"{document_id}"
                    ),
                )

                if st.button(
                    "Complete document review",
                    type="primary",
                    disabled=(
                        not reviewer
                        or document[
                            "review_state"
                        ]
                        == "completed"
                    ),
                ):
                    try:
                        complete_document(
                            api_url,
                            document_id=document_id,
                            reviewer=reviewer,
                            note=(
                                completion_note
                                or None
                            ),
                        )
                        st.success(
                            "Document review completed."
                        )
                        st.rerun()
                    except RuntimeError as error:
                        st.error(str(error))

    with accepted_tab:
        if accepted_fields:
            st.dataframe(
                build_field_table(
                    accepted_fields
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No fields were auto accepted."
            )

    with audit_tab:
        try:
            event_payload = load_events(
                api_url,
                document_id,
            )
        except RuntimeError as error:
            st.error(str(error))
        else:
            events = event_payload[
                "events"
            ]

            if not events:
                st.info(
                    "No audit events found."
                )
            else:
                event_rows = []

                for event in events:
                    event_rows.append(
                        {
                            "Time": event[
                                "created_at"
                            ],
                            "Event": event[
                                "event_type"
                            ],
                            "Field": (
                                event[
                                    "field_name"
                                ]
                                or ""
                            ),
                            "Actor": (
                                event["actor"]
                                or ""
                            ),
                            "Details": str(
                                event["details"]
                            ),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(
                        event_rows
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


if __name__ == "__main__":
    main()
