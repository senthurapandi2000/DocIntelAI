# DocIntelAI

DocIntelAI is a production-style document intelligence platform for U.S. SEC filings and IRS forms. It combines document classification, XBRL validation, hybrid OCR, learned routing, confidence scoring, human review, audit logging, FastAPI, Streamlit, SQLite, automated tests, and Docker.

The IRS workflow is intentionally human-in-the-loop. Low-confidence or high-risk fields are routed to reviewers instead of being silently accepted.

## What the system does

### SEC workflow

- Classifies SEC 8-K, 10-Q, and 10-K filings
- Uses TF-IDF and LinearSVC for filing-type classification
- Extracts and validates XBRL facts
- Applies fallback metric logic when preferred facts are unavailable
- Produces structured validation reports

### IRS workflow

- Supports 2026 W-2 and 1099-NEC layouts
- Accepts PNG, JPG, and PDF uploads
- Detects document type automatically
- Extracts coordinate-based field crops
- Runs hybrid Tesseract OCR
- Uses TrOCR as a fallback
- Uses a learned router to decide when fallback is useful
- Scores field-level correctness probability
- Applies format and cross-field validation
- Routes uncertain fields into a review queue
- Supports approve, correct, assign, complete, retry, and archive actions
- Preserves field-level audit history

## Architecture

```mermaid
flowchart LR
    A[User uploads PDF or image] --> B[FastAPI upload endpoint]
    B --> C[(SQLite processing job)]
    C --> D[Background worker]

    D --> E[PDF or image normalization]
    E --> F[Document-type classifier]
    F --> G[Coordinate-based field crops]

    G --> H[Tesseract OCR]
    H --> I[Learned OCR router]
    I -->|Fallback needed| J[TrOCR]
    I -->|Tesseract retained| K[Candidate value]
    J --> K

    K --> L[Confidence model]
    L --> M[Format validation]
    M --> N[Cross-field validation]

    N -->|Safe to accept| O[Auto-accepted field]
    N -->|Uncertain or high risk| P[Human review queue]

    O --> Q[(SQLite review store)]
    P --> Q

    Q --> R[FastAPI review API]
    R --> S[Streamlit dashboard]
    S --> T[Approve, correct, complete]
    T --> U[Audit events]
```

A more detailed explanation is available in [`docs/architecture.md`](docs/architecture.md).

## Key results

### SEC evaluation

| Metric | Result |
|---|---:|
| Filings processed | 88 |
| 8-K filings | 30 |
| 10-Q filings | 29 |
| 10-K filings | 29 |
| XBRL validation failures | 0 |
| 10-K/10-Q filings without facts | 0 |
| Average preferred-metric completeness | 87.93% |
| Fallback metrics used | 24 |

The SEC classifier achieved 100% accuracy on the project's limited internal and external holdout sets. This result should not be interpreted as universal performance across all SEC filings.

### IRS OCR evaluation

| Metric | Result |
|---|---:|
| Final untouched test documents | 159 |
| Final untouched test fields | 2,784 |
| Hybrid Tesseract exact match | 44.40% |
| Hybrid Tesseract similarity | 75.38 |
| Routed OCR exact match | 49.68% |
| Routed OCR similarity | 81.28 |
| Clean exact match | 64.01% |
| Mild exact match | 54.53% |
| Hard exact match | 30.50% |

### Human-review policy

| Metric | Result |
|---|---:|
| Auto-accepted fields | 767 / 2,784 |
| Fields routed for review | 2,017 / 2,784 |
| Review rate | 72.45% |
| Auto-accepted exact accuracy | 86.31% |
| Incorrect-field capture | 92.51% |
| Critical accepted exact accuracy | 96.97% |
| Critical incorrect-field capture | 99.83% |
| Structured accepted exact accuracy | 91.03% |
| Structured incorrect-field capture | 97.44% |

The system favors review coverage over aggressive automation, particularly for TINs, SSNs, EINs, wages, tax amounts, and other high-risk fields.

## Synthetic IRS dataset

- 500 clean base documents
- 500 mild-degradation variants
- 500 hard-degradation variants
- 1,500 labeled images total
- Parent-based train, validation, and test splits to reduce leakage
- Fictional identifiers using reserved-looking synthetic prefixes
- Samples clearly marked as synthetic and not for filing

No real tax records are required to reproduce the portfolio workflow.

## Technology stack

- Python 3.12
- FastAPI
- Streamlit
- SQLite
- Tesseract OCR
- TrOCR
- PyTorch
- Transformers
- scikit-learn
- pandas
- Pillow
- OpenCV
- ReportLab
- pytest
- Docker and Docker Compose

## Repository structure

```text
DocIntelAI/
├── data/
│   ├── processed/
│   ├── samples/
│   └── uploads/
├── docs/
│   ├── architecture.md
│   ├── demo_script.md
│   ├── release_checklist.md
│   └── screenshots/
├── models/
├── reports/
├── scripts/
├── src/
│   ├── api/
│   ├── services/
│   ├── ui/
│   └── validation/
├── tests/
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run all tests:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1
```

Start FastAPI, the CPU worker, and Streamlit:

```powershell
powershell -ExecutionPolicy Bypass `
    -File scripts\start_all.ps1 `
    -WorkerMode cpu
```

Local URLs:

- Dashboard: `http://localhost:8501`
- FastAPI health: `http://127.0.0.1:8000/health`
- FastAPI docs: `http://127.0.0.1:8000/docs`

## Docker setup

```powershell
docker compose build
docker compose up -d
docker compose ps
```

Docker URLs:

- Dashboard: `http://localhost:8501`
- FastAPI health: `http://localhost:8001/health`
- FastAPI docs: `http://localhost:8001/docs`

Inside Docker, Streamlit connects to FastAPI through `http://api:8000`.

Stop:

```powershell
docker compose down
```

## Main API endpoints

### Review workflow

```text
GET  /health
GET  /api/v1/review/statistics
GET  /api/v1/review/documents
GET  /api/v1/review/documents/{document_id}
POST /api/v1/review/documents/{document_id}/assign
POST /api/v1/review/documents/{document_id}/fields/{field_name}/decision
POST /api/v1/review/documents/{document_id}/complete
GET  /api/v1/review/documents/{document_id}/events
GET  /api/v1/review/documents/{document_id}/source-image
GET  /api/v1/review/documents/{document_id}/fields/{field_name}/crop
```

### Upload and processing

```text
POST /api/v1/documents/upload
GET  /api/v1/documents/jobs
GET  /api/v1/documents/jobs/{job_id}
POST /api/v1/documents/jobs/{job_id}/retry
POST /api/v1/documents/jobs/{job_id}/archive
```

## Automated testing

The integration suite uses a temporary SQLite database and temporary upload directory, so tests do not modify the local production-style review queue.

Current coverage includes health, statistics, document retrieval, reviewer assignment, field approval and correction, completion rules, audit events, uploads, duplicate blocking, job tracking, retry and archive behavior, source-image and crop endpoints, and SQLite persistence.

Expected result:

```text
11 passed
```

## Screenshots

Add final screenshots under `docs/screenshots/` using these names:

```text
01-upload-document.png
02-processing-jobs.png
03-review-queue-overview.png
04-risk-aware-field-table.png
05-source-verification-crop.png
06-audit-history.png
```

Recommended README placement:

```markdown
![Processing jobs](docs/screenshots/02-processing-jobs.png)

![Review queue](docs/screenshots/03-review-queue-overview.png)

![Source verification](docs/screenshots/05-source-verification-crop.png)
```

## Safety and limitations

- This project is a portfolio system, not tax-preparation software.
- OCR output must not be treated as authoritative without verification.
- High-risk fields are intentionally routed to human review.
- The dataset is synthetic and does not represent every real-world IRS layout or scan condition.
- TrOCR is compute-intensive and the Docker worker is configured for CPU reliability.
- SQLite is appropriate for a local portfolio deployment, but a larger deployment should use PostgreSQL or another production database.
- Authentication, role-based access control, encryption, and enterprise observability would be required for real tax documents.
- SEC classification performance was measured on a limited project holdout and should not be generalized without broader evaluation.

## Future improvements

- PostgreSQL persistence
- Authentication and reviewer roles
- Object storage for uploaded documents
- Distributed task queue
- Model monitoring and drift detection
- Expanded IRS form coverage
- Improved image registration and layout normalization
- Better OCR ensembles
- Reviewer analytics and SLA tracking
- CI/CD deployment pipeline
- Cloud deployment with managed secrets and observability

## Project positioning

DocIntelAI demonstrates applied machine learning, OCR and multimodal document processing, model evaluation, human-in-the-loop system design, backend API development, workflow orchestration, database design, testing, Docker packaging, and responsible automation.
