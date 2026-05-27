# DocIntelAI Docker Runbook

## Prerequisites

The project root must contain:

- `requirements.txt`
- `src/`
- `data/processed/irs_review_queue.db`
- `models/irs_ocr_router.joblib`
- the trained field-confidence model used by the worker

The Compose file mounts `data/` and `models/` from the host. It does not copy private uploads or model binaries into the Docker image.

## Validate configuration

```powershell
docker --version
docker compose version
docker compose config
```

## Build

```powershell
docker compose build
```

## Start

```powershell
docker compose up
```

Run in the background:

```powershell
docker compose up -d
```

## Open

- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Logs

```powershell
docker compose logs -f api
docker compose logs -f worker
docker compose logs -f dashboard
```

## Stop

```powershell
docker compose down
```

Keep the Hugging Face cache:

```powershell
docker compose down
```

Remove the Hugging Face cache as well:

```powershell
docker compose down -v
```

## Notes

- The worker is intentionally CPU-only in Docker for predictable behavior.
- The first TrOCR run may download and cache the Hugging Face checkpoint.
- Stop locally running Uvicorn, Streamlit, and worker processes before starting Compose to avoid port conflicts.
- Uploaded files and the SQLite database remain under the host `data/` directory.
