# DocIntelAI Release Checklist

## Source control cleanup

- [ ] Remove nested extraction folders such as `docintelai_release_pack/`
- [ ] Remove `docintelai_docker_pack/`
- [ ] Move backup Python files outside tracked source folders
- [ ] Ensure no backup test file remains under `tests/`
- [ ] Confirm `.venv/` is ignored
- [ ] Confirm local SQLite databases are ignored
- [ ] Confirm uploaded documents are ignored
- [ ] Confirm generated IRS images are ignored
- [ ] Confirm model binaries are intentionally handled
- [ ] Confirm no `.env` or secrets are tracked

## Tests

- [ ] Run `powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1`
- [ ] Confirm `11 passed`
- [ ] Confirm no test modifies the real review database

## Local application

- [ ] Start with `scripts\start_all.ps1 -WorkerMode cpu`
- [ ] Confirm FastAPI health
- [ ] Confirm Streamlit loads
- [ ] Confirm worker polls
- [ ] Confirm upload, process, and review flow

## Docker

- [ ] Run `docker compose config`
- [ ] Run `docker compose build`
- [ ] Run `docker compose up -d`
- [ ] Confirm API is healthy
- [ ] Confirm worker is running
- [ ] Confirm dashboard is running
- [ ] Confirm dashboard uses `http://api:8000`
- [ ] Confirm host API uses `http://localhost:8001`
- [ ] Complete one Docker end-to-end upload

## Documentation

- [ ] Add final `README.md`
- [ ] Add `docs/architecture.md`
- [ ] Add `docs/demo_script.md`
- [ ] Add screenshots
- [ ] Verify all relative links
- [ ] Verify reported metrics
- [ ] Keep limited-holdout caveat for SEC classification
- [ ] Keep human-in-the-loop limitation for IRS processing

## Screenshot names

- [ ] `01-upload-document.png`
- [ ] `02-processing-jobs.png`
- [ ] `03-review-queue-overview.png`
- [ ] `04-risk-aware-field-table.png`
- [ ] `05-source-verification-crop.png`
- [ ] `06-audit-history.png`

## Git review

```powershell
git status
git ls-files
git diff --stat
```

Check for tracked files larger than 10 MB:

```powershell
git ls-files | ForEach-Object {
    if (Test-Path $_) {
        $item = Get-Item $_
        if ($item.Length -gt 10MB) {
            [PSCustomObject]@{
                SizeMB = [math]::Round($item.Length / 1MB, 2)
                Path = $_
            }
        }
    }
} | Sort-Object SizeMB -Descending
```

## Final commit

```powershell
git add .
git commit -m "Complete DocIntelAI document intelligence platform"
git push
```

## Portfolio publishing

- [ ] Add the GitHub repository URL to the resume
- [ ] Add a one-sentence project summary
- [ ] Add two strong resume bullets
- [ ] Record a two-minute demo
- [ ] Add the project to the LinkedIn Featured section
