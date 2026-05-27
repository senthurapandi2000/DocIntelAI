# DocIntelAI Demo Script

## Two-minute walkthrough

### Introduction

“DocIntelAI is an end-to-end document intelligence platform for SEC filings and IRS forms. This demo focuses on the IRS human-in-the-loop workflow for W-2 and 1099-NEC documents.”

### Upload

Open **Upload Document**.

“This interface accepts PDF, PNG, and JPG files. Uploads are hashed to prevent unnecessary duplicate processing, and users can optionally specify the expected form type.”

### Processing jobs

Open **Processing Jobs**.

“The upload creates an asynchronous processing job. A separate worker normalizes the file, classifies the form, creates coordinate-based crops, runs Tesseract, optionally falls back to TrOCR through a learned router, and applies field-confidence and validation logic.”

Show `queued → processing → completed`.

### Review queue

Open **Review Queue**.

“The system does not automatically trust every OCR result. Low-confidence, invalid, inconsistent, or critical fields are routed here.”

Show the review-field count, exception count, auto-accepted-field count, priority, and reviewer assignment.

### Field verification

Select a field.

“The reviewer sees the original document with the selected region highlighted and the exact crop used by OCR. They can approve a correct value or enter a corrected value.”

### Audit history

Open **Audit history**.

“Every assignment, approval, correction, and completion action is retained as an audit event.”

### Engineering summary

“The application is separated into FastAPI, Streamlit, a background worker, SQLite persistence, and model artifacts. It includes automated integration tests, retry and archive behavior, duplicate protection, and Docker Compose deployment.”

### Results

“On the untouched IRS test set, the routed OCR pipeline reached 49.68% exact match and 81.28 similarity. The review policy captured 99.83% of incorrect critical fields while maintaining 96.97% exact accuracy among accepted critical fields.”

### Closing

“This project demonstrates applied OCR, model routing, confidence estimation, responsible automation, backend engineering, testing, and containerized deployment.”
