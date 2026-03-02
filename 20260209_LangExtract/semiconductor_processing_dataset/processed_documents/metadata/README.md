# Metadata Tracking

This directory stores per-document metadata JSON files and collection progress logs.

## Files

- `TEMPLATE.json`
  - Copy this template to create `<document_id>.json`.
- `manifest_documents.jsonl`
  - Canonical state table (one line per `document_id`).
  - Each line is a JSON object.
- `collection_attempts.jsonl`
  - Append-only attempt log.
  - Each line is a JSON object representing one attempt event.

## Manifest Record Schema (`manifest_documents.jsonl`)

Each record should include at least:
```json
{
  "document_id": "geerlings_kyle_yale_2013",
  "source_type": "phd_thesis",
  "title": "Improving Coherence of Superconducting Qubits and Resonators",
  "institution": "Yale University",
  "year": 2013,
  "url": "https://...",
  "source_path": "semiconductor_processing_dataset/raw_documents/theses/yale/geerlings_kyle_yale_2013.pdf",
  "status": "succeeded",
  "last_attempt_at": "2026-02-10T12:34:56Z",
  "attempt_count": 1,
  "last_error": null,
  "quality_assessment": "high_value",
  "priority": "high",
  "notes": "Downloaded from Yale repository"
}
```

Allowed `status` values:
- `discovered`
- `attempted`
- `succeeded`
- `failed`
- `blocked`
- `skipped`

## Attempt Record Schema (`collection_attempts.jsonl`)

Each attempt event should include:
```json
{
  "attempt_id": "2026-02-10T12:34:56Z_geerlings_kyle_yale_2013",
  "timestamp": "2026-02-10T12:34:56Z",
  "worker_id": "collector_eth_01",
  "document_id": "geerlings_kyle_yale_2013",
  "action": "download_pdf",
  "source_url": "https://...",
  "result": "succeeded",
  "http_status": 200,
  "error_type": null,
  "error_message": null,
  "output_path": "semiconductor_processing_dataset/raw_documents/theses/yale/geerlings_kyle_yale_2013.pdf",
  "sha256": "optional_sha256_hash",
  "duration_sec": 3.1,
  "notes": "Public repository direct PDF"
}
```

Allowed `result` values:
- `attempted`
- `succeeded`
- `failed`
- `blocked`
- `skipped`

## Update Rules

1. Always append an event to `collection_attempts.jsonl` for each attempted action.
2. Always upsert the corresponding row in `manifest_documents.jsonl` after each event.
3. Never delete attempt rows.
4. Keep `document_id` consistent across filename, metadata JSON, manifest, and attempts log.
