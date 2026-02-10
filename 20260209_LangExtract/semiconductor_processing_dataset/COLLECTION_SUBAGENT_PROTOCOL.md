# Collection Subagent Protocol

This protocol is mandatory for all collection subagents.

## Objective

Collect publicly available documents and keep complete progress notes with machine-readable metadata.

## Required Logging

For every target document:
1. Write one event to `processed_documents/metadata/collection_attempts.jsonl`.
2. Upsert one state row in `processed_documents/metadata/manifest_documents.jsonl`.
3. Create or update `<document_id>.json` from `processed_documents/metadata/TEMPLATE.json` once a document is downloaded.

## Event Logging Rules

Every attempt event must include:
- `attempt_id`
- `timestamp` (UTC ISO 8601)
- `worker_id`
- `document_id`
- `action`
- `source_url`
- `result` (`attempted|succeeded|failed|blocked|skipped`)
- `error_type` (null when succeeded)
- `error_message` (null when succeeded)
- `output_path` (null when not downloaded)
- `notes`

## Manifest Update Rules

Manifest row must track latest state:
- `document_id`
- `status` (`discovered|attempted|succeeded|failed|blocked|skipped`)
- `last_attempt_at`
- `attempt_count`
- `last_error`
- `source_path` (once downloaded)
- `url`
- `priority`
- `quality_assessment` (if known)

## Failure and Retry Notes

When `result=failed` or `result=blocked`, `notes` must include:
- exact reason (e.g., 403, login wall, broken link),
- next retry strategy,
- whether manual intervention is needed.

## Success Notes

When `result=succeeded`, include:
- final saved path,
- checksum if available,
- quick quality note (e.g., has fabrication chapter, has supplementary).

## Parallel Work

- Subagents can run in parallel on different sources.
- Do not process the same `document_id` concurrently.
- Prefer source partitioning:
  - `collector_theses_*` for repository theses,
  - `collector_papers_*` for review/target papers,
  - `collector_supp_*` for supplementary files.
