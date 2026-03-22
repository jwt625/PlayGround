#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_PATH="${1:-/home/ubuntu/ofc_pdfs_20260321_full.zip}"
SHARD_DIR="$ROOT_DIR/tmp/ofc2026_shards"

mkdir -p "$SHARD_DIR"

CUDA_VISIBLE_DEVICES=0 "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/ofc2026_extract_marker.py" \
  --archive "$ARCHIVE_PATH" \
  --pdf-dir "$ROOT_DIR/data/ofc2026_pdfs" \
  --output-dir "$ROOT_DIR/extracted_text" \
  --index "$SHARD_DIR/paper_text_index.shard0.json" \
  --fail-log "$SHARD_DIR/extraction_failures.shard0.jsonl" \
  --num-shards 2 \
  --shard-index 0 &

PID0=$!

CUDA_VISIBLE_DEVICES=1 "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/ofc2026_extract_marker.py" \
  --archive "$ARCHIVE_PATH" \
  --pdf-dir "$ROOT_DIR/data/ofc2026_pdfs" \
  --output-dir "$ROOT_DIR/extracted_text" \
  --index "$SHARD_DIR/paper_text_index.shard1.json" \
  --fail-log "$SHARD_DIR/extraction_failures.shard1.jsonl" \
  --num-shards 2 \
  --shard-index 1 &

PID1=$!

wait "$PID0"
wait "$PID1"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/merge_ofc2026_indexes.py" \
  --index-glob "tmp/ofc2026_shards/paper_text_index.shard*.json" \
  --output-index "$ROOT_DIR/paper_text_index.json"

cat "$SHARD_DIR"/extraction_failures.shard*.jsonl > "$ROOT_DIR/extraction_failures.jsonl" 2>/dev/null || true

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/ofc2026_build_wordmap.py" \
  --index "$ROOT_DIR/paper_text_index.json" \
  --top-terms "$ROOT_DIR/corpus_top_terms.csv" \
  --top-bigrams "$ROOT_DIR/corpus_top_bigrams.csv" \
  --top-trigrams "$ROOT_DIR/corpus_top_trigrams.csv" \
  --wordmap-png "$ROOT_DIR/wordmap.png" \
  --wordmap-svg "$ROOT_DIR/wordmap.svg" \
  --tfidf-json "$ROOT_DIR/paper_tfidf_keywords.json"
