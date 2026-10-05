#!/bin/bash
# One modeling iteration: build all groups, render, evaluate, print the report.
# Usage: scripts/iterate.sh <run_name> [views=probe] [parts_prefix] [mode=geom|full] [passes=id,rgb,pts]
# Output: outputs/runs/<run_name>/eval/{report.md,regions.png,views.png,parts.json,orphans.json}
set -e
cd "$(dirname "$0")/.."
RUN=$1; VIEWS=${2:-probe}; PARTS=${3:-}; MODE=${4:-geom}; PASSES=${5:-id,rgb,pts}
unset VIRTUAL_ENV
LOG=outputs/runs/$RUN/render_$(date +%Y%m%d_%H%M%S).log
mkdir -p outputs/runs/$RUN
scripts/bslot.sh -b --factory-startup --python scripts/blender/render_views.py -- \
  --out outputs/runs/$RUN --build scripts/model/build_all.py --views "$VIEWS" --passes "$PASSES" > "$LOG" 2>&1 || true
grep -E "BUILD_RESULT|BUILD_FAIL|BUILD_LKG|RENDER_DONE" "$LOG" || { echo "render failed, see $LOG"; tail -30 "$LOG"; exit 1; }
grep -A15 "BUILD_FAIL" "$LOG" | head -40 || true
uv run python scripts/eval/evaluate.py outputs/runs/$RUN --mode "$MODE" ${PARTS:+--parts "$PARTS"} > /dev/null 2>&1
cat outputs/runs/$RUN/eval/report.md
