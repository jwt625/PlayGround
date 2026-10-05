#!/bin/bash
# Milestone evaluation: holdout + probe renders and metrics, photometric comparison with 3DGS, error budget,
# model .blend, new-viewpoint renders. Usage: scripts/final_eval.sh <tag>   -> outputs/snapshots/<tag>/
set -e
cd "$(dirname "$0")/.."
TAG=$1
OUT=outputs/snapshots/$TAG
mkdir -p "$OUT"
unset VIRTUAL_ENV
for V in holdout probe; do
  scripts/iterate.sh ${TAG}_$V $V "" full > "$OUT/eval_$V.log" 2>&1
  cp outputs/runs/${TAG}_$V/eval/report.md "$OUT/eval_$V.md"
  cp outputs/runs/${TAG}_$V/eval/summary.json "$OUT/summary_$V.json"
  uv run python scripts/eval/compare_photometric.py outputs/runs/${TAG}_$V --out "$OUT/compare_$V" > "$OUT/compare_$V.json.txt" 2>/dev/null
  uv run python scripts/eval/error_budget.py outputs/runs/${TAG}_$V > "$OUT/error_budget_$V.json" 2>/dev/null
done
scripts/bslot.sh -b --factory-startup --python scripts/blender/save_model.py -- "$OUT/crt_model_$TAG.blend" > "$OUT/save.log" 2>&1
rm -f "$OUT/crt_model_$TAG.blend1"
scripts/bslot.sh -b --factory-startup --python scripts/blender/export_glb.py -- "$OUT/glb" --with-mat > "$OUT/glb.log" 2>&1
scripts/bslot.sh -b --factory-startup --python scripts/blender/render_orbit.py -- --out "$OUT/orbit" > "$OUT/orbit.log" 2>&1
uv run python - "$OUT" <<'PY'
import sys, cv2, numpy as np
from pathlib import Path
o = Path(sys.argv[1]) / "orbit"
fs = sorted(o.glob("*.png"))
ims = [cv2.resize(cv2.imread(str(f)), (480, 360)) for f in fs]
for im, f in zip(ims, fs):
    cv2.putText(im, f.stem, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
rows = [np.hstack(ims[i:i + 4] + [np.zeros_like(ims[0])] * (4 - len(ims[i:i + 4]))) for i in range(0, len(ims), 4)]
cv2.imwrite(str(Path(sys.argv[1]) / "orbit_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
PY
echo "snapshot: $OUT"; cat "$OUT/compare_holdout.json.txt"
