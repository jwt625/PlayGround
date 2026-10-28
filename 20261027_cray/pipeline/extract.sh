#!/usr/bin/env bash
# Stage 1: page images + OCR for one patent.
# Usage: pipeline/extract.sh <patent-number>   (reads sources/<pn>.pdf)
# Writes work/<pn>/pages/p-NNN.png (native 1-bit scans), work/<pn>/ocr/p-NNN.{txt,tsv}
set -euo pipefail
cd "$(dirname "$0")/.."
pn="$1"
out="work/$pn"
mkdir -p "$out/pages" "$out/ocr"
pdfimages -png "sources/$pn.pdf" "$out/pages/p"
for f in "$out"/pages/p-*.png; do
  b=$(basename "$f" .png)
  tesseract "$f" "$out/ocr/$b" --psm 1 txt tsv >/dev/null 2>&1
done
echo "$pn: $(ls "$out"/pages | wc -l | tr -d ' ') pages"
