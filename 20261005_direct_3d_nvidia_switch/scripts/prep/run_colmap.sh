#!/bin/zsh
# Own SfM on the source photos with the existing Homebrew COLMAP (no CUDA; CPU SIFT).
# One shared SIMPLE_RADIAL camera (single iPhone lens), raw pixel order (COLMAP ignores EXIF orientation).
# Usage: scripts/prep/run_colmap.sh [threads]   (default 8 of 12 cores, so the machine stays usable)
set -euo pipefail
cd "$(dirname "$0")/../.."
T=${1:-8}
DB=data/colmap/database.db
IMG=source/images
OUT=data/colmap/sparse
LOG=outputs/logs/colmap_$(date +%Y%m%d_%H%M%S).log
mkdir -p data/colmap $OUT outputs/logs
if [[ -e $DB ]]; then echo "exists: $DB (move it away to rerun)"; exit 1; fi
{
  echo "start $(date -Iseconds) threads=$T"
  colmap feature_extractor --database_path $DB --image_path $IMG \
    --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL \
    --SiftExtraction.use_gpu 0 --SiftExtraction.num_threads $T --SiftExtraction.max_image_size 3200
  echo "extract done $(date -Iseconds)"
  colmap exhaustive_matcher --database_path $DB \
    --SiftMatching.use_gpu 0 --SiftMatching.num_threads $T --SiftMatching.guided_matching 1
  echo "match done $(date -Iseconds)"
  colmap mapper --database_path $DB --image_path $IMG --output_path $OUT --Mapper.num_threads $T
  echo "map done $(date -Iseconds)"
  for d in $OUT/*/; do colmap model_analyzer --path $d; done
  echo "end $(date -Iseconds)"
} > $LOG 2>&1
echo "log: $LOG"
