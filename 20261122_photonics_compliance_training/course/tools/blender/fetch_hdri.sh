#!/bin/bash
# Fetch the two CC0 Poly Haven HDRIs (2k) used by the render scripts into the cache directory
# read by common.py (BLENDER_HDRI_DIR, default ~/.cache/photonics_compliance_hdri). Skips existing files.
D=${BLENDER_HDRI_DIR:-$HOME/.cache/photonics_compliance_hdri}
mkdir -p "$D"
for n in studio_small_09 hospital_room; do
  f="$D/${n}_2k.hdr"
  [ -s "$f" ] && continue
  curl -sL -A "generic-client" -o "$f" "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/${n}_2k.hdr"
  sleep 3
done
ls -la "$D"
