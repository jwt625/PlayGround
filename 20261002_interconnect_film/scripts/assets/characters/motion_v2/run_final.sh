#!/bin/bash
# Bake all rigs, review sheets (gary: all base actions, manager and npc: subsets), gun muzzle test. One Blender process at a time.
set -e
PROJ="$(cd "$(dirname "$0")/../../../.." && pwd)"
B=/Applications/Blender.app/Contents/MacOS/Blender
CH="$PROJ/assets/components/characters"
M2="$PROJ/scripts/assets/characters/motion_v2"
PV="$CH/previews_motion_v2"
mkdir -p "$PV"
export M2_TMP="${M2_TMP:-/tmp}/m2_tiles"
for r in gary manager npc; do
  $B -b "$CH/$r.blend" --python "$M2/motion_v2.py" -- bake all
done
$B -b "$CH/gary.blend" --python "$M2/m2_review.py" -- base "$PV"
$B -b "$CH/manager.blend" --python "$M2/m2_review.py" -- manager "$PV"
$B -b "$CH/npc.blend" --python "$M2/m2_review.py" -- npc "$PV"
$B -b "$CH/manager.blend" --python "$M2/m2_guntest.py" -- "$PV"
