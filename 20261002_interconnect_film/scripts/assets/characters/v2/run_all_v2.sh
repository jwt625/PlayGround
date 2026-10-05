#!/bin/bash
# Build gary_v2, gary_v2_holes30, manager_v2 and render all v2 previews. Run from the project root (20261002_interconnect_film).
# One Blender process at a time. PV_TMP = scratch directory for tile renders (default /tmp; set it to your session scratchpad).
set -e
BL=/Applications/Blender.app/Contents/MacOS/Blender
S=scripts/assets/characters/v2
O=assets/components/characters
PV=$O/previews_v2
export PV_TMP=${PV_TMP:-/tmp}
mkdir -p $PV
$BL -b --python $S/build_gary_v2.py -- $O
$BL -b --python $S/build_manager_v2.py -- $O
rm -f $O/gary_v2.blend1 $O/gary_v2_holes30.blend1 $O/manager_v2.blend1
$BL -b $O/gary_v2.blend --python $S/render_previews.py -- gary_v2 $PV views,faces,holes,hand,squash,poseboard
$BL -b $O/gary_v2_holes30.blend --python $S/render_previews.py -- gary_v2_holes30 $PV holes
$BL -b $O/manager_v2.blend --python $S/render_previews.py -- manager_v2 $PV views,faces,hand,squash,poseboard,anger
$BL -b --python $S/test_v2_asm.py -- $PV gary compare
$BL -b --python $S/test_v2_asm.py -- $PV manager compare
$BL -b --python $S/test_v2_asm.py -- $PV gary scene
$BL -b --python $S/test_v2_asm.py -- $PV manager scene
$BL -b --python $S/verify_v2.py -- gary $PV/verify_gary_v2.json
$BL -b --python $S/verify_v2.py -- manager $PV/verify_manager_v2.json
echo ALL_DONE > $PV_TMP/run_all_v2.done
