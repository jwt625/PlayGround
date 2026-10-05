#!/bin/bash
# Rebuild only manager_v2 and its previews (same steps as run_all_v2.sh). Run from the project root.
set -e
BL=/Applications/Blender.app/Contents/MacOS/Blender
S=scripts/assets/characters/v2
O=assets/components/characters
PV=$O/previews_v2
export PV_TMP=${PV_TMP:-/tmp}
$BL -b --python $S/build_manager_v2.py -- $O
rm -f $O/manager_v2.blend1
$BL -b $O/manager_v2.blend --python $S/render_previews.py -- manager_v2 $PV views,faces,hand,squash,poseboard,anger
$BL -b --python $S/test_v2_asm.py -- $PV manager compare
$BL -b --python $S/test_v2_asm.py -- $PV manager scene
$BL -b --python $S/verify_v2.py -- manager $PV/verify_manager_v2.json
python3 $S/finalize_meta_v2.py
echo ALL_DONE > $PV_TMP/run_manager_v2.done
