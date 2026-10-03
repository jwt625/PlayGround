#!/bin/bash
# Build every character asset and render previews. Run from the project root (20261002_interconnect_film).
# One Blender process at a time. PV_TMP = scratch directory for tile renders.
set -e
BL=/Applications/Blender.app/Contents/MacOS/Blender
S=scripts/assets/characters
O=assets/components/characters
PV=$O/previews
export PV_TMP=${PV_TMP:-/tmp}
mkdir -p $PV
$BL -b --python $S/build_gary.py -- $O
$BL -b --python $S/build_manager.py -- $O
for v in npc npc_molexx npc_nubiss npc_terahop npc_ayarr npc_nvydia npc_openay; do
  $BL -b --python $S/build_npc.py -- $O $v
done
rm -f $O/*.blend1
# previews
$BL -b $O/gary.blend --python $S/render_previews.py -- gary $PV views,faces,holes,poses
$BL -b $O/gary_holes30.blend --python $S/render_previews.py -- gary_holes30 $PV holes
$BL -b $O/manager.blend --python $S/render_previews.py -- manager $PV views,faces,anger,poses
for v in npc_molexx npc_nubiss npc_terahop npc_ayarr npc_nvydia npc_openay; do
  $BL -b $O/$v.blend --python $S/render_previews.py -- $v $PV views,holes
done
$BL -b $O/npc_molexx.blend --python $S/render_previews.py -- npc_molexx $PV faces,poses shout_loop,punch_loop,shove,aim_gun,topple_back,walk
$BL -b $O/npc_openay.blend --python $S/render_previews.py -- npc_openay $PV faces,poses hug_leg,idle,walk
$BL -b $O/npc.blend --python $S/render_previews.py -- npc $PV views
rm -f $O/*.blend1
echo ALL_DONE > $PV_TMP/run_all.done
