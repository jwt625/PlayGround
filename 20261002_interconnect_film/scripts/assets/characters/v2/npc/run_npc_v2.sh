#!/bin/bash
# Build the NPC v2 family, verify against v1, render previews and asm tests. Run from the project root
# (20261002_interconnect_film). One Blender process at a time. PV_TMP = scratch dir for temp tiles (set it to your
# session scratchpad).
set -e
BL=/Applications/Blender.app/Contents/MacOS/Blender
S=scripts/assets/characters/v2
N=$S/npc
O=assets/components/characters
PV=$O/previews_v2/npc
export PV_TMP=${PV_TMP:-$PV/_tmp}
mkdir -p $PV
for v in npc npc_molexx npc_nubiss npc_terahop npc_ayarr npc_nvydia npc_openay; do
  $BL -b --python $N/build_npc_v2.py -- $O $v
  rm -f $O/${v}_v2.blend1
  $BL -b --python $S/verify_v2.py -- $v $PV/verify_${v}_v2.json
done
for m in views lineup compare asm; do
  $BL -b --python $N/preview_npc_v2.py -- $PV $m
done
$BL -b $O/npc_nubiss_v2.blend --python $S/render_previews.py -- npc_nubiss_v2 $PV faces
[ "$PV_TMP" = "$PV/_tmp" ] && rm -rf "$PV/_tmp" || true
