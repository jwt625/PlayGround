#!/bin/bash
# Run Blender under a cap on concurrent Blender processes for this project (default 2, env BLENDER_SLOTS).
# Usage: scripts/bslot.sh <blender args...>   e.g. scripts/bslot.sh -b --python x.py -- args
# Slots are lock dirs under ${BLENDER_SLOT_DIR:-$TMPDIR/blender_slots_d3d}; a slot whose owner pid is gone is reclaimed.
B=/Applications/Blender.app/Contents/MacOS/Blender
N=${BLENDER_SLOTS:-2}
D=${BLENDER_SLOT_DIR:-${TMPDIR:-/tmp}/blender_slots_d3d}
mkdir -p "$D"
while true; do
  for i in $(seq 0 $((N - 1))); do
    s="$D/slot$i"
    if mkdir "$s" 2>/dev/null; then
      echo $$ > "$s/pid"
      trap 'rm -rf "$s"' EXIT INT TERM
      "$B" "$@"
      exit $?
    fi
    p=$(cat "$s/pid" 2>/dev/null)
    if [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null; then rm -rf "$s"; fi
  done
  sleep 2
done
