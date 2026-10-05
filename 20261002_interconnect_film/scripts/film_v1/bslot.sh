#!/bin/bash
# Run Blender with a machine-wide cap on concurrent Blender processes (default 3, env BLENDER_SLOTS).
# Usage: scripts/film_v1/bslot.sh <blender args...>   e.g. scripts/film_v1/bslot.sh -b --python x.py -- out.blend
# Slots are lock directories under ${BLENDER_SLOT_DIR:-$TMPDIR/blender_slots}; a slot whose owner pid is gone is reclaimed.
B=/Applications/Blender.app/Contents/MacOS/Blender
N=${BLENDER_SLOTS:-3}
D=${BLENDER_SLOT_DIR:-${TMPDIR:-/tmp}/blender_slots}
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
  sleep 3
done
