#!/bin/sh
# Rebuild the feedback-course widget files (head + shared helpers + body) into course/widgets/.
S=$(cd "$(dirname "$0")" && pwd); W="$S/../../../widgets"
cat "$S/fp_head.js" "$S/helpers.js" "$S/fp_body.js" > "$W/fabry_perot.js"
cat "$S/mh_head.js" "$S/helpers.js" "$S/mh_body.js" > "$W/mode_hop.js"
cat "$S/db_head.js" "$S/helpers.js" "$S/db_body.js" > "$W/doublet.js"
for f in fabry_perot mode_hop doublet; do node -e "require('vm').createScript(require('fs').readFileSync('$W/$f.js','utf8'))" || echo "SYNTAX $f"; done
