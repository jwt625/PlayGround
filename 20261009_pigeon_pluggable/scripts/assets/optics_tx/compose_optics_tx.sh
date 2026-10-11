#!/bin/bash
# Compose the final TX optics renders (labels, lane tags, beam note) from the raw renders in outputs/assets/optics_tx/work.
# Run from the project folder: bash scripts/assets/optics_tx/compose_optics_tx.sh
set -e
W=outputs/assets/optics_tx/work
O=outputs/assets/optics_tx
N="* beams invisible (1311 nm); rendered for viewing"
R() { uv run --with pillow python scripts/assets/optics_tx/label_overlay_tx.py "$@"; }
R $W/optics_tx_overview_raw.png $O/optics_tx_overview.png
R $W/optics_tx_exploded_raw.png $O/optics_tx_exploded_labeled.png --labels $W/optics_tx_exploded_labels.json --crop
R $W/optics_tx_pic_top_raw.png $O/optics_tx_pic_top.png
R $W/optics_tx_pic_zoom_mzm_raw.png $O/optics_tx_pic_zoom_mzm.png
R $W/optics_tx_pic_bonds_raw.png $O/optics_tx_pic_bonds.png
R $W/optics_tx_laser_coupling_raw.png $O/optics_tx_laser_coupling.png --note "$N"
R $W/optics_tx_mapping_1x8_2x4_raw.png $O/optics_tx_mapping_1x8_2x4.png --tags $W/optics_tx_mapping_1x8_2x4_tags.json --note "$N"
R $W/optics_tx_mapping_oblique_raw.png $O/optics_tx_mapping_oblique.png --tags $W/optics_tx_mapping_oblique_tags.json --note "$N"
