#!/usr/bin/env bash
# Headless screenshot of index.html at a route. Usage: pipeline/shoot.sh '<hash>' out.png [WxH]
cd "$(dirname "$0")/.."
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars \
  --window-size="${3:-1600,1000}" --virtual-time-budget=4000 --screenshot="$2" "file://$PWD/index.html$1" 2>/dev/null
