#!/usr/bin/env bash
# Headless screenshot at a route. Usage: pipeline/shoot.sh '<hash>' out.png [WxH]
# '#/' (or empty) shoots the gallery index.html; '#/<pn>/...' shoots p/<pn>.html.
cd "$(dirname "$0")/.."
pn=$(printf '%s' "$1" | sed -nE 's@^#/([0-9]+).*@\1@p'); f=index.html; [ -n "$pn" ] && f="p/$pn.html"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars \
  --window-size="${3:-1600,1000}" --virtual-time-budget=4000 --screenshot="$2" "file://$PWD/$f$1" 2>/dev/null
