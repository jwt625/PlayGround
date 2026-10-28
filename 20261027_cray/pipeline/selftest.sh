#!/usr/bin/env bash
# Run the in-page self-test headless and print its report.
cd "$(dirname "$0")/.."
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --window-size=1600,1000 \
  --virtual-time-budget=60000 --dump-dom "file://$PWD/index.html?selftest#/" 2>/dev/null \
  | python3 -I -c "import sys,re,html; m=list(re.finditer(r'<pre id=\"selftest\">(.*?)</pre>', sys.stdin.read(), re.S)); print(html.unescape(m[-1].group(1)) if m else 'NO REPORT (page did not finish)')"
