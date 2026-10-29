#!/usr/bin/env bash
# Run the in-page self-test headless on the gallery and every patent page; print each report.
# Usage: pipeline/selftest.sh [<pn> ...]   (default: index.html + all p/*.html)
cd "$(dirname "$0")/.."
pages=("$@"); [ ${#pages[@]} -eq 0 ] && pages=(index $(ls p/*.html 2>/dev/null | sed 's#p/##; s#\.html##'))
for pg in "${pages[@]}"; do
  f="p/$pg.html"; [ "$pg" = index ] && f=index.html
  echo "== $f"
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --window-size=1600,1000 \
    --virtual-time-budget=60000 --dump-dom "file://$PWD/$f?selftest#/" 2>/dev/null \
    | python3 -I -c "import sys,re,html; m=list(re.finditer(r'<pre id=\"selftest\">(.*?)</pre>', sys.stdin.read(), re.S)); print(html.unescape(m[-1].group(1)) if m else 'NO REPORT (page did not finish)')"
done
