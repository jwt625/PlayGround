"""settex.py name=key:value ...: set one key in config/model/chassis.toml [tex.<name>] (A/B helper).
Example: settex.py floor_rear=suffix:_s2 floor_mid=skip:true   (value '-' removes the key)."""
import re
import sys
from pathlib import Path

p = Path(__file__).resolve().parents[4] / "config" / "model" / "chassis.toml"
s = p.read_text()
for arg in sys.argv[1:]:
    nm, kv = arg.split("=", 1)
    k, v = kv.split(":", 1)
    m = re.search(rf"^\[tex\.{re.escape(nm)}\][^\n]*\n((?:(?!\[)[^\n]*\n)*)", s, re.M)
    assert m, nm
    body = re.sub(rf"^{k} = [^\n]*\n", "", m.group(1), flags=re.M)
    if v != "-":
        val = v if v in ("true", "false") or re.fullmatch(r"-?[\d.]+", v) or v.startswith("[") else f'"{v}"'
        body = f"{k} = {val}\n" + body
    s = s[: m.start(1)] + body + s[m.end(1):]
p.write_text(s)
