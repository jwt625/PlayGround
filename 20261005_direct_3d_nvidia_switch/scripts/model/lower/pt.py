"""Per-part table across runs: edge px, pts median mm, color residual, SSIM.
uv run python scripts/model/lower/pt.py run1,run2 [prefix]"""
import json
import sys

runs = sys.argv[1].split(",")
flt = sys.argv[2] if len(sys.argv) > 2 else "lower."
D = [{x["part"]: x for x in json.load(open(f"outputs/runs/{r}/eval/parts.json"))} for r in runs]
names = sorted(n for n in set().union(*D) if n.startswith(flt))
print("part".ljust(22), " | ".join(f"{r}: e pm cres ssim" for r in runs))
for n in names:
    c = []
    for d in D:
        x = d.get(n)
        c.append("-" if not x else f"{x['edge_mean']:.2f} {x.get('pts_median_mm', float('nan')):.2f} "
                                   f"{x['color_res']:.1f} {x['ssim']:.3f}")
    print(n.ljust(22), " | ".join(c))
