"""cmpc.py RUN_A RUN_B [substr]: chassis parts color residual / SSIM / fp / edge, A -> B."""
import json, sys
def L(r): return {p['part']: p for p in json.load(open(f'outputs/runs/{r}/eval/parts.json')) if p['part'].startswith('chassis.')}
a, b = L(sys.argv[1]), L(sys.argv[2]); sub = sys.argv[3] if len(sys.argv) > 3 else ''
f = lambda p, k, d: (f"{p[k]:.{d}f}" if p.get(k) is not None else '-')
for n in sorted(set(a) | set(b)):
    if sub not in n: continue
    pa, pb = a.get(n, {}), b.get(n, {})
    print(f"{n:30s} res {f(pa,'color_res',1):>5} -> {f(pb,'color_res',1):>5}  ssim {f(pa,'ssim',3)} -> {f(pb,'ssim',3)}  fp {f(pa,'fp_frac',2)} -> {f(pb,'fp_frac',2)}  edge {f(pa,'edge_mean',1)} -> {f(pb,'edge_mean',1)}")
