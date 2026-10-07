import json, sys
runs = sys.argv[1:]
D = [{p['part']: p for p in json.load(open(f'outputs/runs/{r}/eval/parts.json')) if p['part'].startswith('chassis.')} for r in runs]
names = sorted(set().union(*D))
print('part'.ljust(26) + ''.join(f'{r[-3:]:>24}' for r in runs) + '   (fp/edge/pts_med)')
for n in names:
    row = n.ljust(26)
    for d in D:
        p = d.get(n)
        row += f"{(f'{p['fp_frac']:.2f}/{p['edge_mean']:.1f}/' + (f'{p['pts_median_mm']:.1f}' if p.get('pts_median_mm') is not None else '-')) if p else '-':>24}"
    print(row)
