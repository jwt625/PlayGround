"""Generate deterministic renderer inputs and inventory collected FlyBody sources.
Run: python3 scripts/prepare_render_assets.py
No third-party Python dependencies. Does not convert meshes or simulate optics.
"""
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/generated'
OUT.mkdir(parents=True, exist_ok=True)
def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')

channels = []
for r in range(2, -3, -1):
    for q in range(-2, 3):
        if max(abs(q), abs(r), abs(q+r)) <= 2:
            channels.append(dict(id=f'CH{len(channels)+1:02}', q=q, r=r,
                                 position_pitch_units=[q+r/2, math.sqrt(3)*r/2, 0]))
assert len(channels) == 19
save('channel-layout.json', dict(coordinates='XY aperture; forward +Z; positions multiplied by physical pitch', channels=channels))
save('render-style.json', dict(colors=dict(background='#101925', panel='#203044', optical='#45ded0', electrical='#ffba67', observation='#bca7ff', label='#e5eef8'),
     pathStyles=dict(optical='solid', electrical='dashed', observation='dotted'),
     materials=dict(housing=dict(metalness=0.75, roughness=0.32), bench=dict(metalness=0.65, roughness=0.48), lens=dict(transmission=0.85, roughness=0.05)),
     notes='Art defaults only. Phase and measured intensity require separate calibrated legends.'))
base = ROOT / 'assets/vendor/flybody'
files = []
for path in sorted(base.rglob('*')):
    if not path.is_file():
        continue
    data = path.read_bytes()
    item = dict(path=str(path.relative_to(ROOT)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    if path.suffix == '.obj':
        vertices = faces = triangles = 0
        for line in data.decode().splitlines():
            if line.startswith('v '): vertices += 1
            if line.startswith('f '):
                faces += 1
                triangles += len(line.split()) - 3
        item.update(vertices=vertices, faces=faces, triangles_if_fan_triangulated=triangles)
    files.append(item)
xml = ET.parse(base / 'assets/fruitfly.xml').getroot()
missing = []
for el in xml.iter():
    if el.get('file') and not (base / 'assets' / el.get('file')).exists():
        missing.append(el.get('file'))
assert not missing, missing
save('flybody-inventory.json', dict(upstream='https://github.com/TuragaLab/flybody', revision='d015e9bfe441bd90ae431bac24c55cb74bdbce26',
     status='source collected; browser conversion and visual validation pending', files=files,
     summary=dict(files=len(files), bytes=sum(f['bytes'] for f in files), meshes=sum('vertices' in f for f in files),
                  vertices=sum(f.get('vertices',0) for f in files), triangles_if_fan_triangulated=sum(f.get('triangles_if_fan_triangulated',0) for f in files),
                  bodies=len(xml.findall('.//body')), joints=len(xml.findall('.//joint')), missing_direct_asset_references=missing)))
# Preserve body transforms and joints for conversion planning; not an evaluated MuJoCo scene.
save('flybody-hierarchy.json', dict(note='Raw MJCF attributes; defaults, compiler transforms and mesh alignment still require evaluation.',
    bodies=[dict(attributes=b.attrib, joints=[j.attrib for j in b.findall('joint')], geoms=[g.attrib for g in b.findall('geom')]) for b in xml.findall('.//body')]))
print(json.dumps(json.loads((OUT/'flybody-inventory.json').read_text())['summary'], indent=2))
