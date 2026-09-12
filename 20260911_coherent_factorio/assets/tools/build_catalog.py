"""Inspect unchanged PNG originals and rebuild catalog/manifest. Requires Pillow."""
from pathlib import Path
from PIL import Image
import json, html
ROOT=Path(__file__).resolve().parents[1]
specs=[
('field-emitter','Field emitter','defense',[3,3],['power','reference','target']),
('armored-frontier-organism','Armored frontier organism','ecology',[4,4],['target-response']),
('starter-extractor','Starter extractor','production',[3,3],['deposit','power','material-output']),
('compact-assembler','Compact assembler','production',[3,3],['power','material-input','material-output']),
('power-unit','Power unit','power',[3,2],['electrical-output']),
('reference-control-station','Reference / control station','instrumentation',[2,2],['power','reference-output','monitor-data','commissioning-ui']),
('four-port-junction','Four-port junction','field-network',[1,1],['field-port-a','field-port-b','field-port-c','field-port-d']),
('phase-tuner','Phase tuner','field-network',[2,1],['field-input','field-output','control-setting']),
('cooled-dump','Cooled dump','protection',[2,2],['rejected-field-input','thermal-state']),
('starter-ore','Starter ore','resource',[3,3],['extraction','depletion']),
('frontier-crystal-ore','Frontier crystal ore','resource',[3,3],['frontier-access','extraction','depletion'])]
assets=[]; cards=[]
for ident,title,family,footprint,deps in specs:
    version='v2' if ident=='frontier-crystal-ore' and (ROOT/'sprites/frontier-crystal-ore-v2.png').exists() else 'v1'
    rel=f'sprites/{ident}-{version}.png'; path=ROOT/rel
    if not path.exists(): continue
    im=Image.open(path); alpha=im.getchannel('A') if 'A' in im.getbands() else None
    entry=dict(id=ident,title=title,family=family,path=rel,dimensions_px=list(im.size),mode=im.mode,alpha_extrema=list(alpha.getextrema()) if alpha else None,alpha_bbox=list(alpha.getbbox()) if alpha and alpha.getbbox() else None,proposed_footprint_tiles=footprint,state='intact-static',view_count=1,anchor_status='uncalibrated',dependencies=deps)
    assets.append(entry)
    cards.append(f'<figure><div class="stage"><img src="{rel}" alt="{html.escape(title)}"></div><figcaption><strong>{html.escape(title)}</strong><br><small>{family} · suggested {footprint[0]} × {footprint[1]} tiles</small></figcaption><div class="thumbs"><img src="{rel}" alt="48 pixel preview" width="48" height="48"><img src="{rel}" alt="96 pixel preview" width="96" height="96"></div></figure>')
manifest=dict(schema_version=1,status='prototype-static-art',notes='Footprints are provisional. Anchors, ports, common camera and damage states require renderer validation. Dependencies describe integration, not recipes.',assets=assets,ui=dict(preview='ui/index.html',stylesheet='ui/kit.css',icons=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'ui/icons').glob('*.svg'))]))
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fieldworks asset catalog</title><link rel="stylesheet" href="ui/kit.css"><main><h1>Fieldworks · First outpost assets</h1><p>Static artwork, full cards and 48/96 px readability previews. Grid spacing, footprints and relative display scale are provisional. <a href="ui/index.html">UI kit</a> · <a href="../DevLog/002-asset-plan.md">Asset plan</a> · <a href="manifest.json">Manifest</a></p><div class="asset-grid">'''+''.join(cards)+'</div></main></html>\n')
print(f'Catalog contains {len(assets)} raster assets and {len(manifest["ui"]["icons"])} UI icons.')
