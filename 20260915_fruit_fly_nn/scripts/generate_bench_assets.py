"""Original reusable display prefabs. Not optical geometry or physical vendor CAD."""
from pathlib import Path
import json
import trimesh
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'assets/generated/bench';OUT.mkdir(parents=True,exist_ok=True)
colors={'metal':[42,58,76,255],'lens':[50,195,195,145],'electrical':[244,170,74,255],'optical':[50,210,190,255]}
def material(mesh,color):
 mesh.visual=trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=colors[color],metallicFactor=.6 if color=='metal' else 0,roughnessFactor=.32,alphaMode='BLEND' if color=='lens' else 'OPAQUE',doubleSided=True));return mesh
manifest=[]
for name in ['emitter','seed','splitter','phase-amplifier','console']:
 scene=trimesh.Scene();ports=[]
 if name=='emitter':
  scene.add_geometry(material(trimesh.creation.cylinder(radius=.028,height=.075,sections=48),'metal'),node_name='housing')
  lens=material(trimesh.creation.cylinder(radius=.021,height=.002,sections=48),'lens');lens.apply_translation([0,0,.039]);scene.add_geometry(lens,node_name='aperture_lens')
  scene.graph.update(frame_to='pointing_pivot',matrix=np.eye(4));scene.graph.update(frame_to='focus_anchor',matrix=trimesh.transformations.translation_matrix([0,0,.039]))
  ports=[dict(name='fiber_in',position=[0,0,-.04],kind='optical'),dict(name='actuator_command',position=[.028,0,0],kind='electrical')]
 else:
  size={'seed':[.18,.1,.1],'splitter':[.13,.09,.06],'phase-amplifier':[.12,.06,.045],'console':[.3,.18,.055]}[name]
  scene.add_geometry(material(trimesh.creation.box(size),'metal'),node_name=name+'_housing')
  ports=[dict(name='input',position=[-size[0]/2,0,0],kind='electrical' if name=='console' else 'optical'),dict(name='output',position=[size[0]/2,0,0],kind='electrical' if name=='console' else 'optical')]
  if name=='console':
   layout=json.loads((ROOT/'assets/generated/channel-layout.json').read_text())
   for c in layout['channels']:
    x,y,_=c['position_pitch_units'];cell=material(trimesh.creation.cylinder(radius=.008,height=.004,sections=24),'optical');cell.apply_translation([x*.037,y*.037,.03]);scene.add_geometry(cell,node_name=c['id']+'_phase_cell')
 for port in ports: scene.graph.update(frame_to='port_'+port['name'],matrix=trimesh.transformations.translation_matrix(port['position']))
 path=OUT/(name+'.glb');path.write_bytes(scene.export(file_type='glb'));rt=trimesh.load(path,force='scene');assert np.allclose(scene.bounds,rt.bounds,atol=1e-7)
 manifest.append(dict(name=name,file=path.name,bounds=scene.bounds.tolist(),ports=ports,bytes=path.stat().st_size))
(OUT/'manifest.json').write_text(json.dumps(dict(provenance='Original project procedural geometry',units='display meters; authored Z-up; emitters emit +Z; not SI simulation aperture layout',limitations='Prototype PBR prefabs; pointing/focus anchors require runtime control wiring. No vendor dimensions implied.',components=manifest),indent=2)+'\n');print('Exported',len(manifest),'bench prefabs; roundtrip bounds checked.')
