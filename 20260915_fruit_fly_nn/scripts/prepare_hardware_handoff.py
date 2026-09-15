"""Generate a reviewable wiring plan; does not modify or run the application."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1];O=R/'assets/generated/hardware-v2';M=json.loads((O/'manifest.json').read_text());assets={c['name']:c for c in M['components']};nodes=[];edges=[]
def instance(id,asset=None,ports=None):nodes.append(dict(id=id,asset=asset,placeholder_ports=ports or [],status='asset-ready' if asset else 'asset-needed'))
def wire(a,b,kind,channel=None):edges.append(dict(id=f'W{len(edges)+1:03}',source=a,target=b,kind=kind,channel=channel))
instance('seed',ports=['optical_out','dc_power']);instance('supply',ports=['dc_out']);instance('console','fly-console');instance('splitter','splitter-19');instance('phase-command-junction',ports=['in','out']);instance('motor-driver-junction',ports=['command_in','motor_out','focus_out','dc_power'])
wire('seed.optical_out','splitter.input','optical');wire('supply.dc_out','seed.dc_power','power');wire('supply.dc_out','console.dc_power','power');wire('supply.dc_out','motor-driver-junction.dc_power','power');wire('console.phase_bus','phase-command-junction.in','control');wire('console.motor_bus','motor-driver-junction.command_in','control')
for i in range(1,20):
 c=f'CH{i:02}'
 for suffix,asset in [('phase','phase-cassette'),('driver','phase-driver'),('amplifier','optical-amplifier'),('mount','tiptilt-collimator')]:instance(c+'-'+suffix,asset)
 instance(c+'-focus',ports=['command_in','dc_power'])
 for a,b,k in [('splitter.'+c,c+'-phase.optical_in','optical'),(c+'-phase.optical_out',c+'-amplifier.optical_in','optical'),(c+'-amplifier.optical_out',c+'-mount.fiber_in','optical'),('phase-command-junction.out',c+'-driver.command','control'),(c+'-driver.rf_out',c+'-phase.rf','rf'),('phase-command-junction.out',c+'-amplifier.command','control'),('motor-driver-junction.motor_out',c+'-mount.motor_bus','control'),('motor-driver-junction.focus_out',c+'-focus.command_in','control'),('supply.dc_out',c+'-driver.dc_power','power'),('supply.dc_out',c+'-amplifier.dc_power','power'),('supply.dc_out',c+'-focus.dc_power','power')]:wire(a,b,k,c)
lookup={n['id']:set(p['name']for p in assets[n['asset']]['ports']) if n['asset'] else set(n['placeholder_ports'])for n in nodes}
for e in edges:
 for endpoint in ['source','target']:
  n,p=e[endpoint].split('.');assert n in lookup and p in lookup[n],e
plan=dict(schema_version=1,status='Specification only; not integrated',channels=19,notes=['Edges are logical connections. Shared outputs require physical junctions/distribution connectors, not multiple plugs occupying one socket.','Control command junction carries phase/amplitude commands; separate RF driver outputs feed phase cassettes.','Optical connector assemblies/pigtail termination are expanded by cable builder; do not attach an FC plug directly to a pigtail exit.','Focus/supply/seed/junction assets are pending. A focus actuator receives command and power through the aperture driver assembly.'],nodes=nodes,edges=edges,cable_profiles=dict(optical=dict(color='#40baa4',label='PM fiber',jacket_diameter_m=.002,visual_bend_radius_m=.03),rf=dict(color='#8f79c6',label='RF coax',jacket_diameter_m=.003,visual_bend_radius_m=.02),control=dict(color='#e7ab45',label='command/motor harness',jacket_diameter_m=.004,visual_bend_radius_m=.025),power=dict(color='#626b78',label='DC power',jacket_diameter_m=.004,visual_bend_radius_m=.025)),radius_note='Illustrative routing defaults only; manufacturer bend limits and connector clearance must be verified for a buildable design')
(O/'wiring-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
cache=json.loads((R/'assets/reference/hardware/manifest.json').read_text());counts={};total=0
for item in cache['records']:
 if 'file' not in item:continue
 p=R/'assets/reference/hardware'/item['file'];b=p.read_bytes();assert len(b)==item['bytes'];assert hashlib.sha256(b).hexdigest()==item['sha256']
 if p.suffix=='.step':assert b.startswith(b'ISO-10303-21;') and b'END-ISO-10303-21;' in b
 if p.suffix=='.pdf':assert b.startswith(b'%PDF-')
 counts[p.suffix]=counts.get(p.suffix,0)+1;total+=len(b)
report=dict(reference_files=counts,reference_bytes=total,generated_glbs=len(assets),generated_glb_bytes=sum(a['bytes']for a in assets.values()),generated_triangles=sum(a['triangles']for a in assets.values()),wiring_nodes=len(nodes),wiring_edges=len(edges),checks=['reference hashes/byte lengths','STEP exchange-file headers/terminators','PDF magic','wiring endpoint membership'],limitations=['STEP geometry not tessellated or dimensionally inspected','PDF magic check is not complete document validation','Scene integration and foreleg animation not implemented'])
(O/'validation-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
