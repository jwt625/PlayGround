"""Original generic CBC display assets; never a claim of vendor fit or optical scale."""
from pathlib import Path
import json, hashlib, math
import numpy as np
import trimesh
R=Path(__file__).resolve().parents[1]; O=R/'assets/generated/hardware-v2'; O.mkdir(parents=True,exist_ok=True)
C={'case':[45,55,67,255],'lid':[76,89,101,255],'steel':[180,189,195,255],'black':[24,28,32,255],'green':[43,155,91,255],'gold':[190,145,57,255],'ceramic':[237,232,218,255],'lens':[52,168,191,255],'amber':[241,163,47,255]}
manifest=[]; previews=[]
def mesh(m,name,color='case',parent=None):
 m.visual=trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=C[color],metallicFactor=.65 if color in ['case','lid','steel','gold'] else .05,roughnessFactor=.35))
 s.add_geometry(m,node_name=name,geom_name=name,parent_node_name=parent)
def box(name,size,pos,color='case',parent=None):
 m=trimesh.creation.box(size);m.apply_translation(pos);mesh(m,name,color,parent)
def cyl(name,r,h,pos,color='steel',axis=(0,0,1),parent=None,n=24):
 m=trimesh.creation.cylinder(radius=r,height=h,sections=n);m.apply_transform(trimesh.geometry.align_vectors([0,0,1],axis));m.apply_translation(pos);mesh(m,name,color,parent)
def node(name,pos=(0,0,0),parent='world'):
 s.graph.update(frame_to=name,frame_from=parent,matrix=trimesh.transformations.translation_matrix(pos))
def port(name,pos,normal,kind,connector,channel=None,parent='world'):
 m=trimesh.geometry.align_vectors([0,0,1],normal);m[:3,3]=pos;s.graph.update(frame_to='port_'+name,frame_from=parent,matrix=m)
 ports.append(dict(name=name,node='port_'+name,position_m=pos,outward_normal=normal,kind=kind,connector=connector,channel=channel,parent=parent,key_reference_local=[0,1,0]))
def enclosure(x,y,z):
 box('chassis',[x,y,z-.0035],[0,0,(z-.0035)/2]);box('lid_seam',[x+.0004,y+.0004,.001],[0,0,z-.003],'black');box('lid',[x,y,.002],[0,0,z-.001],'lid')
 for i,(a,b) in enumerate([(a,b)for a in [-1,1]for b in [-1,1]]):
  cyl('lid_screw_'+str(i),.0014,.0007,[a*(x/2-.006),b*(y/2-.006),z+.0002]);box('screw_slot_'+str(i),[.002,.00035,.0002],[a*(x/2-.006),b*(y/2-.006),z+.0006],'black');box('foot_'+str(i),[.012,.012,.004],[a*(x/2-.008),b*(y/2-.008),-.002],'black')
def bulk(name,p,axis=(0,-1,0)):
 # Generic flange and socket mouth; local mating plane at supplied position.
 p=np.array(p);a=np.array(axis)
 m=trimesh.creation.annulus(r_min=.0027,r_max=.004,height=.007,sections=24);m.apply_transform(trimesh.geometry.align_vectors([0,0,1],axis));m.apply_translation(p-a*.0035);mesh(m,name+'_barrel','steel')
 m=trimesh.creation.box([.014,.014,.0015]);m.apply_transform(trimesh.geometry.align_vectors([0,0,1],axis));m.apply_translation(p-a*.006);mesh(m,name+'_flange','steel')
 cyl(name+'_socket',.0027,.0003,p-a*.001,'black',axis)
for name in ['fc-apc-plug','fc-bulkhead','sma-plug','phase-cassette','phase-driver','optical-amplifier','splitter-19','tiptilt-collimator','fly-console']:
 s=trimesh.Scene();ports=[];controls=[]
 if name=='fc-apc-plug':
  cyl('coupling_nut',.005,.011,[0,0,-.008]);
  for i in range(32):
   a=i*math.tau/32;cyl('grip_ridge_'+str(i),.00025,.008,[.005*math.cos(a),.005*math.sin(a),-.008],n=6)
  cyl('ferrule',.00125,.003,[0,0,-.0005],'ceramic');box('alignment_key',[.002,.001,.005],[0,.0037,-.006],'steel');cyl('green_boot',.003,.017,[0,0,-.022],'green')
  for i in range(5):cyl('boot_rib_'+str(i),.0033-i*.00015,.0008,[0,0,-.017-i*.0023],'green')
  port('mating',[0,0,0],[0,0,1],'optical','FC/APC');port('cable_exit',[0,0,-.032],[0,0,-1],'optical','jacket-2mm')
 elif name=='fc-bulkhead':
  bulk('front',[0,0,.007],(0,0,1));bulk('rear',[0,0,-.007],(0,0,-1));port('front',[0,0,.007],[0,0,1],'optical','FC/APC');port('rear',[0,0,-.007],[0,0,-1],'optical','FC/APC')
 elif name=='sma-plug':
  cyl('hex_nut',.004,.006,[0,0,-.004],'gold',n=6);cyl('coupling',.003,.003,[0,0,-.0005],'gold');cyl('dielectric',.0021,.0004,[0,0,.001],'ceramic');cyl('pin',.00045,.002,[0,0,.001],'gold');cyl('strain_relief',.0025,.012,[0,0,-.013],'black');port('mating',[0,0,0],[0,0,1],'rf','SMA');port('cable_exit',[0,0,-.019],[0,0,-1],'rf','coax')
 elif name=='phase-cassette':
  enclosure(.080,.030,.016)
  for side,word in [(-1,'in'),(1,'out')]:
   cyl('pigtail_'+word,.001,.018,[side*.048,0,.008],'green',(1,0,0));port('optical_'+word,[side*.057,0,.008],[side,0,0],'optical','PM-pigtail')
  cyl('rf_bulkhead',.003,.008,[0,-.019,.008],'gold',(0,-1,0));port('rf',[0,-.023,.008],[0,-1,0],'rf','SMA')
 elif name in ['phase-driver','optical-amplifier']:
  x,y,z=(.105,.080,.038) if name=='phase-driver' else (.160,.130,.055);enclosure(x,y,z)
  for i in range(10):box('heat_fin_'+str(i),[x-.016,.002,.007],[0,-y/2+.012+i*(y-.024)/9,z+.0035],'black')
  if name=='phase-driver':
   cyl('rf_output',.003,.008,[0,-y/2-.004,z/2],'gold',(0,-1,0));port('rf_out',[0,-y/2-.008,z/2],[0,-1,0],'rf','SMA')
  else:
   for a,nm in [(-.042,'optical_in'),(.042,'optical_out')]:bulk(nm,[a,-y/2-.007,z/2]);port(nm,[a,-y/2-.007,z/2],[0,-1,0],'optical','FC/APC')
  for a,nm in [(-.025,'dc_power'),(.025,'command')]:box(nm+'_socket',[.012,.005,.009],[a,y/2+.002,z/2],'black');port(nm,[a,y/2+.005,z/2],[0,1,0],'power' if nm=='dc_power' else 'control','DC' if nm=='dc_power' else 'multipin')
 elif name=='splitter-19':
  enclosure(.210,.120,.032);bulk('input',[0,-.067,.016]);port('input',[0,-.067,.016],[0,-1,0],'optical','FC/APC')
  for i in range(19):
   p=[(i%10-4.5)*.019,.067,.010+(i//10)*.014];bulk('CH%02d'%(i+1),p,(0,1,0));port('CH%02d'%(i+1),p,[0,1,0],'optical','FC/APC','CH%02d'%(i+1))
 elif name=='tiptilt-collimator':
  box('base',[.045,.042,.006],[0,0,.003]);node('tip_pivot',[0,0,.035]);node('tilt_pivot',parent='tip_pivot')
  box('tip_yoke',[.034,.006,.027],[0,.012,0],'lid','tip_pivot');cyl('collimator',.008,.028,[0,0,0],'steel',parent='tilt_pivot');cyl('lens',.006,.001,[0,0,.0145],'lens',parent='tilt_pivot')
  for i,p in enumerate([[-.017,0,.025],[0,.016,.022]]):cyl('motor_'+str(i),.006,.015,p,'black',(1,0,0) if i==0 else (0,1,0))
  port('fiber_in',[0,0,-.018],[0,0,-1],'optical','FC/APC',parent='tilt_pivot');port('emission',[0,0,.015],[0,0,1],'beam','free-space',parent='tilt_pivot');port('motor_bus',[.022,0,.007],[1,0,0],'control','multipin');controls=[dict(node='tip_pivot',axis=[1,0,0]),dict(node='tilt_pivot',axis=[0,1,0])]
 elif name=='fly-console':
  enclosure(.160,.110,.021)
  layout=json.loads((R/'assets/generated/channel-layout.json').read_text())['channels']
  for ch in layout:
   a,b,_=ch['position_pitch_units'];p=[-.044+a*.011,b*.011,.023];cyl(ch['id']+'_selector',.0035,.003,p,'lens',n=16)
  for i,key in enumerate(['phase','amplitude','tip','tilt','focus']):
   p=[.014+(i%2)*.025,.031-(i//2)*.027,.026];node('knob_'+key,p);cyl(key+'_knob_body',.008,.007,[0,0,0],'black',parent='knob_'+key);box(key+'_pointer',[.001,.005,.0008],[0,.0035,.0038],'amber','knob_'+key);node('touch_'+key,[p[0],p[1],p[2]+.005]);controls.append(dict(node='knob_'+key,axis=[0,0,1],contact_node='touch_'+key))
  for x,nm in [(-.050,'phase_bus'),(0,'motor_bus'),(.050,'dc_power')]:box(nm+'_socket',[.019,.006,.009],[x,.058,.010],'black');port(nm,[x,.062,.010],[0,1,0],'power' if nm=='dc_power' else 'control','multipin')
  node('fly_stance',[0,-.095,0]);node('rest_left',[-.018,-.043,.028]);node('rest_right',[.018,-.043,.028])
 out=O/(name+'.glb');out.write_bytes(s.export(file_type='glb'));rt=trimesh.load(out,force='scene');assert np.allclose(rt.bounds,s.bounds,atol=1e-7)
 expected={p['node'] for p in ports}|{c['node'] for c in controls};assert expected.issubset(set(rt.graph.nodes))
 entry=dict(name=name,file=out.name,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bounds_m=s.bounds.tolist(),triangles=sum(len(m.faces)for m in s.geometry.values()),ports=ports,controls=controls,dimension_status='Original illustrative dimensions; not vendor-exact or fabrication-ready')
 manifest.append(entry)
(O/'manifest.json').write_text(json.dumps(dict(schema_version=1,provenance='Original project procedural geometry, no vendor CAD conversion',units='meters',up_axis='+Z',port_convention='port node local +Z is outward cable tangent; mates have opposed normals; positions are parent-local',limitations=['Display assets only; keep optical simulation coordinates separate','FC/APC ferrule end angle and internal threads simplified','Module dimensions and port layouts illustrative','Motor internals and collision-clearance not validated','No vendor logos or implied manufacturer product matching'],components=manifest),indent=2)+'\n')
print(json.dumps({'assets':len(manifest),'triangles':sum(e['triangles']for e in manifest),'bytes':sum(e['bytes']for e in manifest),'validation':'GLB reload, bounds, named ports and pivots passed'}))
