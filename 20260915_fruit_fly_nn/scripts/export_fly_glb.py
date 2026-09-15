"""Export compiled MuJoCo visual meshes/hierarchy and illustrative wing clip to GLB.
.asset-venv/bin/python scripts/export_fly_glb.py
Uses compiled mesh/geom transforms, not raw OBJ placement. Units: cm -> m.
"""
from pathlib import Path
import json, struct, hashlib
import numpy as np
import mujoco
import trimesh
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/generated/flybody'; OUT.mkdir(parents=True,exist_ok=True)
m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/vendor/flybody/assets/fruitfly.xml'))
d=mujoco.MjData(m); mujoco.mj_forward(m,d)
g=dict(asset=dict(version='2.0',generator='CBC MuJoCo visual exporter',copyright='FlyBody authors; Apache-2.0; see assets/vendor/flybody/LICENSE'),scene=0,scenes=[dict(nodes=[0])],nodes=[],meshes=[],materials=[],accessors=[],bufferViews=[],buffers=[])
buf=bytearray()
def acc(arr,typ,component=5126,target=None):
 arr=np.asarray(arr,dtype='<f4' if component==5126 else '<u4'); buf.extend(b'\0'*((-len(buf))%4)); off=len(buf); buf.extend(arr.tobytes())
 view=dict(buffer=0,byteOffset=off,byteLength=arr.nbytes)
 if target: view['target']=target
 g['bufferViews'].append(view)
 a=dict(bufferView=len(g['bufferViews'])-1,componentType=component,count=len(arr),type=typ)
 if typ in ('VEC3','SCALAR'): a.update(min=np.atleast_1d(arr.min(axis=0)).tolist(),max=np.atleast_1d(arr.max(axis=0)).tolist())
 g['accessors'].append(a);return len(g['accessors'])-1
def local(b):
 p=int(m.body_parentid[b]); rp=d.xmat[p].reshape(3,3); rb=d.xmat[b].reshape(3,3)
 return (rp.T@(d.xpos[b]-d.xpos[p])),Rotation.from_matrix(rp.T@rb).as_quat()
# Explicit root scale and rotation: source +Z up -> glTF +Y up, source +X forward.
g['nodes'].append(dict(name='FlyBody_meters_Y_up',scale=[.01]*3,rotation=Rotation.from_euler('x',-90,degrees=True).as_quat().tolist(),children=[]))
body_nodes={0:0}
for b in range(1,m.nbody):
 t,q=local(b); idx=len(g['nodes']);body_nodes[b]=idx
 g['nodes'].append(dict(name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b),translation=t.tolist(),rotation=q.tolist(),children=[],extras=dict(mujocoBodyId=b)))
 g['nodes'][body_nodes[int(m.body_parentid[b])]]['children'].append(idx)
for mat in range(m.nmat):
 rgba=m.mat_rgba[mat].tolist()
 g['materials'].append(dict(name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_MATERIAL,mat),pbrMetallicRoughness=dict(baseColorFactor=rgba,metallicFactor=0,roughnessFactor=.65),doubleSided=True,alphaMode='BLEND' if rgba[3]<1 else 'OPAQUE'))
errors=[]; expected=[]; joint_meta=[]
for i in range(m.ngeom):
 if m.geom_type[i]!=mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[i]!=1: continue
 mesh=int(m.geom_dataid[i]);v0=int(m.mesh_vertadr[mesh]);nv=int(m.mesh_vertnum[mesh]);f0=int(m.mesh_faceadr[mesh]);nf=int(m.mesh_facenum[mesh])
 v=m.mesh_vert[v0:v0+nv].copy();faces=m.mesh_face[f0:f0+nf].copy()
 tm=trimesh.Trimesh(vertices=v,faces=faces,process=False)
 primitive=dict(attributes=dict(POSITION=acc(v,'VEC3',target=34962),NORMAL=acc(tm.vertex_normals,'VEC3',target=34962)),indices=acc(faces.reshape(-1),'SCALAR',5125,34963),mode=4)
 if m.geom_matid[i]>=0: primitive['material']=int(m.geom_matid[i])
 mesh_idx=len(g['meshes']);g['meshes'].append(dict(name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_MESH,mesh),primitives=[primitive]))
 b=int(m.geom_bodyid[i]);node=len(g['nodes'])
 g['nodes'].append(dict(name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or f'visual_{i}',mesh=mesh_idx,translation=m.geom_pos[i].tolist(),rotation=np.roll(m.geom_quat[i],-1).tolist()))
 g['nodes'][body_nodes[b]]['children'].append(node)
 # Independently compare composed body/local geom transform with mj_forward world geom.
 vworld=v@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i]
 composed=(v@Rotation.from_quat(np.roll(m.geom_quat[i],-1)).as_matrix().T+m.geom_pos[i])@d.xmat[b].reshape(3,3).T+d.xpos[b]
 errors.append(float(np.max(np.abs(composed-vworld))));expected.append(vworld*.01)
for j in range(m.njnt):
 joint_meta.append(dict(name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j),bodyNode=body_nodes[int(m.jnt_bodyid[j])],axis=m.jnt_axis[j].tolist(),position_cm=m.jnt_pos[j].tolist(),range=m.jnt_range[j].tolist(),type=int(m.jnt_type[j])))
# Bake local transforms using actual articulated MuJoCo kinematics. Slow illustration, not flight physics.
times=np.linspace(0,1,61,dtype=np.float32); wing_joints=[j for j in range(m.njnt) if 'wing_' in joint_meta[j]['name']]
wing_bodies=sorted(set(int(m.jnt_bodyid[j]) for j in wing_joints));samples={b:[] for b in wing_bodies}
for t in times:
 d.qpos[:]=m.qpos0
 for j in wing_joints:
  name=joint_meta[j]['name'];phase=2*np.pi*float(t)
  value=.65*np.sin(phase) if 'yaw' in name else (.25+.35*np.sin(phase+np.pi/2) if 'roll' in name else .6+.6*np.sin(phase+np.pi/2))
  d.qpos[m.jnt_qposadr[j]]=np.clip(value,*m.jnt_range[j])
 mujoco.mj_forward(m,d)
 for b in wing_bodies: samples[b].append(local(b))
time_acc=acc(times,'SCALAR');anim=dict(name='illustrative_wing_cycle_1Hz',samplers=[],channels=[],extras=dict(note='Procedural slowed display motion; not measured flight kinematics or learned locomotion.'))
for b in wing_bodies:
 for k,path,typ in [(0,'translation','VEC3'),(1,'rotation','VEC4')]:
  values=np.array([s[k] for s in samples[b]])
  if k==1:
   for z in range(1,len(values)):
    if np.dot(values[z-1],values[z])<0: values[z]*=-1
  out=acc(values,typ);sid=len(anim['samplers']);anim['samplers'].append(dict(input=time_acc,output=out,interpolation='LINEAR'));anim['channels'].append(dict(sampler=sid,target=dict(node=body_nodes[b],path=path)))
g['animations']=[anim]
g['buffers']=[dict(byteLength=len(buf))]
js=json.dumps(g,separators=(',',':')).encode();js+=b' '*((-len(js))%4);buf.extend(b'\0'*((-len(buf))%4))
content=struct.pack('<III',0x46546c67,2,12+8+len(js)+8+len(buf))+struct.pack('<II',len(js),0x4e4f534a)+js+struct.pack('<II',len(buf),0x004e4942)+buf
out=OUT/'flybody-articulated.glb';out.write_bytes(content)
# Roundtrip load glTF independently through trimesh and check world bounds/mesh count.
roundtrip=trimesh.load(out,force='scene');rotation=Rotation.from_euler('x',-90,degrees=True).as_matrix();allpoints=np.concatenate(expected)@rotation.T
bounds=np.array([allpoints.min(0),allpoints.max(0)]);bound_error=float(np.max(np.abs(roundtrip.bounds-bounds)))
assert max(errors)<1e-6 and bound_error<1e-7,(max(errors),bound_error)
report=dict(source_revision='d015e9bfe441bd90ae431bac24c55cb74bdbce26',mujoco=mujoco.__version__,trimesh=trimesh.__version__,numpy=np.__version__,units='meters; Y up; X forward; source centimeter -> meter root scale .01',body_nodes=m.nbody-1,actual_joints=m.njnt,visual_meshes=len(g['meshes']),triangles=int(sum(m.mesh_facenum)),bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),max_transform_error_source_cm=max(errors),roundtrip_bounds_error_m=bound_error,bounds_m=bounds.tolist(),animations=[anim['name']],limitations=['PBR materials approximate MuJoCo shading','Visual browser/source silhouette review pending','Neutral pose is qpos0; no optimized LOD or resting pose yet','Wing clip is illustrative 1 Hz, not biological flight timing'])
(OUT/'conversion-report.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'joint-map.json').write_text(json.dumps(joint_meta,indent=2)+'\n');print(json.dumps(report,indent=2))
