"""Inspect source tables and derive an explicitly filtered, unsigned indexed graph.
No biological sign assumption, gain normalization, random edges, or learned weights.
Run: .asset-venv/bin/python scripts/prepare_connectome_cache.py
"""
from pathlib import Path
from collections import Counter
import json, hashlib
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as feather
ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT/'data/cache/malecns-v1.0';OUT=CACHE/'derived';OUT.mkdir(exist_ok=True)
t=feather.read_table(CACHE/'annotations.feather'); status=t['status'].to_pylist();superclass=t['superclass'].to_pylist()
# Strict, transparent starter selection; not a claim to reproduce the paper's neuron total.
mask=np.array([s=='Traced' and c is not None for s,c in zip(status,superclass)])
selected=t.filter(pa.array(mask));order=np.argsort(selected['bodyId'].to_numpy());selected=selected.take(pa.array(order));ids=selected['bodyId'].to_numpy()
assert len(np.unique(ids))==len(ids)
feather.write_feather(selected,OUT/'nodes.feather');np.save(OUT/'body-ids.npy',ids)
raw=ipc.open_file(pa.memory_map(str(CACHE/'connectome-weights.feather'),'r'));rows=kept=power=0
partial=OUT/'edges.u32.part'
with partial.open('wb') as f:
 for i in range(raw.num_record_batches):
  batch=raw.get_batch(i);a=batch.column(0).to_numpy();b=batch.column(1).to_numpy();w=batch.column(2).to_numpy();rows+=len(a)
  ia=np.searchsorted(ids,a);ib=np.searchsorted(ids,b)
  valid=(ia<len(ids))&(ib<len(ids));valid&=(ids[np.minimum(ia,len(ids)-1)]==a)&(ids[np.minimum(ib,len(ids)-1)]==b)
  assert np.all(w>=0) and np.all(w<2**32)
  edges=np.column_stack([ia[valid],ib[valid],w[valid]]).astype('<u4');f.write(edges.tobytes());kept+=len(edges);power+=int(w[valid].sum())
  if i%400==0: print('batches',i,'/',raw.num_record_batches,flush=True)
partial.replace(OUT/'edges.u32')
report=dict(release='v1.0',annotation_rows=t.num_rows,annotation_status_counts=dict(Counter(str(s) for s in status)),selection="status == 'Traced' AND superclass is not null; both endpoints must belong to selected bodies",selected_nodes=len(ids),raw_segment_edge_rows=rows,selected_edge_rows=kept,selected_synapse_weight_sum=power,edge_format='little-endian uint32 triples: pre_index, post_index, raw_synapse_count; nodes indexed by ascending bodyId in body-ids.npy / nodes.feather',weight_policy='unsigned raw counts; transmitter-to-sign and normalization are modeling decisions not applied here',source_sha256=hashlib.file_digest((CACHE/'connectome-weights.feather').open('rb'),'sha256').hexdigest(),edge_sha256=hashlib.file_digest((OUT/'edges.u32').open('rb'),'sha256').hexdigest())
(OUT/'graph-report.json').write_text(json.dumps(report,indent=2)+'\n')
# Renderable real skeleton geometry; one shared transform, preserving relative anatomy.
neurons=[]; allcoords=[]
for p in sorted((CACHE/'skeletons').glob('*.swc')):
 a=np.loadtxt(p); lookup={int(row[0]):i for i,row in enumerate(a)};segments=[]
 for i,row in enumerate(a):
  parent=int(row[6])
  if parent!=-1:
   assert parent in lookup
   segments.append([lookup[parent],i])
 xyz=a[:,2:5]*8e-9;allcoords.append(xyz)
 neurons.append(dict(bodyId=int(p.stem),node_ids=a[:,0].astype(int).tolist(),positions_m=xyz.tolist(),segments=segments,source=str(p.relative_to(ROOT))))
coords=np.concatenate(allcoords);center=(coords.max(0)+coords.min(0))/2;extent=float(np.ptp(coords,axis=0).max())
render=dict(dataset='MaleCNS v1.0',license='CC-BY-4.0',units='meters; native male CNS EM axes; SWC source unit = 8 nm',display_transform=dict(center_m=center.tolist(),scale_to_unit_extent=1/extent),selection='Official DNge104 examples 12781 and 556329; two illustrative morphologies, not whole-CNS visualization or learned pathway',neurons=neurons)
asset=ROOT/'assets/generated/connectome';asset.mkdir(exist_ok=True);(asset/'DNge104-skeletons.json').write_text(json.dumps(render,separators=(',',':'))+'\n');(asset/'graph-cache-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
