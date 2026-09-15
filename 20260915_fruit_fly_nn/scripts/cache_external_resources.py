"""Cache official MaleCNS v1.0 core tables and representative skeletons.
Run with stdlib Python. Resumable curl transfers, atomic final names, SHA-256 manifest.
Raw segment graph is not filtered into neurons or assigned dynamical weights here.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, subprocess, urllib.request, base64, datetime
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'data/cache/malecns-v1.0'
BASE='https://storage.googleapis.com/flyem-male-cns/v1.0/'
FLAT=BASE+'connectome-data/flat-connectome/'
RESOURCES=[
 ('annotations.feather',FLAT+'body-annotations-male-cns-v1.0-minconf-0.5.feather'),
 ('neurotransmitters.feather',FLAT+'body-neurotransmitters-male-cns-v1.0.feather'),
 ('connectome-weights.feather',FLAT+'connectome-weights-male-cns-v1.0-minconf-0.5.feather'),
 ('skeletons/12781.swc',BASE+'segmentation/skeletons-malecns/skeletons-swc/12781.swc'),
 ('skeletons/556329.swc',BASE+'segmentation/skeletons-malecns/skeletons-swc/556329.swc'),
 ('sources/download.html','https://male-cns.janelia.org/download/'),
 ('sources/release-notes.html','https://male-cns.janelia.org/release/'),
 ('sources/license.html','https://creativecommons.org/licenses/by/4.0/'),
]
def fetch(item):
 name,url=item
 dest=CACHE/name; dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():
  partial=Path(str(dest)+'.part')
  subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','4','--connect-timeout','30','--continue-at','-','--output',str(partial),url],check=True)
  partial.replace(dest)
 sha=hashlib.sha256(); md5=hashlib.md5()
 with dest.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): sha.update(block); md5.update(block)
 headers={}
 if 'storage.googleapis.com' in url:
  with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as resp:
   headers={k.lower():v for k,v in resp.headers.items()}
 if 'storage.googleapis.com' in url:
  assert int(headers['content-length'])==dest.stat().st_size, name
  hashes=headers.get('x-goog-hash','')
  expected=next((x.strip()[4:] for x in hashes.split(',') if x.strip().startswith('md5=')),None)
  if expected: assert expected==base64.b64encode(md5.digest()).decode(),name
 record=dict(path=str(dest.relative_to(ROOT)),url=url,bytes=dest.stat().st_size,sha256=sha.hexdigest(),http_metadata={k:v for k,v in headers.items() if k in ['etag','last-modified','x-goog-generation','x-goog-hash','content-length']})
 print('Cached',name,record['bytes'],flush=True)
 return record
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool: records=list(pool.map(fetch,RESOURCES))
 manifest=dict(dataset='MaleCNS',release='v1.0',license='CC-BY-4.0',source='https://male-cns.janelia.org/download/',retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=records,scope='Full published segment connection table; body annotations and predicted transmitters; two illustrative DNge104 skeletons. EM, complete skeleton collection and individual synapse locations not downloaded.')
 (CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('Wrote',CACHE/'manifest.json')
