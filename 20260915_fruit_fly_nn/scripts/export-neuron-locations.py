"""Join selected real body IDs to cached annotations for point placement."""
import json,sys
from pathlib import Path
import pyarrow.feather as feather
out=Path('public/training');ids=json.loads((out/'selected-body-ids.json').read_text())
t=feather.read_table(Path(sys.argv[1])/'derived/nodes.feather',columns=['bodyId','somaLocation','type','superclass'])
lookup={row['bodyId']:row for row in t.to_pylist()}
rows=[dict(bodyId=i,type=lookup[i]['type'],superclass=lookup[i]['superclass'],soma=lookup[i]['somaLocation']) for i in ids]
(out/'neuron-locations.json').write_text(json.dumps(dict(source='MaleCNS v1.0 body annotations',units='native male CNS EM voxel coordinates (8 nm)',placement='soma location; neurons without a soma coordinate are omitted from the point rendering, retained in dynamics',license='CC-BY-4.0',neurons=rows),separators=(',',':')))
print('Mapped somata:',sum(r['soma'] is not None for r in rows),'/',len(rows))
