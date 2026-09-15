# External resource cache

## MaleCNS v1.0 — cached 2026-09-15

Primary release: https://male-cns.janelia.org/download/ · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The official source supplies the segment connection table, body annotations, aggregate predicted neurotransmitters, and SWC morphology. These are anatomical data, not pretrained dynamical weights. The downloaded release/documentation/license pages are cached alongside the data.

| Local path under `data/cache/malecns-v1.0/` | Content | Bytes |
|---|---|---:|
| `connectome-weights.feather` | Full released segment-to-segment graph; 151,856,684 rows | 1,051,241,946 |
| `annotations.feather` | 211,577 annotated bodies, including non-neuron segments | 14,483,314 |
| `neurotransmitters.feather` | Predicted transmitters, confidence and consensus fields | 43,282,834 |
| `skeletons/12781.swc`, `556329.swc` | Official DNge104 example pair, native EM coordinates | 1,155,306 combined |
| `derived/edges.u32` | Filtered graph, interleaved uint32 triples | 306,704,052 |
| `derived/body-ids.npy` | Sorted original IDs, int64 | See manifest/report |
| `derived/nodes.feather` | Retained annotations in indexed order | See local file |

The raw cache and Python environment are Git-ignored. The [source manifest](manifests/malecns-v1.0.json) is retained separately for handoff, with SHA-256, cloud object generation, size, and URL. Cloud files passed size and published MD5 checks. The complete EM image volume, every neuron skeleton, synapse-level locations, and database dump are not cached; they are unnecessary for this initial graph/controller and asset work.

## Derived graph contract

Selection: `status == 'Traced' AND superclass IS NOT NULL`, retaining connections only when both endpoints are selected. This yields **164,606 nodes, 25,558,671 connection rows, and a total raw synapse weight of 124,009,893** in the cached files. This is an explicit modeling subset and does not claim to reproduce the paper's neuron count. Preserve source IDs; never infer that annotation row count equals neuron count.

`edges.u32` is a headerless, little-endian stream of `(pre_index, post_index, raw_synapse_count)` triples. Indices refer to ascending IDs in `body-ids.npy` and rows in `nodes.feather`. Weights are unsigned raw counts; no transmitter sign assignment, scaling, edge randomization, or threshold beyond the release/filter has been applied. Join neurotransmitters on original body ID, not row index.

```python
import numpy as np
edges = np.memmap('data/cache/malecns-v1.0/derived/edges.u32',
                  dtype='<u4', mode='r').reshape(-1, 3)
body_ids = np.load('data/cache/malecns-v1.0/derived/body-ids.npy')
```

Do not load 25 million edges as browser JSON objects. Add a typed-array/streaming loader, retain ID mapping, define transmitter assumptions explicitly, and benchmark memory/step time. Dataset-cache availability does not mean the active TypeScript controller uses these data yet.

## Reproduction

```sh
uv venv .asset-venv
uv pip install --python .asset-venv/bin/python -r scripts/requirements-assets.lock.txt
python3 scripts/cache_external_resources.py
.asset-venv/bin/python scripts/prepare_connectome_cache.py
```

The cache script uses resumable `.part` downloads and fails on size/hash mismatches. It reuses existing complete files and verifies cloud metadata. Derived graph generation streams Arrow batches and writes edges atomically. Current source metadata may change even under a release path; retain and compare manifests when refreshing, rather than silently treating a new hash as the same snapshot.

Attribution: MaleCNS collaboration — FlyEM/HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, and Google Research; cite the MaleCNS paper linked by the release. Derived graph and skeleton assets remain subject to the dataset's CC BY 4.0 license.
