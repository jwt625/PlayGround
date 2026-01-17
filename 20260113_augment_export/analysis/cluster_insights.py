#!/usr/bin/env python3
"""
Stage 1: Embed and cluster insights into ~64 semantic groups.
Shows samples from each cluster for review before LLM consolidation.
"""
import json
import os
from collections import defaultdict
import numpy as np

# Config
TARGET_CLUSTERS = 64
OUTPUT_FILE = "insight_clusters.json"

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'classification_results/stage2_consolidated.json')

print("Loading insights...")
data = json.load(open(data_path))

# Extract all insights with metadata
insights = []
for r in data['results']:
    if r.get('insights'):
        for i in r['insights']:
            insights.append({
                'content': i['content'],
                'canonical_type': i.get('canonical_type', 'unknown'),
                'confidence': i.get('confidence', 0),
                'message_hash': r.get('message_hash', ''),
                'folder_path': r.get('folder_path', ''),
                'generalizability': r.get('generalizability', 0),
            })

print(f"Loaded {len(insights)} insights")

# Generate embeddings
print("Loading embedding model (sentence-transformers)...")
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'sentence-transformers', '-q'])
    from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
print("Generating embeddings...")
contents = [i['content'] for i in insights]
embeddings = model.encode(contents, show_progress_bar=True, batch_size=128)
print(f"Generated {len(embeddings)} embeddings of dim {embeddings.shape[1]}")

# Cluster using agglomerative clustering
print(f"Clustering into ~{TARGET_CLUSTERS} groups...")
from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(
    n_clusters=TARGET_CLUSTERS,
    metric='cosine',
    linkage='average'
)
labels = clustering.fit_predict(embeddings)

# Group insights by cluster
clusters = defaultdict(list)
for idx, label in enumerate(labels):
    clusters[label].append({
        **insights[idx],
        'embedding_idx': idx
    })

# Sort clusters by size (descending)
sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))

# Compute cluster centroids and find representative samples
print("Computing cluster statistics...")
cluster_stats = []
for cluster_id, members in sorted_clusters:
    member_embeddings = embeddings[[m['embedding_idx'] for m in members]]
    centroid = member_embeddings.mean(axis=0)
    
    # Find samples closest to centroid
    distances = np.linalg.norm(member_embeddings - centroid, axis=1)
    sorted_indices = np.argsort(distances)
    
    # Get representative samples (closest to centroid)
    representative_samples = [members[i]['content'] for i in sorted_indices[:5]]
    
    # Type distribution within cluster
    type_dist = defaultdict(int)
    for m in members:
        type_dist[m['canonical_type']] += 1
    
    cluster_stats.append({
        'cluster_id': int(cluster_id),
        'size': len(members),
        'type_distribution': dict(type_dist),
        'representative_samples': representative_samples,
        'all_members': [{'content': m['content'], 'confidence': m['confidence'], 
                         'generalizability': m['generalizability']} for m in members]
    })

# Save full results
output_path = os.path.join(script_dir, 'classification_results', OUTPUT_FILE)
with open(output_path, 'w') as f:
    json.dump({'clusters': cluster_stats, 'total_insights': len(insights)}, f, indent=2)
print(f"\nSaved full cluster data to {output_path}")

# Print summary
print(f"\n{'='*80}")
print(f"CLUSTER SUMMARY ({TARGET_CLUSTERS} clusters, {len(insights)} insights)")
print(f"{'='*80}\n")

for i, cs in enumerate(cluster_stats[:64]):  # Show all 64
    dominant_type = max(cs['type_distribution'].items(), key=lambda x: x[1])[0]
    print(f"Cluster {i:2d} | Size: {cs['size']:4d} | Dominant: {dominant_type:12s}")
    for j, sample in enumerate(cs['representative_samples'][:3]):
        print(f"    [{j+1}] {sample[:75]}")
    print()

