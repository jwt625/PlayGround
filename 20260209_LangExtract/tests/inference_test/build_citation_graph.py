#!/usr/bin/env python3
"""
Phase 3: Reference Deduplication and Citation Graph Building

Deduplication strategy (priority order):
1. DOI (normalized to lowercase)
2. arXiv ID (normalized, version suffix removed)
3. Composite key: first_author_last_name + year + first_significant_title_word

Output files:
- unique_references.jsonl: Canonical reference records with citation counts
- citation_graph.json: Adjacency list for graph analysis
- dedup_stats.json: Summary statistics
"""

import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Common words to skip when extracting first significant word from title
STOP_WORDS = {
    'a', 'an', 'the', 'on', 'in', 'of', 'for', 'to', 'and', 'or', 'with',
    'from', 'by', 'at', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'using', 'via', 'towards', 'toward', 'into', 'onto', 'upon'
}


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Normalize DOI to lowercase, strip whitespace."""
    if not doi or not isinstance(doi, str):
        return None
    doi = doi.strip().lower()
    if not doi:
        return None
    # Remove common DOI URL prefixes
    for prefix in ['https://doi.org/', 'http://doi.org/', 'doi:', 'doi.org/']:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi if doi else None


def normalize_arxiv(arxiv_id: Optional[str]) -> Optional[str]:
    """Normalize arXiv ID: lowercase, remove version suffix (e.g., v1, v2)."""
    if not arxiv_id or not isinstance(arxiv_id, str):
        return None
    arxiv_id = arxiv_id.strip().lower()
    if not arxiv_id:
        return None
    # Remove 'arxiv:' prefix if present
    if arxiv_id.startswith('arxiv:'):
        arxiv_id = arxiv_id[6:]
    # Remove version suffix (e.g., 'v1', 'v2')
    arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
    return arxiv_id if arxiv_id else None


def extract_last_name(author: str) -> Optional[str]:
    """
    Extract last name from various author formats:
    - "J. Koch" -> "koch"
    - "Jens Koch" -> "koch"
    - "Koch, J." -> "koch"
    - "Koch, Jens" -> "koch"
    - "et al." -> None
    """
    if not author or not isinstance(author, str):
        return None
    
    author = author.strip()
    if not author or author.lower() in ['et al.', 'et al', 'others']:
        return None
    
    # Normalize unicode characters (e.g., accented characters)
    author = unicodedata.normalize('NFKD', author)
    
    # Check if format is "LastName, FirstName" or "LastName, F."
    if ',' in author:
        last_name = author.split(',')[0].strip()
    else:
        # Format is "FirstName LastName" or "F. LastName" or "F. M. LastName"
        parts = author.split()
        if not parts:
            return None
        # Last part is typically the last name
        last_name = parts[-1]
    
    # Clean up: remove non-alphabetic characters, lowercase
    last_name = re.sub(r'[^a-zA-Z]', '', last_name).lower()
    return last_name if last_name else None


def extract_first_significant_word(title: str) -> Optional[str]:
    """
    Extract first significant word from title (skip stop words).
    - "Charge-insensitive qubit design..." -> "charge"
    - "A quantum engineer's guide..." -> "quantum"
    - "The flux qubit revisited..." -> "flux"
    """
    if not title or not isinstance(title, str):
        return None
    
    title = title.strip().lower()
    if not title:
        return None
    
    # Split on whitespace and punctuation
    words = re.split(r'[\s\-_:;,.()\[\]{}]+', title)
    
    for word in words:
        # Clean the word
        word = re.sub(r'[^a-z]', '', word)
        if word and word not in STOP_WORDS and len(word) > 1:
            return word
    
    return None


def generate_composite_key(ref: dict) -> Optional[str]:
    """Generate composite key: last_name_year_titleword."""
    authors = ref.get('authors', [])
    year = ref.get('year')
    title = ref.get('title', '')
    
    if not year:
        return None
    
    # Get first author's last name
    last_name = None
    if authors and len(authors) > 0:
        last_name = extract_last_name(authors[0])
    
    if not last_name:
        last_name = 'unknown'
    
    # Get first significant word from title
    title_word = extract_first_significant_word(title)
    if not title_word:
        title_word = 'untitled'
    
    return f"{last_name}_{year}_{title_word}"


def generate_ref_id(ref: dict) -> tuple[str, str]:
    """
    Generate reference ID using priority:
    1. DOI
    2. arXiv ID
    3. Composite key
    
    Returns: (ref_id, match_method)
    """
    # Priority 1: DOI
    doi = normalize_doi(ref.get('doi'))
    if doi:
        return f"doi:{doi}", "doi"
    
    # Priority 2: arXiv ID
    arxiv = normalize_arxiv(ref.get('arxiv_id'))
    if arxiv:
        return f"arxiv:{arxiv}", "arxiv"
    
    # Priority 3: Composite key
    composite = generate_composite_key(ref)
    if composite:
        return composite, "composite"
    
    # Fallback: use hash of the reference
    return f"unknown_{hash(json.dumps(ref, sort_keys=True)) % 10**8}", "hash"


def merge_references(refs: list[dict]) -> dict:
    """
    Merge multiple references into a canonical record.
    Prefer non-empty values, longer author lists, etc.
    """
    canonical = {
        'title': None,
        'authors': [],
        'year': None,
        'journal': None,
        'volume': None,
        'pages': None,
        'doi': None,
        'arxiv_id': None,
    }

    for ref in refs:
        # Title: prefer longer, non-journal-name titles
        title = ref.get('title', '')
        if title and isinstance(title, str):
            title = title.strip()
            # Skip if title looks like a journal name
            if title.lower() not in ['phys. rev. lett.', 'nature', 'science',
                                      'applied physics letters', 'appl. phys. lett.',
                                      'physical review a', 'physical review b',
                                      'physical review letters', 'none', '']:
                if not canonical['title'] or len(title) > len(canonical['title']):
                    canonical['title'] = title

        # Authors: prefer longer list
        authors = ref.get('authors', [])
        if authors and len(authors) > len(canonical['authors']):
            canonical['authors'] = authors

        # Year: prefer non-null
        year = ref.get('year')
        if year and not canonical['year']:
            canonical['year'] = year

        # Journal: prefer non-empty
        journal = ref.get('journal', '')
        if journal and not canonical['journal']:
            canonical['journal'] = journal

        # Volume, pages: prefer non-empty
        for field in ['volume', 'pages']:
            val = ref.get(field, '')
            if val and not canonical[field]:
                canonical[field] = val

        # DOI: prefer normalized non-empty
        doi = normalize_doi(ref.get('doi'))
        if doi and not canonical['doi']:
            canonical['doi'] = doi

        # arXiv: prefer normalized non-empty
        arxiv = normalize_arxiv(ref.get('arxiv_id'))
        if arxiv and not canonical['arxiv_id']:
            canonical['arxiv_id'] = arxiv

    return canonical


def load_source_paper_metadata(metadata_dir: Path) -> dict[str, dict]:
    """
    Load metadata for source papers from JSON files.
    Returns: {document_id: metadata_dict}
    """
    source_metadata = {}
    if not metadata_dir.exists():
        print(f"  Warning: Metadata directory not found: {metadata_dir}")
        return source_metadata

    for json_file in metadata_dir.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                meta = json.load(f)
            doc_id = meta.get('document_id') or json_file.stem
            source_metadata[doc_id] = meta
        except Exception as e:
            print(f"  Warning: Failed to load {json_file}: {e}")

    return source_metadata


def load_phase2_data(input_path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Load Phase 2 extracted references.
    Returns: (all_refs_with_source, doc_to_refs)
    """
    all_refs = []
    doc_to_refs = {}

    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = doc['document_id']
            refs = doc.get('references', [])
            doc_to_refs[doc_id] = refs

            for ref in refs:
                ref['_source_doc'] = doc_id
                all_refs.append(ref)

    return all_refs, doc_to_refs


def build_source_paper_id_map(source_metadata: dict[str, dict]) -> tuple[dict[str, str], dict[str, dict]]:
    """
    Generate canonical IDs for source papers using the same logic as references.
    Returns:
        - doc_id_to_ref_id: {document_id: canonical_ref_id}
        - source_as_refs: {canonical_ref_id: ref_data} for source papers
    """
    doc_id_to_ref_id = {}
    source_as_refs = {}

    for doc_id, meta in source_metadata.items():
        # Build a reference-like dict from metadata
        ref_like = {
            'title': meta.get('title', ''),
            'authors': meta.get('authors', []),
            'year': meta.get('year'),
            'doi': meta.get('doi'),
            'arxiv_id': meta.get('arxiv_id'),
            'journal': meta.get('journal', ''),
            '_source_doc': None,  # This IS a source paper
            '_is_source_paper': True,
            '_document_id': doc_id,
        }

        # Generate canonical ID
        ref_id, match_method = generate_ref_id(ref_like)
        doc_id_to_ref_id[doc_id] = ref_id

        # Store as a reference record
        if ref_id not in source_as_refs:
            source_as_refs[ref_id] = {
                'ref_id': ref_id,
                'canonical': {
                    'title': ref_like['title'],
                    'authors': ref_like['authors'],
                    'year': ref_like['year'],
                    'journal': ref_like['journal'],
                    'volume': meta.get('volume'),
                    'pages': meta.get('pages'),
                    'doi': normalize_doi(ref_like['doi']),
                    'arxiv_id': normalize_arxiv(ref_like['arxiv_id']),
                },
                'citation_count': 0,
                'cited_by': [],
                'match_method': match_method,
                'merged_from': 1,
                'in_dataset': True,
                'document_ids': [doc_id],
            }
        else:
            # Multiple doc_ids map to same ref_id (shouldn't happen often)
            source_as_refs[ref_id]['document_ids'].append(doc_id)

    return doc_id_to_ref_id, source_as_refs


def deduplicate_references(all_refs: list[dict], source_as_refs: dict[str, dict] = None) -> dict[str, dict]:
    """
    Deduplicate references and build canonical records.
    Three-pass approach:
    1. Group by primary ID (DOI > arXiv > composite)
    2. Merge groups that share DOI or arXiv across different primary IDs
    3. Merge source papers into the pool, matching references to source papers

    Args:
        all_refs: List of references extracted from source papers
        source_as_refs: {ref_id: ref_data} for source papers (optional)

    Returns: {ref_id: {canonical, citation_count, cited_by, match_method, merged_from, in_dataset}}
    """
    source_as_refs = source_as_refs or {}
    # Pass 1: Group references by their generated ID
    ref_groups = defaultdict(list)
    ref_methods = {}

    for ref in all_refs:
        ref_id, method = generate_ref_id(ref)
        ref_groups[ref_id].append(ref)
        ref_methods[ref_id] = method

    # Pass 2: Build DOI and arXiv indexes to find cross-group matches
    # Map DOI/arXiv -> list of ref_ids that have this DOI/arXiv
    doi_to_refids = defaultdict(set)
    arxiv_to_refids = defaultdict(set)

    for ref_id, refs in ref_groups.items():
        for ref in refs:
            doi = normalize_doi(ref.get('doi'))
            if doi:
                doi_to_refids[doi].add(ref_id)
            arxiv = normalize_arxiv(ref.get('arxiv_id'))
            if arxiv:
                arxiv_to_refids[arxiv].add(ref_id)

    # Build union-find structure to merge ref_ids that share DOI/arXiv
    parent = {ref_id: ref_id for ref_id in ref_groups}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            # Prefer DOI-based ID, then arXiv, then composite
            # Priority: doi > arxiv > composite > hash
            def priority(ref_id):
                if ref_id.startswith('doi:'):
                    return 0
                elif ref_id.startswith('arxiv:'):
                    return 1
                elif ref_id.startswith('unknown_'):
                    return 3
                else:
                    return 2
            if priority(px) <= priority(py):
                parent[py] = px
            else:
                parent[px] = py

    # Union ref_ids that share DOI
    for doi, refids in doi_to_refids.items():
        refids = list(refids)
        for i in range(1, len(refids)):
            union(refids[0], refids[i])

    # Union ref_ids that share arXiv
    for arxiv, refids in arxiv_to_refids.items():
        refids = list(refids)
        for i in range(1, len(refids)):
            union(refids[0], refids[i])

    # Pass 3: Match by normalized title + year + first author last name
    # This catches cases where some refs have DOI and some don't
    def normalize_title_for_matching(title: str) -> str:
        """Normalize title for fuzzy matching."""
        if not title:
            return ''
        # Lowercase, remove punctuation, collapse whitespace
        title = title.lower()
        title = re.sub(r'[^a-z0-9\s]', '', title)
        title = re.sub(r'\s+', ' ', title).strip()
        # Take first 50 chars to handle truncation
        return title[:50]

    # Build title+year+author index
    title_key_to_refids = defaultdict(set)
    for ref_id, refs in ref_groups.items():
        for ref in refs:
            title = ref.get('title', '')
            year = ref.get('year')
            authors = ref.get('authors', [])

            norm_title = normalize_title_for_matching(title)
            if not norm_title or len(norm_title) < 15:  # Skip short/empty titles
                continue
            if not year:
                continue

            last_name = None
            if authors:
                last_name = extract_last_name(authors[0])
            if not last_name:
                last_name = 'unknown'

            title_key = f"{last_name}_{year}_{norm_title}"
            title_key_to_refids[title_key].add(ref_id)

    # Union ref_ids that share title key
    for title_key, refids in title_key_to_refids.items():
        refids = list(refids)
        for i in range(1, len(refids)):
            union(refids[0], refids[i])

    # Group by canonical parent
    merged_groups = defaultdict(list)
    for ref_id in ref_groups:
        canonical_id = find(ref_id)
        merged_groups[canonical_id].extend(ref_groups[ref_id])

    # Build canonical records from merged groups
    unique_refs = {}
    for ref_id, refs in merged_groups.items():
        # Get all source documents that cite this reference
        cited_by = list(set(r['_source_doc'] for r in refs if r.get('_source_doc')))

        # Merge into canonical record
        canonical = merge_references(refs)

        # Determine best match method
        method = ref_methods.get(ref_id, 'composite')

        unique_refs[ref_id] = {
            'ref_id': ref_id,
            'canonical': canonical,
            'citation_count': len(cited_by),
            'cited_by': cited_by,
            'match_method': method,
            'merged_from': len(refs),
            'in_dataset': False,  # Will be updated below
        }

    # Pass 4: Merge source papers into the pool
    # Check if any reference matches a source paper's ID
    for src_ref_id, src_data in source_as_refs.items():
        if src_ref_id in unique_refs:
            # A reference matches a source paper - merge them
            unique_refs[src_ref_id]['in_dataset'] = True
            unique_refs[src_ref_id]['document_ids'] = src_data.get('document_ids', [])
            # Update canonical with source paper metadata (more reliable)
            for key in ['title', 'authors', 'year', 'journal', 'doi', 'arxiv_id']:
                src_val = src_data['canonical'].get(key)
                if src_val:
                    unique_refs[src_ref_id]['canonical'][key] = src_val
        else:
            # Source paper not found in references - add it
            unique_refs[src_ref_id] = src_data.copy()

    return unique_refs


def build_citation_graph(unique_refs: dict, doc_id_to_ref_id: dict[str, str], all_doc_ids: set) -> dict:
    """
    Build citation graph adjacency list.
    All nodes are now canonical ref_ids (both source papers and external refs).

    Args:
        unique_refs: {ref_id: ref_data} - all deduplicated references including source papers
        doc_id_to_ref_id: {document_id: ref_id} - mapping from source doc IDs to canonical ref IDs
        all_doc_ids: set of all document_ids from Phase 2 data (some may lack metadata)
    """
    nodes = {}
    edges = []
    edge_set = set()  # To avoid duplicate edges

    # Add all references as nodes
    for ref_id, ref_data in unique_refs.items():
        nodes[ref_id] = {
            'in_dataset': ref_data.get('in_dataset', False),
            'year': ref_data['canonical'].get('year'),
            'title': ref_data['canonical'].get('title'),
        }

    # Add source papers that don't have metadata as nodes (using doc_id as fallback)
    for doc_id in all_doc_ids:
        if doc_id not in doc_id_to_ref_id and doc_id not in nodes:
            # This source paper doesn't have metadata - add it with doc_id as the node ID
            nodes[doc_id] = {
                'in_dataset': True,  # It's a source paper
                'year': None,
                'title': doc_id,  # Use doc_id as title placeholder
            }

    # Build edges: citing_doc -> cited_ref
    for ref_id, ref_data in unique_refs.items():
        for citing_doc in ref_data['cited_by']:
            # Convert citing_doc (document_id) to its canonical ref_id
            citing_ref_id = doc_id_to_ref_id.get(citing_doc, citing_doc)

            # Ensure the citing node exists
            if citing_ref_id not in nodes:
                nodes[citing_ref_id] = {
                    'in_dataset': citing_doc in all_doc_ids,
                    'year': None,
                    'title': citing_ref_id,
                }

            # Avoid self-loops and duplicate edges
            if citing_ref_id != ref_id:
                edge_key = (citing_ref_id, ref_id)
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edges.append({
                        'from': citing_ref_id,
                        'to': ref_id,
                    })

    return {
        'nodes': nodes,
        'edges': edges,
    }


def compute_stats(all_refs: list, unique_refs: dict, doc_id_to_ref_id: dict, graph: dict) -> dict:
    """Compute deduplication statistics."""
    method_counts = defaultdict(int)
    for ref_data in unique_refs.values():
        method_counts[ref_data['match_method']] += 1

    # Separate in-dataset and external references
    in_dataset_refs = [r for r in unique_refs.values() if r.get('in_dataset', False)]
    external_refs = [r for r in unique_refs.values() if not r.get('in_dataset', False)]

    # Sort external by citation count
    external_refs.sort(key=lambda x: x['citation_count'], reverse=True)
    top_100 = external_refs[:100]

    # Count inter-dataset citations (edges where both from and to are in-dataset)
    # Use graph nodes to include all source papers (including those without metadata)
    in_dataset_node_ids = {
        node_id for node_id, data in graph['nodes'].items()
        if data.get('in_dataset', False)
    }
    inter_dataset_edges = [
        e for e in graph['edges']
        if e['from'] in in_dataset_node_ids and e['to'] in in_dataset_node_ids
    ]

    return {
        'total_raw_references': len(all_refs),
        'unique_references': len(unique_refs),
        'dedup_ratio': round(len(all_refs) / len(unique_refs), 2) if unique_refs else 0,
        'documents_in_dataset_with_metadata': len(doc_id_to_ref_id),
        'total_in_dataset_nodes': len(in_dataset_node_ids),
        'in_dataset_references': len(in_dataset_refs),
        'external_references': len(external_refs),
        'inter_dataset_citations': len(inter_dataset_edges),
        'total_graph_nodes': len(graph['nodes']),
        'total_graph_edges': len(graph['edges']),
        'match_methods': dict(method_counts),
        'top_100_external': [
            {
                'ref_id': r['ref_id'],
                'title': r['canonical'].get('title'),
                'authors': r['canonical'].get('authors', [])[:3],
                'year': r['canonical'].get('year'),
                'citation_count': r['citation_count'],
                'doi': r['canonical'].get('doi'),
                'arxiv_id': r['canonical'].get('arxiv_id'),
            }
            for r in top_100
        ],
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Build citation graph from Phase 2 extracted references')
    parser.add_argument('--input', '-i', type=Path,
                        default=Path('output/phase2_extracted_refs.jsonl'),
                        help='Input JSONL file from Phase 2')
    parser.add_argument('--metadata-dir', '-m', type=Path,
                        default=Path('../../semiconductor_processing_dataset/processed_documents/metadata'),
                        help='Directory containing source paper metadata JSON files')
    parser.add_argument('--output-dir', '-o', type=Path,
                        default=Path('output'),
                        help='Output directory')
    args = parser.parse_args()

    # Load Phase 2 data (references extracted from source papers)
    print(f"Loading Phase 2 data from {args.input}...")
    all_refs, doc_to_refs = load_phase2_data(args.input)
    doc_ids = set(doc_to_refs.keys())
    print(f"  Loaded {len(all_refs)} references from {len(doc_ids)} documents")

    # Load source paper metadata
    print(f"Loading source paper metadata from {args.metadata_dir}...")
    source_metadata = load_source_paper_metadata(args.metadata_dir)
    print(f"  Loaded metadata for {len(source_metadata)} source papers")

    # Generate canonical IDs for source papers
    print("Generating canonical IDs for source papers...")
    doc_id_to_ref_id, source_as_refs = build_source_paper_id_map(source_metadata)
    print(f"  Generated {len(doc_id_to_ref_id)} source paper IDs")

    # Deduplicate references (including matching against source papers)
    print("Deduplicating references...")
    unique_refs = deduplicate_references(all_refs, source_as_refs)
    in_dataset_count = sum(1 for r in unique_refs.values() if r.get('in_dataset', False))
    print(f"  Found {len(unique_refs)} unique references ({in_dataset_count} in-dataset, {len(unique_refs) - in_dataset_count} external)")
    print(f"  Dedup ratio: {len(all_refs)/len(unique_refs):.2f}x")

    # Build citation graph using canonical ref_ids
    print("Building citation graph...")
    graph = build_citation_graph(unique_refs, doc_id_to_ref_id, doc_ids)
    print(f"  Graph has {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")

    # Compute statistics
    print("Computing statistics...")
    stats = compute_stats(all_refs, unique_refs, doc_id_to_ref_id, graph)

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # unique_references.jsonl
    output_refs = args.output_dir / 'unique_references.jsonl'
    with open(output_refs, 'w') as f:
        for ref_data in sorted(unique_refs.values(), key=lambda x: -x['citation_count']):
            f.write(json.dumps(ref_data) + '\n')
    print(f"  Wrote {output_refs}")

    # citation_graph.json
    output_graph = args.output_dir / 'citation_graph.json'
    with open(output_graph, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"  Wrote {output_graph}")

    # dedup_stats.json
    output_stats = args.output_dir / 'dedup_stats.json'
    with open(output_stats, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Wrote {output_stats}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Total raw references: {stats['total_raw_references']}")
    print(f"Unique references: {stats['unique_references']}")
    print(f"  - In-dataset: {stats['in_dataset_references']}")
    print(f"  - External: {stats['external_references']}")
    print(f"Dedup ratio: {stats['dedup_ratio']}x")
    print(f"Inter-dataset citations: {stats['inter_dataset_citations']}")
    print(f"Match methods: {stats['match_methods']}")
    print(f"\nTop 10 most-cited external references:")
    for i, ref in enumerate(stats['top_100_external'][:10], 1):
        title = ref['title'][:50] + '...' if ref['title'] and len(ref['title']) > 50 else ref['title']
        print(f"  {i}. [{ref['citation_count']} citations] {title} ({ref['year']})")


if __name__ == '__main__':
    main()

