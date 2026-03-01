#!/usr/bin/env python3
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / 'tests/inference_test/output'

UNIQUE_REFS = OUT_DIR / 'unique_references.jsonl'
R1_MANIFEST = ROOT / 'data/collection_r1/manifest_documents.jsonl'
R1_META_DIR = ROOT / 'data/collection_r1/metadata'
R2_MANIFEST = ROOT / 'data/collection_r2/manifest_documents_r2.jsonl'
R3_MANIFEST = ROOT / 'semiconductor_processing_dataset/raw_documents_R3/manifest_documents_r3.jsonl'
R3NEW_MANIFEST = ROOT / 'semiconductor_processing_dataset/raw_documents_R3_new/manifest_documents_r3new.jsonl'

OUT_JSONL = OUT_DIR / 'canonical_reference_registry.jsonl'
OUT_SUMMARY = OUT_DIR / 'canonical_reference_registry_summary.json'


def ndoi(v):
    if not v:
        return None
    s = str(v).strip().lower()
    s = re.sub(r'^https?://(dx\.)?doi\.org/', '', s)
    s = s.rstrip('.')
    return s or None


def narxiv(v):
    if not v:
        return None
    s = str(v).strip().lower()
    s = re.sub(r'^arxiv:\s*', '', s)
    s = re.sub(r'v\d+$', '', s)
    if not re.match(r'^[a-z0-9.\-\/]+$', s):
        return None
    if not re.search(r'\d', s):
        return None
    return s or None


def ntitle(v):
    if not v:
        return ''
    s = str(v).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def title_sim(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, ntitle(a), ntitle(b)).ratio()


def title_year_key(title, year):
    t = ntitle(title)
    if not t or not year:
        return None
    return f"{year}:{t[:120]}"


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_r1_metadata(meta_dir):
    out = {}
    if not meta_dir.exists():
        return out
    for p in sorted(meta_dir.glob('*.json')):
        try:
            row = json.loads(p.read_text())
        except Exception:
            continue
        doc_id = row.get('document_id') or p.stem
        out[doc_id] = row
    return out


def merge_r1_manifest_with_metadata(manifest_rows, meta_by_doc):
    merged = []
    for r in manifest_rows:
        rr = dict(r)
        m = meta_by_doc.get(rr.get('document_id'))
        if m:
            rr['doi'] = rr.get('doi') or m.get('doi')
            rr['arxiv'] = rr.get('arxiv') or m.get('arxiv_id')
            rr['authors'] = rr.get('authors') or m.get('authors')
            rr['journal'] = rr.get('journal') or m.get('journal')
            rr['year'] = rr.get('year') or m.get('year')
            rr['title'] = rr.get('title') or m.get('title')
        merged.append(rr)
    return merged


def build_round_index(rows, round_name):
    idx_doi = defaultdict(list)
    idx_arx = defaultdict(list)
    idx_ty = defaultdict(list)
    for r in rows:
        entry = {
            'round': round_name,
            'document_id': r.get('document_id'),
            'status': r.get('status'),
            'source_path': r.get('source_path'),
            'url': r.get('url'),
            'year': r.get('year'),
            'title': r.get('title'),
            'doi': ndoi(r.get('doi')),
            'arxiv': narxiv(r.get('arxiv') or r.get('arxiv_id')),
        }
        if entry['doi']:
            idx_doi[entry['doi']].append(entry)
        if entry['arxiv']:
            idx_arx[entry['arxiv']].append(entry)
        ty = title_year_key(entry['title'], entry['year'])
        if ty:
            idx_ty[ty].append(entry)
    return {'doi': idx_doi, 'arxiv': idx_arx, 'ty': idx_ty}


def pick_status(matches):
    if not matches:
        return None
    order = {'succeeded': 3, 'skipped': 2, 'attempted': 1, 'failed': 0}
    best = None
    best_score = -1
    for m in matches:
        s = (m.get('status') or '').lower()
        sc = order.get(s, -1)
        if sc > best_score:
            best_score = sc
            best = s
    return best or matches[0].get('status')


def pick_best_match_for_metadata(matches):
    """Pick best match row for metadata enrichment (prefer succeeded)."""
    if not matches:
        return None
    order = {'succeeded': 3, 'skipped': 2, 'attempted': 1, 'failed': 0}
    best = None
    best_score = -1
    for m in matches:
        s = (m.get('status') or '').lower()
        sc = order.get(s, -1)
        if sc > best_score:
            best_score = sc
            best = m
    return best or matches[0]


def choose_best_strict_metadata(matches, target_doi=None, target_arxiv=None):
    """Pick best strict-match metadata row for canonical correction."""
    if not matches:
        return None

    order = {'succeeded': 4, 'skipped': 3, 'attempted': 2, 'failed': 1}
    best = None
    best_key = None
    for m in matches:
        m_doi = ndoi(m.get('doi'))
        m_arx = narxiv(m.get('arxiv'))
        if target_doi and m_doi and m_doi != target_doi:
            continue
        if target_arxiv and m_arx and m_arx != target_arxiv:
            continue

        title = m.get('title') or ''
        year = m.get('year')
        status = (m.get('status') or '').lower()
        key = (
            order.get(status, 0),
            1 if year else 0,
            len(title),
        )
        if best_key is None or key > best_key:
            best = m
            best_key = key
    return best


def collect_matches(ref_canon, idx):
    doi = ndoi(ref_canon.get('doi'))
    arx = narxiv(ref_canon.get('arxiv_id'))
    ty = title_year_key(ref_canon.get('title'), ref_canon.get('year'))

    by_key = {'doi': [], 'arxiv': [], 'title_year': []}
    if doi and doi in idx['doi']:
        by_key['doi'].extend(idx['doi'][doi])
    if arx and arx in idx['arxiv']:
        by_key['arxiv'].extend(idx['arxiv'][arx])
    # Keep title+year fallback available, but do not mix it silently with strict keys
    if ty and ty in idx['ty']:
        by_key['title_year'].extend(idx['ty'][ty])

    all_found = by_key['doi'] + by_key['arxiv']
    # use fallback only when strict keys found nothing
    if not all_found:
        all_found = by_key['title_year']

    # de-dup by document_id
    uniq = {}
    for f in all_found:
        k = f.get('document_id') or f.get('title')
        uniq[k] = f
    all_unique = list(uniq.values())

    strict = by_key['doi'] + by_key['arxiv']
    uniq_strict = {}
    for f in strict:
        k = f.get('document_id') or f.get('title')
        uniq_strict[k] = f
    strict_unique = list(uniq_strict.values())

    return {
        'all': all_unique,
        'strict': strict_unique,
        'by_key_counts': {
            'doi': len(by_key['doi']),
            'arxiv': len(by_key['arxiv']),
            'title_year': len(by_key['title_year']),
        }
    }


def main():
    unique = load_jsonl(UNIQUE_REFS)

    r1_manifest = load_jsonl(R1_MANIFEST)
    r1_meta = load_r1_metadata(R1_META_DIR)
    r1_rows = merge_r1_manifest_with_metadata(r1_manifest, r1_meta)

    r2_rows = load_jsonl(R2_MANIFEST)
    r3_rows = load_jsonl(R3_MANIFEST)
    r3new_rows = load_jsonl(R3NEW_MANIFEST)

    idx_r1 = build_round_index(r1_rows, 'r1')
    idx_r2 = build_round_index(r2_rows, 'r2')
    idx_r3 = build_round_index(r3_rows, 'r3')
    idx_r3n = build_round_index(r3new_rows, 'r3new')

    out = []
    ctr = Counter()

    for row in unique:
        canon = row.get('canonical', {})
        m1 = collect_matches(canon, idx_r1)
        m2 = collect_matches(canon, idx_r2)
        m3 = collect_matches(canon, idx_r3)
        m3n = collect_matches(canon, idx_r3n)
        all_matches = m1['all'] + m2['all'] + m3['all'] + m3n['all']
        best_meta = pick_best_match_for_metadata(all_matches)
        strict_matches_all = m1['strict'] + m2['strict'] + m3['strict'] + m3n['strict']

        canonical_enriched = dict(canon)
        canonical_corrections = []
        ref_id = row.get('ref_id') or ''

        # Rule 1: strict ref_id identity is authoritative for DOI/arXiv.
        if ref_id.startswith('doi:'):
            doi_from_id = ref_id[4:]
            if ndoi(canonical_enriched.get('doi')) != doi_from_id:
                canonical_enriched['doi'] = doi_from_id
                canonical_corrections.append('enforced_doi_from_ref_id')
        elif ref_id.startswith('arxiv:'):
            arx_from_id = ref_id[6:]
            if narxiv(canonical_enriched.get('arxiv_id')) != arx_from_id:
                canonical_enriched['arxiv_id'] = arx_from_id
                canonical_corrections.append('enforced_arxiv_from_ref_id')

        strict_target_doi = ndoi(canonical_enriched.get('doi'))
        strict_target_arxiv = narxiv(canonical_enriched.get('arxiv_id'))
        best_strict_meta = choose_best_strict_metadata(
            strict_matches_all, target_doi=strict_target_doi, target_arxiv=strict_target_arxiv
        )

        if best_meta:
            # Enrich missing fields from sourced metadata when available
            if not canonical_enriched.get('title') and best_meta.get('title'):
                canonical_enriched['title'] = best_meta.get('title')
                canonical_corrections.append('filled_title_from_match')
            if not canonical_enriched.get('year') and best_meta.get('year'):
                canonical_enriched['year'] = best_meta.get('year')
                canonical_corrections.append('filled_year_from_match')
            if not canonical_enriched.get('doi') and best_meta.get('doi'):
                canonical_enriched['doi'] = best_meta.get('doi')
                canonical_corrections.append('filled_doi_from_match')
            if not canonical_enriched.get('arxiv_id') and best_meta.get('arxiv'):
                canonical_enriched['arxiv_id'] = best_meta.get('arxiv')
                canonical_corrections.append('filled_arxiv_from_match')

        # Rule 2: for strict matches, allow correcting title/year drift.
        if best_strict_meta:
            strict_title = best_strict_meta.get('title')
            strict_year = best_strict_meta.get('year')
            cur_title = canonical_enriched.get('title')
            cur_year = canonical_enriched.get('year')

            # If title similarity is low, trust strict metadata title.
            if strict_title and (not cur_title or title_sim(cur_title, strict_title) < 0.55):
                canonical_enriched['title'] = strict_title
                canonical_corrections.append('corrected_title_from_strict_match')

            # Correct year when clearly inconsistent.
            if strict_year:
                if not cur_year:
                    canonical_enriched['year'] = strict_year
                    canonical_corrections.append('filled_year_from_strict_match')
                else:
                    try:
                        if abs(int(cur_year) - int(strict_year)) >= 2:
                            canonical_enriched['year'] = strict_year
                            canonical_corrections.append('corrected_year_from_strict_match')
                    except Exception:
                        canonical_enriched['year'] = strict_year
                        canonical_corrections.append('corrected_year_from_strict_match')

        rec = {
            'ref_id': row.get('ref_id'),
            'canonical': canon,
            'canonical_enriched': canonical_enriched,
            'citation_count': row.get('citation_count', 0),
            'match_method': row.get('match_method'),
            'merged_from': row.get('merged_from'),
            'in_dataset': bool(row.get('in_dataset', False)),
            'external': not bool(row.get('in_dataset', False)),

            # relaxed = strict DOI/arXiv plus optional title+year fallback
            'sourced_in_r1': bool(m1['all']),
            'sourced_in_r2': bool(m2['all']),
            'sourced_in_r3': bool(m3['all']),
            'sourced_in_r3new': bool(m3n['all']),

            # strict = DOI/arXiv only
            'sourced_in_r1_strict': bool(m1['strict']),
            'sourced_in_r2_strict': bool(m2['strict']),
            'sourced_in_r3_strict': bool(m3['strict']),
            'sourced_in_r3new_strict': bool(m3n['strict']),

            'r1_status': pick_status(m1['all']),
            'r2_status': pick_status(m2['all']),
            'r3_status': pick_status(m3['all']),
            'r3new_status': pick_status(m3n['all']),

            'r1_status_strict': pick_status(m1['strict']),
            'r2_status_strict': pick_status(m2['strict']),
            'r3_status_strict': pick_status(m3['strict']),
            'r3new_status_strict': pick_status(m3n['strict']),

            'r1_doc_ids': sorted([m.get('document_id') for m in m1['all'] if m.get('document_id')]),
            'r2_doc_ids': sorted([m.get('document_id') for m in m2['all'] if m.get('document_id')]),
            'r3_doc_ids': sorted([m.get('document_id') for m in m3['all'] if m.get('document_id')]),
            'r3new_doc_ids': sorted([m.get('document_id') for m in m3n['all'] if m.get('document_id')]),

            'match_counts_by_round': {
                'r1': m1['by_key_counts'],
                'r2': m2['by_key_counts'],
                'r3': m3['by_key_counts'],
                'r3new': m3n['by_key_counts'],
            },
            'canonical_corrections': sorted(set(canonical_corrections)),
        }

        rec['downloaded_any_round'] = any(
            s == 'succeeded' for s in [rec['r1_status'], rec['r2_status'], rec['r3_status'], rec['r3new_status']]
        )
        rec['downloaded_any_round_strict'] = any(
            s == 'succeeded' for s in [rec['r1_status_strict'], rec['r2_status_strict'], rec['r3_status_strict'], rec['r3new_status_strict']]
        )
        rec['seen_in_any_round'] = any([
            rec['sourced_in_r1'], rec['sourced_in_r2'], rec['sourced_in_r3'], rec['sourced_in_r3new']
        ])
        rec['seen_in_any_round_strict'] = any([
            rec['sourced_in_r1_strict'], rec['sourced_in_r2_strict'], rec['sourced_in_r3_strict'], rec['sourced_in_r3new_strict']
        ])
        rec['in_dataset_effective'] = bool(rec['in_dataset'] or rec['seen_in_any_round'])
        rec['in_dataset_effective_strict'] = bool(rec['in_dataset'] or rec['seen_in_any_round_strict'])
        rec['external_effective'] = not rec['in_dataset_effective']
        rec['external_effective_strict'] = not rec['in_dataset_effective_strict']
        out.append(rec)

        ctr['total'] += 1
        ctr['external' if rec['external'] else 'in_dataset'] += 1
        ctr['external_effective' if rec['external_effective'] else 'in_dataset_effective'] += 1
        ctr['external_effective_strict' if rec['external_effective_strict'] else 'in_dataset_effective_strict'] += 1
        if rec['sourced_in_r1']:
            ctr['sourced_refs_in_r1_relaxed'] += 1
        if rec['sourced_in_r2']:
            ctr['sourced_refs_in_r2_relaxed'] += 1
        if rec['sourced_in_r3']:
            ctr['sourced_refs_in_r3_relaxed'] += 1
        if rec['sourced_in_r3new']:
            ctr['sourced_refs_in_r3new_relaxed'] += 1
        if rec['sourced_in_r1_strict']:
            ctr['sourced_refs_in_r1_strict'] += 1
        if rec['sourced_in_r2_strict']:
            ctr['sourced_refs_in_r2_strict'] += 1
        if rec['sourced_in_r3_strict']:
            ctr['sourced_refs_in_r3_strict'] += 1
        if rec['sourced_in_r3new_strict']:
            ctr['sourced_refs_in_r3new_strict'] += 1
        if rec['downloaded_any_round']:
            ctr['downloaded_refs_any_round_relaxed'] += 1
        if rec['downloaded_any_round_strict']:
            ctr['downloaded_refs_any_round_strict'] += 1
        if rec.get('canonical_corrections'):
            ctr['canonical_records_corrected'] += 1
            for c in rec['canonical_corrections']:
                ctr[f'correction_{c}'] += 1

    with OUT_JSONL.open('w') as f:
        for r in sorted(out, key=lambda x: -x.get('citation_count', 0)):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    summary = {
        'source_unique_references': str(UNIQUE_REFS),
        'output_registry': str(OUT_JSONL),
        'counts': dict(ctr),
        'inputs': {
            'r1_manifest': str(R1_MANIFEST),
            'r1_metadata_dir': str(R1_META_DIR),
            'r2_manifest': str(R2_MANIFEST),
            'r3_manifest': str(R3_MANIFEST),
            'r3new_manifest': str(R3NEW_MANIFEST),
        },
        'note': 'counts with sourced_* are reference-level counts, not document counts'
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + '\n')

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
