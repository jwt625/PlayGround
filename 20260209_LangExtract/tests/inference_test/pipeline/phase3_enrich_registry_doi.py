#!/usr/bin/env python3
import json
import re
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REG = ROOT / 'tests/inference_test/output/canonical_reference_registry.jsonl'
OUT_CAND = ROOT / 'tests/inference_test/output/relaxed_only_doi_candidates.jsonl'
OUT_SUM = ROOT / 'tests/inference_test/output/relaxed_only_doi_candidates_summary.json'
OUT_REG_ENRICH = ROOT / 'tests/inference_test/output/canonical_reference_registry_enriched.jsonl'
UA = 'LangExtract-DOI-Enricher/1.0'


def ntitle(s):
    s = (s or '').lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def sim(a, b):
    return SequenceMatcher(None, ntitle(a), ntitle(b)).ratio()


def crossref_query(title, year=None, rows=5):
    q = urllib.parse.quote(title)
    url = f'https://api.crossref.org/works?query.title={q}&rows={rows}'
    if year:
        url += f'&filter=from-pub-date:{year-1},until-pub-date:{year+1}'
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode('utf-8', 'replace'))


def pick_best_crossref(cands, title, year):
    best = None
    bs = -1
    for it in cands:
        t = (it.get('title') or [''])[0] if isinstance(it.get('title'), list) else (it.get('title') or '')
        doi = it.get('DOI')
        if not doi:
            continue
        yr = None
        for k in ('issued', 'published-print', 'published-online', 'created'):
            d = it.get(k) or {}
            dp = d.get('date-parts') if isinstance(d, dict) else None
            if dp and dp[0] and dp[0][0]:
                yr = int(dp[0][0]); break
        s = sim(title, t)
        if year and yr:
            if abs(year - yr) > 2:
                s -= 0.15
        score = s
        if score > bs:
            bs = score
            best = {'doi': doi.lower(), 'title': t, 'year': yr, 'score': round(score, 4), 'source': 'crossref'}
    return best


records = []
for line in REG.open():
    r = json.loads(line)
    any_rel = r['sourced_in_r1'] or r['sourced_in_r2'] or r['sourced_in_r3'] or r['sourced_in_r3new']
    any_str = r['sourced_in_r1_strict'] or r['sourced_in_r2_strict'] or r['sourced_in_r3_strict'] or r['sourced_in_r3new_strict']
    doi = (r.get('canonical') or {}).get('doi')
    if any_rel and not any_str and not doi:
        records.append(r)

resolved = {}
out_rows = []
errors = 0
for i, r in enumerate(records, 1):
    title = (r.get('canonical') or {}).get('title') or (r.get('canonical_enriched') or {}).get('title') or ''
    year = (r.get('canonical') or {}).get('year') or (r.get('canonical_enriched') or {}).get('year')
    best = None
    try:
        cr = crossref_query(title, year=year, rows=8)
        best_cr = pick_best_crossref((cr.get('message') or {}).get('items') or [], title, year)
        if best_cr and best_cr['score'] >= 0.78:
            best = best_cr
    except Exception:
        errors += 1

    row = {
        'ref_id': r.get('ref_id'),
        'citation_count': r.get('citation_count', 0),
        'title': title,
        'year': year,
        'found_doi': (best or {}).get('doi'),
        'match_score': (best or {}).get('score'),
        'match_source': (best or {}).get('source'),
        'matched_title': (best or {}).get('title'),
        'matched_year': (best or {}).get('year'),
    }
    out_rows.append(row)
    if best:
        resolved[r.get('ref_id')] = best

    if i % 25 == 0:
        print(f'processed {i}/{len(records)} resolved={len(resolved)}', flush=True)
    time.sleep(0.08)

with OUT_CAND.open('w') as f:
    for row in sorted(out_rows, key=lambda x: -(x.get('citation_count') or 0)):
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

# create enriched registry copy
with REG.open() as fin, OUT_REG_ENRICH.open('w') as fout:
    for line in fin:
        r = json.loads(line)
        m = resolved.get(r.get('ref_id'))
        if m:
            r['doi_inferred'] = m['doi']
            r['doi_inferred_source'] = m['source']
            r['doi_inferred_score'] = m['score']
            ce = r.get('canonical_enriched') or {}
            if not ce.get('doi'):
                ce['doi'] = m['doi']
            r['canonical_enriched'] = ce
        fout.write(json.dumps(r, ensure_ascii=False) + '\n')

summary = {
    'target_relaxed_only_missing_doi': len(records),
    'resolved_with_confident_doi': len(resolved),
    'unresolved': len(records) - len(resolved),
    'query_errors': errors,
    'input_registry': str(REG),
    'output_candidates': str(OUT_CAND),
    'output_enriched_registry': str(OUT_REG_ENRICH),
}
OUT_SUM.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
