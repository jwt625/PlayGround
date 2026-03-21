#!/usr/bin/env python3
"""Check for missing topics in the OFC metadata."""

import json

data = json.load(open('output/full_metadata/ofc_full_metadata.json'))
presentations = [r for r in data if r['record_kind'] == 'presentation']

print("Searching for potentially missing topics...\n")

# Search for various topics
search_terms = {
    'CPO/co-packaged': ['cpo', 'co-packaged', 'copackaged', 'co packaged'],
    'pluggable': ['pluggable'],
    'transceiver': ['transceiver'],
    'interconnect': ['interconnect'],
    'chiplet': ['chiplet'],
    'neuromorphic': ['neuromorphic'],
    'photonic computing': ['photonic computing', 'optical computing'],
    'microring': ['microring', 'micro-ring', 'ring resonator'],
    'AWG': ['awg', 'arrayed waveguide'],
    'MZI': ['mzi', 'mach-zehnder interferometer'],
    'EIC': ['eic', 'electronic-photonic'],
    'TSV': ['tsv', 'through-silicon via'],
}

results = {}
for topic, patterns in search_terms.items():
    matches = []
    for pres in presentations:
        text = f"{pres.get('session_title', '')} {pres.get('presentation_title', '')} {pres.get('abstract_text', '')}".lower()
        if any(pattern in text for pattern in patterns):
            matches.append(pres)
    results[topic] = matches

# Print summary
for topic, matches in results.items():
    print(f"{topic}: {len(matches)} presentations")

# Show CPO examples
print("\n" + "="*80)
print("CPO/Co-packaged Optics Examples:")
print("="*80)
for i, pres in enumerate(results['CPO/co-packaged'][:5]):
    print(f"\n{i+1}. {pres['presentation_code']}")
    print(f"   Session: {pres['session_title']}")
    print(f"   Title: {pres['presentation_title']}")
    print(f"   Abstract: {pres['abstract_text'][:200]}...")

# Show session titles
print("\n" + "="*80)
print("Unique session titles (first 30):")
print("="*80)
sessions = set()
for pres in presentations:
    if pres.get('session_title'):
        sessions.add(pres['session_title'])

for i, session in enumerate(sorted(sessions)[:30]):
    print(f"{i+1}. {session}")

