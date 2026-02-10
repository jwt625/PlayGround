#!/usr/bin/env python3
"""
Audit OFS-extracted papers against existing dataset.
Compare papers extracted from OFS blog posts with PDFs already in the dataset.
"""

import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def normalize_url(url: str) -> str:
    """Normalize URL for comparison."""
    if not url:
        return ""
    # Remove protocol
    url = re.sub(r'^https?://', '', url)
    # Remove www.
    url = re.sub(r'^www\.', '', url)
    # Remove trailing slash
    url = url.rstrip('/')
    # Remove query params for arxiv
    if 'arxiv.org' in url:
        # Extract just the arxiv ID
        match = re.search(r'(\d{4}\.\d{4,5})', url)
        if match:
            return f"arxiv:{match.group(1)}"
    return url.lower()


def extract_arxiv_id(url: str) -> Optional[str]:
    """Extract arXiv ID from URL if present."""
    if not url:
        return None
    match = re.search(r'(\d{4}\.\d{4,5})', url)
    return match.group(1) if match else None


def load_ofs_papers(jsonl_path: Path) -> list[dict]:
    """Load all papers from OFS extractions."""
    papers = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get('papers'):
                for paper in entry['papers']:
                    paper['source_ofs'] = entry['filename']
                    papers.append(paper)
    return papers


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load existing manifest entries."""
    entries = []
    with open(manifest_path) as f:
        for line in f:
            if not line.strip():
                continue
            entries.append(json.loads(line))
    return entries


def main():
    base_dir = Path(__file__).parent.parent.parent
    ofs_path = base_dir / "tests/inference_test/output/ofs_extractions.jsonl"
    manifest_path = base_dir / "semiconductor_processing_dataset/processed_documents/metadata/manifest_documents.jsonl"
    
    print("=" * 80)
    print("OFS Papers vs Existing Dataset Audit")
    print("=" * 80)
    
    # Load OFS papers
    ofs_papers = load_ofs_papers(ofs_path)
    print(f"\nTotal papers extracted from OFS: {len(ofs_papers)}")
    
    # Load manifest
    manifest_entries = load_manifest(manifest_path)
    print(f"Total entries in manifest: {len(manifest_entries)}")
    
    # Build lookup sets from manifest
    manifest_urls = set()
    manifest_arxiv_ids = set()
    manifest_titles = set()
    
    for entry in manifest_entries:
        url = entry.get('url', '')
        if url:
            manifest_urls.add(normalize_url(url))
            arxiv_id = extract_arxiv_id(url)
            if arxiv_id:
                manifest_arxiv_ids.add(arxiv_id)
        title = entry.get('title', '').lower().strip()
        if title:
            manifest_titles.add(title)
    
    # Compare
    matched = []
    new_papers = []
    
    for paper in ofs_papers:
        url = paper.get('url', '')
        title = paper.get('title', '').lower().strip()
        arxiv_id = extract_arxiv_id(url)
        
        is_match = False
        match_reason = None
        
        if arxiv_id and arxiv_id in manifest_arxiv_ids:
            is_match = True
            match_reason = f"arxiv:{arxiv_id}"
        elif normalize_url(url) in manifest_urls:
            is_match = True
            match_reason = "url"
        elif title and title in manifest_titles:
            is_match = True
            match_reason = "title"
        
        if is_match:
            matched.append((paper, match_reason))
        else:
            new_papers.append(paper)
    
    print(f"\nMatched (already in dataset): {len(matched)}")
    print(f"New papers (not in dataset): {len(new_papers)}")
    
    print("\n" + "=" * 80)
    print("MATCHED PAPERS")
    print("=" * 80)
    for paper, reason in matched:
        print(f"  [{reason}] {paper.get('title', 'No title')[:70]}")
    
    print("\n" + "=" * 80)
    print("NEW PAPERS TO DOWNLOAD")
    print("=" * 80)
    for i, paper in enumerate(new_papers, 1):
        print(f"\n{i}. {paper.get('title', 'No title')}")
        print(f"   URL: {paper.get('url', 'N/A')}")
        print(f"   Year: {paper.get('year', 'N/A')}")
        print(f"   Source: {paper.get('source_ofs', 'N/A')}")
    
    # Save new papers to JSON for download script
    output_path = base_dir / "tests/inference_test/output/new_papers_to_download.json"
    with open(output_path, 'w') as f:
        json.dump(new_papers, f, indent=2)
    print(f"\n\nSaved {len(new_papers)} new papers to: {output_path}")


if __name__ == "__main__":
    main()

