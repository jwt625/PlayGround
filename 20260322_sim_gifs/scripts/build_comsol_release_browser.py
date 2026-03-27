#!/usr/bin/env python3
"""Generate a compact static browser for scraped COMSOL release videos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>COMSOL Release Video Browser</title>
  <style>
    :root {{
      --bg: #f2efe8;
      --panel: rgba(255,255,255,0.82);
      --panel-strong: rgba(255,255,255,0.94);
      --ink: #172129;
      --muted: #5f6d78;
      --line: rgba(23,33,41,0.12);
      --accent: #c4522f;
      --accent-soft: rgba(196,82,47,0.12);
      --accent-2: #1f6d78;
      --shadow: 0 20px 50px rgba(23,33,41,0.10);
      --radius: 18px;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(31,109,120,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(196,82,47,0.16), transparent 26%),
        linear-gradient(180deg, #f8f5ef 0%, #f2efe8 54%, #ebe7de 100%);
    }}

    .page {{
      width: min(1500px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }}

    .hero {{
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 26px;
      background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(250,244,236,0.92));
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}

    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 8px;
    }}

    h1 {{
      margin: 0;
      font-size: clamp(28px, 4vw, 48px);
      line-height: 1;
      letter-spacing: -0.04em;
    }}

    .subtitle {{
      margin: 12px 0 0;
      max-width: 820px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.5;
    }}

    .toolbar {{
      display: grid;
      gap: 14px;
      grid-template-columns: 1.3fr auto auto auto;
      margin-top: 24px;
      align-items: center;
    }}

    .control, .toggle {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      min-height: 48px;
      box-shadow: 0 10px 30px rgba(23,33,41,0.06);
    }}

    .control input, .control select {{
      width: 100%;
      height: 100%;
      border: 0;
      background: transparent;
      color: var(--ink);
      padding: 0 14px;
      font: inherit;
      outline: none;
    }}

    .toggle {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 14px;
      color: var(--muted);
      font-size: 14px;
    }}

    .toggle input {{
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }}

    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}

    .selection-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-top: 18px;
    }}

    .pill {{
      padding: 10px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}

    .button {{
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel-strong);
      color: var(--ink);
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
      box-shadow: 0 10px 24px rgba(23,33,41,0.05);
      transition: transform 120ms ease, background 120ms ease;
    }}

    .button:hover {{
      transform: translateY(-1px);
      background: #fff;
    }}

    .button.accent {{
      background: var(--accent);
      color: #fff;
      border-color: rgba(0,0,0,0.04);
    }}

    .button.ghost {{
      background: transparent;
    }}

    .version-nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 18px 0 0;
    }}

    .version-link {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 10px 14px;
      border-radius: 999px;
      color: var(--ink);
      text-decoration: none;
      font-size: 14px;
      transition: transform 120ms ease, background 120ms ease;
    }}

    .version-link:hover {{
      transform: translateY(-1px);
      background: var(--panel-strong);
    }}

    .release-section {{
      margin-top: 26px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(255,255,255,0.62);
      box-shadow: 0 12px 32px rgba(23,33,41,0.05);
    }}

    .release-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin-bottom: 14px;
    }}

    .release-title {{
      margin: 0;
      font-size: 24px;
      letter-spacing: -0.03em;
    }}

    .release-meta {{
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 14px;
    }}

    .card {{
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel-strong);
      overflow: hidden;
      box-shadow: 0 10px 24px rgba(23,33,41,0.05);
    }}

    .card.selected {{
      border-color: rgba(196,82,47,0.55);
      box-shadow: 0 16px 34px rgba(196,82,47,0.16);
    }}

    .player-wrap {{
      aspect-ratio: 16 / 9;
      background: linear-gradient(145deg, #d3d9db, #bcc6ca);
    }}

    video {{
      width: 100%;
      height: 100%;
      display: block;
      object-fit: cover;
      background: #c8d1d4;
    }}

    .card-body {{
      padding: 12px 12px 14px;
    }}

    .card-topline {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }}

    .pick {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      user-select: none;
    }}

    .pick input {{
      width: 16px;
      height: 16px;
      accent-color: var(--accent);
    }}

    .card-title {{
      margin: 0 0 8px;
      font-size: 14px;
      line-height: 1.35;
      font-weight: 700;
      word-break: break-word;
    }}

    .meta-line {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}

    .meta-line strong {{
      color: var(--accent-2);
      font-weight: 700;
    }}

    .hidden {{
      display: none !important;
    }}

    .empty {{
      margin-top: 18px;
      padding: 20px;
      border: 1px dashed var(--line);
      border-radius: 18px;
      color: var(--muted);
      background: rgba(255,255,255,0.54);
    }}

    .export-panel {{
      margin-top: 22px;
      border: 1px solid var(--line);
      border-radius: 22px;
      background: rgba(255,255,255,0.72);
      box-shadow: 0 12px 32px rgba(23,33,41,0.05);
      overflow: hidden;
    }}

    .export-head {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.52);
    }}

    .export-title {{
      margin: 0;
      font-size: 18px;
      letter-spacing: -0.02em;
    }}

    .export-meta {{
      color: var(--muted);
      font-size: 13px;
    }}

    .export-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .export-text {{
      width: 100%;
      min-height: 260px;
      border: 0;
      resize: vertical;
      padding: 16px 18px 18px;
      background: transparent;
      color: var(--ink);
      font: 12px/1.45 ui-monospace, "SFMono-Regular", Menlo, monospace;
      outline: none;
    }}

    @media (max-width: 900px) {{
      .toolbar {{
        grid-template-columns: 1fr;
      }}
      .release-header {{
        flex-direction: column;
        align-items: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="eyebrow">Review Browser</div>
      <h1>COMSOL Release Video Browser</h1>
      <p class="subtitle">
        Compact review grid for the scraped COMSOL release highlight videos. Filter by title, version, or short-duration montage candidates and play each clip directly in place.
      </p>

      <div class="toolbar">
        <label class="control">
          <input id="searchInput" type="search" placeholder="Search title, module, or release">
        </label>
        <label class="control">
          <select id="durationSelect">
            <option value="all">All durations</option>
            <option value="sub10" selected>Sub 10 s</option>
            <option value="sub30">Sub 30 s</option>
            <option value="gte30">30 s and up</option>
          </select>
        </label>
        <label class="toggle">
          <input id="onlyPlayable" type="checkbox" checked>
          <span>Only downloaded clips</span>
        </label>
        <label class="toggle">
          <input id="autopause" type="checkbox" checked>
          <span>Pause others on play</span>
        </label>
      </div>

      <div id="stats" class="stats"></div>
      <div class="selection-bar">
        <button id="selectVisibleBtn" class="button accent" type="button">Select visible</button>
        <button id="clearVisibleBtn" class="button" type="button">Unselect visible</button>
        <button id="clearAllBtn" class="button ghost" type="button">Clear all selections</button>
        <div id="selectionSummary" class="pill">0 selected</div>
      </div>
      <nav id="versionNav" class="version-nav"></nav>
    </section>

    <div id="emptyState" class="empty hidden">No videos match the current filters.</div>
    <div id="sections"></div>

    <section class="export-panel">
      <div class="export-head">
        <div>
          <h2 class="export-title">Selection Export</h2>
          <div id="exportMeta" class="export-meta">No clips selected yet.</div>
        </div>
        <div class="export-controls">
          <button id="exportTsvBtn" class="button" type="button">TSV</button>
          <button id="exportJsonBtn" class="button" type="button">JSON</button>
          <button id="copyExportBtn" class="button accent" type="button">Copy export</button>
        </div>
      </div>
      <textarea id="exportOutput" class="export-text" spellcheck="false" placeholder="Selected clips will appear here."></textarea>
    </section>
  </main>

  <script>
    const DATA = __DATA__;
    const STORAGE_KEY = 'comsol-release-browser-selection-v1';

    const searchInput = document.getElementById('searchInput');
    const durationSelect = document.getElementById('durationSelect');
    const onlyPlayable = document.getElementById('onlyPlayable');
    const autopause = document.getElementById('autopause');
    const selectVisibleBtn = document.getElementById('selectVisibleBtn');
    const clearVisibleBtn = document.getElementById('clearVisibleBtn');
    const clearAllBtn = document.getElementById('clearAllBtn');
    const selectionSummary = document.getElementById('selectionSummary');
    const exportMeta = document.getElementById('exportMeta');
    const exportOutput = document.getElementById('exportOutput');
    const exportTsvBtn = document.getElementById('exportTsvBtn');
    const exportJsonBtn = document.getElementById('exportJsonBtn');
    const copyExportBtn = document.getElementById('copyExportBtn');
    const sectionsEl = document.getElementById('sections');
    const statsEl = document.getElementById('stats');
    const emptyStateEl = document.getElementById('emptyState');
    const versionNavEl = document.getElementById('versionNav');
    let exportMode = 'tsv';
    let currentVisibleIds = [];
    let selectedIds = loadSelection();

    function loadSelection() {{
      try {{
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return new Set();
        return new Set(JSON.parse(raw));
      }} catch (error) {{
        return new Set();
      }}
    }}

    function persistSelection() {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify([...selectedIds]));
    }}

    function fmt(sec) {{
      return `${{sec.toFixed(3).replace(/\\.000$/, '')}} s`;
    }}

    function matchesDuration(duration, mode) {{
      if (mode === 'sub10') return duration < 10;
      if (mode === 'sub30') return duration < 30;
      if (mode === 'gte30') return duration >= 30;
      return true;
    }}

    function cardMarkup(item) {{
      const pageLabel = item.page_slug.replace(/-/g, ' ');
      const title = item.name || item.media_id;
      const checked = selectedIds.has(item.id) ? 'checked' : '';
      const selectedClass = selectedIds.has(item.id) ? 'selected' : '';
      return `
        <article class="card ${{selectedClass}}" data-card data-id="${{item.id}}">
          <div class="player-wrap">
            <video controls preload="metadata" src="${{item.src}}" title="${{title}}"></video>
          </div>
          <div class="card-body">
            <div class="card-topline">
              <label class="pick">
                <input class="pick-input" type="checkbox" data-pick-id="${{item.id}}" ${{checked}}>
                <span>Pick</span>
              </label>
            </div>
            <h3 class="card-title">${{title}}</h3>
            <div class="meta-line"><strong>${{item.release}}</strong> · ${{fmt(item.duration)}}</div>
            <div class="meta-line">${{pageLabel}}</div>
            <div class="meta-line">${{item.media_id}}</div>
          </div>
        </article>
      `;
    }}

    function buildVersionNav(releases) {{
      versionNavEl.innerHTML = releases.map(release =>
        `<a class="version-link" href="#release-${{release.replace('.', '-')}}">${{release}}</a>`
      ).join('');
    }}

    function allItems() {{
      return DATA.releases.flatMap(section => section.items);
    }}

    function selectedItems() {{
      const byId = new Map(allItems().map(item => [item.id, item]));
      return [...selectedIds].map(id => byId.get(id)).filter(Boolean);
    }}

    function tsvFor(items) {{
      const head = ['id', 'release', 'page_slug', 'duration', 'name', 'media_id', 'src'];
      const lines = [head.join('\\t')];
      items.forEach(item => {{
        lines.push([
          item.id,
          item.release,
          item.page_slug,
          item.duration,
          item.name,
          item.media_id,
          item.src || ''
        ].map(value => String(value).replace(/\\t|\\n/g, ' ')).join('\\t'));
      }});
      return lines.join('\\n');
    }}

    function updateExport() {{
      const items = selectedItems().sort((a, b) =>
        a.release.localeCompare(b.release, undefined, {{ numeric: true }}) ||
        a.page_slug.localeCompare(b.page_slug) ||
        a.duration - b.duration ||
        a.name.localeCompare(b.name)
      );
      selectionSummary.textContent = `${{items.length}} selected`;
      exportMeta.textContent = items.length
        ? `${{items.length}} clip${{items.length === 1 ? '' : 's'}} ready for export`
        : 'No clips selected yet.';
      exportOutput.value = exportMode === 'json'
        ? JSON.stringify(items, null, 2)
        : tsvFor(items);
    }}

    function attachCardHandlers() {{
      document.querySelectorAll('.pick-input').forEach(input => {{
        input.addEventListener('change', event => {{
          const id = event.target.dataset.pickId;
          if (event.target.checked) {{
            selectedIds.add(id);
          }} else {{
            selectedIds.delete(id);
          }}
          persistSelection();
          const card = event.target.closest('.card');
          if (card) card.classList.toggle('selected', event.target.checked);
          updateExport();
        }});
      }});
    }}

    function attachVideoHandlers() {{
      const videos = Array.from(document.querySelectorAll('video'));
      videos.forEach(video => {{
        video.addEventListener('play', () => {{
          if (!autopause.checked) return;
          videos.forEach(other => {{
            if (other !== video) other.pause();
          }});
        }});
      }});
    }}

    function mutateVisibleSelection(checked) {{
      currentVisibleIds.forEach(id => {{
        if (checked) {{
          selectedIds.add(id);
        }} else {{
          selectedIds.delete(id);
        }}
      }});
      persistSelection();
      render();
    }}

    function render() {{
      const query = searchInput.value.trim().toLowerCase();
      const durationMode = durationSelect.value;
      const requirePlayable = onlyPlayable.checked;
      const releases = [];
      let totalVisible = 0;
      let totalSections = 0;
      currentVisibleIds = [];

      const sectionsHtml = DATA.releases.map(section => {{
        const items = section.items.filter(item => {{
          if (requirePlayable && !item.src) return false;
          if (!matchesDuration(item.duration, durationMode)) return false;
          if (!query) return true;
          const haystack = `${{item.release}} ${{item.page_slug}} ${{item.name}} ${{item.media_id}}`.toLowerCase();
          return haystack.includes(query);
        }});

        if (!items.length) return '';
        releases.push(section.release);
        totalVisible += items.length;
        totalSections += 1;
        currentVisibleIds.push(...items.map(item => item.id));
        return `
          <section class="release-section" id="release-${{section.release.replace('.', '-')}}">
            <div class="release-header">
              <h2 class="release-title">Version ${{section.release}}</h2>
              <div class="release-meta">${{items.length}} clips shown</div>
            </div>
            <div class="grid">
              ${{items.map(cardMarkup).join('')}}
            </div>
          </section>
        `;
      }}).join('');

      sectionsEl.innerHTML = sectionsHtml;
      emptyStateEl.classList.toggle('hidden', totalVisible !== 0);

      buildVersionNav(releases);

      const downloaded = DATA.releases.flatMap(r => r.items).filter(item => item.src).length;
      const shortCount = DATA.releases.flatMap(r => r.items).filter(item => item.duration < 10).length;
      statsEl.innerHTML = [
        `<div class="pill">${{DATA.total_clips}} total clips</div>`,
        `<div class="pill">${{downloaded}} downloaded</div>`,
        `<div class="pill">${{shortCount}} under 10 s</div>`,
        `<div class="pill">${{totalVisible}} currently shown</div>`,
        `<div class="pill">${{totalSections}} versions visible</div>`
      ].join('');

      attachCardHandlers();
      attachVideoHandlers();
      updateExport();
    }}

    [searchInput, durationSelect, onlyPlayable, autopause].forEach(el => {{
      el.addEventListener('input', render);
      el.addEventListener('change', render);
    }});

    selectVisibleBtn.addEventListener('click', () => mutateVisibleSelection(true));
    clearVisibleBtn.addEventListener('click', () => mutateVisibleSelection(false));
    clearAllBtn.addEventListener('click', () => {{
      selectedIds = new Set();
      persistSelection();
      render();
    }});

    exportTsvBtn.addEventListener('click', () => {{
      exportMode = 'tsv';
      updateExport();
    }});

    exportJsonBtn.addEventListener('click', () => {{
      exportMode = 'json';
      updateExport();
    }});

    copyExportBtn.addEventListener('click', async () => {{
      try {{
        await navigator.clipboard.writeText(exportOutput.value);
        copyExportBtn.textContent = 'Copied';
        setTimeout(() => {{
          copyExportBtn.textContent = 'Copy export';
        }}, 1200);
      }} catch (error) {{
        copyExportBtn.textContent = 'Copy failed';
        setTimeout(() => {{
          copyExportBtn.textContent = 'Copy export';
        }}, 1200);
      }}
    }});

    render();
  </script>
</body>
</html>
"""


def collect_data(index_dir: Path, output_dir: Path) -> dict:
    releases = []
    total = 0
    for path in sorted(index_dir.glob("release_*.json"), key=lambda p: [int(x) for x in p.stem.split("_", 1)[1].split(".")]):
        data = json.loads(path.read_text())
        items = []
        for page in data["pages"]:
            for media in page["media"]:
                asset = media.get("best_asset") or {}
                saved_path = asset.get("saved_path")
                rel_src = None
                if saved_path:
                    rel_src = os.path.relpath(Path(saved_path).resolve(), output_dir.resolve())
                    rel_src = Path(rel_src).as_posix()
                duration = media.get("duration")
                if duration is None:
                    continue
                items.append(
                    {
                        "id": f"{data['release']}::{page['page_slug']}::{media.get('media_id')}",
                        "release": data["release"],
                        "page_slug": page["page_slug"],
                        "name": media.get("name") or media.get("media_id"),
                        "media_id": media.get("media_id"),
                        "duration": float(duration),
                        "src": rel_src,
                    }
                )
        items.sort(key=lambda item: (item["duration"], item["page_slug"], item["name"]))
        total += len(items)
        releases.append({"release": data["release"], "items": items})
    return {"total_clips": total, "releases": releases}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="build/comsol_release_scrape/indexes")
    parser.add_argument("--output", default="build/comsol_release_browser/index.html")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = collect_data(Path(args.index_dir), output_path.parent)
    html = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}").replace("__DATA__", json.dumps(data))
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Embedded {data['total_clips']} clips across {len(data['releases'])} releases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
