# References Folder

This folder stores source artifacts and extracted markdown used for planning research.

## Structure
- `raw/`: raw downloaded artifacts (`.html`, `.pdf`).
- `md/`: extracted markdown with metadata frontmatter (`key`, `url`, `retrieved_at_utc`, etc.).
- `manifest.json`: Python fetch pipeline status.
- `manifest.playwright.json`: Playwright manual-browser pipeline status.

## Pipelines
1. Python fast pipeline (threaded workers):
```bash
.venv/bin/python scripts/fetch_references.py
```

2. Playwright manual-browser pipeline (Firefox persistent profile in `references/.pw-firefox-profile`):
```bash
node scripts/fetch_references_playwright.js
```

## Notes
- Playwright pipeline is useful for pages that occasionally block plain HTTP clients.
- PDF text extraction is done with `pypdf` in `.venv`.
