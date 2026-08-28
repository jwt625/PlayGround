# CQD equipment catalog

Static, frontend-only catalog built from `source/CQD-Daily-responsibility-August21-2026.pdf`.

The source PDF and `extraction/` outputs are working files, not public assets. The source contains lab layouts, hazardous-gas procedures, internal contact details, and chemical inventory. Do not publish those directories with the catalog.

## Run locally

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

Build the publishable bundle with:

```bash
node scripts/build_static.mjs
```

Point GitHub Pages (or a Pages deployment action) at `dist/`. It contains only the site and curated equipment photos.

## Comments and inquiries

The current implementation opens a prefilled issue in `jwt625/cqd-equipment`, so it works on GitHub Pages without a server. It is public and requires a GitHub account.

For lower-friction private responses, replace the issue link with Formspree, Basin, or a Google Form. Those services store submissions, so they avoid a custom backend but introduce a third-party data processor.

## Regenerate source images

```bash
uv run --with pymupdf python scripts/extract_pdf_assets.py
uvx --from docling docling convert source/CQD-Daily-responsibility-August21-2026.pdf \
  --from pdf --to md --to json --image-export-mode referenced --output extraction
```

The public catalog intentionally excludes chemical inventory and toxic-gas handling details. All listings should be treated as unverified until the institution confirms ownership, availability, completeness, condition, and removal requirements.
