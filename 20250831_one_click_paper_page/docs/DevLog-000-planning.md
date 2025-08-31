# Academic Paper to Website Service – Plan

Time: Sun Aug 31 01:35:58 PDT 2025
Author: Wentao & GPT-5

## Overview
A lightweight web service enabling users to one-click (or very few clicks) convert their academic papers into free static websites hosted on GitHub Pages or Cloudflare Pages.

Supported input:
- Overleaf project (via Git clone + Personal Access Token)
- LaTeX zip file
- Word (.docx) or PDF

Output:
- Clean static website (HTML + assets), with optional PDF.

---

## Conversion Engine

### Primary
- **Docling**: Efficient DOCX/PDF/HTML → structured HTML/Markdown.
- **Marker**: High-quality PDF/DOCX → Markdown/HTML with math, tables, images. Runs on CPU/MPS/GPU.

### Fallbacks
- **Pandoc**: general-purpose DOCX/LaTeX → HTML/PDF.
- **LaTeXML/tex4ht**: for edge cases where Docling/Marker fail.
- **Tectonic**: minimal TeX engine for PDF generation if needed.

### Notes
- GitHub-hosted runners have no GPU, but Docling/Marker work on CPU.
- Cache models and dependencies between runs to reduce build time.

---

## Scale & Execution

- Expected usage: **~1 build/hour** (low volume).
- Builds & conversions happen in **GitHub Actions workflows** inside the user’s repository.
- **GitHub Pages** serves static artifacts only; it does not execute builds.

---

## Hosting Options

### GitHub Pages
- **Limits**:  
  - 1 GB site size  
  - 100 GB/month soft bandwidth cap  
  - 10 builds/hour soft limit  
- Best for: personal academic sites with minimal traffic.

### Cloudflare Pages
- **Not just DNS**: full static site hosting + build system.  
- **Free Plan Limits**:  
  - 500 builds/month  
  - 20-min build timeout  
  - Unlimited static asset requests  
  - 20,000 files / 25 MB per file  
  - Optional Functions (100k requests/day on free tier)  
- Best for: unlimited traffic hosting with global CDN.

---

## Architecture

1. **Auth & Repo Provisioning**
   - User signs in with GitHub.
   - GitHub App creates a new repo with minimal permissions.
   - Pages enabled; initial workflow committed.

2. **Ingestion**
   - **Overleaf**: user provides Git URL + PAT; workflow clones from Overleaf during build.
   - **Zip/DOCX/PDF**: uploaded via frontend; committed to repo.

3. **Build**
   - GitHub Actions workflow runs Docling/Marker.
   - Fallback to Pandoc/LaTeXML if needed.
   - Generates `/dist/index.html` and assets.
   - Deploys via `actions/deploy-pages` (GitHub Pages) or Cloudflare Pages.

4. **Hosting**
   - Default: GitHub Pages (simpler for academics).
   - Alternative: Cloudflare Pages (unlimited static bandwidth, CDN performance).

---

## Updated Workflow (convert-and-deploy.yml)

- Detect input type (DOCX/PDF/zip/Overleaf).
- Run Docling/Marker conversion.
- Insert theme/template.
- Deploy artifacts to hosting provider.

---

## Security
- Overleaf tokens stored as GitHub Secrets.
- No persistent server needed if using only GitHub Actions/Pages.
- Optional tiny backend (OCI Always Free) for orchestration.

---

## Template Options for Final Site

Offer users a choice of ready-made GitHub Pages themes:

- **Academic Pages (Jekyll)** – full academic personal site with publications, talks, CV, portfolio.  
- **Academic-project-page-template (JS)** – streamlined project/paper presentation page.  
- **al-folio (Jekyll)** – clean, responsive minimal academic landing page.

Users pick their preferred template during onboarding; the system copies it into their new repo. Allow later theme switching via configuration.


## Next Steps
1. Build a **template repo** with:
   - `convert-and-deploy.yml` workflow.
   - Basic themes (Paper, Minimal).
2. Build a **frontend** (Next.js) that:
   - Authenticates with GitHub.
   - Creates a repo from template.
   - Commits uploaded files or Overleaf config.
3. Add **preview builds** to temporary branches.
4. Implement **fallback switch** for Pandoc/LaTeXML.
