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


## Development Phases

### Phase 1: MVP Development in Playground Repo ✅ COMPLETED
**Goal**: Build and test core functionality locally

**Repository Structure**:
```
20250831_one_click_paper_page/
├── frontend/                 # Vite + React/TypeScript app (run locally)
├── backend/                  # Python scripts and API (if needed)
├── template/                 # Template for user repos
├── scripts/                  # Conversion tools (Python)
└── docs/                     # Planning and documentation
```

**Development Tools & Standards**:
- **Frontend**: Vite + React + TypeScript + pnpm
- **Backend/Scripts**: Python + uv (venv/packages) + ruff (linting) + mypy (type checking)
- **No npm or pip** - use pnpm and uv respectively

**TODOs**:

**Setup & Tooling**:
- [x] Set up frontend with Vite (`pnpm create vite@latest frontend --template react-ts`)
- [x] Configure frontend linting and type checking:
  - [x] ESLint + Prettier configuration
  - [x] TypeScript strict mode
  - [x] Pre-commit hooks with lint-staged
- [x] Set up Python environment:
  - [x] Initialize with uv (`uv init backend`)
  - [x] Configure ruff for linting (`ruff.toml`)
  - [x] Configure mypy for type checking (`mypy.ini`)
  - [x] Set up pre-commit hooks for Python

**Core Development**:
- [x] Create template workflow structure (`template/.github/workflows/convert-and-deploy.yml`)
- [x] Build conversion scripts (Python + type hints):
  - [x] Docling integration (`scripts/docling_converter.py`)
  - [x] Marker integration (`scripts/marker_converter.py`)
  - [x] Pandoc fallback (`scripts/fallback_pandoc.py`)
- [x] Implement GitHub API integration (TypeScript):
  - [x] GitHub App authentication
  - [x] Repository creation from template
  - [x] File upload and commit functionality
- [x] Create basic UI components (React + TypeScript):
  - [x] File upload interface
  - [x] Template selection
  - [x] GitHub authentication flow
- [x] Test conversion pipeline locally with sample files
- [x] End-to-end test: create real user repo and trigger build
- [x] Fix Tailwind CSS v4 PostCSS integration issue
- [x] Fix TypeScript module import issues with type-only imports

### Phase 2: Integration Testing & Validation
**Goal**: Validate full user journey with real GitHub integration

**TODOs**:
- [ ] Test with multiple file formats (PDF, DOCX, LaTeX zip)
- [ ] Validate all three template options work correctly
- [ ] Test Overleaf integration (Git clone + PAT)
- [ ] Implement error handling and user feedback
- [ ] Add conversion quality validation
- [ ] Test GitHub Pages deployment from user repos
- [ ] Performance testing (build times, file size limits)
- [ ] Security review (token handling, file validation)

### Phase 3: Production Migration
**Goal**: Deploy as subpage of personal GitHub Pages site

**Migration Options**:
- **Option A**: Subpage of personal site (`username.github.io/paper-converter`)
- **Option B**: Dedicated repository (`paper-to-website.github.io`)

**TODOs**:
- [ ] Configure Next.js for static export with basePath
- [ ] Set up GitHub Actions for deployment
- [ ] Update all API endpoints for production URLs
- [ ] Add analytics and usage tracking
- [ ] Create user documentation and examples
- [ ] Set up monitoring and error reporting
- [ ] Beta testing with real users
- [ ] Performance optimization and caching

### Phase 4: Enhancement & Scale (Future)
**Goal**: Add advanced features and improve user experience

**TODOs**:
- [ ] Preview builds on temporary branches
- [ ] Custom theme editor
- [ ] Batch processing for multiple papers
- [ ] Integration with academic platforms (arXiv, ResearchGate)
- [ ] Advanced customization options
- [ ] Plugin system for custom conversion steps
- [ ] Analytics dashboard for users
- [ ] API for programmatic access

---

## Development Standards & Configuration

### Frontend (Vite + React + TypeScript)
**Package Manager**: pnpm only
```bash
# Setup
pnpm create vite@latest frontend --template react-ts
cd frontend
pnpm install

# Development
pnpm dev
pnpm build
pnpm lint
pnpm type-check
```

**Required Configuration Files**:
- `eslint.config.js` - ESLint with TypeScript rules
- `prettier.config.js` - Code formatting
- `tsconfig.json` - TypeScript strict mode
- `vite.config.ts` - Vite configuration
- `.gitignore` - Include node_modules, dist, .env

**Linting & Type Checking**:
```json
// package.json scripts
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "lint:fix": "eslint . --ext ts,tsx --fix",
    "type-check": "tsc --noEmit",
    "format": "prettier --write .",
    "format:check": "prettier --check ."
  }
}
```

### Backend/Scripts (Python)
**Package Manager**: uv only (no pip)
```bash
# Setup
uv init backend
cd backend
uv add docling marker-pdf pandas  # example dependencies
uv add --dev ruff mypy pytest

# Development
uv run python script.py
uv run ruff check .
uv run mypy .
```

**Required Configuration Files**:
- `pyproject.toml` - Project metadata and tool configuration
- `ruff.toml` or `pyproject.toml` - Ruff linting rules
- `mypy.ini` or `pyproject.toml` - MyPy type checking
- `.gitignore` - Include __pycache__, .venv, .mypy_cache

**Linting & Type Checking**:
```toml
# pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.ruff.isort]
known-first-party = ["scripts"]
```

### Pre-commit Hooks
**Setup for both frontend and backend**:
```bash
# Install pre-commit
uv add --dev pre-commit  # for Python projects
pnpm add -D lint-staged husky  # for frontend

# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ruff-check
        name: ruff-check
        entry: uv run ruff check --fix
        language: system
        types: [python]
      - id: mypy
        name: mypy
        entry: uv run mypy
        language: system
        types: [python]
      - id: eslint
        name: eslint
        entry: pnpm lint:fix
        language: system
        types: [typescript, tsx]
```
