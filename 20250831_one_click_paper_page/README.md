# Academic Paper to Website Converter

A comprehensive service that automatically converts academic papers into professional websites and deploys them to GitHub Pages. The system supports multiple input formats including PDF, DOCX, LaTeX, ZIP archives, and Overleaf projects.

## Overview

This project provides a complete pipeline for transforming academic papers into beautiful, responsive websites. Users can upload their papers through a modern web interface, select from professional academic templates, and have their content automatically converted and deployed to GitHub Pages.

### Key Features

- **Multi-format Support**: PDF, DOCX, LaTeX (.tex), ZIP archives, and Overleaf projects
- **Professional Templates**: Three curated academic website templates
- **Automatic Conversion**: High-quality document processing using Docling, Marker, and Pandoc
- **GitHub Integration**: Seamless repository creation and GitHub Pages deployment
- **Modern Interface**: React-based frontend with step-by-step workflow
- **Type Safety**: Full TypeScript implementation with comprehensive type definitions

## Architecture

The system consists of three main components:

1. **Frontend**: React + TypeScript application with Tailwind CSS styling
2. **Backend**: Python-based conversion pipeline using uv for dependency management
3. **Template System**: GitHub Actions workflows for automated deployment

## Prerequisites

### System Dependencies

- **Node.js** (v18 or higher)
- **pnpm** (package manager)
- **Python** (3.11 or higher)
- **uv** (Python package manager)
- **Pandoc** (for document conversion)
- **LaTeX** (for LaTeX document processing)

### Installation Commands

```bash
# Install Node.js dependencies
brew install node pnpm

# Install Python and uv
brew install python@3.11 uv

# Install Pandoc and LaTeX
brew install pandoc
brew install --cask mactex  # or brew install texlive
```

## Setup Instructions

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd 20250831_one_click_paper_page
```

### 2. Frontend Setup

```bash
cd frontend
pnpm install
```

### 3. Backend Setup

```bash
cd ../backend
uv sync
```

### 4. Environment Configuration

Create environment files for GitHub integration:

```bash
# Frontend environment
cp frontend/.env.example frontend/.env.local
```

Edit `frontend/.env.local`:
```
VITE_GITHUB_CLIENT_ID=your_github_app_client_id
```

## Running the Application

### Development Mode

Start both frontend and backend services:

```bash
# Terminal 1: Frontend development server
cd frontend
pnpm dev

# Terminal 2: Backend services (when implemented)
cd backend
uv run python main.py
```

The frontend will be available at `http://localhost:5173/`

### Production Build

```bash
cd frontend
pnpm build
pnpm preview
```

## Project Structure

```
20250831_one_click_paper_page/
├── frontend/                    # React + TypeScript frontend
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── auth/          # GitHub authentication
│   │   │   ├── upload/        # File upload interface
│   │   │   └── templates/     # Template selection
│   │   ├── lib/               # Core libraries
│   │   │   └── github/        # GitHub API integration
│   │   └── types/             # TypeScript definitions
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
├── backend/                     # Python backend
│   ├── src/                   # Source code
│   ├── pyproject.toml         # Python dependencies
│   └── test_sample.tex        # Sample test file
├── scripts/                     # Conversion scripts
│   ├── docling_converter.py   # Docling integration
│   ├── marker_converter.py    # Marker PDF converter
│   └── fallback_pandoc.py     # Pandoc fallback
├── template/                    # GitHub Actions templates
│   └── .github/
│       ├── workflows/         # Deployment workflows
│       └── scripts/           # Conversion scripts
└── docs/                       # Documentation
    └── DevLog-000-planning.md # Project planning
```

## Conversion Pipeline

The system uses a multi-tier conversion approach:

1. **Primary**: Docling for high-quality PDF/DOCX extraction
2. **Secondary**: Marker for advanced PDF to Markdown conversion
3. **Fallback**: Pandoc for various document formats

### Supported Input Formats

- **PDF**: Research papers, preprints, published articles
- **DOCX**: Microsoft Word documents
- **LaTeX**: Source files (.tex) with full project support
- **ZIP**: Compressed LaTeX projects or document bundles
- **Overleaf**: Direct integration with Overleaf projects

## Available Templates

### 1. Academic Pages (Jekyll)
- Full academic personal website
- Publications, talks, CV sections
- Blog functionality and portfolio showcase

### 2. Academic Project Page (JavaScript)
- Streamlined project presentation
- Results visualization and methodology sections
- Code and data integration

### 3. al-folio (Jekyll)
- Clean, minimal academic landing page
- Responsive design with dark mode support
- Publication lists and project showcases

## GitHub Integration

The system integrates with GitHub to:

- Authenticate users via OAuth
- Create repositories automatically
- Upload converted content and assets
- Configure GitHub Pages deployment
- Trigger conversion workflows

### Required GitHub App Permissions

- `repo`: Repository creation and management
- `user:email`: User identification
- `pages`: GitHub Pages configuration

## Development

### Code Quality

The project maintains high code quality standards:

- **TypeScript**: Full type safety in frontend
- **ESLint + Prettier**: Code formatting and linting
- **Ruff + MyPy**: Python code quality and type checking

### Testing

```bash
# Frontend tests
cd frontend
pnpm test

# Backend tests
cd backend
uv run pytest
```

### Linting and Formatting

```bash
# Frontend
cd frontend
pnpm lint
pnpm format

# Backend
cd backend
uv run ruff check
uv run mypy .
```

## Deployment

### GitHub Actions Workflow

The system automatically creates GitHub Actions workflows in user repositories that:

1. Detect input file types
2. Run appropriate conversion tools
3. Apply selected themes
4. Deploy to GitHub Pages

### Manual Deployment

For manual deployment of the service itself:

1. Build the frontend: `pnpm build`
2. Deploy to your preferred hosting service
3. Configure environment variables
4. Set up GitHub App credentials

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Ensure code quality checks pass
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:

1. Check the documentation in `/docs`
2. Review existing GitHub issues
3. Create a new issue with detailed information

## Acknowledgments

This project uses several open-source tools:

- **Docling**: High-quality document processing
- **Marker**: Advanced PDF to Markdown conversion
- **Pandoc**: Universal document converter
- **React**: Frontend framework
- **Tailwind CSS**: Utility-first CSS framework
