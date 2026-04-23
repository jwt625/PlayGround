# Kizuna

Kizuna is a local-first personal relationship management system for people, organizations, interactions, reminders, and lightweight pipelines.

## Requirements

- Python 3.12+
- `uv`
- Docker
- Node `^20.19`, `^22.12`, or `>=24`
- `pnpm`

Use the version in `.nvmrc` for the smoothest frontend setup. The repo does not enforce engines strictly because odd Node releases such as `23.x` may still run locally even when some packages only declare LTS support.

## Local Development

The normal development loop uses Docker only for Postgres. Run the backend and frontend directly from your shell so both support hot reload.

Start the database:

```sh
docker compose up -d postgres
```

Run the backend directly:

```sh
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

Run the frontend directly:

```sh
cd frontend
pnpm install
pnpm dev
```

Development URLs:

- Backend API: `http://localhost:8000`
- Backend health check: `http://localhost:8000/health`
- Frontend dev server: `http://localhost:5173`

You do not need `pnpm build` for local development. Use `pnpm build` only when checking a production bundle.

Useful backend checks:

```sh
cd backend
uv run pytest
uv run ruff check .
uv run mypy app tests
```

Useful frontend checks:

```sh
cd frontend
pnpm check
pnpm lint
pnpm format:check
```

Or run all frontend checks together:

```sh
pnpm check:all
```
