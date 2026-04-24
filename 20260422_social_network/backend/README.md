# Kizuna Backend

FastAPI backend for Kizuna.

## Development

Run the backend directly during local development:

```sh
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000`.

Only Postgres is expected to run in Docker for the MVP local dev loop.

## Demo Seed

```sh
curl -X POST http://localhost:8000/api/demo/seed
```

## Checks

```sh
uv run pytest
uv run ruff check .
uv run mypy app tests
```
