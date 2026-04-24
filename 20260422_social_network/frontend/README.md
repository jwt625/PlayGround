# Kizuna Frontend

SvelteKit frontend for Kizuna.

## Development

Run the frontend directly during local development:

```sh
pnpm install
pnpm dev
```

The dev server runs at `http://localhost:5173`.

## Backend

The backend should be run separately from `../backend` with `uv run uvicorn app.main:app --reload`.

The frontend should call the backend at `http://localhost:8000`.

## Current Pages

- `/`
- `/people`
- `/organizations`
- `/events`
- `/reminders`
- `/pipelines`
- `/search`
- `/imports`
- `/exports`

## Build

Production builds are not required for daily local development.

```sh
pnpm build
```

## Checks

```sh
pnpm check
pnpm lint
pnpm format:check
```

Or run all frontend checks together:

```sh
pnpm check:all
```
