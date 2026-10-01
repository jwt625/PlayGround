# EO Modulator Atlas

A traceable literature database and static browser explorer for electro-optic
modulators. The verified pilot contains **3 papers, 7 devices and 8 organizations**;
198 candidates await further distillation. Each reported metric has an evidence
locator and basis. Missing values stay empty.

## Run the app

Use Node 22.12+ (tested with Node 25.9) and pnpm 10.12.4:

```sh
cd engine
npx pnpm@10.12.4 install --frozen-lockfile
cd ../app
npx pnpm@10.12.4 install --frozen-lockfile
npm run dev
```

Open the local URL printed by Vite. Routes:

- `/table`: expandable paper/device table, filters, representative selection,
  evidence details and CSV export.
- `/explore`: metric comparisons, disclosure and organization views.
- `/sim?id=chen2022-c`: editable YAML, geometry preview and cancellable browser
  cross-section solver. The legacy `?config=sims/chen2022/config.yaml` URL also works.
- `/about`: metric conventions, evidence and current model limitations.

`npm run build` produces `app/build/`; `npm run preview` serves it locally.
`BASE_PATH=/eo-atlas npm run build` supports a subdirectory deployment. Dev/build
copy input YAML into ignored `app/static/sims/`; source PDFs are not copied to the
website. Install the engine dependencies before building the app.

## Database workflow

With Python 3.12+ and uv:

```sh
uv sync --extra dev
uv run python scripts/validate_db.py
uv run pytest -q
uv run python scripts/build_views.py
```

Without uv, create a virtual environment and install the script dependencies:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install pydantic pyyaml requests pytest
.venv/bin/python scripts/validate_db.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/build_views.py
```

The CSVs and `data/evidence/` are authoritative; `app/static/data/atlas.json` is
a generated view, not a second database. Refresh it after accepted data changes.
The [distillation skill](.claude/skills/eo-modulator-distill/SKILL.md),
[batch contract](data/_staging/BATCH_INSTRUCTIONS.md) and
[manual download list](data/manual_downloads.md) describe ingestion.
Reference PDFs, text and figures are tracked under the revised policy recorded in
[references/README.md](references/README.md); rights remain source-specific.

## Simulation status and checks

```sh
cd engine
npm test
cd ..
node engine/cli.mjs sims/chen2022/config.yaml
node engine/cli.mjs sims/chen2022/config.yaml --section unloaded
cd app
npm test
npm run check
npm run build
npm run smoke
```

The smoke suite requires an installed Google Chrome. It starts/stops its own
local preview server and temporary browser profile, tests the built site and
compares browser/Node calculations. For a subdirectory build, use the same
`BASE_PATH=/eo-atlas` for both `npm run build` and `npm run smoke`.
Optional `SMOKE_SCREENSHOTS=../logs/ui` keeps screenshots in ignored local scratch.

The engine runs **electrostatics and optional scalar optical modes**. Analytic
gates cover parallel plates, differential/half-domain capacitance, anisotropic
rotation, TE/TM slab modes and waveguide group index. CLI results go to stdout;
browser results stay in memory. Do not commit solver outputs.

EO overlap, RF loss, the periodic loaded-line model and EO response remain pending.
A cross-section's RF index and impedance are not the effective values of a
periodically loaded device. Such targets are reported as `not_evaluated`.
The Chen config remains `unvalidated`. Its optical window intersects gold, so
`--optical` currently stops with an explicit unsupported-material error; it does
not substitute an invented optical index. Tests of the optical solver use
dielectric slab fixtures.

CLI exit codes: 0 = requested stages completed (not a literature pass), 1 = input
or solver error, 2 = an evaluated target missed its tolerance. Mesh scale changes
numerical resolution only. Results need mesh/domain convergence checks before
physical interpretation. See [sims/SPEC.md](sims/SPEC.md) and
[engine/README.md](engine/README.md).

The shared runner reports target coverage separately from pass/fail. Unspecified
line topology, frequency-specific targets, alternate-section targets and configured
fixed group index are excluded from prediction comparisons. Each mesh has an
80,000-vertex default budget; exceeding it produces an explicit error.

## Next work

1. Add metal-aware/vector optical handling and explicit EO drive/arm conventions.
2. Implement EO overlap, RF loss and periodic-cell/EO-response stages with analytic gates.
3. Resume batches `p1_01`–`p1_11`, merging only verified rows.
4. Complete literature regressions and independent evidence audits.

Current work and remaining phases are recorded in [DevLog-000](DevLog/DevLog-000-plan.md)
and [DevLog-002](DevLog/DevLog-002-work-plan-and-ownership.md).
E1/U1 progress, final checks and handoff: [DevLog-003](DevLog/DevLog-003-cross-section-progress.md).
Read [WORKBOARD.md](WORKBOARD.md) and the active claims under
`coordination/claims/` before starting concurrent work.
