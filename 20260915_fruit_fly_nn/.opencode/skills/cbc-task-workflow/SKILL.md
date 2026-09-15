---
name: cbc-task-workflow
description: Use when implementing coding tasks in the fruit-fly coherent-beam-combining (CBC) simulation repo from the `docs/CODING_TASKS.md` handoff queue (T01-T10). Covers verifying devlog/spec claims against real code, working in small green batches, the validation gate (optics/reference tests plus headless and browser commands once T01 pins them), evidence capture, and handing open decisions to the planning agent and visual/complex review to inspection agents.
---

# Fruit-fly CBC coding-task workflow

This is the operating contract for implementation agents picking up coding tasks
in this repo. Follow it unless the user explicitly overrides it. The project has
planning, coding, and complex/visual-inspection agents working asynchronously on
a shared workspace, so hand-offs and evidence matter as much as the code.

Provenance: this skill was copied from the Phieldworks repo
(`.opencode/skills/phieldworks-task-workflow/SKILL.md`) and retargeted to this
project's queue and boundaries.

## 0. Scope discipline

- Work only the tasks assigned by the current queue or directly by the user.
  Start from [`docs/CODING_TASKS.md`](../../../docs/CODING_TASKS.md), backed by
  [`DevLog/000-initial-discussion.md`](../../../DevLog/000-initial-discussion.md)
  and [`DevLog/001-cbc-optics-rendering-spec.md`](../../../DevLog/001-cbc-optics-rendering-spec.md);
  `001` supersedes conflicting defaults in `000`. Begin with **T01-T03**, then
  proceed through the dependent tasks. Do **not** jump ahead to rendering,
  fly assets, or video before the optics/control slice works.
- If a task requires a design/physics decision or missing art, **do not invent
  it**. Record an explicit TBD question in the devlog with candidate options and
  consequences, and leave the code untouched or clearly unimplemented.
- Preserve unrelated work. This project lives inside the larger `PlayGround`
  Git repository alongside sibling projects (including Phieldworks). **Do not
  modify unrelated sibling projects.** Stage and commit only paths under
  `20260915_fruit_fly_nn/`.
- Keep the simulation-only boundary: the "laser", target fly, and connectome are
  virtual. Never add code that interfaces with or controls physical laser
  hardware.
- Keep the architecture contract: separate the simulation core from the
  renderer. Optics, connectome, environment, and learning must run headless and
  independently testable without importing renderer/DOM code. `optics/` must not
  depend on the presentation layer.

## 1. Orient and verify before coding

- Read the handoff queue, the relevant devlogs, current source and tests — not
  just the prose. Docs drift; code is the source of truth.
- Verify the unproven source claims before depending on them: the MaleCNS
  connectome and FlyBody availability/version/license are explicitly **not yet
  audited** (T07, T09). Do not state them as fact until checked.
- Reconcile documentation with reality: run the commands, count the tests, and
  check checkboxes, test counts, and "implemented / next priority" claims against
  the code. Report every discrepancy you find.
- A review finding is not fixed until code plus evidence say so. Conversely, a
  passing automated assertion is **not** visual or physical approval.

## 2. Implement in small green batches

- Prefer the smallest change that satisfies the stated contract. Add or adjust
  tests with the behavior change in the same batch so every commit is green.
- Deterministic and conservation-preserving by default. Fix random seeds; make
  replay reproducible. Fail explicitly rather than silently falling back or
  dropping work. Do not silently change limits or physical models to hit a
  target.
- Respect the optical contract: one source of truth supplies fields, intensity
  maps, metrics, beam envelopes, and actuator telemetry. Never animate beams
  toward a target independently of actuator state. Never conflate piston, phase
  slope (tip/tilt), and curvature.
- Use read-only exploration subagents for reconnaissance. Do not run concurrent
  edits against the same files.

## 3. Validation gate (run before calling anything done)

T01 pins the actual commands and dependencies; use those. Until then the default
expected gate is:

- `npm test`
- `npm run build`
- `npm run test:browser` (once the browser client exists)
- the Python/NumPy reference tests (`pytest`, once the reference lands)
- `git diff --check`

When performance or physical fidelity is involved, run the relevant benchmark
and record machine + language/runtime versions + sample counts + fixture:

- optics field-evaluator throughput (target: fast enough for thousands of
  control steps),
- reference-vs-fast evaluator convergence fixtures,
- SPGD convergence across at least the 10 fixed seeds required by T03.

Rules for measurements:

- State the fixture: geometry, wavelength, pitch, grid spacing/extent, seeds,
  controller mode, disturbances on/off.
- "Algebraic invariant" tests are not power-conservation tests of distinct
  apertures; label them correctly.
- If a tolerance or target is missed, report the miss with numbers; never claim
  success and never relax a threshold merely to pass.
- Record hardware and note that single-machine numbers are not cross-hardware
  claims.

## 4. Capture evidence

- Save raw benchmark/report JSON under `DevLog/evidence/<id>-<topic>.json`.
- Save field arrays and convergence data, not only screenshots. A screenshot is
  not proof of correct interference or successful learning.
- Reference screenshots from `test-results/` by filename in the devlog; do not
  commit `test-results/` (it is gitignored).
- In the handoff report, list changed files (with line refs), exact commands run,
  results, and remaining limitations.

## 5. Update the docs (required, not optional)

- Update the task checkboxes in `docs/CODING_TASKS.md` and the relevant devlog,
  plus an implementation journal entry once one exists, with a dated entry: what
  changed, evidence, checkboxes, and an explicit **"still open"** list.
- Keep the review ledger honest: add dated follow-ups rather than rewriting prior
  findings. Do not mark a reviewed failure as fixed without code and evidence.
- Propose physics-contract changes explicitly; do not silently redefine metrics
  such as PIB, Strehl, or phase RMS to make a number look better.

## 6. Hand off to the planning agent

Leave unresolved decisions as explicit, answerable questions with options and
consequences, for example:

- default numerical fixtures (wavelength, pitch, aperture, steering range) that
  still need the T02 convergence checks before acceptance,
- whether wide-angle power on the full hemisphere is claimed or explicitly
  approximated,
- action-space expansion (19 piston channels vs 19x5 full command) and when,
- reward weights and curriculum gating.

Do not guess numbers. Tag each item with the blocking dependency and owner.

## 7. Hand off to complex/visual-inspection agents

- Maintain a **"Visual/complex verification needed"** list naming the artifact
  (screenshot path or review view) and exactly what to confirm. Example items:
  selecting CH01/CH10/CH19 highlights the complete corresponding path in both
  the 3D console and the 2D schematic; the movable section samples match direct
  field queries; interference overlays derive from the summed complex field.
- Separate automated assertions from visual review. Never mark generated art or a
  screenshot as integrated or approved on the strength of a test.
- Call out complex/numerical references needing independent review: complex
  fields, Gouy/curvature phase, dome solid-angle quadrature, centroid vs
  strongest-lobe behavior, power-conservation identities, singular/zero-power
  failure behavior.

## 8. Commit and push hygiene

- Commit only when the user asks. Inspect `git status` and `git diff` first, stage
  only the intended files under `20260915_fruit_fly_nn/`, and never commit
  secrets.
- One logical batch per commit with a concise imperative message. When changes
  interleave across concerns, split by file/concern where possible and note when
  hunk-level splitting would be required.
- The workspace is shared and the remote may advance while you work. Re-check
  status/diff immediately before committing, and after pushing confirm the
  working tree is clean and `HEAD` matches upstream.
- Never commit `test-results/` artifacts; do commit `DevLog/evidence/` JSON.

## 9. Done report template

Keep it concise and factual:

1. **Task / scope** — which handoff item (T-id).
2. **Changed** — files with line refs and the behavior change.
3. **Impact** — before/after numbers where performance or optics fidelity is involved.
4. **Validation** — exact commands and results (tests/build/browser/reference/benchmarks).
5. **Still open** — remaining gaps, untested paths, and TBD decisions.
6. **Verification needed** — visual/device/complex checks delegated to reviewers.

## 10. Anti-patterns (do not do these)

- Treating stale devlog counts or unchecked source/fidelity claims as fact without checking.
- Claiming a performance or visual result without a measurement/screenshot.
- Animating beams toward the target independently of actuator state, or faking a
  connectome while calling it biological control.
- Relaxing tolerances or physical models to make a test pass.
- Committing mixed, unscoped changes, or touching sibling projects in `PlayGround`.
- Marking a task complete while its verification is delegated and unanswered.
