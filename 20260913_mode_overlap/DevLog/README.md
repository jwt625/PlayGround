# Mode Overlap — execution index

Date: 2026-09-13. Preserve [000-initial-discussion.md](000-initial-discussion.md) as the original product brief. These logs split implementation from physics research and game decisions. Checked items mean delivered artifacts, not proposed work.

| Sequence | Track / log | Outcome / dependency |
|---|---|---|
| 001 | [Coding: contracts and roadmap](coding/001-contracts-and-roadmap.md) | Module boundaries, milestone gates, requirement coverage |
| 002 | [Knowledge: physics contract](knowledge-base/002-physics-and-validation.md) | Normalization, approximations, validation gates; needed by C1/C3 |
| 003 | [Game: question and scoring design](game-design/003-gameplay-and-curriculum.md) | Scoring and content decisions; needed by C2/C4 |
| 004 | [Coding: UI implementation](coding/004-ui-and-game-engine.md) | Concrete SLM-Guessr mapping; C2 |
| 005 | [Knowledge: presets and RNG](knowledge-base/005-presets-and-randomization.md) | Sourced nominal values and educational ranges; C1/C4 |
| 006 | [Coding: cache and example bank](coding/006-cache-and-example-bank.md) | Implemented analytic cache; numerical recipe backlog |
| 007 | [Coding: vector solver](coding/007-vector-solver-and-validation.md) | C3 implementation and physics release gates |
| 008 | [Game: coverage and advanced modes](game-design/008-content-and-advanced-modes.md) | Required 20 cases; advanced backlog |
| 009 | [Coding: implementation record](coding/009-foundation-results.md) | What was built and measured in this pass |

Execution order: C1 foundation → C2 playable Gaussian speedrun → C3 validated vector solver → C4 numerical scenarios → C5 vector/heterogeneous curriculum → C6 adaptive/optimization modes. Cache work is front-loaded within C1 to make examples concrete. This pass implements the data/analytic/cache foundation, not the entire game or an unvalidated vector solver.

Track ownership: coding logs contain executable tasks and acceptance criteria; knowledge logs contain source-backed facts, assumptions and research gaps; game logs contain learning objectives and UX/scoring policy. Update the owning log when a decision changes, then reference its ID from dependents.

Presentation rule (UI4.8 / G3.4): every paired question — analytic Gaussian, on-the-fly Gaussian variant, or waveguide mode — MUST be shown as a single toggling/flashing panel by default (shared µm extent, shared color scale, adjustable rate, manual toggle and Space). Side-by-side is optional, never the default. Any new pair/question type is not done until it toggles in the speedrun, practice and gallery.
