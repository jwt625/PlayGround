# 004 — SLM-Guessr UI translation and implementation tasks

Status: reference inspected; frontend planned. Date: 2026-09-13.

Reference repository: `/Users/wentaojiang/Documents/GitHub/SLM-Guessr`. Inspected actual Svelte sources, not only README. Follow its dark backgrounds (#0a0a0a / #141414), thin #2a2a2a borders, square cards, blue #4a9eff accent, compact monospace labels and 1200px content width. Adapt patterns; no new branding work.

| Reference file | Mode-overlap adaptation |
|---|---|
| `src/lib/components/Header.svelte` | Gallery / Speedrun / Practice navigation |
| `src/lib/components/SampleCard.svelte` | Paired Mode A / Mode B panels with geometry overlays and physical axes |
| `src/lib/components/SampleModal.svelte` | Inspect fields and explanation; reveal-gated integrand |
| `src/routes/gallery/+page.svelte` | Category/level sidebar, cards, counts, ready-only default filter |
| `src/routes/quiz/+page.svelte` | setup → playing → feedback → results state machine |
| `src/app.css` | Technical dark palette, spacing and typography |

Do not transfer its Math.random shuffling, binary correctness/streak scoring, or GIF pixels as physics. Use seeded scenario modules, G003 scoring and complex field data.

- [ ] UI4.1 Scaffold static SvelteKit routes and port visual tokens by reference; verify license if copying code. Deliver real gallery and speedrun views, not a landing-page-only shell.
- [ ] UI4.2 Implement field canvas with independent physics/render grids. Shared µm extent by default; clearly label optional independently fitted panels. Geometry dimensions are distinct from MFD. Canvas resize cannot change overlap.
- [ ] UI4.3 Intensity/amplitude/sign/phase/Sz controls; sequential/diverging/cyclic maps, shared vs individual color scale, linear/log. Unsupported components disabled with clear labels.
- [ ] UI4.4 Implement session module outside Svelte components; answer adapters for slider/interval/rank and one-shot submission.
- [ ] UI4.5 Ready question queue with analytic fallback; loading/error recovery and keyboard focus after reveal.
- [ ] UI4.6 Results and local player statistics; deterministic share IDs include versions/full persisted recipe.
- [ ] UI4.7 Waveguide vector modes (C3+): show an inset of the cross-section geometry drawing colored by refractive index, and outline the material/geometry boundaries as a dashed line over the field plot. Analytic Gaussian questions keep the MFD overlay; this inset/outline is required once validated waveguide fields exist and must not be faked for pending recipes.
- [x] UI4.8 Flash/toggle is the mandatory default comparison for every mode pair, in every surface: analytic Gaussian, on-the-fly Gaussian variants, waveguide questions, gallery modal, speedrun and practice. One shared panel, shared µm extent, shared color scale, adjustable rate plus manual toggle and Space; side-by-side is an optional fallback. Reduced-motion starts paused. Any new pair type MUST ship with a toggling view rather than a static two-up or a palette view. Applied: `ModeFlashView.svelte` and `WaveguidePairView.svelte`. Flash rate and pause state are shared (`src/lib/flash.svelte.ts`) and MUST persist across questions and across the Gaussian↔waveguide viewer switch; they must not reset per question.
- [x] UI4.9 On-the-fly waveguide variants must be **independently solved** fields (different guide geometry or mode order), never a translated/scaled copy of one solved mode. Pair two distinct entries and integrate their real overlap; pre-render both layers so the flash rate equals the slider (no per-toggle repaint). Applied: `src/lib/waveguide.ts` + variant guides in `scripts/build-modes.mjs`.

Acceptance: complete 20-question session and replay; zero and unit overlap format correctly; refresh gallery loads shipped bank; before reveal no answer/eta-derived filename or answer bucket in rendered labels; mobile panels remain readable and scale labels visible; keyboard-only completion; field cache failure does not prevent analytic play. Browser E2E covers these behaviors when UI exists.
