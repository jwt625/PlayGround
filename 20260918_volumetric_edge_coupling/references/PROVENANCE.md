# Source cache and provenance

Cache prepared on 2026-09-18 America/Los_Angeles (2026-09-19 UTC). The workstation reported 2026-09-19T01:48:36Z during caching.

A local `references/.gitignore` overrides the parent repository's PDF/text/PNG/HTML exclusions so this source cache can accompany the handoff. Nothing was committed.

## Paper

- Title: **Volumetric Evanescent Edge Coupling for Fiber-to-Chip Optical I/O**.
- Authors: Hamdy Elshehaby, Omar Bakheet, Mohamed A. Swillam, Mohamed Elkabbash.
- Identifier: arXiv:2609.20686v1, physics.optics / physics.app-ph.
- arXiv metadata: submitted 17 September 2026, 16:55:39 UTC.
- User URL: https://arxiv.org/pdf/2609.20686.
- Pinned URL: https://arxiv.org/pdf/2609.20686v1.
- Abstract URL: https://arxiv.org/abs/2609.20686v1.
- `2609.20686/paper.pdf`: 1,728,671 bytes, six pages.
- SHA-256: `226a40ba34978f2094c70146ac7c931321c635577efbccc100c06b7afd065203`.
- `paper-v1.pdf`: independently downloaded pinned version; identical SHA-256.
- `arxiv-abs.html`: cached version-pinned abstract page.

The endpoint https://arxiv.org/src/2609.20686v1 returned a PDF with the same checksum, not a TeX archive or simulation source. Its response is retained as `source-response.pdf`. It does not supply additional geometry or code. A title-based search did not locate a separate simulation repository or supplement during this review; that is a search result, not proof none exists.

## Derived paper artifacts

Text extraction used the PyMuPDF command-line tool via an isolated `uvx` invocation. Ghostscript rendered six page images at 160 dpi; a 400 dpi page-2 render is also retained. These tools were used only for document inspection, not simulation implementation.

| Local artifact | Content |
|---|---|
| `2609.20686/paper.txt` | Layout-preserving text extraction; some two-column order artifacts |
| `2609.20686/paper-reading-order.txt` | Simpler text extraction; visual figures remain authoritative for labels |
| `2609.20686/page-01.png` … `page-06.png` | Full-page raster reference views |
| `2609.20686/page-02-detail.png` | Higher-resolution page-2 view |
| `2609.20686/embedded/img-50.png` | Embedded Figure 1 illustration |
| `2609.20686/embedded/img-51.png` | Embedded Figure 2 geometry/apodization illustration |
| `2609.20686/embedded/img-58.png` | Embedded Figure 3 spectral plot |
| `2609.20686/embedded/img-59.png` | Embedded Figure 3 inset |
| `2609.20686/embedded/img-68.png` | Embedded Figure 4 fabrication spectra |
| `2609.20686/embedded/img-75.png` | Embedded Figure 5 alignment/reflection panels |

Pages 2–6 and the original Figures 2–5 were visually inspected. The complete extracted paper text was read. Numeric curve readings in PAPER_REVIEW.md are approximate visual readings, not completed digitization.

## Solver sources

Official project sources accessed during planning; cached pages are evidence snapshots, not a dependency lock. Their `latest`/branch URLs can change. Resolve actual installed versions during T01.

| Source | Cached artifact | Use |
|---|---|---|
| https://meep.readthedocs.io/en/latest/Installation/ | `solver-docs/meep-installation.html` | supported packages, ARM/macOS and MPI installation |
| https://meep.readthedocs.io/en/latest/Python_Tutorials/Eigenmode_Source/ | `solver-docs/meep-eigenmode-source.html` | oblique guide and PML considerations |
| https://meep.readthedocs.io/en/latest/Python_User_Interface/ | `solver-docs/meep-python-interface.html` | Gaussian source classes and monitor API |
| https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/ | `solver-docs/meep-subpixel-smoothing.html` | geometric interface convergence |
| https://meep.readthedocs.io/en/latest/Mode_Decomposition/ | `solver-docs/meep-mode-decomposition.html` | modal power and material restrictions |
| https://raw.githubusercontent.com/ymahlau/fdtdx/main/README.md | `solver-docs/fdtdx-readme.md` | alternative JAX/accelerator solver |
| https://raw.githubusercontent.com/flaport/fdtd/master/README.md | `solver-docs/flaport-fdtd-readme.md` | lightweight Python FDTD alternative |

Additional primary pages consulted:

- [Meep repository](https://github.com/NanoComp/meep).
- [Meep license clarification](https://github.com/NanoComp/meep/blob/master/doc/docs/License_and_Copyright.md).
- [Meep materials](https://meep.readthedocs.io/en/latest/Materials/).
- [MPB documentation](https://mpb.readthedocs.io/en/latest/).
- [FDTDX mode-source/detector tutorial](https://fdtdx.readthedocs.io/en/latest/notebooks/components/02_mode_source_detector.html).

## Current completion state

Completed: source cache, full paper review, figure inspection, parameter extraction, missing-input audit, solver research, numerical specification and implementation plan.

Not performed: implementation, solver installation, curve digitization, material fitting, FDTD runs, or numerical reproduction. No messages were sent to the authors. All future numerical thresholds are proposed acceptance criteria, not observed results.
