# Volumetric edge-coupler simulation reproduction

Reproduction workspace for **Volumetric Evanescent Edge Coupling for Fiber-to-Chip Optical I/O**, Elshehaby et al., [arXiv:2609.20686v1](https://arxiv.org/abs/2609.20686v1). Initial planning package prepared 2026-09-18 Pacific time. A coding agent has since implemented a running Meep pipeline and provisional results; numerical/physical acceptance is still in progress.

**Current next action (2026-09-22):** [Coding-session restart handoff](NEXT_SESSION.md). The physical C0/C1 pair is unrun. Repair the common-cell nominal termination, incident calibration, per-dataset coordinates and output persistence before the two coarse Hz runs; save downstream complex E/H and usable traces. The existing res-8 files are smoke artifacts only. Actual-port identity proceeds alongside C0/C1; full-domain 50/75 remains deferred and T02 independent. The [port/channel/overlap decision](reports/R6_port_and_overlap_decision.md) remains the measurement contract.

**Latest branch decision:** investigate Hz first, with one matched Ez reciprocal control. The previous root-residual and band/frequency indexing defects have been corrected in the inspected implementation; loaded-mode identity and accumulated phase/leakage remain unresolved. [Earlier R5/R6 review](reports/R5_R6_branch_priority.md) records the diagnostic rationale. T02 digitization can proceed independently now.

**Recommendation:** use Python-controlled Meep for the 2D reproduction, then extend the validated model to 3D if reproducing the finite-width and transverse-alignment results. A homebuilt 2D solver is an optional independent check; it adds substantial source, boundary, geometry, and modal-normalization work.

The paper reports 88% coupling at 1550 nm, measured in the inclined receiving waveguide **before the bend into the planar photonic layer**. Its input is a Gaussian beam, and the described TIR occurs from silicon into oxide. The paper does not fully specify the geometry or input-interface normalization. Reproducing the internal coupling calculation and predicting packaged fiber-to-planar-waveguide loss are distinct deliverables.

Read these in order:

1. [Paper extraction and visual review](PAPER_REVIEW.md): known parameters, every reported simulation, missing inputs, and interpretation issues.
2. [Solver decision and numerical specification](SIMULATION_SPEC.md): recommended solver, geometry contract, physics checks, normalization, convergence, and compute sizing.
3. [Implementation TODOs and acceptance criteria](TODO.md): ordered work packages, dependencies, deliverables, comparison tolerances, and handoff instructions.

Cached evidence:

- [Original PDF](references/2609.20686/paper.pdf), [version-pinned PDF](references/2609.20686/paper-v1.pdf), [extracted text](references/2609.20686/paper-reading-order.txt).
- [Cache provenance and source index](references/PROVENANCE.md).
- [Figure 2: coupling geometry](references/2609.20686/embedded/img-51.png), [Figure 3: width comparison](references/2609.20686/embedded/img-58.png), [Figure 4: fabrication sweeps](references/2609.20686/embedded/img-68.png), [Figure 5: alignment sweeps](references/2609.20686/embedded/img-75.png).

The workspace was initially empty. The inspected host is macOS arm64 with 24 GiB RAM. Continue in 2D; full uniform-grid 3D at the needed resolution can exceed this host's memory by a large factor. The next coding agent should follow the current R6 handoff above and retain uncertainty labels throughout.
