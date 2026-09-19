"""Material models.

Primary reproduction uses the frozen, lossless, non-dispersive indices chosen
for debugging (per orchestration decision 2026-09-18). Dispersive datasets are
retained as a named sensitivity hypothesis but are not the headline model.

All indices are provisional paper-review inputs, not extracted author data.
"""

from __future__ import annotations

from dataclasses import dataclass

import meep as mp


@dataclass(frozen=True)
class IndexSet:
    """Refractive indices for the nominal constant-index model."""

    n_si: float = 3.48
    n_oxide: float = 1.444
    n_clad: float = 1.0
    cladding: str = "air"

    def epsilon(self, name: str) -> float:
        n = self.n(name)
        return n * n

    def n(self, name: str) -> float:
        if name == "si":
            return self.n_si
        if name == "oxide":
            return self.n_oxide
        if name == "clad":
            if self.cladding == "air":
                return self.n_clad
            if self.cladding == "oxide":
                return self.n_oxide
            raise ValueError(f"unknown cladding {self.cladding!r}")
        raise ValueError(f"unknown material {name!r}")

    def medium(self, name: str) -> mp.Medium:
        return mp.Medium(epsilon=self.epsilon(name))

    def provenance(self) -> dict:
        return {
            "kind": "constant_lossless",
            "n_si": self.n_si,
            "n_oxide": self.n_oxide,
            "cladding": self.cladding,
            "n_clad": self.n("clad"),
            "label": "provisional, not author-supplied",
        }
