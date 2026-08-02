# Nonuniform thermally chirped FBG reconstruction

This folder contains a parameterized coupled-mode/transfer-matrix reconstruction of Figure 5(a)
from the supplied paper, plus a semi-analytic cylindrical-fin temperature solution.

Run it with:

```bash
uv sync
uv run python reproduce_spectra.py
uv run python -m unittest discover -s tests
```

The script writes plot-ready numerical data and figures under `outputs/`. The main model is in
`fbg_model.py`; all measured inputs, inferred parameters, inconsistencies, and identifiability
limits are recorded in `MODEL_NOTES.md`.

The reconstruction digitizes a cold bandwidth of approximately 0.30 nm and uses a realistic
Gaussian-apodized high-reflectivity 10 mm sensing FBG. The temperature profile is recovered from
six heated spectral envelopes as a strongly asymmetric 2.743 mm-FWHM hotspot rather than
assumed Gaussian. Run `uv run python fit_temperature_profile.py` to repeat the inversion.
`digitized_constraints.csv` records quantities read from the figures. Exact ripple phase remains
non-unique because the raw complex spectrum, polarization, strain transfer, and grating phase
are unavailable.
