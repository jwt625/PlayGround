# Multipole Cow Reconstruction

This project reproduces the libigl benchmark cow with the distance-gradient-flow surface expansion from Section III.1 of *Higher multipoles of the cow*, using spherical harmonics up to order `l = 24`.

The pipeline is:

1. Download the canonical mesh from `libigl-tutorial-data` as `data/cow.off`.
2. Reorder and rescale the mesh to the requested bounding box `(1.0440, 0.6397, 0.3403)`.
3. Numerically integrate the signed-distance gradient flow and search for a flowed shell that is as close as possible to star-shaped from the origin under dense ray tests.
4. Use the angular coordinates of the flowed star shell to define the sphere map `h : S^2 -> Σ`, then fit `f_r`, `f_{Δθ}`, and `f_{Δφ}` in a spherical-harmonic basis.
5. Reconstruct meshes for `l = 0..24` and export each one as OBJ.

## Math

Following Section III of the paper, the cow boundary `Σ` is represented through a map `h : S^2 -> Σ` and three real-valued component fields on the sphere:

- `f_r(Ω)`: radial distance
- `f_Δθ(Ω)`: polar-angle shift
- `f_Δφ(Ω)`: azimuthal-angle shift

For a sphere point `Ω = (θ, φ)`, the implementation uses

```math
f_r(\Omega) = r(h(\Omega)),
\qquad
f_{\Delta\theta}(\Omega) = \theta(h(\Omega)) - \theta,
\qquad
f_{\Delta\phi}(\Omega) = \operatorname{wrap}\!\left(\phi(h(\Omega)) - \phi\right).
```

The `l = 0` coefficients of the angular fields are then set to zero, matching the paper's convention that the monopole of the angular correction vanishes.

Each field is expanded in spherical harmonics:

```math
f(\Omega) = \sum_{l=0}^{L}\sum_{m=-l}^{l} a_{lm} Y_l^m(\Omega)
```

with the sphere inner product

```math
\langle f, g \rangle = \int_{S^2} f(\Omega) g(\Omega)^*\, d\Omega,
```

so the ideal coefficients are the projections

```math
a_{lm} = \int_{S^2} f(\Omega)\,Y_l^m(\Omega)^*\,d\Omega
```

To construct `h`, the script follows the paper's distance-gradient-flow method. Let `d(x)` be the signed distance to the original cow surface. The flow `Φ_t` solves

```math
\frac{d}{dt}\Phi_t(x) = \nabla d(\Phi_t(x)),
\qquad
\Phi_0(x) = x.
```

For sufficiently large `t`, `Φ_t(\Sigma)` becomes star-shaped in the ideal continuous construction. The code advances the mesh vertices along this flow, measures ray-intersection diagnostics on candidate shells, and uses the best-tested shell. If a numerically star-shaped shell is found within the search budget, that fact is recorded in `outputs/metrics.json`.

Once the flowed shell is star-shaped, each flowed vertex defines a sphere sample `\Omega_i` by radial projection, and the corresponding original vertex supplies the value `h(\Omega_i)`.

The code then estimates coefficients from sampled data by weighted least squares. Given samples `(\Omega_i, f_i)` and basis matrix

```math
A_{i\alpha} = Y_{l_\alpha}^{m_\alpha}(\Omega_i),
```

it solves

```math
\min_a \sum_i w_i \left| f_i - \sum_\alpha a_\alpha A_{i\alpha} \right|^2
```

or equivalently

```math
a = \arg\min_a \left\| W^{1/2}(Aa - f) \right\|_2^2.
```

The weights `w_i` are not cow-surface areas. They are vertex areas on the unit sphere obtained by radially projecting the star shell, so the discrete fit approximates the paper's `S^2` inner product rather than an area integral over the cow.

This fit is done separately for `f_r`, `f_{Δθ}`, and `f_{Δφ}`.

The order `l` controls angular detail:

- `l = 0`: spherical average only
- `l = 1`: broad offset / tilt corrections
- `l = 2`: large-scale elongation / compression
- higher `l`: progressively finer angular structure

The number of modes up to order `L` is

```math
\sum_{l=0}^{L}(2l+1) = (L+1)^2,
```

so `L = 24` gives `625` modes per scalar field, or `1875` coefficients across the three fitted fields.

## Run

```bash
uv run python reconstruct_cow.py
```

By default the script now fails fast if dense ray checks do not find a numerically star-shaped shell, or if a higher-order reconstruction exceeds the built-in geometric safety thresholds.

If you explicitly want exploratory output even when the shell test fails, opt in with:

```bash
uv run python reconstruct_cow.py --allow-non-star-shell
```

Generate the reconstruction grid only when you explicitly want it:

```bash
uv run python reconstruct_cow.py --plot
```

## Interactive Viewer

Start a simple static server from the project root and open the viewer in your browser:

```bash
uv run python -m http.server 8000
```

Then visit `http://localhost:8000/viewer/`.

## Outputs

- `data/cow.obj`: prepared benchmark mesh in OBJ format
- `outputs/cow_reconstruction_l*.obj` up to the last safe completed order
- `outputs/coefficients.json`
- `outputs/metrics.json`
- `outputs/reconstruction_grid.png` when `--plot` is used
- `viewer/`: lightweight Three.js front end for interactive inspection

## RCA

On March 28, 2026, a regeneration run contributed to a full machine reboot instead of failing gracefully.

Observed evidence:

- The macOS panic report at `/Library/Logs/DiagnosticReports/panic-full-2026-03-28-100427.0002.panic` records a watchdog timeout rather than a normal Python exception.
- The reconstruction process spent most of its time in repeated ray-intersection queries triggered by `trimesh.ray.intersects_location(...)`.
- `outputs/metrics.json` recorded that the flowed shell was not numerically star-shaped:
  - `numerically_star_shaped = 0.0`
  - `max_hits_along_any_verified_ray = 5.0`
- High-order reconstructions became geometrically unstable instead of converging visually. The reconstructed bounding-box max extent grew rapidly:
  - `l = 18`: about `1.8`
  - `l = 19`: about `3.5`
  - `l = 20`: about `18`
  - `l = 21`: about `48`
  - `l = 22`: about `177`
  - `l = 23`: about `1036`
  - `l = 24`: about `4372`

Root cause:

- The code continued with coefficient fitting and mesh export even though the star-shell prerequisite of the paper had not been achieved.
- At high order, the unstable angular/radial reconstruction produced extreme spikes and huge triangles.
- The script combined that unstable geometry with expensive proximity and ray queries, which created sustained system load.
- The current `surface_rmse` metric is one-way, so it understated severe outward spikes and did not act as a reliable safety signal.

Contributing factors:

- Dense repeated ray diagnostics in the distance-gradient-flow search.
- No hard stop based on geometric blow-up.
- Plotting and export still proceed after unstable high-order meshes are formed.
- Other active system load can make the same script behavior much less safe.

Corrective actions for future runs:

- Abort reconstruction if a numerically star-shaped shell is not found.
- Use a symmetric surface distance metric, not only original-to-reconstruction nearest distances.
- Add explicit safety caps on bounding-box growth, edge-length outliers, and vertex norms.
- Stop order escalation once quality degrades or geometry blows up.
- Keep plotting optional and off by default for heavy runs.

## FMEA

| Failure mode | Cause | Effect | Detection | Mitigation |
| --- | --- | --- | --- | --- |
| Star-shell search never reaches a valid shell | Distance-gradient flow proxy is insufficient for this mesh and origin choice | Sphere map is not well-defined; downstream fit becomes unreliable | `numerically_star_shaped = 0.0`, multi-hit rays in `metrics.json` | Fail fast instead of continuing; improve shell construction before fitting |
| High-order harmonic fit blows up | Ill-conditioned fit on an invalid or weakly parameterized shell | Spiky meshes, huge extents, misleading apparent improvement in one-way RMSE | Bounding-box growth, extreme edge lengths, visual spikes | Clamp or stop at a safe order; add geometric blow-up thresholds |
| RMSE understates bad reconstructions | Metric only measures original vertices to candidate surface | Large outward spikes are weakly penalized | Compare one-way RMSE against symmetric RMSE and max distance | Replace with symmetric RMSE plus a worst-case distance metric |
| Ray diagnostics overwhelm CPU | Many repeated `intersects_location` calls on dense rays | Machine becomes unresponsive; watchdog risk under load | High CPU usage, long stalls before first output | Reduce ray density adaptively, cache where possible, add per-stage runtime limits |
| Proximity queries become too expensive on unstable meshes | Blown-up geometry increases nearest-point query cost | Slow export/metrics/plot stages, excessive memory churn | Sudden runtime jump at higher `l` | Refuse to evaluate meshes whose size/edge statistics exceed thresholds |
| Plot generation magnifies instability cost | Large malformed meshes are still rendered | Extra memory and CPU use after reconstruction already degraded | Late-stage slowdown or failure during plotting | Keep plotting disabled by default for exploratory or heavy runs |
