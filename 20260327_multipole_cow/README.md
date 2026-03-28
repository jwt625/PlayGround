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

## Interactive Viewer

Start a simple static server from the project root and open the viewer in your browser:

```bash
uv run python -m http.server 8000
```

Then visit `http://localhost:8000/viewer/`.

## Outputs

- `data/cow.obj`: prepared benchmark mesh in OBJ format
- `outputs/cow_reconstruction_l0.obj` through `outputs/cow_reconstruction_l24.obj`
- `outputs/coefficients.json`
- `outputs/metrics.json`
- `outputs/reconstruction_grid.png`
- `viewer/`: lightweight Three.js front end for interactive inspection
