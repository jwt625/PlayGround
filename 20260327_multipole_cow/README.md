# Multipole Cow Reconstruction

This project reproduces the libigl benchmark cow with a spherical-harmonic surface expansion up to order `l = 24`.

The pipeline is:

1. Download the canonical mesh from `libigl-tutorial-data` as `data/cow.off`.
2. Reorder and rescale the mesh to the requested bounding box `(1.0440, 0.6397, 0.3403)`.
3. Build a practical distance-gradient-flow proxy by offsetting the mesh along its normals until it becomes star-shaped from the origin under a ray test.
4. Use the star shell directions as the sphere parameterization and fit `f_r`, `f_{Δθ}`, and `f_{Δφ}` in a spherical-harmonic basis.
5. Reconstruct meshes for `l = 0..24` and export each one as OBJ.

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
