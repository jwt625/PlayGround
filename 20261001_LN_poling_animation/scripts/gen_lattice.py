"""Generate lattice_data.js for the LN poling animation from the Hsu 1997 R3c coordinates (COD 2101845).

Up state  = experimental R3c structure.
Down state = inversion of the up state about the centre of the Nb oxygen octahedron (the paraelectric R-3c
inversion centre); each atom is paired with the nearest same-species atom of the inverted structure
(minimum image), giving its displacement vector in the polarization flip.
"""
import json
import pathlib

import numpy as np

A, C = 5.148, 13.863
LI, NB, O = (0, 0, 0.2806), (0, 0, -0.0010), (0.04751, 0.34301, 0.0625)

OPS = [
    lambda x, y, z: (x, y, z),
    lambda x, y, z: (-y, x - y, z),
    lambda x, y, z: (-x + y, -x, z),
    lambda x, y, z: (-y, -x, z + 0.5),
    lambda x, y, z: (x, x - y, z + 0.5),
    lambda x, y, z: (-x + y, y, z + 0.5),
]
CENT = [(0, 0, 0), (2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)]

M = np.array([[A, -A / 2, 0], [0, A * np.sqrt(3) / 2, 0], [0, 0, C]])  # columns = a1, a2, c


def cart(f):
    return M @ np.asarray(f)


def gen(site):
    out = []
    for op in OPS:
        p = np.array(op(*site))
        for t in CENT:
            q = (p + t) % 1.0
            if not any(np.allclose(((q - o + 0.5) % 1.0) - 0.5, 0, atol=1e-6) for o in out):
                out.append(q)
    return np.array(out)


up = {"Li": gen(LI), "Nb": gen(NB), "O": gen(O)}
print({k: len(v) for k, v in up.items()})

# octahedron centre of the Nb at (0,0,-0.001): centroid of its 6 nearest O (minimum image)
nb0 = np.array(NB)
best = []
for o in up["O"]:
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                best.append((np.linalg.norm(cart(o + [i, j, k] - nb0)), o + [i, j, k]))
best.sort(key=lambda b: b[0])
oct6 = np.array([b[1] for b in best[:6]])
print("Nb-O nearest 6 (A):", [round(b[0], 3) for b in best[:6]])
centre = oct6.mean(axis=0)
print("octahedron centre frac:", centre, " Nb offset from centre (A):", cart(nb0 - centre))

down = {}
disp = {}
for sp, pos in up.items():
    inv = (2 * centre - pos) % 1.0
    d = np.zeros_like(pos)
    dn = np.zeros_like(pos)
    for i, p in enumerate(pos):
        diff = (inv - p + 0.5) % 1.0 - 0.5
        j = np.argmin(np.linalg.norm(diff @ M.T, axis=1))
        d[i] = diff[j]
    disp[sp] = d
    print(sp, "displacement magnitude (A) min/max:", np.linalg.norm(d @ M.T, axis=1).min().round(3),
          np.linalg.norm(d @ M.T, axis=1).max().round(3), " dz (A):", np.unique((d[:, 2] * C).round(3)))

# Li travel relative to its oxygen plane: Li z vs. mean z of its 3 nearest O in the plane
li = up["Li"][0]
cands = []
for o in up["O"]:
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                cands.append((np.linalg.norm(cart(o + [i, j, k] - li)), o + [i, j, k]))
cands.sort(key=lambda b: b[0])
print("Li nearest O (A):", [round(b[0], 3) for b in cands[:6]])
tri = np.array([b[1] for b in cands[:3]])
print("Li height above O3 plane (A), up state:", ((li - tri.mean(axis=0))[2] * C).round(3))

data = {"a": A, "c": C, "centre": centre.tolist(), "species": {}}
for sp in up:
    data["species"][sp] = {"frac": up[sp].round(6).tolist(), "disp": disp[sp].round(6).tolist()}
out = pathlib.Path(__file__).resolve().parent.parent / "data" / "lattice_data.js"
out.write_text("window.LATTICE = " + json.dumps(data) + ";\n")
print("wrote", out)
