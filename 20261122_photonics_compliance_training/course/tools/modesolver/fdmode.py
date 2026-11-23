"""Full-vector finite-difference eigenmode solver for anisotropic dielectric waveguides.

Formulation
-----------
Fields vary as exp(i (beta z - omega t)); z is the propagation direction, x is lateral, y is vertical.
Unknowns are all three electric field components (Ex, Ey, Ez) on a 2D Yee grid (Yee, IEEE Trans. Antennas
Propag. 14, 302 (1966)): Ex at (x_i + h/2, y_j), Ey at (x_i, y_j + h/2), Ez at (x_i, y_j). With d/dz -> i beta the
discrete curl is C(beta) = C0 + beta C1 and the vector wave equation curl curl E = k0^2 eps E becomes

    (K0 + beta K1 + beta^2 K2) e = 0,   K0 = C0^T C0 - k0^2 M,  K1 = C0^T C1 + C1^H C0,  K2 = C1^H C1,

a quadratic eigenvalue problem in beta. M is the discrete permittivity operator. The permittivity is a full
symmetric tensor with the in-plane components eps_xx, eps_yy, eps_zz and the lateral-longitudinal component
eps_xz (eps_xy = eps_yz = 0 for an x-cut film whose crystal x axis is the film normal). eps_xz couples Ex and Ez,
which sit at different Yee positions; it enters M as the symmetric block
    M_xz = 1/2 [diag(eps_xz at Ex) P + P diag(eps_xz at Ez)],  M_zx = M_xz^T,
where P averages the two Ez neighbours onto an Ex position. M stays real symmetric, so the problem is Hermitian.

With eps_xz != 0 the substitution that makes the usual transverse formulations linear in beta^2 is not available
(the longitudinal field is no longer in quadrature with the transverse field), so the quadratic problem is solved
exactly: companion linearization (Tisseur and Meerbergen, SIAM Review 43, 235 (2001)) of size 6N, shift-invert
Arnoldi (scipy eigs, ARPACK) about beta = sigma. The shifted 6N operator is applied through one sparse LU of the
3N matrix Q(sigma) = K0 + sigma K1 + sigma^2 K2 (Schur complement of the companion pencil), so the cost is that of
a single 3N factorization.

Boundaries: perfect electric conductor on all four walls, or periodic in x (used for the laterally uniform slab
validation). Material interfaces: sub-cell averaging over the dual cell of each field component (8 x 8 samples):
eps_xx harmonic along x then arithmetic along y, eps_yy harmonic along y then arithmetic along x, eps_zz and eps_xz
arithmetic. Horizontal interfaces are placed on grid lines, which makes the averaging exact there; slanted
sidewalls are sub-pixel averaged.

transverse_diag() is an independent code path for diagonal tensors only: the classical linear transverse-field
eigenproblem in beta^2 with Ez eliminated through the discrete Gauss law. It is used to check the quadratic solver.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from dataclasses import dataclass, field  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import LinearOperator, eigs, splu  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# Materials (wavelength in micrometres)


def n_ln_zelmon(lam_um: float) -> tuple[float, float]:
    """Congruent undoped LiNbO3, ordinary and extraordinary index. Zelmon, Small and Jundt, JOSA B 14, 3319 (1997),
    three-term Sellmeier n^2 - 1 = sum A_i lam^2 / (lam^2 - B_i), lam in um."""
    l2 = lam_um**2
    o = 2.6734 * l2 / (l2 - 0.01764) + 1.2290 * l2 / (l2 - 0.05914) + 12.614 * l2 / (l2 - 474.60)
    e = 2.9804 * l2 / (l2 - 0.02047) + 0.5981 * l2 / (l2 - 0.0666) + 8.9543 * l2 / (l2 - 416.08)
    return float(np.sqrt(1 + o)), float(np.sqrt(1 + e))


def n_sio2_malitson(lam_um: float) -> float:
    """Fused silica. Malitson, JOSA 55, 1205 (1965)."""
    l2 = lam_um**2
    s = 0.6961663 * l2 / (l2 - 0.0684043**2) + 0.4079426 * l2 / (l2 - 0.1162414**2) + 0.8974794 * l2 / (l2 - 9.896161**2)
    return float(np.sqrt(1 + s))


def ln_tensor_xcut(no: float, ne: float, theta_deg: float) -> dict[str, float]:
    """Relative permittivity of x-cut LiNbO3 in waveguide coordinates (x lateral, y = crystal x = film normal,
    z = propagation). theta = in-plane angle between the propagation direction and the crystal y axis; the c axis
    (crystal z) is then at 90 - theta from the propagation direction: c = cos(theta) x_hat + sin(theta) z_hat."""
    t = np.deg2rad(theta_deg)
    eo, ee = no**2, ne**2
    d = ee - eo
    c, s = np.cos(t), np.sin(t)
    c = 0.0 if abs(c) < 1e-12 else c
    s = 0.0 if abs(s) < 1e-12 else s
    return {"xx": eo + d * c * c, "yy": eo, "zz": eo + d * s * s, "xz": d * s * c}


# ---------------------------------------------------------------------------------------------------------------
# Geometry


@dataclass
class Grid:
    x0: float
    y0: float
    h: float
    nx: int
    ny: int
    periodic_x: bool = False

    @property
    def xn(self) -> np.ndarray:
        return self.x0 + self.h * np.arange(self.nx)

    @property
    def yn(self) -> np.ndarray:
        return self.y0 + self.h * np.arange(self.ny)


def make_grid(xmin: float, xmax: float, ymin: float, ymax: float, h: float, periodic_x: bool = False) -> Grid:
    nx = int(round((xmax - xmin) / h))
    ny = int(round((ymax - ymin) / h))
    return Grid(xmin, ymin, h, nx, ny, periodic_x)


@dataclass
class Structure:
    """Material function: returns the relative permittivity tensor components on arrays of points."""

    # each region: (mask_fn(x, y) -> bool array, tensor dict); later regions override earlier ones
    background: dict[str, float]
    regions: list = field(default_factory=list)

    def eps(self, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        out = {k: np.full(x.shape, v, dtype=float) for k, v in self.background.items()}
        for mask_fn, tens in self.regions:
            m = mask_fn(x, y)
            for k in out:
                out[k][m] = tens[k]
        return out


def iso(n: float) -> dict[str, float]:
    return {"xx": n * n, "yy": n * n, "zz": n * n, "xz": 0.0}


def rib_structure(t_film: float, t_slab: float, w_top: float, sidewall_deg: float, ln: dict[str, float],
                  n_box: float, n_top: float) -> Structure:
    """x-cut film on buried oxide (y < 0); LN slab 0 <= y <= t_slab everywhere; rib t_slab < y <= t_film with a
    trapezoidal cross-section of top width w_top and sidewall angle sidewall_deg measured from the substrate plane;
    cover index n_top above."""
    cot = 1.0 / np.tan(np.deg2rad(sidewall_deg))

    def box(x, y):
        return y < 0

    def slab(x, y):
        return (y >= 0) & (y <= t_slab)

    def rib(x, y):
        half = 0.5 * w_top + (t_film - y) * cot
        return (y > t_slab) & (y <= t_film) & (np.abs(x) <= half)

    return Structure(iso(n_top), [(box, iso(n_box)), (slab, ln), (rib, ln)])


def rib_outline(t_film: float, t_slab: float, w_top: float, sidewall_deg: float, xmin: float, xmax: float):
    cot = 1.0 / np.tan(np.deg2rad(sidewall_deg))
    wb = 0.5 * w_top + (t_film - t_slab) * cot
    return [[xmin, 0.0], [xmax, 0.0], [xmax, t_slab], [wb, t_slab], [0.5 * w_top, t_film], [-0.5 * w_top, t_film],
            [-wb, t_slab], [xmin, t_slab]]


# ---------------------------------------------------------------------------------------------------------------
# Discretization


def _avg_eps(g: Grid, st: Structure, ox: float, oy: float, S: int = 8) -> dict[str, np.ndarray]:
    """Sub-cell averaged tensor at component positions (x_i + ox h, y_j + oy h), over a dual cell of size h x h."""
    h = g.h
    xs = g.xn + ox * h
    ys = g.yn + oy * h
    off = (np.arange(S) + 0.5) / S - 0.5
    X = xs[:, None, None, None] + h * off[None, None, :, None]
    Y = ys[None, :, None, None] + h * off[None, None, None, :]
    X, Y = np.broadcast_arrays(X, Y)
    e = st.eps(X, Y)  # arrays (nx, ny, Sx, Sy)
    out = {
        "xx": np.mean(1.0 / np.mean(1.0 / e["xx"], axis=2), axis=2),  # harmonic along x, arithmetic along y
        "yy": np.mean(1.0 / np.mean(1.0 / e["yy"], axis=3), axis=2),  # harmonic along y, arithmetic along x
        "zz": e["zz"].mean(axis=(2, 3)),
        "xz": e["xz"].mean(axis=(2, 3)),
    }
    return out


def _d1(n: int, h: float, periodic: bool) -> sp.csr_matrix:
    """Forward difference (f[i+1] - f[i]) / h; f[n] = 0 (PEC) or f[0] (periodic)."""
    d = sp.diags([-np.ones(n), np.ones(n - 1)], [0, 1], shape=(n, n), format="lil")
    if periodic:
        d[n - 1, 0] = 1.0
    return (d.tocsr() / h)


def _avg1(n: int, periodic: bool) -> sp.csr_matrix:
    """(f[i] + f[i+1]) / 2 with the same end rule as _d1."""
    a = sp.diags([0.5 * np.ones(n), 0.5 * np.ones(n - 1)], [0, 1], shape=(n, n), format="lil")
    if periodic:
        a[n - 1, 0] = 0.5
    return a.tocsr()


@dataclass
class Operators:
    g: Grid
    K0: sp.csr_matrix
    K1: sp.csr_matrix
    K2: sp.csr_matrix
    keep: np.ndarray  # indices of kept DOFs in the full 3N vector
    k0: float
    eps: dict


def build(g: Grid, st: Structure, lam_um: float) -> Operators:
    nx, ny, h = g.nx, g.ny, g.h
    N = nx * ny
    Ix, Iy = sp.identity(nx, format="csr"), sp.identity(ny, format="csr")
    Dx = sp.kron(_d1(nx, h, g.periodic_x), Iy, format="csr")
    Dy = sp.kron(Ix, _d1(ny, h, False), format="csr")
    Px = sp.kron(_avg1(nx, g.periodic_x), Iy, format="csr")  # Ez (x_i) -> Ex (x_i + h/2)
    Z = sp.csr_matrix((N, N))
    I = sp.identity(N, format="csr")
    C0 = sp.bmat([[Z, Z, Dy], [Z, Z, -Dx], [-Dy, Dx, Z]], format="csr")
    C1 = sp.bmat([[Z, -1j * I, Z], [1j * I, Z, Z], [Z, Z, Z]], format="csr")
    ex = _avg_eps(g, st, 0.5, 0.0)
    ey = _avg_eps(g, st, 0.0, 0.5)
    ez = _avg_eps(g, st, 0.0, 0.0)
    exz_x = sp.diags(ex["xz"].ravel())
    exz_z = sp.diags(ez["xz"].ravel())
    Mxz = 0.5 * (exz_x @ Px + Px @ exz_z)
    M = sp.bmat([[sp.diags(ex["xx"].ravel()), Z, Mxz], [Z, sp.diags(ey["yy"].ravel()), Z],
                 [Mxz.T, Z, sp.diags(ez["zz"].ravel())]], format="csr")
    k0 = 2 * np.pi / lam_um
    K0 = (C0.T @ C0 - k0**2 * M).tocsr()
    K1 = (C0.T @ C1 + C1.conj().T @ C0).tocsr()
    K2 = (C1.conj().T @ C1).tocsr()
    # PEC: remove tangential components on the walls at x = x0 (Ey, Ez) and y = y0 (Ex, Ez); the walls at
    # x0 + nx h and y0 + ny h are enforced by the truncated differences.
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    kx = np.ones(N, bool)
    ky = np.ones(N, bool)
    kz = np.ones(N, bool)
    if not g.periodic_x:
        ky &= ii != 0
        kz &= ii != 0
    kx &= jj != 0
    kz &= jj != 0
    keep = np.flatnonzero(np.concatenate([kx, ky, kz]))
    sel = lambda A: A[keep][:, keep].tocsr()  # noqa: E731
    return Operators(g, sel(K0), sel(K1), sel(K2), keep, k0, {"ex": ex, "ey": ey, "ez": ez})


@dataclass
class Mode:
    neff: float
    Ex: np.ndarray  # on nodes (nx, ny), complex
    Ey: np.ndarray
    Ez: np.ndarray
    te_frac: float


def _to_nodes(op: Operators, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = op.g
    N = g.nx * g.ny
    full = np.zeros(3 * N, complex)
    full[op.keep] = v
    Ex = full[:N].reshape(g.nx, g.ny)
    Ey = full[N:2 * N].reshape(g.nx, g.ny)
    Ez = full[2 * N:].reshape(g.nx, g.ny)
    Exn = Ex.copy()
    Exn[1:, :] = 0.5 * (Ex[1:, :] + Ex[:-1, :])
    Eyn = Ey.copy()
    Eyn[:, 1:] = 0.5 * (Ey[:, 1:] + Ey[:, :-1])
    return Exn, Eyn, Ez


def solve(op: Operators, n_sigma: float, k: int = 8, n_min: float = 1.0, n_max: float = 10.0,
          tol: float = 1e-12) -> list[Mode]:
    """Modes with real neff in (n_min, n_max) nearest to n_sigma, sorted by neff descending."""
    sigma = n_sigma * op.k0
    K0, K1, K2 = op.K0, op.K1, op.K2
    n = K0.shape[0]
    Q = (K0 + sigma * K1 + sigma**2 * K2).tocsc()
    lu = splu(Q, permc_spec="COLAMD")
    K1s = (K1 + sigma * K2).tocsr()

    def mv(z):
        z1, z2 = z[:n], z[n:]
        r1, r2 = z1, K2 @ z2
        u = -lu.solve(r2 + K1s @ r1)
        v = r1 + sigma * u
        return np.concatenate([u, v])

    opl = LinearOperator((2 * n, 2 * n), matvec=mv, dtype=complex)
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(2 * n) + 0j
    nu, vecs = eigs(opl, k=k, which="LM", tol=tol, v0=v0, ncv=max(2 * k + 1, 30))
    beta = sigma + 1.0 / nu
    modes = []
    for b, z in zip(beta, vecs.T):
        ne = b / op.k0
        if abs(ne.imag) > 1e-7 or not (n_min < ne.real < n_max):
            continue
        Ex, Ey, Ez = _to_nodes(op, z[:n])
        px = float(np.sum(np.abs(Ex) ** 2))
        py = float(np.sum(np.abs(Ey) ** 2))
        # fix global phase: largest transverse component real positive
        a = Ex if px >= py else Ey
        idx = np.unravel_index(np.argmax(np.abs(a)), a.shape)
        ph = np.exp(-1j * np.angle(a[idx]))
        modes.append(Mode(float(ne.real), Ex * ph, Ey * ph, Ez * ph, px / (px + py)))
    modes.sort(key=lambda m: -m.neff)
    return modes


def transverse_diag(g: Grid, st: Structure, lam_um: float, n_sigma: float, k: int = 6) -> list[float]:
    """Independent check for diagonal tensors: linear eigenproblem L Et = beta^2 Et with
    L = k0^2 eps_t - [[Dy'Dy + Dx ez^-1 Dx' ex, -Dy'Dx + Dx ez^-1 Dy' ey], [-Dx'Dy + Dy ez^-1 Dx' ex, Dx'Dx + Dy ez^-1 Dy' ey]]
    (Ez eliminated with the discrete Gauss law Dx' ex Ex + Dy' ey Ey - i beta ez Ez = 0)."""
    nx, ny, h = g.nx, g.ny, g.h
    N = nx * ny
    Ix, Iy = sp.identity(nx, format="csr"), sp.identity(ny, format="csr")
    Dx = sp.kron(_d1(nx, h, g.periodic_x), Iy, format="csr")
    Dy = sp.kron(Ix, _d1(ny, h, False), format="csr")
    ex = _avg_eps(g, st, 0.5, 0.0)
    ey = _avg_eps(g, st, 0.0, 0.5)
    ez = _avg_eps(g, st, 0.0, 0.0)
    if np.max(np.abs(ex["xz"])) > 0 or np.max(np.abs(ez["xz"])) > 0:
        raise ValueError("transverse_diag handles diagonal tensors only")
    Ex_ = sp.diags(ex["xx"].ravel())
    Ey_ = sp.diags(ey["yy"].ravel())
    iEz = sp.diags(1.0 / ez["zz"].ravel())
    k0 = 2 * np.pi / lam_um
    Lxx = Dy.T @ Dy + Dx @ iEz @ Dx.T @ Ex_
    Lxy = -Dy.T @ Dx + Dx @ iEz @ Dy.T @ Ey_
    Lyx = -Dx.T @ Dy + Dy @ iEz @ Dx.T @ Ex_
    Lyy = Dx.T @ Dx + Dy @ iEz @ Dy.T @ Ey_
    L = k0**2 * sp.bmat([[Ex_, None], [None, Ey_]]) - sp.bmat([[Lxx, Lxy], [Lyx, Lyy]])
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    kx = jj != 0
    ky = np.ones(N, bool) if g.periodic_x else (ii != 0)
    keep = np.flatnonzero(np.concatenate([kx, ky]))
    L = L.tocsr()[keep][:, keep].tocsc()
    vals = eigs(L, k=k, sigma=(k0 * n_sigma) ** 2, which="LM", tol=1e-12, return_eigenvectors=False)
    ne = np.sqrt(vals.astype(complex)) / k0
    ne = sorted([v.real for v in ne if abs(v.imag) < 1e-7], reverse=True)
    return ne
