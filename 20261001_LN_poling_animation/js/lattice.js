// LiNbO3 R3c lattice helpers (hexagonal setting). Atom data comes from data/lattice_data.js (scripts/gen_lattice.py).
const LAT = (() => {
  const D = window.LATTICE, a = D.a, c = D.c, h = a * Math.sqrt(3) / 2;
  const cart = (f) => [f[0] * a - f[1] * a / 2, f[1] * h, f[2] * c]; // linear: also converts fractional displacements
  // Display radii are schematic (not ionic radii); q = formal charge.
  const SP = {
    Li: { q: 1, color: 0xcc80ff, r: 0.42 },
    Nb: { q: 5, color: 0x73c2c9, r: 0.5 },
    O: { q: -2, color: 0xff2a2a, r: 0.58 },
  };
  const base = [];
  for (const sp of ['Li', 'Nb', 'O']) {
    D.species[sp].frac.forEach((f, k) => base.push({ sp, f, d: cart(D.species[sp].disp[k]) }));
  }
  // Atoms (periodic images included) for cell offsets in range, kept where test(frac, cart) is true.
  const gen = (test, ri, rj, rk) => {
    const out = [];
    for (let i = ri[0]; i <= ri[1]; i++) for (let j = rj[0]; j <= rj[1]; j++) for (let k = rk[0]; k <= rk[1]; k++) {
      for (const b of base) {
        const fr = [b.f[0] + i, b.f[1] + j, b.f[2] + k], p = cart(fr);
        if (test(fr, p)) out.push({ sp: b.sp, fr, p, d: b.d });
      }
    }
    return out;
  };
  return { a, c, h, cart, SP, gen, base };
})();
