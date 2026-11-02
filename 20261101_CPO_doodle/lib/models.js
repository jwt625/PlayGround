// Doodle-LOD models. Units are mm for the OE parts. Heights x2 and fiber pitch x2 versus the source assets, for
// legibility; ring and bump counts are illustrative (see DevLog-000).
(() => {
  const { box, cyl, prism, rotBox, tubes, rectTop, discTop, text, guide, label, rng } = DD;

  // Generic open OE, part list and footprints from the Gary film asset oe_module_cpo: carrier 24 x 20, PIC 7 x 9,
  // EIC 2.4 x 4.2 flip-chip on the PIC, FAU with 16 fibers on the +X PIC edge, MT-style ferrule.
  // stage: 'die' = PIC + EIC only, 'oe' = with FAU and fibers.
  function oeGeneric(o = [0, 0, 0], { explode = false, s = 1, stage = 'oe', fibers = true } = {}) {
    const [ox, oy, oz] = o;
    const X = (x) => ox + s * x, Y = (y) => oy + s * y, Z = (z) => oz + s * z;
    const B = (x0, y0, z0, x1, y1, z1, key, opt) => box(X(x0), Y(y0), Z(z0), X(x1), Y(y1), Z(z1), key, opt);
    const zc = 2.0;
    B(-12, -10, 0, 12, 10, zc, 'carrier');
    for (const [x, y] of [[-10, 5], [-10, 7.6], [-10, -7.2], [-10, -4.6], [7, -8.4], [7, 6.8]]) {
      B(x, y, zc, x + 1.8, y + 1.6, zc + 1.0, 'smd', { tier: 'fine' });
    }
    const liftPic = explode ? 9 : 0, liftEic = explode ? 18 : 0, shiftFau = explode ? 10 : 0;
    const zp = zc + 0.1 + liftPic;
    const pic = B(-5.5, -4.5, zp, 1.5, 4.5, zp + 1.5, 'pic');
    const ztop = zp + 1.5;
    for (let i = 0; i < 7; i++) {
      const y = -3.9 + 1.3 * i;
      rectTop(pic, X(-5.3), Y(y - 0.2), X(-4.9), Y(y + 0.2), 'pad');
    }
    for (let k = 0; k < 16; k++) { // waveguide at the FAU pitch, one ring each
      const y = -3.75 + 0.5 * k;
      rectTop(pic, X(-1.2), Y(y - 0.07), X(1.5), Y(y + 0.07), 'wg');
      discTop(pic, X(k % 2 ? -0.2 : 0.6), Y(y + 0.22), s * 0.2, 'ring');
    }
    if (explode) {
      for (let i = 0; i < 4; i++) for (let j = 0; j < 6; j++) { // microbump sites under the EIC
        discTop(pic, X(-4.0 + 0.6 * i), Y(-1.75 + 0.7 * j), s * 0.14, 'pad');
      }
    }
    const zeic = ztop + 0.1 + liftEic;
    B(-4.3, -2.1, zeic, -1.9, 2.1, zeic + 1.0, 'eic');
    if (stage === 'die') return;
    const fx0 = 1.5 + shiftFau, zf = zc + liftPic;
    B(fx0, -4.2, zf, fx0 + 5, 4.2, zf + 1.3, 'glass');
    B(fx0, -4.2, zf + 1.3, fx0 + 4, 4.2, zf + 2.1, 'glass');
    if (fibers) {
      const zax = zf + 1.0, xend = fx0 + 15;
      const paths = [];
      for (let k = 0; k < 16; k++) {
        const y0 = -3.75 + 0.5 * k, y1 = (-1.875 + 0.25 * k) * 0.9, path = [];
        for (let i = 0; i <= 14; i++) {
          const u = i / 14, e = u * u * (3 - 2 * u);
          path.push([X(fx0 + 5 + (xend - fx0 - 5) * u), Y(y0 + (y1 - y0) * e), Z(zax - 0.6 * u)]);
        }
        paths.push(path);
      }
      tubes(paths, s * 0.26, 'fiber');
      B(xend, -3.2, zax - 1.8, xend + 5, 3.2, zax + 0.7, 'ferrule');
      if (explode) label([X(xend + 3), Y(-3.2), Z(zax - 1.8)], '16 fibers', 60, 58);
    }
    if (explode) {
      const g = (a, b) => guide([X(a[0]), Y(a[1]), Z(a[2])], [X(b[0]), Y(b[1]), Z(b[2])]);
      for (const [x, y] of [[-1.9, -2.1], [-4.3, -2.1], [-1.9, 2.1]]) g([x, y, zeic], [x, y, ztop]);
      for (const [x, y] of [[1.5, -4.5], [-5.5, -4.5], [1.5, 4.5]]) g([x, y, zp], [x, y, zc]);
      for (const y of [-4.2, 4.2]) g([fx0, y, zf + 0.6], [1.5, y, zf + 0.6]);
      label([X(-4.3), Y(-2.1), Z(zeic + 0.5)], 'EIC', -70, -6);
      label([X(-5.5), Y(-4.5), Z(zp + 0.7)], 'PIC', -70, -6);
      label([X(fx0 + 2), Y(4.2), Z(zf + 2.1)], 'FAU', 46, -44);
      label([X(-12), Y(-10), Z(1)], 'carrier', -60, 24);
    }
  }

  // Spectrum-6 look, own photo IMG_5711. 6.8 x 11.2 mm footprint. Olive die with QR on the ASIC side (-Y), stamped
  // metal receptacle on the outer side (+Y): two angled posts, V notch, central slot, center tab, two round posts.
  function oeSpectrum(o = [0, 0, 0], s = 1) {
    const [ox, oy, oz] = o;
    const X = (x) => ox + s * x, Y = (y) => oy + s * y, Z = (z) => oz + s * z;
    box(X(-4.4), Y(-6.6), Z(0), X(4.4), Y(6.6), Z(0.8), 'substrate');
    const die = box(X(-3.4), Y(-5.6), Z(0.8), X(3.4), Y(-0.4), Z(1.8), 'gold_die');
    rectTop(die, X(0.2), Y(-4.9), X(2.2), Y(-2.9), 'qr');
    const r = rng(7);
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) {
      if (r() < 0.5) rectTop(die, X(0.32 + 0.3 * i), Y(-4.78 + 0.3 * j), X(0.56 + 0.3 * i), Y(-4.54 + 0.3 * j), 'ink');
    }
    const plate = [[-3.4, -0.1], [3.4, -0.1], [3.4, 4.1], [2.7, 5.6], [1.5, 5.6], [0.6, 4.2], [-0.6, 4.2], [-1.5, 5.6], [-2.7, 5.6], [-3.4, 4.1]];
    const body = prism(plate.map(([x, y]) => [X(x), Y(y)]), Z(0.8), Z(2.4), 'metal', { convex: false });
    rectTop(body, X(-2.6), Y(2.0), X(2.6), Y(3.2), 'screen');
    box(X(-0.45), Y(-0.1), Z(2.4), X(0.45), Y(1.4), Z(2.9), 'metal', { tier: 'fine' });
    cyl(X(-2.3), Y(0.7), Z(2.4), Z(3.0), s * 0.38, 'metal', { tier: 'fine' });
    cyl(X(2.3), Y(0.7), Z(2.4), Z(3.0), s * 0.38, 'metal', { tier: 'fine' });
    for (const sign of [-1, 1]) rotBox(X(sign * 2.2), Y(4.7), s * 0.55, s * 0.22, Z(2.4), Z(2.75), sign * 0.7, 'metal', { tier: 'fine' });
    label([X(4.4), Y(-6.6), Z(0)], 'Spectrum-6 photo', -40, 70);
  }

  // As drawn on the Broadcom flow slide IMG_4084: castellated pale body with two dark strips (EEngine), two square
  // blocks on one short edge added at the OEngine step. Dimensions unknown, proportions read off the icon.
  function oeBroadcom(o = [0, 0, 0], s = 1) {
    const [ox, oy, oz] = o;
    const X = (x) => ox + s * x, Y = (y) => oy + s * y, Z = (z) => oz + s * z;
    const body = box(X(-8), Y(-5.2), Z(0), X(8), Y(5.2), Z(1.2), 'bcm_body');
    rectTop(body, X(-7), Y(-3.6), X(4), Y(-2.6), 'bcm_strip');
    rectTop(body, X(-7), Y(2.6), X(4), Y(3.6), 'bcm_strip');
    for (let i = 0; i < 16; i++) { // castellations along the visible -Y edge and the far +Y edge
      const x = -7.5 + i;
      box(X(x - 0.3), Y(-5.6), Z(0), X(x + 0.3), Y(-5.2), Z(0.9), 'bcm_body', { tier: 'fine' });
      box(X(x - 0.3), Y(5.2), Z(0), X(x + 0.3), Y(5.6), Z(0.9), 'bcm_body', { tier: 'fine' });
    }
    box(X(4.6), Y(-4.6), Z(1.2), X(8), Y(-0.4), Z(2.8), 'bcm_block');
    box(X(4.6), Y(0.4), Z(1.2), X(8), Y(4.6), Z(2.8), 'bcm_block');
    label([X(8), Y(-5.2), Z(0)], 'Broadcom slide icon', -40, 70);
  }

  // Compact line: first half of the slide's row 1, machines behind the belt so the work stays visible. Machine shapes
  // are style-test placeholders, not reference-accurate tools.
  function lineScene() {
    box(-60, -22, -2, 54, 22, 0, 'floor', { ground: true });
    const zb = 6;
    box(-52, -6, zb - 1, 48, 6, zb, 'belt');
    box(-52, -7, zb - 2.4, 48, -6, zb + 0.5, 'rail');
    box(-52, 6, zb - 2.4, 48, 7, zb + 0.5, 'rail');
    for (const x of [-48, -20, 8, 36]) {
      box(x - 1, -7, 0, x + 1, -5, zb - 2.4, 'rail');
      box(x - 1, 5, 0, x + 1, 7, zb - 2.4, 'rail');
    }

    // EIC and PIC trays behind the belt, feeding in
    const table = box(-58, 9, 0, -36, 21, 7, 'frame');
    text(table, 'EIC + PIC', [-47, 9, 4.8], 'my', 2.4);
    for (const [x, key] of [[-56.5, 'eic'], [-47, 'pic']]) {
      box(x, 10.5, 7, x + 8, 19.5, 8, 'tray');
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        box(x + 0.9 + 2.3 * i, 11.4 + 2.5 * j, 8, x + 2.5 + 2.3 * i, 13 + 2.5 * j, 8.5, key, { tier: 'fine' });
      }
    }

    // Column behind the belt, arm over it, tool head on the arm
    function station(x, lines, armKey, tool, lamp) {
      const col = box(x - 6, 9, 0, x + 9, 16, 24, armKey);
      text(col, lines, [x + 1.2, 9, 11], 'my', 2.3);
      box(x - 2, -2, 21, x + 2, 9, 24, armKey);
      box(x - 1.6, -1.6, 15, x + 1.6, 1.6, 21, 'rail');
      if (tool === 'nozzle') cyl(x, 0, 12, 15, 0.8, 'ink', { tier: 'fine' });
      if (tool === 'fau') box(x - 1.3, -1.3, 12, x + 1.3, 1.3, 15, 'glass', { tier: 'fine' });
      if (tool === 'probe') box(x - 2.2, -2.2, 12.6, x + 2.2, 2.2, 15, 'pad');
      if (lamp) cyl(x - 3, 12.5, 24, 26.4, 1.5, lamp);
    }
    station(-25, 'KNOWN\nGOOD DIE', 'frame2', 'probe', 'ok');
    station(-6, 'EENGINE\nASSEMBLY', 'frame', 'nozzle');
    station(15, 'OENGINE\nASSEMBLY', 'frame', 'fau');
    cyl(19, 12.5, 24, 26.6, 3, 'fiber');
    station(35, 'KNOWN\nGOOD OE', 'frame2', 'probe', 'ok');
    const scr = box(29, 10, 24, 43, 15, 33, 'screen');
    text(scr, 'PASS', [36, 10, 28.5], 'my', 4.4);

    // Reject bin in front of the last gate
    const bin = box(27, -21, 0, 41, -11, 5, 'bin');
    text(bin, 'REJECT', [34, -21, 2.6], 'my', 2.6);
    rotBox(31, -18, 3, 2.4, 5, 6, 0.3, 'carrier', { tier: 'fine' });
    rotBox(37, -14, 3, 2.4, 5, 6, 0.9, 'carrier', { tier: 'fine' });

    // Work on the belt: bare stacks before the FAU step, finished engines after it
    const us = 0.34;
    for (const x of [-46, -36, -25, -15, -6, 5, 15]) oeGeneric([x, 0, zb], { s: us, stage: 'die' });
    for (const x of [25, 35, 44]) oeGeneric([x, 0, zb], { s: us, fibers: false });
  }

  window.MODELS = { oeGeneric, oeSpectrum, oeBroadcom, lineScene };
})();
