/* Course: Feedback, Reflection and Zero Tolerance for Retaliation in the Optical Workplace (FB-1550)
   Engine contract: course/COURSE_SPEC.md. Classic script, no DOM access at load time. */
(function () {
  'use strict';

  var MANTRA = 'Every interface reflects. Every reflection is a cavity. Every cavity hops. Isolate, angle, and verify.';
  var MSHORT = ['Every interface reflects.', 'Every reflection is a cavity.', 'Every cavity hops.', 'Isolate. Angle. Verify.'];
  var GARY = 'Gary';
  var MGR = 'Morgan (Manager)';
  var BY = 'Alex (Coworker)';

  /* ---------- reference table (all from sourced research notes) ---------- */
  var R = {
    tkach: { cite: 'Tkach and Chraplyvy, "Regimes of feedback effects in 1.5-um distributed feedback lasers", J. Lightwave Technol. 4, 1655 (1986)', doi: '10.1109/JLT.1986.1074666' },
    marinins: { cite: 'Marinins et al., "Wafer-scale hybrid integration of InP DFB lasers on Si photonics by flip-chip bonding with sub-300nm alignment precision", IEEE J. Sel. Top. Quantum Electron., 2022', doi: '10.1109/jstqe.2022.3223641' },
    kurczveil: { cite: 'Kurczveil et al., "Characterization of insertion loss and back reflection in passive hybrid silicon tapers", IEEE Photonics Journal, 2013', doi: '10.1109/jphot.2013.2246559' },
    moscoso: { cite: 'Moscoso-Martir et al., "Hybrid silicon photonics flip-chip laser integration with vertical self-alignment", CLEO Pacific Rim, 2017', doi: '10.1109/cleopr.2017.8118971' },
    dekoninck: { cite: 'De Koninck et al., "GaAs nano-ridge laser diodes fully fabricated in a 300-mm CMOS pilot line", Nature, 2025', doi: '10.1038/s41586-024-08364-2' },
    bao: { cite: 'Bao et al., "An erbium-doped waveguide amplifier on thin film lithium niobate with an output power exceeding 100 mW", Laser Photonics Rev., 2024', doi: '10.1002/lpor.202400765' },
    celo: { cite: 'Celo et al., Group IV Photonics Conference (GFP), 2017', doi: '10.1109/group4.2017.8082188' },
    tian: { cite: 'Tian et al., Optical and Quantum Electronics, 2025', doi: '10.1007/s11082-024-07958-4' },
    cherchi: { cite: 'Cherchi et al., Optics Express, 2014', doi: '10.1364/oe.22.009245' },
    vaerenbergh: { cite: 'Van Vaerenbergh et al., Optics Express, 2021', doi: '10.1364/oe.433744' },
    contreras: { cite: 'Contreras-Teran et al., "High-resolution strain fiber laser-sensor based on core-offset Mach-Zehnder interferometer", Meas. Sci. Technol., 2023', doi: '10.1088/1361-6501/acb3ea' },
    vanrees: { cite: 'van Rees et al., "Ring resonator enhanced mode-hop-free wavelength tuning of an integrated extended-cavity laser", Optics Express, 2020', doi: '10.1364/oe.386356' },
    xu2021: { cite: 'Xu et al., "Hybrid external-cavity lasers (ECL) using photonic wire bonds as coupling elements", Scientific Reports, 2021', doi: '10.1038/s41598-021-95981-w' },
    franken: { cite: 'Franken et al., "High-power and narrow-linewidth laser on thin-film lithium niobate enabled by photonic wire bonding", APL Photonics, 2025', doi: '10.1063/5.0231827' },
    jalas: { cite: 'Jalas et al., "What is - and what is not - an optical isolator", Nature Photonics, 2013', doi: '10.1038/nphoton.2013.185' },
    shi2015: { cite: 'Shi, Yu and Fan, "Limitations of nonlinear optical isolators due to dynamic reciprocity", Nature Photonics 9, 388 (2015)', doi: '10.1038/nphoton.2015.79' },
    lapointe: { cite: 'Lapointe et al., "Passive broadband Faraday isolator for hybrid integration to photonic circuits without lens and external magnet", Nature Photonics, 2025', doi: '10.1038/s41566-024-01601-0' },
    kashyap: { cite: 'Kashyap and Blow, "Observation of catastrophic self-propelled self-focusing in optical fibres", Electronics Letters, 1988', doi: '10.1049/el:19880032' },
    ji2017: { cite: 'Ji et al., "Ultra-low-loss on-chip resonators with sub-milliwatt parametric oscillation threshold", Optica, 2017', doi: '10.1364/optica.4.000619' },
    hausmann: { cite: 'Hausmann et al., "Diamond nonlinear photonics", Nature Photonics, 2014', doi: '10.1038/nphoton.2014.72' },
    raja: { cite: 'Raja et al., "Electrically pumped photonic integrated soliton microcomb", Nature Communications, 2019', doi: '10.1038/s41467-019-08498-2' },
    white: { cite: 'White et al., "Unified laser stabilization and isolation on a silicon chip", Nature Photonics, 2024', doi: '10.1038/s41566-024-01539-3' },
    kramnik: { cite: 'Kramnik et al., "Scalable feedback stabilization of quantum light sources on a CMOS chip", arXiv:2411.05921, 2024', url: 'https://arxiv.org/abs/2411.05921' },
    chen2026: { cite: 'Chen et al., "Ultrahigh-Q integrated flame-hydrolysis-deposited germano-silicate resonators on silicon", Light: Science and Applications, 2026', doi: '10.1038/s41377-026-02353-y' },
    chowdhury: { cite: 'Chowdhury et al., Optics Express, 2024', doi: '10.1364/oe.510036' },
    lenstra: { cite: 'Lenstra, Verbeek and den Boef, "Coherence collapse in single-mode semiconductor lasers due to optical feedback", IEEE J. Quantum Electron. 21, 674 (1985)', doi: '10.1109/JQE.1985.1072725' },
    malitson: { cite: 'Malitson, "Interspecimen comparison of the refractive index of fused silica", J. Opt. Soc. Am. 55, 1205 (1965)', doi: '10.1364/JOSA.55.001205' },
    li1980: { cite: 'Li, "Refractive index of silicon and germanium and its wavelength and temperature derivatives", J. Phys. Chem. Ref. Data 9, 561 (1980)', doi: '10.1063/1.555624' },
    saleh: { cite: 'Saleh and Teich, Fundamentals of Photonics, Wiley (1991)', doi: '10.1002/0471213748' },
    weiss: { cite: 'Weiss et al., "Splitting of high-Q Mie modes induced by light backscattering in silica microspheres", Optics Letters 20, 1835 (1995)', doi: '10.1364/OL.20.001835' },
    kippenberg: { cite: 'Kippenberg, Spillane and Vahala, "Modal coupling in traveling-wave resonators", Optics Letters 27, 1669 (2002)', doi: '10.1364/OL.27.001669' },
    yariv: { cite: 'Yariv, "Universal relations for coupling of optical power between microresonators and dielectric waveguides", Electronics Letters 36, 321 (2000)', doi: '10.1049/el:20000340' }
  };
  function refs() { var a = []; for (var i = 0; i < arguments.length; i++) a.push(R[arguments[i]]); return { type: 'refs', title: 'References for This Chapter', items: a }; }

  /* ---------- content helpers ---------- */
  /* list items may be [shown on screen, spoken by the voice] */
  function shown(it) { return Array.isArray(it) ? it[0] : it; }
  function spoken(it) { return Array.isArray(it) ? it[1] : it; }
  function objectives(title, items) {
    var n = ['Welcome to this chapter: ' + title + '.', 'By the end of this chapter, you will be able to do the following.'];
    items.forEach(function (it) { n.push(spoken(it)); });
    n.push('Please remember our policy mantra. ' + MANTRA);
    return { type: 'slide', layout: 'bullets', title: 'Learning Objectives', bullets: items.map(shown), callout: MANTRA, narration: n };
  }
  function takeaways(items, extra) {
    var n = ['Let us review the key takeaways.'];
    items.forEach(function (it) { n.push(spoken(it)); });
    n.push('And one more time, together. ' + MANTRA);
    if (extra) n.push(extra);
    n.push('Click Next to continue.');
    return { type: 'takeaways', title: 'Key Takeaways', items: items.map(shown).concat([MANTRA]), narration: n };
  }
  function ack(text) {
    return { type: 'ack', text: text + ' I acknowledge the policy mantra: ' + MANTRA + ' I understand that this attestation is retained in accordance with Policy PHC-1550.4 and may be reviewed during my annual Return Loss Conversation.', narration: ['Please read the attestation carefully, and tick the box to confirm.', 'Your attestation is recorded.'] };
  }
  function isThisOk(line) { return [line, 'Pause and ask yourself. Is this okay?', 'Remember. ' + MANTRA]; }

  /* =================================================================== */
  /* INCIDENT REVIEWS (dramatizations; Gary is a fictional composite)      */
  /* Numbers: Fresnel R = ((n-1)/(n+1))^2 with n = 1.444 fused silica      */
  /* (Malitson 1965) -> 3.3 %, -14.8 dB; n = 3.48 silicon (Li 1980) ->     */
  /* 31 %; air-gap etalon period lambda^2 / (2 L); Lapointe 2025 isolator  */
  /* > 25 dB isolation, < 1.5 dB insertion loss (stated as an assumption); */
  /* doublet numbers from the chapter-5 coupled-mode model.                */
  /* =================================================================== */
  var APPROVED = 'I noticed something on the optical path. Can we pause and check it together?';
  var DISCLAIM = 'The following incident is a dramatization. Gary is a fictional composite. Any resemblance to engineers, living or lasing, is coincidental.';
  function OK(t) { return { text: t, correct: true, why: '' }; }
  function W(t, why) { return { text: t, correct: false, why: why + ' Isolate, angle, and verify. And log it.' }; }
  function WB(t, why) { return { text: t, correct: false, why: why + ' The approved phrase is: ' + APPROVED + '' }; }
  function tester(qn, opts, say) {
    return { type: 'check', kicker: 'What would you do?', question: qn, options: opts, correctSay: say,
      narration: ['What would you do?', 'Put yourself in Gary\'s position. There are no wrong answers. There are only answers that get struck out.'] };
  }
  function bystander(qn, opts, say) {
    return { type: 'check', kicker: 'You are the bystander', question: qn, options: opts, correctSay: say,
      narration: ['You are the bystander.', 'Every optical path has bystanders. So does every lab.', 'Remember the approved phrase. ' + APPROVED] };
  }
  function felt(situation, bullets, narr) {
    return { type: 'slide', layout: 'host', title: 'How Do You Think Gary Felt?',
      bullets: [situation, 'Gary felt ashamed', 'Shame is a normal response to a reflection, including your own', 'This is a judgment-free zone', 'The incident has still been logged'].concat(bullets || []),
      callout: 'Judgment-free. Still logged.',
      narration: ['Let us pause for a moment.', situation + '.', 'How do you think Gary felt?', 'Gary felt ashamed.', 'That is normal. Shame is a normal response to a reflection, including your own.', 'This is a judgment-free zone. Nobody in this course is judging Gary.', 'The incident has still been logged.'].concat(narr || []).concat(['Remember. ' + MANTRA]) };
  }
  function feltTabs(situation, tabs) {
    return { type: 'reveal', title: 'How Do You Think Gary Felt?', prompt: situation + '. How do you think Gary felt? Open each feeling. All feelings are valid. All feelings are logged.', tabs: tabs,
      narration: ['Let us pause for a moment.', situation + '.', 'How do you think Gary felt? Please open each feeling.', 'All feelings are valid. All feelings are logged.'] };
  }
  function telegraph(x) { return (Math.sin(x * 37) + Math.sin(x * 91) > 0.3 ? 0.75 : 0.42); }


  /* ---------- animatic drawing helpers (runtime only) ---------- */
  function C(name, fb) {
    try { var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim(); return v || fb; } catch (e) { return fb; }
  }
  function cols() {
    return { bg: C('--panel-2', '#14171b'), panel: C('--panel', '#1d2228'), fg: C('--ink', '#e3e6ea'), muted: C('--ink-2', '#8b939c'),
      accent: C('--accent', '#4ea1ff'), bad: C('--bad', '#e5534b'), ok: C('--ok', '#3fb950'), warn: C('--warn', '#d29922'), border: C('--line', '#3a414a') };
  }
  function clear(ctx, w, h) { var c = cols(); ctx.fillStyle = c.panel; ctx.fillRect(0, 0, w, h); return c; }
  function text(ctx, s, x, y, size, color, align) {
    var sz = size || 14, cw = (ctx.canvas && (ctx.canvas.clientWidth || ctx.canvas.width)) || 800;
    ctx.font = sz + 'px system-ui, sans-serif';
    while (sz > 8 && ctx.measureText(s).width > cw - 12) { sz -= 1; ctx.font = sz + 'px system-ui, sans-serif'; }
    ctx.fillStyle = color; ctx.textAlign = align || 'left'; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y);
  }
  function stamp(ctx, w, h, u, s, color) {
    if (u < 0.55) return;
    var k = Math.min(1, (u - 0.55) / 0.1);
    ctx.save(); ctx.translate(w * 0.78, h * 0.72); ctx.rotate(-0.18); ctx.scale(Math.min(1, w / 640), Math.min(1, w / 640)); ctx.globalAlpha = k;
    ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.strokeRect(-90, -22, 180, 44);
    text(ctx, s, 0, 0, 20, color, 'center'); ctx.restore();
  }
  function stick(ctx, x, y, s, name, color) {
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y - s * 1.6, s * 0.35, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, y - s * 1.25); ctx.lineTo(x, y - s * 0.4); ctx.moveTo(x - s * 0.5, y - s); ctx.lineTo(x + s * 0.5, y - s);
    ctx.moveTo(x, y - s * 0.4); ctx.lineTo(x - s * 0.35, y); ctx.moveTo(x, y - s * 0.4); ctx.lineTo(x + s * 0.35, y); ctx.stroke();
    text(ctx, name, x, y + 12, 11, color, 'center');
  }
  /* chip with laser, bus waveguide, reflectors; forward pulse and returning echoes */
  function dChip(ctx, w, h, u, o) {
    var c = clear(ctx, w, h); o = o || {};
    var y = h * 0.55, x0 = w * 0.12, x1 = w * 0.88, xl = w * 0.24;
    ctx.fillStyle = c.bg; ctx.fillRect(x0, y - h * 0.12, x1 - x0, h * 0.24);
    ctx.strokeStyle = c.border; ctx.strokeRect(x0, y - h * 0.12, x1 - x0, h * 0.24);
    ctx.fillStyle = c.warn; ctx.fillRect(x0 + 4, y - 10, xl - x0 - 8, 20); text(ctx, 'laser', (x0 + xl) / 2, y - 22, 11, c.muted, 'center');
    ctx.strokeStyle = c.accent; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xl, y); ctx.lineTo(x1 - 6, y); ctx.stroke();
    var refl = o.refl || [0.95];
    refl.forEach(function (f, i) {
      var xr = xl + f * (x1 - 6 - xl);
      ctx.fillStyle = i === refl.length - 1 && o.hotFar ? c.bad : c.muted; ctx.fillRect(xr - 2, y - 9, 4, 18);
      var back = xr - ((u * 1.6) % 1) * (xr - xl);
      ctx.globalAlpha = 0.35 + 0.4 * (i === refl.length - 1 && o.hotFar ? 1 : 0.3);
      ctx.fillStyle = c.bad; ctx.beginPath(); ctx.arc(back, y + 5, 4, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
    });
    var fw = xl + ((u * 1.6) % 1) * (x1 - 6 - xl); ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(fw, y - 5, 5, 0, 7); ctx.fill();
    if (o.ruler) {
      var k = Math.min(1, u * 1.4); ctx.strokeStyle = c.ok; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(xl, y + h * 0.2); ctx.lineTo(xl + k * (x1 - 6 - xl), y + h * 0.2); ctx.stroke(); ctx.setLineDash([]);
      text(ctx, 'parasitic cavity length', (xl + x1) / 2, y + h * 0.26, 12, c.ok, 'center');
    }
    if (o.gary) stick(ctx, w * 0.06, h * 0.9, 18, 'Gary', c.fg);
    if (o.label) text(ctx, o.label, w / 2, h * 0.12, 15, c.fg, 'center');
    if (o.stamp) stamp(ctx, w, h, u, o.stamp, o.stampColor === 'ok' ? c.ok : c.bad);
  }
  /* plotted trace that draws itself progressively */
  function dTrace(ctx, w, h, u, o) {
    var c = clear(ctx, w, h), L = w * 0.1, Rr = w * 0.92, T = h * 0.2, B = h * 0.82;
    ctx.strokeStyle = c.border; ctx.lineWidth = 1; ctx.strokeRect(L, T, Rr - L, B - T);
    text(ctx, o.x || '', (L + Rr) / 2, B + 16, 11, c.muted, 'center');
    ctx.save(); ctx.translate(L - 14, (T + B) / 2); ctx.rotate(-Math.PI / 2); text(ctx, o.y || '', 0, 0, 11, c.muted, 'center'); ctx.restore();
    var n = 300, end = Math.max(2, Math.floor(n * Math.min(1, u * 1.25)));
    (o.fs || [o.f]).forEach(function (f, j) {
      ctx.strokeStyle = j === 0 ? c.accent : c.warn; ctx.lineWidth = 2; ctx.beginPath();
      for (var i = 0; i < end; i++) { var xx = i / (n - 1), yy = f(xx, u); var px = L + xx * (Rr - L), py = B - yy * (B - T); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
      ctx.stroke();
    });
    if (o.label) text(ctx, o.label, w / 2, h * 0.09, 15, c.fg, 'center');
    if (o.stamp) stamp(ctx, w, h, u, o.stamp, o.stampColor === 'ok' ? c.ok : c.bad);
  }
  /* comb of cavity modes under a moving filter */
  function dComb(ctx, w, h, u, o) {
    var c = clear(ctx, w, h); o = o || {};
    var B = h * 0.8, sp = w / 11, off = (u * sp * 1.2) % sp, fc = w * (0.25 + 0.5 * u), fw = w * 0.12;
    ctx.strokeStyle = c.warn; ctx.lineWidth = 2; ctx.beginPath();
    for (var x = 0; x <= w; x += 3) { var g = 1 / (1 + Math.pow((x - fc) / fw, 2)); var py = B - g * h * 0.55; if (x) ctx.lineTo(x, py); else ctx.moveTo(x, py); }
    ctx.stroke();
    var best = null, bd = 1e9;
    for (var k = -1; k < 13; k++) { var xm = k * sp + off; var d = Math.abs(xm - fc); if (d < bd) { bd = d; best = xm; } }
    for (var k2 = -1; k2 < 13; k2++) {
      var xm2 = k2 * sp + off, g2 = 1 / (1 + Math.pow((xm2 - fc) / fw, 2));
      ctx.strokeStyle = xm2 === best ? c.accent : c.muted; ctx.lineWidth = xm2 === best ? 4 : 1.5;
      ctx.beginPath(); ctx.moveTo(xm2, B); ctx.lineTo(xm2, B - (xm2 === best ? 0.7 : 0.2 + 0.25 * g2) * h * 0.6); ctx.stroke();
    }
    ctx.strokeStyle = c.border; ctx.beginPath(); ctx.moveTo(0, B); ctx.lineTo(w, B); ctx.stroke();
    text(ctx, o.label || 'cavity modes (gray), filter (amber), lasing mode (blue)', w / 2, h * 0.1, 14, c.fg, 'center');
    if (o.stamp) stamp(ctx, w, h, u, o.stamp, o.stampColor === 'ok' ? c.ok : c.bad);
  }
  /* ring resonator with CW and CCW light, optional scatterer and standing wave */
  function dRing(ctx, w, h, u, o) {
    var c = clear(ctx, w, h); o = o || {};
    var cx = w / 2, cy = h * 0.5, r = Math.min(w, h) * 0.28;
    ctx.strokeStyle = c.accent; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(cx, cy, r, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(w * 0.08, cy + r + 12); ctx.lineTo(w * 0.92, cy + r + 12); ctx.stroke();
    var a = u * Math.PI * 4;
    ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(cx + r * Math.cos(a), cy + r * Math.sin(a), 6, 0, 7); ctx.fill();
    if (o.ccw) { ctx.fillStyle = c.bad; ctx.beginPath(); ctx.arc(cx + r * Math.cos(-a + 1), cy + r * Math.sin(-a + 1), 6, 0, 7); ctx.fill(); }
    if (o.speck) { ctx.fillStyle = c.warn; ctx.beginPath(); ctx.arc(cx + r * Math.cos(-0.6), cy + r * Math.sin(-0.6), 4, 0, 7); ctx.fill(); }
    if (o.seams) { ctx.strokeStyle = c.panel; ctx.lineWidth = 5; for (var s = 0; s < 8; s++) { var t = s * Math.PI / 4; ctx.beginPath(); ctx.moveTo(cx + (r - 6) * Math.cos(t), cy + (r - 6) * Math.sin(t)); ctx.lineTo(cx + (r + 6) * Math.cos(t), cy + (r + 6) * Math.sin(t)); ctx.stroke(); } }
    if (o.standing && u > 0.4) {
      ctx.fillStyle = c.fg; for (var n = 0; n < 24; n++) { var t2 = n * Math.PI / 12; ctx.globalAlpha = 0.3 + 0.7 * Math.abs(Math.cos(t2 * 3 + u * 2)); ctx.beginPath(); ctx.arc(cx + (r - 14) * Math.cos(t2), cy + (r - 14) * Math.sin(t2), 2.5, 0, 7); ctx.fill(); }
      ctx.globalAlpha = 1;
    }
    if (o.gary) stick(ctx, w * 0.1, h * 0.75, 18, 'Gary', c.fg);
    if (o.label) text(ctx, o.label, w / 2, h * 0.08, 15, c.fg, 'center');
    if (o.stamp) stamp(ctx, w, h, u, o.stamp, o.stampColor === 'ok' ? c.ok : c.bad);
  }
  function dCard(ctx, w, h, u, o) {
    var c = clear(ctx, w, h);
    text(ctx, o.title, w / 2, h * 0.42, 22, c.fg, 'center');
    if (o.sub) text(ctx, o.sub, w / 2, h * 0.56, 14, c.muted, 'center');
    if (o.people) o.people.forEach(function (p, i) { stick(ctx, w * (0.2 + 0.6 * i / Math.max(1, o.people.length - 1)), h * 0.92, 16, p, c.muted); });
    if (o.stamp) stamp(ctx, w, h, u, o.stamp, o.stampColor === 'ok' ? c.ok : c.bad);
  }
  function lor(x, x0, wd) { return 1 / (1 + Math.pow((x - x0) / wd, 2)); }
  /* isolator scene: laser, isolator (normal, reversed, removed or rotated 90 degrees), downstream reflector or source */
  function dIso(ctx, w, h, u, o) {
    var c = clear(ctx, w, h); o = o || {};
    var y = h * 0.52, xl = w * 0.16, xi = w * 0.5, xr = w * 0.86, bw = Math.min(80, w * 0.12);
    var k = (u * 1.4) % 1, back = !o.rot && (o.none || o.rev);
    ctx.strokeStyle = c.accent; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xl, y); ctx.lineTo(xr, y); ctx.stroke();
    ctx.fillStyle = back && u > 0.5 ? c.bad : c.warn; ctx.fillRect(xl - 44, y - 13, 44, 26); text(ctx, 'laser', xl - 22, y - 26, 11, c.muted, 'center');
    ctx.fillStyle = c.muted; ctx.fillRect(xr - 2, y - 14, 5, 28); text(ctx, o.right || 'reflector', xr, y - 26, 11, c.muted, 'center');
    if (o.none) { ctx.setLineDash([4, 4]); ctx.strokeStyle = c.muted; ctx.strokeRect(xi - bw / 2, y - 18, bw, 36); ctx.setLineDash([]); text(ctx, 'isolator removed', xi, y - 30, 11, c.muted, 'center'); }
    else {
      ctx.fillStyle = c.bg; ctx.fillRect(xi - bw / 2, y - 18, bw, 36); ctx.strokeStyle = c.border; ctx.strokeRect(xi - bw / 2, y - 18, bw, 36);
      var d = o.rev ? -1 : 1; ctx.strokeStyle = c.fg; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xi - d * bw * 0.3, y); ctx.lineTo(xi + d * bw * 0.3, y);
      ctx.moveTo(xi + d * bw * 0.3, y); ctx.lineTo(xi + d * bw * 0.15, y - 7); ctx.moveTo(xi + d * bw * 0.3, y); ctx.lineTo(xi + d * bw * 0.15, y + 7); ctx.stroke();
      text(ctx, o.rot ? 'isolator, rotated 90 degrees' : 'isolator', xi, y - 30, 11, c.muted, 'center');
      if (o.rot) { ctx.strokeStyle = c.bad; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xi - bw / 2 - 10, y - 12); ctx.lineTo(xi - bw / 2 - 10, y + 12); ctx.stroke(); text(ctx, 'input polarizer crossed with the laser polarization', xi, y + 34, 11, c.bad, 'center'); }
    }
    var blockF = !o.none && (o.rev || o.rot), fx = xl + k * (xr - xl);
    if (blockF && fx > xi - bw / 2) { ctx.globalAlpha = Math.max(0, 1 - (fx - xi + bw / 2) / 40); fx = xi - bw / 2; }
    ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(fx, y - 5, 5, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
    var bx = xr - k * (xr - xl);
    if (!o.rot && (back || bx > xi + bw / 2)) { ctx.fillStyle = c.bad; ctx.globalAlpha = 0.8; ctx.beginPath(); ctx.arc(bx, y + 6, 4, 0, 7); ctx.fill(); ctx.globalAlpha = 1; }
    if (o.gary) stick(ctx, w * 0.06, h * 0.92, 18, 'Gary', c.fg);
    if (o.label) text(ctx, o.label, w / 2, h * 0.12, 15, c.fg, 'center');
    if (o.stamp) stamp(ctx, w, h, u, o.stamp, o.stampColor === 'ok' ? c.ok : c.bad);
  }

  /* =================================================================== */
  /* CHAPTER 0: Welcome and Course Navigation                              */
  /* =================================================================== */
  var ch0 = {
    id: 'ch0', title: 'Welcome and Course Navigation', subtitle: 'How to complete this mandatory training',
    objectives: ['Navigate the course', 'Understand enforcement features', 'Recite the policy mantra'],
    steps: [
      { type: 'slide', layout: 'title', title: 'Feedback, Reflection and Zero Tolerance for Retaliation in the Optical Workplace',
        bullets: ['Course FB-1550, version 2026.4', 'Policy PHC-1550.4: Speak-Up, Non-Retaliation and Return Loss Standard', 'Mandatory for all staff who emit, guide, couple, reflect, or are near light'],
        narration: ['Welcome to Feedback, Reflection, and Zero Tolerance for Retaliation in the Optical Workplace.', 'This course is mandatory for all employees of Intracavity Photonics, Incorporated.', 'It is also mandatory for contractors, interns, visitors, and any laser on the premises.', 'This course takes approximately one hundred and fifty-five minutes.', 'Thank you for taking the time to keep our photons safe, respected, and travelling forward.'] },
      { type: 'slide', layout: 'host', title: 'A Message From Pat Reynolds',
        bullets: ['Director, Office of Photonic Conduct', 'Every photon deserves to travel forward without fear', 'Feedback is welcome. Unsolicited optical feedback is not'],
        callout: MANTRA,
        narration: ['Hi. I am Pat Reynolds, Director of the Office of Photonic Conduct.', 'At Intracavity Photonics, we believe in feedback.', 'We believe in open doors, open communication, and an open speak-up culture.', 'What we do not believe in is optical feedback that nobody asked for.', 'Light that comes back into a laser cavity is not a conversation. It is retaliation.', 'And we have zero tolerance for retaliation.', 'Over the next seven chapters, we will look at where retaliation comes from, how it forms cavities you never designed, and how those cavities make lasers hop.', 'Our mantra for this course is simple. ' + MANTRA, 'You will hear it often. That is intentional.'] },
      { type: 'slide', layout: 'policy', title: 'Course Enforcement Features',
        bullets: ['The Next button stays locked until narration ends', 'Playback speed cannot be changed. Light does not hurry; neither will you', 'Leaving this browser tab pauses the course', 'All hypothetical scenarios are mandatory, including the extra ones', 'Knowledge checks allow unlimited retries', 'A certificate is issued after the final assessment (80 percent to pass)'],
        callout: 'These controls exist for your benefit and cannot be disabled.',
        narration: ['Before we begin, a few words about how this course works.', 'The Next button stays locked until the narration ends.', 'Playback speed cannot be changed.', 'If you leave this browser tab, the course will pause and wait for you, like a laser waiting for its cavity phase to be set.', 'All hypothetical scenarios are mandatory. Including the extra ones. Especially the extra ones.', 'Knowledge checks allow unlimited retries. Wrong answers come with a detailed explanation, read aloud in full.', 'These controls exist for your benefit and cannot be disabled.'] },
      { type: 'reveal', title: 'Key Terms Used in This Course', prompt: 'Open each tab to review the policy definitions.',
        tabs: [
          { label: 'Feedback', body: '<p><b>Optical feedback</b>: any light that re-enters a laser cavity after leaving it. It adds a delayed field to the field inside the cavity. Its effect depends on how much returns, with what phase, and from how far away.</p>', narration: ['Optical feedback is any light that re-enters a laser cavity after leaving it.', 'It adds a delayed copy of the field to the field inside the cavity.', 'What happens next depends on how much light returns, with what phase, and from how far away.'] },
          { label: 'Retaliation', body: '<p><b>Retaliation</b> (policy term): a back-reflection. Every refractive index step, facet, coupler end, crossing, splice and connector can retaliate. Retaliation does not require intent.</p>', narration: ['In this course, retaliation is our policy term for a back-reflection.', 'Every refractive index step can retaliate. Facets, coupler ends, crossings, splices and connectors.', 'Retaliation does not require intent.'] },
          { label: 'Parasitic cavity', body: '<p><b>Parasitic cavity</b>: any two reflections on the same optical path. Together they form a Fabry-Perot etalon that nobody drew on the layout. Its free spectral range is c / (2 n L), with n the group index and L the spacing.</p>', narration: ['A parasitic cavity is any two reflections on the same optical path.', 'Together they form a Fabry-Perot etalon that nobody drew on the layout.', 'Its free spectral range is the speed of light divided by two times the group index times the spacing.'] },
          { label: 'Mode hop', body: '<p><b>Mode hop</b>: a sudden jump of a laser from one longitudinal cavity mode to another, usually by one free spectral range, when the mode with the most net gain changes. Mode hops often show hysteresis.</p>', narration: ['A mode hop is a sudden jump of a laser from one cavity mode to another.', 'It happens when a different mode starts to win the competition for gain.', 'Mode hops often show hysteresis, so the direction you sweep in matters.'] }
        ], narration: ['Before we begin, let us agree on some terms.'] },
      { type: 'slide', layout: 'policy', title: 'Speak-Up Procedure for Optical Incidents',
        bullets: ['Notice: a ripple, a hop, a missing isolator, an arrow pointing the wrong way, a fit term with no physical meaning', 'Name it, using the approved phrase: "' + APPROVED + '"', 'Log it in the lab log, with the setup exactly as it was, before anything is changed', 'Tell the owner of the device or setup the same day', 'Retaliation against people who speak up is prohibited. Retaliation by light is covered in Chapters 1 to 7'],
        callout: APPROVED,
        narration: ['Policy section one point one. The Speak-Up Procedure for Optical Incidents.', 'Step one. Notice. A ripple. A hop. A missing isolator. An arrow pointing the wrong way.', 'Step two. Name it, using the approved phrase. ' + APPROVED, 'Step three. Log it, with the setup exactly as it was, before anything is changed.', 'Step four. Tell the owner of the device or setup, the same day.', 'Retaliation against people who speak up is prohibited. Retaliation by light is covered in the rest of this course.', 'Let us practice the approved phrase together. ' + APPROVED, 'Once more. ' + APPROVED, 'You will be asked to use it later. That is intentional.'] },
      { type: 'check', question: 'Practice question. Which of the following is a reportable event under Policy PHC-1550.4?',
        options: [
          { text: 'Light reflected from a far-end grating coupler re-enters a laser and ripples its light versus current curve', correct: true, why: '' },
          { text: 'A coworker gives you constructive verbal feedback on your layout', correct: false, why: 'Verbal feedback from a coworker is welcome and encouraged under our speak-up culture. This policy covers optical feedback: light that returns into a laser cavity. Constructive comments on your layout do not re-enter any cavity, do not beat with any intracavity field, and do not cause mode hops. Please speak up, and please keep your reflections to yourself.' },
          { text: 'Nothing is reportable if the laser is a distributed feedback laser, because it already has feedback', correct: false, why: 'A distributed feedback laser has feedback that was designed: a grating, with a known phase, inside the cavity. Light returning from outside the cavity is a different matter. It arrives with a delay and a phase that nobody controls, and it competes with the designed feedback. Distributed feedback lasers were the subject of the classic study of external feedback regimes by Tkach and Chraplyvy in 1986. Having feedback does not exempt you from feedback.' }
        ], correctSay: 'Unsolicited optical feedback is always reportable.', narration: ['Let us try a practice question. It does not count toward your score.'] },
      takeaways(['This course is mandatory, and the extra scenarios are mandatory too', 'Feedback is light returning into a laser cavity', 'Retaliation means back-reflection, and it does not require intent', 'The approved phrase: ' + APPROVED], 'You are now ready to begin Chapter One.'),
      ack('I have reviewed the course navigation and enforcement features.')
    ]
  };

  /* =================================================================== */
  /* CHAPTER 1: Recognizing Unsolicited Feedback                           */
  /* =================================================================== */
  var ch1 = {
    id: 'ch1', title: 'Recognizing Unsolicited Feedback', subtitle: 'Regimes, distances, and the reflector you did not suspect',
    objectives: ['Describe the five feedback regimes', 'Identify far-end reflectors as feedback sources', 'Apply return-loss budgets to tapers and couplers'],
    steps: [
      objectives('Recognizing Unsolicited Feedback', [
        'Describe the five regimes of external optical feedback in semiconductor lasers',
        'Recognize that the strongest feedback can come from the far end of the chip, not the nearest gap',
        'Apply a return-loss budget to tapers, couplers and splitters',
        'Explain why the distance to a reflector matters as much as its strength'
      ]),
      { type: 'video', title: 'Cold Open: The Ripple', duration: 26,
        keyframes: [
          { t: 0, cast: ['Gary'], caption: 'Gary flip-chip bonds an InP laser onto a silicon photonics chip.', shot: 'Lab bench, medium shot from 35 degrees above bench height, 50 mm lens. Flip-chip bonder with a small InP laser die held by the collet, lowering onto a silicon photonics chip with silicon nitride waveguides and grating couplers, on the chuck. Gary leans in, safety glasses on forehead. Too-clean training-video lighting, soft key from left.', draw: function (ctx, w, h, u) { dChip(ctx, w, h, u, { gary: true, label: 'Flip-chip InP laser on a silicon photonics chip' }); } },
          { t: 7, cast: ['Gary'], caption: 'The light versus current curve has periodic ripples.', shot: 'Over-the-shoulder on Gary to a bench monitor. Light versus current curve plotting live, with a periodic ripple. Slow push-in on the monitor. Gary points at the gap between laser and chip on a second monitor showing a microscope view.', draw: function (ctx, w, h, u) { dTrace(ctx, w, h, u, { label: 'Light versus current (illustrative)', x: 'current', y: 'power', f: function (x) { var p = Math.max(0, x - 0.15); return 0.05 + 0.9 * p + 0.04 * Math.sin(x * 60) * (p > 0 ? 1 : 0); } }); } },
          { t: 13, cast: ['Gary'], caption: 'Gary blames the sub-micron bond gap and reworks the bond. Three times.', shot: 'Montage, locked-off wide. Gary lifts the die, re-bonds, re-measures. Wall clock advances between cuts. The ripple on the monitor does not change. Freeze frame, red X stamp.', draw: function (ctx, w, h, u) { dChip(ctx, w, h, u, { refl: [0.03], gary: true, label: 'Suspect: the gap next to the laser', stamp: 'VIOLATION' }); } },
          { t: 19, cast: ['Pat Reynolds'], caption: 'Replay: measure the whole path. The far-end grating coupler is the second mirror.', shot: 'Top-down macro of the chip. An animated ruler extends from the laser facet across the entire chip to the grating coupler at the far end. The coupler glows red. Green check stamp.', draw: function (ctx, w, h, u) { dChip(ctx, w, h, u, { refl: [0.03, 0.97], hotFar: true, ruler: true, label: 'The whole chip is the cavity', stamp: 'COMPLIANT', stampColor: 'ok' }); } }
        ],
        narration: ['Meet Gary.', 'Gary has bonded a laser onto a silicon photonics chip.', 'The light versus current curve has ripples. Regular, periodic ripples.', 'Gary blames the tiny gap between the laser and the chip, because it is the closest interface.', 'He reworks the bond. Then he reworks it again.', 'Let us replay that. This time, Gary measures the whole optical path.', 'The second mirror was at the far end of the chip all along.'] },
      { type: 'slide', layout: 'host', title: 'What Is Unsolicited Feedback?',
        bullets: ['Light returning into the laser adds a delayed field to the cavity field', 'The effect depends on returned power, its phase, and the delay (distance)', 'Changing current or temperature changes the phase, so the effect is periodic', 'Feedback does not need your permission, or your layout'],
        callout: MANTRA,
        narration: ['Let us take a closer look.', 'When light returns into a laser, it adds a delayed copy of the field to the field already inside the cavity.', 'Whether that helps or hurts depends on three things. How much light returns. Its phase. And how long it took to come back.', 'Current and temperature both change the phase, so the symptoms repeat periodically as you tune.', 'That is why a ripple with a regular period is a confession, not a coincidence.', 'Remember. ' + MANTRA] },
      { type: 'reveal', title: 'The Five Regimes of Feedback', prompt: 'Tkach and Chraplyvy (1986) classified feedback effects on 1.5 micron distributed feedback lasers into five regimes, ordered by increasing feedback. Open each regime.',
        tabs: [
          { label: 'Regime I', body: '<p>Very weak feedback. The linewidth narrows or broadens depending on the phase of the returning light.</p><p class="muted">The thresholds between regimes depend on the laser and on the distance to the reflector. This course does not quote numbers for them.</p>', narration: ['Regime one. Very weak feedback.', 'The linewidth narrows or broadens, depending on the phase of the returning light.', 'Even a whisper of retaliation changes the laser.'] },
          { label: 'Regime II', body: '<p>Stronger feedback. The laser can split into or hop between external-cavity modes. This is where mode hopping from feedback lives.</p>', narration: ['Regime two. Stronger feedback.', 'The laser can hop between modes of the new, larger cavity that now includes the reflector.', 'Every reflection is a cavity. Every cavity hops.'] },
          { label: 'Regime III', body: '<p>A narrow range where the laser settles on a single external-cavity mode with a narrow linewidth, insensitive to the feedback phase.</p>', narration: ['Regime three. A narrow window where the laser settles on one external cavity mode, with a narrow line.', 'This regime is narrow. Do not plan your career around it.'] },
          { label: 'Regime IV', body: '<p><b>Coherence collapse.</b> The linewidth broadens dramatically and the laser becomes noisy. In this regime the behavior is largely insensitive to the distance and phase of the reflector.</p>', narration: ['Regime four. Coherence collapse.', 'The linewidth broadens dramatically and the laser becomes noisy.', 'Here, changing the distance to the reflector does not save you.'] },
          { label: 'Regime V', body: '<p>Strong feedback into a laser with an anti-reflection coated facet: stable operation as an external-cavity laser. This is feedback that was designed, with consent.</p>', narration: ['Regime five. Strong feedback into a laser whose facet is anti-reflection coated.', 'The laser now runs stably as an external cavity laser.', 'This is feedback with consent. It was designed, documented, and approved.'] }
        ], narration: ['Feedback is not one thing. It is five things.'] },
      { type: 'slide', layout: 'bullets', title: 'Retaliation From an Unexpected Direction',
        bullets: ['Flip-chip InP distributed feedback lasers on silicon photonics showed light versus current ripples with a 25 to 30 mA period (Marinins et al., 2022)', 'The cause: about 4 percent (-14 dB) feedback from a silicon nitride grating coupler at the far end of the waveguide', 'Not the sub-micron gap between the laser and the chip', 'Everyone suspects the closest interface. Measure the whole path'],
        callout: 'The far end of your chip is part of your laser until proven otherwise.',
        narration: ['This is not hypothetical.', 'Marinins and coworkers flip-chip bonded indium phosphide distributed feedback lasers onto silicon photonics.', 'The light versus current curves showed ripples with a period of twenty-five to thirty milliamps.', 'The cause was about four percent feedback, minus fourteen decibels, from a silicon nitride grating coupler at the far end of the waveguide.', 'Not the gap of less than one micron right next to the laser.', 'Everyone suspects the closest interface. Please measure the whole path.', 'Every interface reflects. Every reflection is a cavity.'] },
      { type: 'slide', layout: 'policy', title: 'Return Loss Budgets and Placement',
        bullets: ['Reflections from a III-V to silicon mode converter are cited as needing to be below -50 dB for feedback-sensitive circuits; measured passive tapers were below -41 dB (Kurczveil et al., 2013)', 'Tapers are usually optimized for insertion loss, not return loss. Budget both', 'Laser sensitivity to back-reflection depends strongly on the distance to the reflector; one argument is to place the isolator at the chip output rather than right after the laser (Moscoso-Martir et al., 2017)', 'An isolator anywhere is not the same as an isolator in the right place'],
        narration: ['Policy section four point two, Return Loss Budgets.', 'For feedback-sensitive circuits, reflections from a mode converter between the laser material and silicon are cited as needing to be below minus fifty decibels.', 'Kurczveil and coworkers measured passive hybrid tapers below minus forty-one decibels.', 'Read that again. The measured number and the cited requirement are not the same number.', 'Tapers are usually optimized for insertion loss. Please budget return loss too.', 'Placement also matters. Laser sensitivity depends strongly on the distance to the reflector.', 'That is one argument for placing the isolator at the chip output, after the reflectors, rather than directly after the laser.', 'Isolate, angle, and verify.'] },
      { type: 'chat', title: 'Scenario 1: The Rework', app: 'phone', meta: { header: 'Messages', participants: [GARY, MGR] },
        messages: [
          { from: GARY, text: 'L-I curve has ripples again. Regular period, maybe 25 mA', side: 'right' },
          { from: GARY, text: 'Must be the bond gap. Its the closest interface. Reworking', side: 'right' },
          { from: MGR, text: 'Approved. Is this the third rework?', side: 'left' },
          { from: GARY, text: 'Fourth. The gap is under a micron, it cannot get smaller', side: 'right' },
          { from: MGR, text: 'Then make it smaller', side: 'left' }
        ], narration: isThisOk('Gary has reworked the bond four times, and the ripple period has not changed.') },
      { type: 'chat', title: 'Scenario 2: The Flat Cleave', app: 'email', meta: { header: 'Inbox', subject: 'RE: RE: Laser defective, requesting replacement', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Hi all, the laser is defective. The output power jumps when I bring the fiber close to the facet. Requesting a replacement.', side: 'right' },
          { from: BY, text: 'Is the fiber flat cleaved? A bare glass to air end reflects a few percent by the Fresnel formula. Your fiber tip is now a mirror, and the gap is a cavity.', side: 'left' },
          { from: GARY, text: 'It is a very good cleave. Perfectly flat.', side: 'right' },
          { from: BY, text: 'That is the problem.', side: 'left' },
          { from: MGR, text: 'Can we just order a better laser?', side: 'left' }
        ], narration: isThisOk('Gary is coupling to a laser with a perfectly flat cleaved fiber, and the power jumps as the gap changes.') },
      { type: 'chat', title: 'Scenario 3: Isolator Placement', app: 'thread', meta: { header: '#laser-integration', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Good news, we have isolator budget now', side: 'right' },
          { from: GARY, text: 'I put it at the very end of the fiber spool, there was space there', side: 'right' },
          { from: BY, text: 'The taper, the crossings and the grating coupler are all between the laser and your isolator', side: 'left' },
          { from: GARY, text: 'But the isolator is in the link. It counts', side: 'right' },
          { from: MGR, text: 'Isolator is checked on the checklist. Closing ticket', side: 'left' }
        ], narration: isThisOk('Gary installed the isolator after every reflector on the path, at the far end of a fiber spool.') },
      { type: 'check', question: 'A flip-chip InP laser on a silicon photonics chip, with silicon nitride waveguides and grating couplers, shows light versus current ripples with a regular 25 to 30 mA period. Reworking the sub-micron laser-to-chip gap does not change the period. What should Gary investigate first?',
        options: [
          { text: 'Reflectors further along the path, such as the grating coupler at the far end of the waveguide', correct: true, why: '' },
          { text: 'Rework the gap again; it is the closest interface so it must dominate', correct: false, why: 'Being the closest interface does not make a reflector the strongest one. The period of the ripple is set by the round-trip phase of the parasitic cavity, so it tells you about the cavity length, not about which interface is nearest. If reworking the gap does not change the period, the gap is probably not one of the two mirrors. In the published case by Marinins and coworkers, a far-end silicon nitride grating coupler returned about four percent, minus fourteen decibels, and that set the ripple. Please measure the whole path. Every interface reflects, and the far end of the chip is an interface.' },
          { text: 'Ignore it; ripples on a light versus current curve are normal laser physics', correct: false, why: 'A clean distributed feedback laser has a smooth light versus current curve above threshold. Regular ripples mean the laser is seeing a phase that changes periodically as the current tunes it. That is the signature of feedback from a parasitic cavity. Ignoring it means accepting mode hops and intensity noise in the field. Retaliation that is ignored is still retaliation.' },
          { text: 'Replace the laser with a higher power laser to overpower the feedback', correct: false, why: 'Feedback effects scale with the fraction of light that returns, not with the absolute power. A higher power laser sends more light to the same reflector, and more light comes back. The ratio is unchanged. The fix is to find the reflector, reduce it, angle it, or isolate it.' }
        ], correctSay: 'Find the second mirror. It is often the far one.', narration: ['Knowledge check.'] },
      { type: 'check', question: 'In which feedback regime does the laser linewidth broaden dramatically, in a way that is largely insensitive to the distance to the reflector?',
        options: [
          { text: 'Regime four: coherence collapse', correct: true, why: '' },
          { text: 'Regime one: very weak feedback', correct: false, why: 'In regime one, the feedback is very weak, and the linewidth narrows or broadens depending on the phase of the returning light. The effect depends on phase, so it is sensitive to the reflector position. The dramatic, distance-insensitive broadening is regime four, called coherence collapse.' },
          { text: 'Regime five: stable external-cavity operation', correct: false, why: 'Regime five is strong feedback into a laser with an anti-reflection coated facet. The laser then runs stably as an external cavity laser. That is designed feedback, with consent. The dramatic broadening you are looking for is regime four, coherence collapse.' },
          { text: 'There is no such regime; feedback only changes power', correct: false, why: 'Feedback changes far more than power. It changes the linewidth, it can make the laser hop between modes, and in the regime called coherence collapse it broadens the line dramatically. Tkach and Chraplyvy classified five regimes in 1986.' }
        ], correctSay: 'Coherence collapse. Regime four.', narration: ['One more knowledge check.'] },
      takeaways(['Feedback acts through returned power, phase, and delay', 'The strongest reflector may be the far end of the chip, not the nearest gap', 'Budget return loss for tapers, couplers and splitters, not only insertion loss', 'Put the isolator where it isolates: after the reflectors you can control, before the ones you cannot', 'Never trade an isolator for its insertion loss. Fix the link, and speak up when someone offers the trade']),
      ack('I have completed Chapter 1, Recognizing Unsolicited Feedback.'),
      refs('tkach', 'marinins', 'kurczveil', 'moscoso', 'lenstra', 'lapointe', 'malitson', 'dekoninck')
    ]
  };

  /* =================================================================== */
  /* CHAPTER 2: Every Interface Is a Cavity                                */
  /* =================================================================== */
  var ch2 = {
    id: 'ch2', title: 'Every Interface Is a Cavity', subtitle: 'Parasitic Fabry-Perot etalons and the cavities you did not design',
    objectives: ['Compute a parasitic cavity length from a ripple period', 'Estimate ripple depth from two reflectances', 'List common hidden reflectors on a chip'],
    steps: [
      objectives('Every Interface Is a Cavity', [
        'Compute the length of a parasitic cavity from the period of a ripple',
        'Estimate the ripple depth from the two reflectances',
        'List common hidden reflectors: coupler ends, crossings, splitters, multimode interferometer walls, amplifier facets',
        'Treat any regular ripple as a cavity until proven otherwise'
      ]),
      { type: 'video', title: 'Cold Open: The Wiggles', duration: 24,
        keyframes: [
          { t: 0, cast: ['Gary'], caption: 'Gary measures a modulator transfer curve. It has wiggles.', shot: 'Office cubicle, medium close-up on Gary at a dual monitor. Left monitor: Mach-Zehnder modulator transfer curve with a fine periodic ripple riding on the cosine. Gary squints, then smiles.', draw: function (ctx, w, h, u) { dTrace(ctx, w, h, u, { label: 'Transmission versus wavelength (illustrative)', x: 'wavelength', y: 'transmission', f: function (x) { return 0.5 + 0.38 * Math.cos(x * 12) + 0.05 * Math.cos(x * 140); } }); } },
          { t: 7, cast: ['Gary'], caption: 'Gary adds a sinusoid to his fit model. The fit is excellent.', shot: 'Screen insert: fit dialog with a new sinusoid term being typed in. Fit residuals collapse to zero. Gary does a small fist pump. Red X stamp freezes the frame.', draw: function (ctx, w, h, u) { dTrace(ctx, w, h, u, { label: 'Fit: cosine plus one extra sinusoid', x: 'wavelength', y: 'transmission', fs: [function (x) { return 0.5 + 0.38 * Math.cos(x * 12) + 0.05 * Math.cos(x * 140); }, function (x) { return 0.5 + 0.38 * Math.cos(x * 12) + 0.05 * Math.cos(x * 140) + 0.005; }], stamp: 'VIOLATION' }); } },
          { t: 13, cast: ['Pat Reynolds'], caption: 'Replay: the ripple period is a ruler. It points to two reflectors.', shot: 'Camera cranes up out of the monitor into a stylized top view of the chip, which becomes a hall of mirrors: each facet, crossing and coupler end flashes as a reflector, and pairs of them are joined by glowing arcs labelled with their implied spacing.', draw: function (ctx, w, h, u) { dChip(ctx, w, h, u, { refl: [0.2, 0.55, 0.97], hotFar: true, ruler: true, label: 'Ripple period to reflector spacing: L = c / (2 n df)', stamp: 'COMPLIANT', stampColor: 'ok' }); } }
        ],
        narration: ['Gary measures the transfer curve of his modulator.', 'It has wiggles. Small, regular, periodic wiggles.', 'Gary adds a sinusoid to his fit model. The fit is excellent.', 'Let us replay that.', 'The period of the wiggle is a ruler. It tells you the spacing between two reflectors.', 'Every interface reflects. Every reflection is a cavity.'] },
      { type: 'slide', layout: 'bullets', title: 'The Parasitic Etalon, in Two Formulas',
        bullets: ['Two reflectors with power reflectances R1 and R2, spacing L, group index n', 'Ripple period: FSR = c / (2 n L); in wavelength, d-lambda = lambda^2 / (2 n L)', 'Lossless Airy transmission: T = (1 - R1)(1 - R2) / (1 + R1 R2 - 2 sqrt(R1 R2) cos(phase))', 'Peak-to-valley ripple: 20 log10[(1 + r) / (1 - r)] dB, with r = sqrt(R1 R2)', 'Weak mirrors still make a cavity. The ripple is small, but the period is exact'],
        callout: 'Measure the period. Compute the length. Find the two reflectors.',
        narration: ['Every parasitic cavity is described by two formulas.', 'The ripple period, or free spectral range, is the speed of light divided by two times the group index times the spacing.', 'The ripple depth depends on the geometric mean of the two reflectances.', 'Weak mirrors still make a cavity. The ripple may be small, but its period is exact.', 'So measure the period, compute the length, and go find the two reflectors.', MANTRA] },
      { type: 'widget', title: 'Interactive: Parasitic Fabry-Perot Ripple', widget: 'fabryPerot', params: {},
        instructions: 'Adjust the two reflectances, the spacing and the group index. Watch the ripple period and depth. Then use the ripple-to-length calculator.',
        narration: ['In this activity, you will build a cavity you did not design.', 'Move the sliders for the two reflectances, the spacing, and the group index.', 'Notice that the depth depends on the reflectances, and the period depends only on the spacing and the group index.'] },
      { type: 'reveal', title: 'Cavities You Did Not Design', prompt: 'Open each tab to meet a reflector that was not on the schematic.',
        tabs: [
          { label: 'Amplifier facets', body: '<p>In a high-gain erbium-doped thin film lithium niobate amplifier, facet back-reflections reinforced amplified spontaneous emission, and parasitic lasing set in at the 1531.5 nm fluorescence peak (Bao et al., 2024).</p><p>An amplifier has no mirrors only until it has gain.</p>', narration: ['An erbium-doped amplifier on thin film lithium niobate.', 'Facet back-reflections reinforced the amplified spontaneous emission, and parasitic lasing set in at fifteen thirty-one point five nanometers, at the fluorescence peak.', 'An amplifier has no mirrors only until it has enough gain.', 'Every interface reflects.'] },
          { label: 'Crossings and splitters', body: '<p>A poorly designed waveguide crossing contributes crosstalk, parasitic cavities from back-reflection, and multipath (Celo et al., 2017). Splitter trees accumulate back-reflection with depth (Tian et al., 2025).</p>', narration: ['Waveguide crossings.', 'A poorly designed crossing contributes crosstalk, parasitic cavities, and multipath interference.', 'And splitter trees accumulate back-reflection with every level of depth.', 'Crossings and splitters are usually budgeted for loss only. Budget their reflections too.'] },
          { label: 'Multimode interferometer walls', body: '<p>Back-reflection ripples from the end walls of multimode interferometers only appear in simulation when enough modes are included; one double-interferometer study needed more than 30 (Cherchi et al., 2014).</p><p>If your simulation has no ripple, check that it is allowed to have one.</p>', narration: ['Multimode interferometers have end walls, and end walls reflect.', 'In one study, the reflection ripples only appeared in simulation once more than thirty modes were included.', 'If your simulation shows no ripple, first check that it was allowed to show one.'] },
          { label: 'Vertical grating couplers', body: '<p>A perfectly vertical grating coupler gives a large back-reflection into the waveguide, which is why grating couplers are usually tilted (Van Vaerenbergh et al., 2021).</p>', narration: ['Perfectly vertical grating couplers.', 'They give a large back-reflection into the waveguide, which is one reason grating couplers are usually tilted.', 'Angle is a policy, not a style choice.'] },
          { label: 'Offset splices', body: '<p>A single-mode fiber spliced with a deliberate lateral core offset excites cladding modes and becomes an in-fiber Mach-Zehnder interferometer, used as a strain sensor (Contreras-Teran et al., 2023).</p><p class="muted">Course inference, not from the paper: an accidentally offset splice would behave the same way, as an unintended interferometer in your link.</p>', narration: ['An offset splice.', 'A single mode fiber spliced with a deliberate lateral offset excites cladding modes, and becomes an interferometer inside the fiber. It has been used as a strain sensor.', 'This next sentence is our inference, not the paper. An accidental offset would do the same thing, uninvited.'] },
          { label: 'The dark port', body: '<p>In an on-chip interferometer that recombines through a 1x2 junction, there is no second guided output. Light that does not interfere constructively is not routed to another port: it is radiated or scattered, and part of it can reflect back toward the source.</p>', narration: ['The dark port of a one by two combiner.', 'There is no second guided output for the light to go to.', 'So the light that does not interfere constructively is radiated, or scattered, and part of it can come back.', 'Energy is conserved. Your budget is not.'] }
        ], narration: ['Parasitic cavities are everywhere. Let us meet some of them.'] },
      { type: 'chat', title: 'Scenario 1: The Excellent Fit', app: 'thread', meta: { header: '#modulator-team', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Transfer curve fit is done. Added a small sinusoidal term for the wiggles, R squared is 0.999', side: 'right' },
          { from: BY, text: 'What is the period of the wiggle? Convert it to a length with lambda squared over 2 n delta lambda', side: 'left' },
          { from: GARY, text: 'It is a fit parameter, it does not need a physical meaning', side: 'right' },
          { from: BY, text: 'It has one anyway', side: 'left' },
          { from: MGR, text: 'Great fit. Put it in the deck', side: 'left' }
        ], narration: isThisOk('Gary fit the ripple away instead of asking where it came from.') },
      { type: 'chat', title: 'Scenario 2: The Accidental Laser', app: 'phone', meta: { header: 'Messages', participants: [GARY, MGR] },
        messages: [
          { from: GARY, text: 'The new erbium amplifier has a super sharp line at the fluorescence peak!!', side: 'right' },
          { from: GARY, text: 'We invented a laser by accident. Should I file a disclosure', side: 'right' },
          { from: MGR, text: 'Is it supposed to be an amplifier', side: 'left' },
          { from: GARY, text: 'It was', side: 'right' }
        ], narration: isThisOk('Gary is celebrating parasitic lasing in his amplifier as a new invention.') },
      { type: 'chat', title: 'Scenario 3: Certified Single-Mode', app: 'call', meta: { header: 'Video call', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'The fringes on my pigtail spectrum cannot be modal. It is single-mode fiber. It says so on the reel', side: 'right' },
          { from: BY, text: 'How was the splice? If the cores are offset, you launch cladding modes, and core plus cladding is an interferometer', side: 'left' },
          { from: GARY, text: 'The splicer said the splice was fine', side: 'right' },
          { from: MGR, text: 'If the reel says single mode, I am comfortable with single mode', side: 'left' }
        ], narration: isThisOk('Gary believes the label on the reel instead of the fringes on his spectrum.') },
      { type: 'check', question: 'A measured spectrum shows a ripple with a period of 0.1 nm near 1550 nm. Assume a group index of 4. What spacing between two reflectors does this imply?',
        options: [
          { text: 'About 3 mm', correct: true, why: '' },
          { text: 'About 3 micrometers', correct: false, why: 'Use the formula: spacing equals lambda squared divided by two times the group index times the ripple period. Lambda squared is one point five five micrometers squared, which is about two point four times ten to the minus twelve square meters. Two times four times zero point one nanometers is eight times ten to the minus ten meters. The ratio is about three times ten to the minus three meters. That is three millimeters, not three micrometers. A three micrometer cavity would have a ripple period of about one hundred nanometers.' },
          { text: 'About 30 cm', correct: false, why: 'A thirty centimeter cavity with a group index of four would have a ripple period of about one picometer, one hundred times finer than the ripple you see. Use the formula: spacing equals lambda squared over two times the group index times the ripple period. That gives about three millimeters, which is a typical chip-scale distance, for example between a facet and a coupler.' },
          { text: 'It cannot be determined; ripples are device physics', correct: false, why: 'The period of a Fabry-Perot ripple is set only by the round-trip optical length between two reflectors. If you know the group index and the period, you know the spacing. That is the entire point of this chapter. Every reflection is a cavity, and every cavity leaves its length in the period of its ripple.' }
        ], correctSay: 'Three millimeters. Now go find the two reflectors.', narration: ['Knowledge check. You may use the calculator from the activity.'] },
      { type: 'check', question: 'A high-gain erbium-doped waveguide amplifier starts emitting a narrow line at its fluorescence peak. Which explanation is consistent with published observations?',
        options: [
          { text: 'Facet back-reflections close a parasitic cavity around the gain, and it lases', correct: true, why: '' },
          { text: 'The amplifier is saturated by the input signal', correct: false, why: 'Gain saturation by the signal reduces the gain, it does not create a new narrow line at the fluorescence peak. A narrow line at the gain peak means something is providing feedback. In the thin film lithium niobate amplifier reported by Bao and coworkers, facet back-reflections reinforced the amplified spontaneous emission, and parasitic lasing set in at fifteen thirty-one point five nanometers. Two facets and enough gain make a laser.' },
          { text: 'Amplifiers cannot lase because they have no mirrors', correct: false, why: 'An amplifier has no designed mirrors. But every facet reflects a little, and when the round-trip gain overcomes the facet losses, the amplifier becomes a laser. That is exactly what parasitic lasing is. Every interface reflects. Every reflection is a cavity.' }
        ], correctSay: 'Two facets plus gain equals an unauthorized laser.', narration: ['One more knowledge check.'] },
      takeaways([['A regular ripple is a cavity: its period gives the spacing, L = c / (2 n FSR)', 'A regular ripple is a cavity. Its period gives the spacing between the two reflectors.'], ['Ripple depth grows with sqrt(R1 R2); weak mirrors still set an exact period', 'Ripple depth grows with the square root of R one times R two. Weak mirrors still set an exact period.'], 'Facets, coupler ends, crossings, splitters and interferometer walls all reflect', 'Gain plus two facets is a laser, whether or not you ordered one', 'If the ripple period moves when the fiber moves, the ripple belongs to the gap, not the device']),
      ack('I have completed Chapter 2, Every Interface Is a Cavity.'),
      refs('bao', 'celo', 'tian', 'cherchi', 'vaerenbergh', 'contreras', 'malitson', 'li1980')
    ]
  };

  /* =================================================================== */
  /* CHAPTER 3: Mode Hop Awareness                                          */
  /* =================================================================== */
  var ch3 = {
    id: 'ch3', title: 'Mode Hop Awareness', subtitle: 'Hysteresis, sweep direction, and the fine print on tuning range',
    objectives: ['Explain why lasers hop', 'Recognize hysteresis and sweep-direction dependence', 'Distinguish tuning range from mode-hop-free range'],
    steps: [
      objectives('Mode Hop Awareness', [
        'Explain why a laser hops: the cavity comb and the filter tune at different rates',
        'Recognize hysteresis: sweeping up and sweeping down hop at different points',
        'Distinguish the total tuning range from the mode-hop-free tuning range',
        'Treat mode-hop-free as an operating regime, not a property of the laser'
      ]),
      { type: 'video', title: 'Cold Open: The Thermometer Ticket', duration: 26,
        keyframes: [
          { t: 0, cast: ['Gary'], caption: 'Gary sweeps the laser temperature up and logs the wavelength.', shot: 'Lab bench, wide. A butterfly-packaged laser on a temperature controller, fiber to a wavemeter. Gary turns the controller knob up in steps, writing values on a clipboard. The wavemeter readout climbs smoothly, then jumps.', draw: function (ctx, w, h, u) { dComb(ctx, w, h, u, { label: 'Sweep up: comb (gray) and filter (amber) move at different rates' }); } },
          { t: 9, cast: ['Gary', 'Alex'], caption: 'Alex repeats the sweep downward. The jump happens at a different temperature.', shot: 'Same set, Alex now at the controller turning the knob down. Split screen: Gary clipboard and Alex clipboard, the jump temperatures circled in red, not matching.', draw: function (ctx, w, h, u) { dComb(ctx, w, h, 1 - u, { label: 'Sweep down: the hop happens somewhere else' }); } },
          { t: 16, cast: ['Gary'], caption: 'Gary files a ticket against the thermometer.', shot: 'Close-up on a laptop screen: a ticket titled "Thermometer inconsistent" being submitted. Red X stamp.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: 'Ticket: thermometer is inconsistent', sub: 'Filed by Gary', people: ['Gary'], stamp: 'VIOLATION' }); } },
          { t: 21, cast: ['Pat Reynolds'], caption: 'Replay: the hop points differ because the laser remembers. That is hysteresis.', shot: 'Host stands beside a large wall chart showing two staircases, up and down, offset from each other. Green check stamp.', draw: function (ctx, w, h, u) { dTrace(ctx, w, h, u, { label: 'Lasing frequency versus temperature: up (blue) and down (amber)', x: 'temperature', y: 'frequency', fs: [function (x) { return (x * 3 % 1) * 0.3 + Math.floor(x * 3) * 0.22 + 0.1; }, function (x) { return ((x * 3 + 0.3) % 1) * 0.3 + Math.floor(x * 3 + 0.3) * 0.22 + 0.01; }], stamp: 'COMPLIANT', stampColor: 'ok' }); } }
        ],
        narration: ['Gary sweeps the laser temperature up, and logs the wavelength.', 'It climbs smoothly. Then it jumps.', 'Alex repeats the sweep, going down. The jump happens at a different temperature.', 'Gary files a ticket against the thermometer.', 'Let us replay that. The thermometer is fine. The laser remembers which mode it was in.', 'Every cavity hops.'] },
      { type: 'slide', layout: 'host', title: 'Why Lasers Hop',
        bullets: ['The laser cavity supports a comb of longitudinal modes, one free spectral range apart', 'A filter (grating, ring, gain peak, or a parasitic etalon) selects which mode wins', 'Temperature, current and phase move the comb and the filter at different rates', 'When another mode wins, the laser jumps. Once it has jumped, it tends to stay: hysteresis'],
        callout: MANTRA,
        narration: ['Let us take a closer look.', 'A laser cavity supports a comb of modes, one free spectral range apart.', 'A filter selects which mode wins. That filter might be a grating, a ring, the gain peak, or a parasitic etalon from Chapter Two.', 'Temperature and current move the comb and the filter, but at different rates.', 'When a different mode starts to win, the laser jumps.', 'And once it has jumped, it tends to stay there. That is hysteresis. Sweep direction matters.', MANTRA] },
      { type: 'widget', title: 'Interactive: Mode Hop Map', widget: 'modeHop', params: {},
        instructions: 'Run a sweep up, then a sweep down. Compare where the hops happen. Then change the hysteresis margin and the filter tuning rate.',
        narration: ['In this activity, you will sweep a laser in both directions.', 'Press Sweep up, then press Sweep down.', 'Compare where the hops happen. This is an illustrative model with normalized units.'] },
      { type: 'reveal', title: 'Mode Hop Fine Print', prompt: 'Open each tab to review the fine print.',
        tabs: [
          { label: 'Tuning range is not mode-hop-free range', body: '<p>Mode-hop-free tuning is bounded by the laser cavity free spectral range unless the cavity phase follows the filter. A ring-enhanced extended-cavity laser reached 0.22 nm (28 GHz) of mode-hop-free tuning, six times its cavity free spectral range (van Rees et al., 2020).</p><p>Read the abstract for the tuning range. Read the figures for the continuous range.</p>', narration: ['The total tuning range and the mode-hop-free range are different numbers.', 'Without help, continuous tuning is bounded by the cavity free spectral range.', 'One ring-enhanced extended cavity laser reached zero point two two nanometers, or twenty-eight gigahertz, of mode-hop-free tuning. That was six times its cavity free spectral range.', 'Read the abstract for the tuning range. Read the figures for the continuous range.'] },
          { label: 'Cavity phase', body: '<p>A kink-free, mode-hop-free power versus current curve in a photonic-wire-bond external-cavity laser required a proper initial cavity-phase setting, and reports of hops in the literature differ because they depend on design (Xu et al., 2021).</p>', narration: ['Cavity phase.', 'A kink-free power versus current curve in a photonic wire bond external cavity laser required setting the cavity phase properly first.', 'Single mode at one current does not mean single mode at every current.'] },
          { label: 'A regime, not a property', body: '<p>Residual wiggles on a thin film lithium niobate photonic-wire-bond laser power curve were longitudinal mode hops, even though the same laser later ran 58 hours mode-hop-free with 4.4 MHz per hour drift (Franken et al., 2025).</p><p>Mode-hop-free: conditions apply.</p>', narration: ['Mode-hop-free is a regime, not a property.', 'In one thin film lithium niobate laser, wiggles on the power curve were longitudinal mode hops.', 'The same laser later ran fifty-eight hours mode-hop-free, with four point four megahertz per hour of drift.', 'Both statements are true. Conditions apply.'] },
          { label: 'Feedback hops too', body: '<p>External feedback adds a second cavity. In the feedback regime classification of Tkach and Chraplyvy (1986), one regime is hopping between external-cavity modes. A parasitic etalon is a filter you did not choose.</p>', narration: ['Feedback hops too.', 'External feedback adds a second cavity, and with it a second comb.', 'A parasitic etalon is a filter you did not choose. Every reflection is a cavity. Every cavity hops.'] }
        ], narration: ['Mode hops come with fine print. Let us read it together.'] },
      { type: 'chat', title: 'Scenario 1: The Datasheet', app: 'phone', meta: { header: 'Messages', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'Datasheet says wide tuning range so I will do one continuous sweep for the spectroscopy run', side: 'right' },
          { from: BY, text: 'Wide tuning range, or wide mode-hop-free range? Those are different lines on the datasheet', side: 'left' },
          { from: GARY, text: 'There is only one line', side: 'right' },
          { from: BY, text: 'Then the other one is in a footnote', side: 'left' }
        ], narration: isThisOk('Gary is planning a continuous sweep based on the total tuning range.') },
      { type: 'chat', title: 'Scenario 2: Up and Down', app: 'email', meta: { header: 'Inbox', subject: 'Wavelength calibration table (final_v3_down)', participants: [GARY, MGR, BY] },
        messages: [
          { from: GARY, text: 'Attached: wavelength versus temperature, measured going up. Alex measured going down and got different hop points, so one of us is wrong. I propose we average.', side: 'right' },
          { from: BY, text: 'Neither of us is wrong. It is hysteresis. The calibration table needs a sweep direction column.', side: 'left' },
          { from: MGR, text: 'Averaging sounds fair to everyone.', side: 'left' }
        ], narration: isThisOk('Gary wants to average two hop maps that differ because of hysteresis.') },
      { type: 'chat', title: 'Scenario 3: Qualified at One Current', app: 'thread', meta: { header: '#product-qual', participants: [GARY, MGR, BY] },
        messages: [
          { from: GARY, text: 'Laser qualified mode-hop-free. Overnight run, zero hops', side: 'right' },
          { from: BY, text: 'At what current and temperature? Product runs hotter and at higher current', side: 'left' },
          { from: GARY, text: 'Mode-hop-free is mode-hop-free', side: 'right' },
          { from: MGR, text: 'Mark it qualified', side: 'left' }
        ], narration: isThisOk('Gary qualified the laser at one operating point and declared it mode-hop-free everywhere.') },
      { type: 'check', question: 'Gary sweeps the temperature up and records hop points. Alex sweeps down and records different hop points. What is the most likely explanation?',
        options: [
          { text: 'Hysteresis: once the laser is in a mode, it stays there past the midpoint, so hop points depend on sweep direction', correct: true, why: '' },
          { text: 'The temperature sensor is inconsistent', correct: false, why: 'A faulty sensor would give inconsistent readings in both directions, scattered at random. A systematic offset between up and down sweeps, with clean staircases in each direction, is the signature of hysteresis. Once the laser is in a mode, gain competition and nonlinear effects tend to keep it there past the point where another mode would have won from a cold start. So the hop happens later going up and later going down, which means at different temperatures. Please record the sweep direction.' },
          { text: 'One of the two measurements is wrong; average them', correct: false, why: 'Both measurements can be right. With hysteresis, the hop points genuinely depend on the direction of approach. Averaging them produces a calibration table that is wrong in both directions. The correct fix is to record the sweep direction, or to approach every operating point from the same side.' },
          { text: 'Mode hops are random and cannot be mapped', correct: false, why: 'Mode hops are deterministic in a stable setup. They happen where a different mode starts to win the gain competition, plus a hysteresis margin. You can map them, as long as you record the direction of approach. Near a hop point, noise can make the outcome look random, which is one more reason to stay away from hop points in operation.' }
        ], correctSay: 'Hysteresis. Record the sweep direction.', narration: ['Knowledge check.'] },
      { type: 'check', question: 'Without synchronized tuning of the cavity phase, what bounds the continuous mode-hop-free tuning range of a laser?',
        options: [
          { text: 'Roughly the free spectral range of the laser cavity', correct: true, why: '' },
          { text: 'The full tuning range of the filter', correct: false, why: 'The filter can tune very far, but if the cavity comb does not move with it, the laser has to jump from one cavity mode to the next every time the filter walks about one free spectral range. That is a mode hop. The full tuning range is reached by a staircase of hops, not continuously. This is why schemes that tune the cavity phase together with the filter are needed for long continuous sweeps, and why a ring-enhanced design reaching six times the cavity free spectral range was worth publishing.' },
          { text: 'The bandwidth of the wavemeter', correct: false, why: 'The wavemeter only reports the wavelength. It does not set what the laser does. The continuous range is set by the laser cavity: without synchronized cavity phase tuning, the laser jumps to the next mode after about one free spectral range.' }
        ], correctSay: 'One free spectral range, unless the cavity phase follows.', narration: ['One more knowledge check.'] },
      takeaways(['Hops happen when the comb and the filter walk apart by about half a free spectral range', 'Hysteresis makes hop points depend on sweep direction: record it', 'Total tuning range is not mode-hop-free range; read the figure, not the abstract', 'Mode-hop-free is a regime defined by current, temperature and feedback, not a property', 'A restart re-selects a mode; it does not remove the hop. Log every hop, and report anyone who asks you not to']),
      ack('I have completed Chapter 4, Mode Hop Awareness.'),
      refs('vanrees', 'xu2021', 'franken', 'tkach')
    ]
  };

  /* =================================================================== */
  /* CHAPTER 4: Reciprocity Is Not Optional                                */
  /* =================================================================== */
  var ch4 = {
    id: 'ch4', title: 'Reciprocity Is Not Optional', subtitle: 'What is, and what is not, an isolator',
    objectives: ['State what an isolator must break', 'Recognize mode-converting diodes as reciprocal', 'Angle facets correctly'],
    steps: [
      objectives('Reciprocity Is Not Optional', [
        'State that an optical isolator must break Lorentz reciprocity',
        'Recognize that asymmetric mode conversion is not isolation',
        'Explain why a reflection retraces the forward path back to the laser',
        'Angle a facet with the refraction correction applied'
      ]),
      { type: 'video', title: 'Cold Open: The Diode', duration: 24,
        keyframes: [
          { t: 0, cast: ['Gary'], caption: 'Gary installs a one-way mode converter between his laser and his chip.', shot: 'Lab bench, macro on a chip under a probe station. A label maker prints "ISOLATOR" and Gary sticks it onto a section of waveguide that tapers asymmetrically. Proud nod.', draw: function (ctx, w, h, u) { dChip(ctx, w, h, u, { refl: [0.4, 0.97], gary: true, label: 'Asymmetric mode converter, labelled ISOLATOR' }); } },
          { t: 7, cast: ['Gary'], caption: 'Forward and backward transmission differ on a single-mode detector. Gary declares victory.', shot: 'Monitor insert: two bars, forward tall, backward short, measured through a single-mode fiber detector. Gary prints the plot.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: 'Forward: high. Backward (fundamental mode only): low', sub: 'Measured with a single-mode detector', people: ['Gary'], stamp: 'VIOLATION' }); } },
          { t: 13, cast: ['Pat Reynolds'], caption: 'Replay: the reflection travels back in the mode it left in, and arrives home as the fundamental mode.', shot: 'Animated cutaway of the waveguide. Forward: fundamental mode enters, exits as a higher-order mode. A downstream reflection sends the higher-order mode back; it reconverts to the fundamental mode and enters the laser. Red glow on the laser facet. Then a Faraday rotator and polarizer slide into place. Green check.', draw: function (ctx, w, h, u) { dChip(ctx, w, h, u, { refl: [0.97], hotFar: true, label: 'The reflection retraces the path, in reverse', stamp: 'COMPLIANT', stampColor: 'ok' }); } }
        ],
        narration: ['Gary has built an isolator. It says so on the label.', 'It is a mode converter that treats the two directions differently.', 'On a single-mode detector, forward transmission is high, and backward transmission is low.', 'Let us replay that.', 'Light reflected downstream travels back in the same mode it left in. The converter undoes its own work, and the reflection arrives home in the fundamental mode.', 'Reciprocity is not optional. Isolate, angle, and verify.'] },
      { type: 'slide', layout: 'policy', title: 'Policy 4.1: Definition of an Isolator',
        bullets: ['An optical isolator must break Lorentz reciprocity (Jalas et al., 2013)', 'Linear, time-invariant, non-magnetic structures are reciprocal: the scattering matrix is symmetric', 'Options that break reciprocity: magneto-optic materials with a bias field, or time modulation. Nonlinearity only partly: it works in a power window and is limited by dynamic reciprocity (Shi et al., 2015)', 'Asymmetric mode conversion, measured with a single-mode detector, can look like isolation. It is not', 'The test: launch backward into the forward output mode(s), count all ports and modes, and compare with the forward transmission. Equal means reciprocal'],
        callout: 'If the scattering matrix is symmetric, the device is not an isolator. Labels do not change matrices.',
        narration: ['Policy section four point one. Definition of an isolator.', 'An optical isolator must break Lorentz reciprocity. Jalas and coworkers wrote this down in two thousand thirteen, in a paper titled, What is, and what is not, an optical isolator.', 'A structure that is linear, time-invariant, and non-magnetic is reciprocal. Its scattering matrix is symmetric.', 'To break reciprocity you need a magneto-optic material with a bias field, or time modulation.', 'Nonlinearity counts only partly. A nonlinear isolator works only within a window of power, and it is limited by dynamic reciprocity, as Shi and coworkers showed in two thousand fifteen.', 'A device that converts modes differently in each direction can look like an isolator if you only measure one mode. It is not one.', 'The proper test is to launch light backward into the mode, or modes, that the forward light exits in. Count every port and every mode, and compare the transmission with the forward direction. If they match, the device is reciprocal.', 'Isolate, angle, and verify.'] },
      { type: 'reveal', title: 'Isolation Facts and Myths', prompt: 'Open each tab.',
        tabs: [
          { label: 'Why the reflection comes home', body: '<p>A reflector downstream returns light in the mode that arrived there. In a reciprocal device, light launched backward in the forward output mode reaches the forward input mode with the same transmission as the forward path (S<sub>ij</sub> = S<sub>ji</sub>): straight into the laser. Asymmetric mode conversion only changes what happens to <i>other</i> backward inputs.</p>', narration: ['Why the reflection comes home.', 'A downstream reflector sends light back in the mode that arrived there.', 'In a reciprocal device, that light reaches the mode that went in, with the same transmission as the forward path. Straight into the laser.', 'Mode conversion only changes what happens to other backward inputs. Not to your own reflection.'] },
          { label: 'Simulated isolation', body: '<p>A large isolation number from a simulation is not a working isolator. Check the boundary conditions, the materials (is anything actually nonreciprocal?), and which ports and modes were counted.</p>', narration: ['Simulated isolation.', 'A large number in a simulation is not a working isolator.', 'Check whether anything in the model is actually nonreciprocal, and which ports and modes were counted.'] },
          { label: 'Real isolators', body: '<p>The usual fix for feedback-sensitive semiconductor lasers is a polarizer plus a Faraday rotator in a magnetic field, the classic bulk isolator. A lens-free, magnet-free version for hybrid integration reached more than 25 dB isolation and less than 1.5 dB insertion loss across the C-band (Lapointe et al., 2025).</p>', narration: ['Real isolators.', 'The classic fix is a polarizer plus a Faraday rotator in a magnetic field.', 'A compact version for hybrid integration, with no lens and no external magnet, reached more than twenty-five decibels of isolation, with less than one and a half decibels of insertion loss, across the C band.'] },
          { label: 'Angle, with refraction', body: '<p>Output waveguides are tilted at the facet so that the reflection does not couple back into the guided mode. By Snell\'s law, n_wg sin(theta_wg) = n_out sin(theta_out), the beam leaves at a larger angle than the waveguide tilt. Rotate the chip or the waveguide so the <i>exiting</i> beam points where your fiber is.</p>', narration: ['Angle, with refraction.', 'Output waveguides are tilted at the facet so that the reflection does not couple back into the guided mode.', 'But by Snell\'s law, the beam leaves at a larger angle than the waveguide tilt.', 'So rotate things so the exiting beam points at your fiber. Not the waveguide.'] },
          { label: 'Retaliation at scale', body: '<p>At high power, a fiber can carry a self-propelled damage front, the fiber fuse, that travels back along the fiber toward the source (Kashyap and Blow, 1988).</p>', narration: ['Retaliation at scale.', 'At high enough power, a fiber can carry a self-propelled damage front, called a fiber fuse.', 'It travels back along the fiber, toward the source.', 'Retaliation always knows where you live.'] }
        ], narration: ['Let us separate isolation facts from isolation myths.'] },
      { type: 'chat', title: 'Scenario 1: The Contrast Ratio', app: 'thread', meta: { header: '#integration-wins', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'We have an on-chip isolator! Big forward to backward contrast, no magnets, no modulation, plain silicon', side: 'right' },
          { from: BY, text: 'Measured with what detector? Single-mode fiber?', side: 'left' },
          { from: GARY, text: 'Yes, the standard setup', side: 'right' },
          { from: BY, text: 'Linear, static, non-magnetic. It is reciprocal. You measured mode conversion', side: 'left' },
          { from: MGR, text: 'Can we call it an isolator in the press release and a mode converter in the paper', side: 'left' }
        ], narration: isThisOk('Gary is calling a linear, static, non-magnetic mode converter an isolator.') },
      { type: 'chat', title: 'Scenario 2: The Directional Modulator', app: 'call', meta: { header: 'Video call', participants: [GARY, MGR, BY] },
        messages: [
          { from: GARY, text: 'We can drop the isolator. Our traveling-wave modulator only modulates light going forward, so it is directional', side: 'right' },
          { from: BY, text: 'It modulates forward light efficiently. A continuous-wave reflection off the facet still goes straight back through it', side: 'left' },
          { from: MGR, text: 'Directional is basically one-way, right', side: 'left' },
          { from: BY, text: 'No', side: 'left' }
        ], narration: isThisOk('Gary wants to remove the isolator because the modulator is directional.') },
      { type: 'chat', title: 'Scenario 3: The Straight Arrow', app: 'email', meta: { header: 'Inbox', subject: 'Fiber attach drawing for review', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'Attached is the fiber attach drawing. Output waveguide is tilted at the facet like an angled connector, and the fiber is aligned along the waveguide.', side: 'right' },
          { from: BY, text: 'The beam refracts at the facet. It leaves at a larger angle than the waveguide. Your fiber is pointing where the light is not.', side: 'left' },
          { from: GARY, text: 'The arrow in the drawing goes straight though.', side: 'right' }
        ], narration: isThisOk('Gary aligned the fiber to the waveguide tilt and forgot refraction.') },
      { type: 'check', question: 'A passive, linear, time-invariant silicon device without magneto-optic material shows high forward and low backward transmission when measured with a single-mode detector. Is it an isolator?',
        options: [
          { text: 'No. It is reciprocal; it converts modes asymmetrically, and a reflection of its forward output still returns to the source', correct: true, why: '' },
          { text: 'Yes. Forward and backward transmission differ, so it isolates', correct: false, why: 'Forward and backward transmission measured in one mode can differ in a reciprocal device, because the device can convert modes differently in each direction. Reciprocity says the scattering matrix is symmetric when all ports and modes are counted. The test that matters for your laser is this: take the light that leaves in the forward direction, reflect it, and send it back in the mode, or modes, it left in. In a reciprocal device, its transmission back into the input mode equals the forward transmission, so it comes out where it started, at the laser. Jalas and coworkers spelled this out in two thousand thirteen.' },
          { text: 'Yes, if the simulated contrast is large enough', correct: false, why: 'No amount of contrast in a reciprocal structure makes it an isolator. Isolation requires breaking Lorentz reciprocity, through a magneto-optic material with a bias field or time modulation. Nonlinearity helps only within a power window and is limited by dynamic reciprocity. A large number from a linear, static simulation is still a mode-conversion number.' },
          { text: 'Yes, as long as it is labelled ISOLATOR', correct: false, why: 'Labels do not change scattering matrices. A linear, time-invariant, non-magnetic device is reciprocal, whatever is printed on it. Please refer to Policy section four point one.' }
        ], correctSay: 'Reciprocity is not optional.', narration: ['Knowledge check.'] },
      { type: 'check', question: 'An output waveguide is tilted at the chip facet to reduce back-reflection. In which direction does the beam leave the chip?',
        options: [
          { text: 'At a larger angle to the facet normal than the waveguide, because it refracts into the lower index', correct: true, why: '' },
          { text: 'Along the waveguide direction, like the arrow in the drawing', correct: false, why: 'The beam crosses an index step at the facet, so it refracts. By Snell\'s law, the waveguide index times the sine of the internal angle equals the outside index times the sine of the external angle. Going from a high index to a lower index, the external angle is larger. So the beam leaves more tilted than the waveguide. Align the fiber to the exiting beam, not to the waveguide.' },
          { text: 'Exactly along the facet normal, because the tilt only affects the reflection', correct: false, why: 'The tilt affects both the reflection and the transmitted beam. The reflection is steered away from the guided mode, which is the point of the tilt. The transmitted beam refracts by Snell\'s law and leaves at a larger angle than the waveguide tilt, not along the normal.' }
        ], correctSay: 'Angle, then correct for refraction.', narration: ['One more knowledge check.'] },
      takeaways(['An isolator must break Lorentz reciprocity: magneto-optics or time modulation; nonlinearity only within a power window', 'Asymmetric mode conversion is reciprocal; your reflection still comes home', 'Verify isolation with all modes counted, on real hardware, not a single-mode detector or a simulation alone', 'Angle facets, then point the fiber at the refracted beam', 'Direction is not angle: the arrow points away from the laser, and the input polarizer follows the laser polarization']),
      ack('I have completed Chapter 5, Reciprocity Is Not Optional.'),
      refs('jalas', 'shi2015', 'lapointe', 'kashyap', 'saleh')
    ]
  };

  /* =================================================================== */
  /* CHAPTER 5: Backscatter in High-Q Resonators                           */
  /* =================================================================== */
  var ch5 = {
    id: 'ch5', title: 'Backscatter in High-Q Resonators', subtitle: 'Doublets, self-injection locking, and the thermal cliff',
    objectives: ['Recognize backscatter doublets', 'Fit doublets correctly', 'Know the power and thermal limits of backscatter locking'],
    steps: [
      objectives('Backscatter in High-Q Resonators', [
        'Recognize a resonance doublet from clockwise and counter-clockwise mode coupling',
        'Avoid misreporting Q by fitting a single Lorentzian to a doublet',
        'Explain why backscatter is both a defect and a requirement',
        'Recognize the power and thermal limits of self-injection locking and thermal locks'
      ]),
      { type: 'video', title: 'Cold Open: The Split Personality', duration: 26,
        keyframes: [
          { t: 0, cast: ['Gary'], caption: 'Gary measures a ring resonator with a very high Q. The dip has two bottoms.', shot: 'Dark lab, laser sweep across a ring on a chip under a microscope. Monitor shows a transmission dip with two closely spaced minima. Gary leans in, frowning.', draw: function (ctx, w, h, u) { dRing(ctx, w, h, u, { gary: true, speck: true, ccw: true, label: 'A scatterer couples clockwise and counter-clockwise light' }); } },
          { t: 8, cast: ['Gary'], caption: 'Gary fits one Lorentzian through both. The reported Q drops.', shot: 'Screen insert: a single wide Lorentzian fit drawn over a doublet. The Q value field shows a smaller number. Gary emails "wafer is bad". Red X stamp.', draw: function (ctx, w, h, u) { dTrace(ctx, w, h, u, { label: 'Doublet with a single-Lorentzian fit (illustrative)', x: 'detuning', y: 'transmission', fs: [function (x) { return 1 - 0.85 * lor(x, 0.45, 0.04) - 0.85 * lor(x, 0.55, 0.04); }, function (x) { return 1 - 0.75 * lor(x, 0.5, 0.09); }], stamp: 'VIOLATION' }); } },
          { t: 15, cast: ['Gary', 'Morgan'], caption: 'Months later: the laser team needs backscatter for self-injection locking.', shot: 'Conference room, wide. Morgan unveils a slide: "Backscatter is now mandatory." Gary looks at his wafer-scrap request on his laptop.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: 'Policy update: backscatter is now mandatory', sub: 'Self-injection locking needs light to come back', people: ['Gary', 'Morgan'] }); } },
          { t: 21, cast: ['Pat Reynolds'], caption: 'Replay: fit the doublet with a coupled-mode model. Know when backscatter helps and when it hurts.', shot: 'Host beside a ring model with standing-wave nodes glowing around the circumference. Green check stamp.', draw: function (ctx, w, h, u) { dRing(ctx, w, h, u, { standing: true, ccw: true, speck: true, label: 'Two standing waves, two resonances', stamp: 'COMPLIANT', stampColor: 'ok' }); } }
        ],
        narration: ['Gary measures a ring resonator with a very high quality factor.', 'The dip has two bottoms.', 'Gary fits a single Lorentzian through both. The reported Q drops, and Gary asks to scrap the wafer.', 'Months later, the laser team announces that backscatter is now mandatory.', 'Let us replay that. A tiny scatterer couples clockwise and counter-clockwise light. The two form standing waves, and the resonance splits.', 'Every interface reflects. Even a speck of dust is an interface.'] },
      { type: 'slide', layout: 'host', title: 'Why Resonances Split',
        bullets: ['A ring supports two degenerate traveling waves: clockwise and counter-clockwise', 'Any backscatter couples them; at high Q, a tiny coupling exceeds the linewidth', 'The new eigenmodes are standing waves, split in frequency by twice the coupling rate', 'Backscattered light also leaves the ring backward: toward your laser'],
        callout: MANTRA,
        narration: ['Let us take a closer look.', 'A ring supports two traveling waves with the same frequency, clockwise and counter-clockwise.', 'Any backscatter couples them. At high Q, even a tiny coupling can be larger than the linewidth.', 'The new eigenmodes are standing waves, and their frequencies split by twice the coupling rate.', 'And the backscattered light also leaves the ring backward. Toward your laser.', MANTRA] },
      { type: 'widget', title: 'Interactive: Backscatter Doublet', widget: 'doublet', params: {},
        instructions: 'Increase the backscatter coupling until the resonance splits. Watch the apparent linewidth a single-Lorentzian fit would report, and the light reflected back toward the laser.',
        narration: ['In this activity, you will add backscatter to a ring.', 'Move the backscatter slider to the right, until the dip splits.', 'Watch the reflected light. That is the light going back toward your laser.'] },
      { type: 'reveal', title: 'Backscatter: Defect and Requirement', prompt: 'Open each tab.',
        tabs: [
          { label: 'The doublet misreports Q', body: '<p>At high Q with tight confinement, very small defects or roughness split the clockwise and counter-clockwise degeneracy into a visible doublet; a single-Lorentzian fit then misreports Q (Ji et al., 2017).</p>', narration: ['At high Q, very small defects split the resonance into a doublet.', 'A single Lorentzian fit then misreports the Q.', 'Fit a doublet model. Report the intrinsic linewidth and the coupling separately.'] },
          { label: 'Your lithography is a scatterer', body: '<p>Electron-beam writing discontinuities, jumps and pattern segmentation along the ring circumference, produced split resonances in diamond rings; continuous writing removed them (Hausmann et al., 2014).</p>', narration: ['Your lithography is a scatterer.', 'In diamond rings, discontinuities in electron beam writing around the circumference produced split resonances.', 'Continuous writing removed them. Each stitch was an interface, and every interface reflects.'] },
          { label: 'Now mandatory', body: '<p>Self-injection locking relies on backscatter: the resonances that self-injection locked all showed backscatter splitting (Raja et al., 2019). The defect is the feature.</p>', narration: ['Policy reversal.', 'Self-injection locking relies on backscatter to send light back into the laser.', 'In one integrated soliton microcomb, the resonances that locked all showed backscatter splitting.', 'This is feedback with consent.'] },
          { label: 'Until you turn it up', body: '<p>Backscatter-based self-injection locking works at low power, but as power increases the ring\'s nonlinear response detunes it, and the backscattered power no longer reaches the laser (White et al., 2024).</p>', narration: ['Until you turn it up.', 'Backscatter-based locking works at low power.', 'At higher power, the nonlinear response of the ring detunes it, and the backscattered light no longer reaches the laser.', 'Characterize at the power you will operate at.'] },
          { label: 'The thermal cliff', body: '<p>Thermally locked microrings sit near the edge of instability: past a point, self-heating flips from negative to positive feedback and the resonance snaps to shorter wavelength. The balance point cannot be predicted, so a hot-to-cold calibration sweep is required (Kramnik et al., 2024).</p>', narration: ['The thermal cliff.', 'A thermally locked ring sits near the edge of instability.', 'Past a certain point, self-heating flips from negative to positive feedback, and the resonance snaps away to shorter wavelength.', 'The balance point cannot be predicted from the design, so a hot to cold calibration sweep is required.'] },
          { label: 'Not every split is backscatter', body: '<p>Fano lineshapes from multimode interference can coexist with backscatter doublets in the same resonator, doublets at one wavelength and Fano shapes at others (Chen et al., 2026). Not every distorted dip is backscatter.</p>', narration: ['Not every split is backscatter.', 'In one resonator, backscatter doublets appeared at one wavelength, and Fano lineshapes from multimode interference at others.', 'Label your diagnosis as an inference until you have checked it.'] }
        ], narration: ['Backscatter is a defect. Backscatter is a requirement. Both are policy.'] },
      { type: 'chat', title: 'Scenario 1: Scrap the Wafer', app: 'email', meta: { header: 'Inbox', subject: 'Ring Q below target, request to scrap lot', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Single-Lorentzian fits give Q well below target on every high-Q ring. Requesting to scrap the lot.', side: 'right' },
          { from: BY, text: 'Are the dips split? If there are two minima, the single fit is measuring the splitting, not the loss.', side: 'left' },
          { from: GARY, text: 'There are two minima on most of them. I assumed noise.', side: 'right' },
          { from: MGR, text: 'Scrapping approved pending one more opinion.', side: 'left' }
        ], narration: isThisOk('Gary is about to scrap a lot because he fit doublets with a single Lorentzian.') },
      { type: 'chat', title: 'Scenario 2: Conflicting Policies', app: 'thread', meta: { header: '#all-hands', participants: [MGR, GARY, BY] },
        messages: [
          { from: MGR, text: 'Reminder: resonator team target is zero backscatter', side: 'left' },
          { from: MGR, text: 'Reminder: laser team target is enough backscatter to self-injection lock', side: 'left' },
          { from: GARY, text: 'Same ring?', side: 'right' },
          { from: MGR, text: 'Same ring', side: 'left' },
          { from: BY, text: 'Specify the coupling rate relative to the linewidth, and the operating power. Then both teams can sign', side: 'left' }
        ], narration: isThisOk('The same ring has two opposite backscatter targets, and no operating power is specified.') },
      { type: 'chat', title: 'Scenario 3: The Snap', app: 'phone', meta: { header: 'Messages', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'Thermal lock was perfect, then the resonance jumped away to the blue', side: 'right' },
          { from: GARY, text: 'I computed the lock point from the design so it should be fine', side: 'right' },
          { from: BY, text: 'The balance point is not predictable from the design. Do a hot to cold calibration sweep', side: 'left' }
        ], narration: isThisOk('Gary computed his thermal lock point from the design instead of calibrating it.') },
      { type: 'check', question: 'A high-Q ring shows a resonance with two closely spaced minima. Gary fits one Lorentzian through both and reports a low Q. What should he do?',
        options: [
          { text: 'Fit a coupled-mode doublet model, and report the linewidth and the backscatter coupling separately', correct: true, why: '' },
          { text: 'Report the single-Lorentzian Q; it is conservative', correct: false, why: 'The single Lorentzian width of a doublet includes the splitting. It is not a conservative estimate of loss, it is a different quantity. When the backscatter coupling is comparable to or larger than the linewidth, the apparent width is dominated by the splitting, and the reported Q can be far below the real one. Ji and coworkers noted exactly this for high-Q resonators. Fit a doublet model, report the intrinsic linewidth, and report the coupling rate as its own number.' },
          { text: 'Scrap the wafer; split resonances mean it is defective', correct: false, why: 'At high Q, a very small scatterer is enough to split the resonance, because the coupling only needs to exceed a very small linewidth. A doublet often means the loss is low, not high. And for self-injection locking, the backscatter is required. Fit properly before scrapping anything.' },
          { text: 'Average the two minima positions and fit only the left half of the dip', correct: false, why: 'Fitting half a doublet with a single Lorentzian gives a width that depends on how you choose the cut, so it is not reproducible. Use a coupled-mode model with both supermodes, which has a known closed form.' }
        ], correctSay: 'Fit the doublet, report both numbers.', narration: ['Knowledge check.'] },
      { type: 'check', question: 'Backscatter-based self-injection locking works in the lab at low power but fails at high power. Which published mechanism explains this?',
        options: [
          { text: 'The ring\'s nonlinear response detunes it at high power, so the backscattered light no longer reaches the laser', correct: true, why: '' },
          { text: 'Backscatter increases with power, so the laser is overwhelmed', correct: false, why: 'Backscatter is a linear scattering process; the fraction backscattered does not grow with power. White and coworkers reported the opposite problem: as power increases, the nonlinear response of the ring detunes it, so the backscattered light no longer returns to the laser in the right way. Characterize at the operating power.' },
          { text: 'Self-injection locking is unrelated to backscatter', correct: false, why: 'Backscatter is what sends light from the ring back into the laser in self-injection locking. Raja and coworkers observed that the resonances that self-injection locked all showed backscatter splitting.' }
        ], correctSay: 'Characterize at the power you will operate at.', narration: ['One more knowledge check.'] },
      takeaways(['Backscatter couples clockwise and counter-clockwise modes into split standing waves', 'Fit doublets with a coupled-mode model; a single Lorentzian misreports Q', 'Backscatter is a defect for resonator teams and a requirement for self-injection locking: specify it', 'Self-injection and thermal locks change with power; calibrate hot to cold, at the operating point', 'Transmission alone cannot choose under- or over-coupling; the bigger number is not a coupling regime']),
      ack('I have completed Chapter 6, Backscatter in High-Q Resonators.'),
      refs('ji2017', 'hausmann', 'raja', 'white', 'kramnik', 'chen2026', 'weiss', 'kippenberg', 'yariv')
    ]
  };

  /* =================================================================== */
  /* CHAPTER 6: Your Instruments Are Averaging                             */
  /* =================================================================== */
  var ch6 = {
    id: 'ch6', title: 'Your Instruments Are Averaging', subtitle: 'Measurement setups that feed back, heat, and hide',
    objectives: ['Recognize instrument averaging', 'Recognize measurement setups that feed back or heat the device', 'Label inferences'],
    steps: [
      objectives('Your Instruments Are Averaging', [
        'Recognize that slow or coarse instruments average away hops and ripples',
        'Recognize that the measurement setup can itself reflect light into the laser',
        'Recognize that an on-chip source can heat the device it measures',
        'Label your own inferences as inferences'
      ]),
      { type: 'video', title: 'Cold Open: The Steady Number', duration: 24,
        keyframes: [
          { t: 0, cast: ['Gary'], caption: 'The wavemeter shows a steady number. Gary concludes the laser is not hopping.', shot: 'Lab bench, close on a wavemeter display with a calm, steady reading. Behind it, out of focus, an oscilloscope trace of the laser intensity flickers between two levels. Gary gives a thumbs up to the camera.', draw: function (ctx, w, h, u) { dTrace(ctx, w, h, u, { label: 'Fast intensity (blue) versus slow instrument average (amber)', x: 'time', y: 'signal', fs: [function (x) { return (Math.sin(x * 90) > 0.2 ? 0.75 : 0.45); }, function () { return 0.62; }] }); } },
          { t: 8, cast: ['Gary'], caption: 'Gary measures the ring with an on-chip laser. The free spectral range looks larger than designed.', shot: 'Probe station, chip with an attached laser and a ring. A thermal camera inset shows the region around the laser and the ring warming as the laser sweeps. Gary writes "new physics?" on a sticky note. Red X stamp.', draw: function (ctx, w, h, u) { dRing(ctx, w, h, u, { gary: true, label: 'The source heats the device it is measuring', stamp: 'VIOLATION' }); } },
          { t: 15, cast: ['Pat Reynolds'], caption: 'Replay: compare against a separate benchtop laser, add an isolator, and look at fast signals.', shot: 'Same bench, now with a benchtop tunable laser through an isolator, a fast photodiode on the oscilloscope, and the thermal camera showing a cool chip. Green check stamp.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: 'Separate source. Isolator. Fast detector.', sub: 'Then compare', people: ['Pat Reynolds'], stamp: 'COMPLIANT', stampColor: 'ok' }); } }
        ],
        narration: ['The wavemeter shows a steady number. Gary concludes the laser is not hopping.', 'Meanwhile, the intensity flickers between two levels, faster than the wavemeter updates.', 'Next, Gary measures a ring with a laser on the same chip, and the free spectral range looks larger than designed.', 'Let us replay that. Use a separate source, add an isolator, and look at fast signals.', 'Isolate, angle, and verify. Verify is the part with the instruments.'] },
      { type: 'slide', layout: 'host', title: 'Every Instrument Integrates',
        bullets: ['A wavemeter reports one number per update; a hop faster than the update is averaged or missed', 'An optical spectrum analyzer reports power within its resolution bandwidth; ripples finer than that disappear', 'A power meter averages over its response time', 'The setup itself has interfaces: fiber tips, connectors, the detector face. They reflect too'],
        callout: MANTRA,
        narration: ['Let us take a closer look.', 'Every instrument integrates over something.', 'A wavemeter reports one number per update. A hop faster than the update can be averaged or missed.', 'An optical spectrum analyzer reports power within its resolution bandwidth. A ripple finer than that disappears.', 'A power meter averages over its response time.', 'And the setup has interfaces of its own. Fiber tips, connectors, the detector face. Every interface reflects.', MANTRA] },
      { type: 'reveal', title: 'When the Measurement Is the Problem', prompt: 'Open each tab.',
        tabs: [
          { label: 'The setup feeds back', body: '<p>Without an isolator, parasitic reflections lead to an overestimate of laser linewidth, as stated for 300-mm GaAs nano-ridge lasers (De Koninck et al., 2025).</p>', narration: ['The setup feeds back.', 'Without an isolator in the measurement path, parasitic reflections lead to an overestimate of the laser linewidth.', 'Your measurement fiber is an interface. It reflects.'] },
          { label: 'The source heats the sample', body: '<p>Sweeping an on-chip, photonic-wire-bonded distributed feedback laser heated the chip and red-shifted the rings under test: the free spectral range read 160 pm, against 120 pm with a benchtop laser and 110 pm simulated (Chowdhury et al., 2024).</p>', narration: ['The source heats the sample.', 'Sweeping a laser on the same chip heated the chip and red-shifted the rings under test.', 'The free spectral range read one hundred sixty picometers. With a separate benchtop laser, it read one hundred twenty. The simulation said one hundred ten.', 'The instrument was cooking the sample.'] },
          { label: 'Resolution hides ripple', body: '<p>Use the parasitic Fabry-Perot calculator from Chapter 2: a long fiber pigtail gives a ripple period far finer than a typical spectrum analyzer resolution. The ripple is still there; the instrument averages it.</p>', narration: ['Resolution hides ripple.', 'A long fiber pigtail makes a ripple period much finer than a typical analyzer resolution.', 'The ripple is still there. The instrument is averaging it for you, without asking.'] },
          { label: 'Simulations average too', body: '<p>An eigenmode solver can return a mode bound to the edge of the simulation domain. Before reporting a higher-order mode, check where its field lives and whether it moves when you move the boundary.</p>', narration: ['Simulations average too, in their own way.', 'An eigenmode solver can return a mode that lives at the edge of the simulation box.', 'Before reporting it, check whether it moves when you move the boundary.'] },
          { label: 'Label your inferences', body: '<p>When you attribute a ripple to a specific pair of reflectors, or a split to backscatter, write down whether you measured it or inferred it. This course labels its own inferences. So should you.</p>', narration: ['Label your inferences.', 'When you blame a ripple on two specific reflectors, write down whether you measured it, or inferred it.', 'This course labels its own inferences. So should you.'] }
        ], narration: ['Sometimes the measurement is the problem.'] },
      { type: 'chat', title: 'Scenario 1: The Calm Wavemeter', app: 'call', meta: { header: 'Video call', participants: [GARY, MGR, BY] },
        messages: [
          { from: GARY, text: 'No mode hops. The wavemeter has been steady all morning', side: 'right' },
          { from: BY, text: 'What is the update rate? The intensity on the fast photodiode is switching between two levels', side: 'left' },
          { from: GARY, text: 'I do not have a fast photodiode connected', side: 'right' },
          { from: MGR, text: 'Steady all morning sounds very stable to me', side: 'left' }
        ], narration: isThisOk('Gary concludes there are no hops because a slow instrument shows a steady average.') },
      { type: 'chat', title: 'Scenario 2: New Physics', app: 'thread', meta: { header: '#ring-characterization', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Measured free spectral range is way bigger than the design using our on-chip laser. Possible new physics', side: 'right' },
          { from: BY, text: 'Did you compare with a benchtop laser? An on-chip laser heats the chip while it sweeps, and the rings shift with it', side: 'left' },
          { from: GARY, text: 'The on-chip laser is more integrated, so it is more accurate', side: 'right' },
          { from: MGR, text: 'Let us call it new physics in the abstract and check later', side: 'left' }
        ], narration: isThisOk('Gary is reporting a heating artifact from his own source as new physics.') },
      { type: 'chat', title: 'Scenario 3: The Record Linewidth', app: 'email', meta: { header: 'Inbox', subject: 'Linewidth result (no isolator, saves loss)', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'Linewidth measurement attached. I removed the isolator to get more power into the interferometer.', side: 'right' },
          { from: BY, text: 'Without an isolator, reflections from the setup feed back into the laser. Your number is a property of the laser plus your setup.', side: 'left' },
          { from: GARY, text: 'The setup is not on the chip, so it does not count.', side: 'right' }
        ], narration: isThisOk('Gary removed the isolator from his linewidth measurement setup.') },
      { type: 'check', question: 'A ring measured with a laser on the same chip shows a free spectral range of 160 pm; a benchtop laser gives 120 pm; simulation gives 110 pm. What is the most likely cause?',
        options: [
          { text: 'The on-chip laser heats the chip while sweeping and red-shifts the rings during the measurement', correct: true, why: '' },
          { text: 'New physics in the ring', correct: false, why: 'The ring is the same ring in both measurements. Only the source changed. When the measured value depends on which source you use, suspect the source. Chowdhury and coworkers reported exactly these numbers: one hundred sixty picometers with the on-chip laser, one hundred twenty with a benchtop laser, and one hundred ten simulated. Sweeping the on-chip laser heated the chip and red-shifted the rings as it swept, which stretches the apparent spacing.' },
          { text: 'The benchtop laser is less accurate because it is not integrated', correct: false, why: 'Integration does not make a source more accurate. A separate benchtop laser does not dump heat into the chip, so it disturbs the device under test less. Its result is also closer to the simulation.' }
        ], correctSay: 'The source was heating the sample.', narration: ['Knowledge check.'] },
      { type: 'check', question: 'Gary measures laser linewidth without an isolator in the setup. What does published work warn about this?',
        options: [
          { text: 'Parasitic reflections from the setup can lead to an overestimate of the linewidth', correct: true, why: '' },
          { text: 'Nothing; the isolator only reduces power', correct: false, why: 'The isolator protects the laser from light reflected by the measurement setup itself: fiber tips, connectors, and the interferometer. Without it, that feedback changes the laser while you measure it. De Koninck and coworkers state that without an isolator, parasitic reflections lead to an overestimate of the linewidth.' },
          { text: 'The linewidth will always be underestimated, so it is a conservative choice', correct: false, why: 'The published warning goes the other way: parasitic reflections lead to an overestimate. More generally, feedback can either narrow or broaden a line depending on its phase and strength, as described in Chapter One. Either way, you are no longer measuring the laser alone.' }
        ], correctSay: 'Isolate the measurement, too.', narration: ['One more knowledge check.'] },
      takeaways(['Slow and coarse instruments average away hops and ripples: add a fast detector', 'The setup reflects too: isolate the measurement path', 'An on-chip source can heat the device it measures: cross-check with a separate source', 'Label measured facts and inferences separately', 'Keep every run. A clean run is the run where the hop happened while the instrument was looking elsewhere'], 'This completes the instructional portion of the course. The final assessment follows.'),
      ack('I have completed Chapter 7, Your Instruments Are Averaging.'),
      refs('dekoninck', 'chowdhury')
    ]
  };

  /* ---------- Incident 1 (Chapter 1): The Missing Isolator ---------- */
  var INC1 = [
    { type: 'video', title: 'Incident Review 1: The Missing Isolator', duration: 30,
      keyframes: [
        { t: 0, cast: ['Gary', 'Morgan'], caption: 'Friday afternoon. The customer demo is on Monday, and the link is short on power.', shot: 'Lab bench, wide, 35 mm lens, eye level. A fiber-pigtailed laser in a butterfly package on a temperature-controlled mount, a small cylindrical inline fiber isolator with a printed arrow, a fiber spool, a benchtop power meter. Gary seen from behind in a lab coat, gloved hands on the bench. Morgan in the doorway, out of focus, holding a laptop. Wall calendar with Monday circled in red.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: 'Link budget: short on power', sub: 'Customer demo on Monday', people: ['Gary', 'Morgan'] }); } },
        { t: 7, cast: ['Gary'], caption: 'Gary unplugs the isolator to win back its insertion loss. The power meter goes up.', shot: 'Macro, 100 mm lens. Gloved hands disconnect two fiber connectors, lift out the inline isolator and drop it into a small parts bag. The laser pigtail is mated directly to the link patch cord. Rack focus to the power meter as its reading rises. Slow push-in.', draw: function (ctx, w, h, u) { dIso(ctx, w, h, u, { none: true, gary: true, label: 'Isolator removed: every reflection now reaches the laser' }); } },
        { t: 14, cast: ['Gary'], caption: 'Monday, at the customer. An unused splitter port on the evaluation board ends in an open, flat-polished connector. The laser hears it.', shot: 'Customer lab, medium shot. Evaluation board on an anti-static mat; one connector with its dust cap off, flat ferrule end catching a highlight. Oscilloscope beside it: laser intensity jumping between two levels. Electrical spectrum analyzer: a broad self-heterodyne beat note. Customer engineers only as out-of-focus silhouettes. Red stamp.', draw: function (ctx, w, h, u) { dTrace(ctx, w, h, u, { label: 'Laser intensity versus time (illustrative)', x: 'time', y: 'intensity', f: telegraph, stamp: 'VIOLATION' }); } },
        { t: 22, cast: ['Pat Reynolds'], caption: 'Replay: the isolator stays in. A link budget is fixed by fixing the link, not by removing the protection.', shot: 'Still (Blender render): butterfly laser bench with the inline isolator back in line, arrow pointing away from the laser. Planned live shot: the customer board with a dust cap and an angled connector on the open port; power meter lower but steady; beat note narrow.', image: { src: 'media/render/butterfly_isolator_fc_bench.jpg' }, draw: function (ctx, w, h, u) { dIso(ctx, w, h, u, { label: 'Isolator in line, arrow away from the laser', stamp: 'COMPLIANT', stampColor: 'ok' }); } }
      ],
      narration: [DISCLAIM, 'Friday afternoon. The customer demo is on Monday, and the link is short on power.', 'Gary looks at the link budget. The isolator costs its insertion loss.', 'Gary unplugs the isolator. The power meter goes up. Gary feels good about this.', 'Monday, at the customer. An unused splitter port on the evaluation board ends in an open, flat-polished connector.', 'A flat end of fused silica reflects about three point three percent of the light that reaches it.', 'Assume an even one by two splitter. The light passes it twice, so about zero point eight percent of the laser output comes home. That is about minus twenty-one decibels.', 'With no isolator, nothing stops it. The intensity flickers, and the line broadens.', 'Let us replay that. The isolator stays in. A link budget is fixed by fixing the link.', MANTRA] },
    { type: 'chat', title: 'Incident 1: How It Unfolded', app: 'thread', meta: { header: '#demo-prep', participants: [GARY, MGR, BY] },
      messages: [
        { from: MGR, text: 'We are short on power for Monday. Find it', side: 'left' },
        { from: GARY, text: 'The isolator costs us its insertion loss. I can take it out', side: 'right' },
        { from: BY, text: 'Without it, every reflection on the customer board goes straight back into the laser', side: 'left' },
        { from: MGR, text: 'It is a distributed feedback laser. It already has feedback', side: 'left' },
        { from: GARY, text: 'Isolator out. Power is up. Looks great on the meter', side: 'right' },
        { from: BY, text: 'The meter averages. Can we check the spectrum and the linewidth before it goes to the demo?', side: 'left' },
        { from: MGR, text: 'No time. And Alex, let us keep the negativity offline', side: 'left' }
      ],
      narration: isThisOk('Gary removed the isolator to recover its insertion loss, and Morgan told Alex to keep the negativity offline.').concat(['Please note. Discouraging a coworker who points out a reflection is retaliation.', 'Zero tolerance applies to people as well as to photons.']) },
    felt('At the customer, the laser line was broad, the intensity flickered, and the customer asked whether this was the product',
      ['The isolator was in Gary\'s bag the whole time'],
      ['The isolator was in Gary\'s bag the whole time. It was isolating the bag.', 'Gary will not be punished for reporting this incident. He will be enrolled in this course again.', 'In Chapter Three, you will learn the official name for what happened on Friday. Isolation, traded for insertion loss. It is called Quid Pro Quo Feedback.']),
    tester('You are Gary. It is Monday morning at the customer. The self-heterodyne beat note of your laser is far broader than the datasheet linewidth, and the intensity flickers. The isolator is in your bag. What would you do?', [
      OK('Stop, put the isolator back in line, re-measure the beat note, and tell the customer and Morgan that the earlier setup had no isolator. Then log it'),
      W('Blame the laser vendor: the datasheet linewidth must be wrong', 'The laser vendor measured the laser under their own conditions, with their own isolation. A laser\'s linewidth under feedback depends on the reflector outside the laser, and that reflector is on the customer board, which the vendor did not ship. Tkach and Chraplyvy classified how external feedback narrows or broadens a distributed feedback laser line, and in the regime called coherence collapse, described by Lenstra and coworkers in nineteen eighty-five, the line broadens dramatically. The open connector returns about zero point eight percent of the laser output through the splitter, about minus twenty-one decibels. If your isolator performs like the hybrid integration isolator reported by Lapointe and coworkers, with more than twenty-five decibels of isolation, removing it lets more than three hundred times more of that reflected light back in. The vendor did not remove your isolator. You did.'),
      W('Blame the self-heterodyne setup: the delay fiber must be faulty', 'The delayed self-heterodyne setup did exactly what it was built to do. It beat the laser against a delayed, frequency-shifted copy of itself, and showed you the laser\'s phase noise as the width of a beat note. If the beat note is broad, the laser\'s phase noise is high. The measurement path has interfaces of its own, which is why it should also be isolated, as De Koninck and coworkers note. But the measurement setup is not what changed since Friday. The isolator is what changed.'),
      W('Say the laser is just warming up, and keep presenting', 'Warming up changes the laser temperature. That moves the phase of the light returning from the open connector, so the symptoms may come and go as the laser warms. That is not the laser settling. That is the feedback phase sweeping through its cycle, which is exactly the periodic behavior described in this chapter. In coherence collapse, the broadening is largely insensitive to the phase anyway. Waiting does not put the isolator back.'),
      W('Turn the laser current up so the laser is stronger than the reflection', 'Feedback effects are set by the fraction of light that returns, not by the absolute power. More current sends more light to the open connector, and more light comes back. The ratio is the same. You also move the laser to a different operating point that nobody characterized, in front of a customer.'),
      W('Average the beat note for longer until it looks narrower', 'Averaging reduces the noise on the displayed trace. It does not reduce the width of the line, because the width is the physics, not the noise on the display. A longer average gives you a smoother picture of the same broad line. The instrument did what it was told. It averaged.')
    ], 'Restore the isolator, re-measure, disclose, and log it.'),
    bystander('You are Alex. On Friday you saw Gary unplug the isolator, and Morgan asked you to keep the negativity offline. It is now Friday evening. What do you do?', [
      OK('Speak up now, to Gary and Morgan, using the approved phrase, and log the setup change before the demo'),
      WB('Say nothing. Morgan approved it', 'Morgan approved a schedule. Nobody can approve a reflection. The open connector on the customer board reflects the same three point three percent regardless of who signed off, and after the splitter, about zero point eight percent of the laser output still comes home. And being told to keep it offline does not change your responsibility. Asking you to stay quiet about a safety or quality concern is itself a reportable event under our non-retaliation policy.'),
      WB('Mention it at lunch on Tuesday, after the demo', 'Light reflected from a few meters of fiber returns to the laser within tens of nanoseconds. Lunch on Tuesday is several days later. The speak-up has to arrive before the demo, not after the light. Speak up while it can still change the outcome.'),
      WB('Quietly put the isolator back on Sunday without telling anyone', 'The intention is good. But now Gary\'s measurements from Friday describe a setup that no longer exists, and nobody knows it. Untracked changes to an optical path are how ripples become mysteries. Put it back, yes, but say so, and log it.'),
      WB('Write "building vibration" in the lab log so nobody gets in trouble', 'Vibration can modulate the phase of light returning from a reflector, and it can make an existing feedback problem flicker. But vibration does not create the reflector, and it does not remove the isolator. Writing a false root cause into the log means the next engineer will chase vibration while the isolator sits in a bag.'),
      WB('Submit an anonymous hotline report saying that some lasers are being mistreated', 'The anonymous hotline is always available, and we thank everyone who uses it. But a report that names no setup and no component cannot be acted on before Monday. Describe the optical path, not the feelings of the laser.')
    ], 'Speak up early, use the approved phrase, and log it.')
  ];

  /* ---------- Incident 2 (Chapter 2): The Ripple in the Report ---------- */
  var INC2 = [
    { type: 'video', title: 'Incident Review 2: The Ripple in the Report', duration: 30,
      keyframes: [
        { t: 0, cast: ['Gary'], caption: 'Gary edge-couples a flat-cleaved fiber to a cleaved silicon chip facet and sweeps a tunable laser across the band.', shot: 'Edge-coupling stage, top-down through a stereo microscope, then cut to the monitor feed: a bare flat-cleaved fiber tip approaching a chip facet, the air gap visible as a dark band. Gloved hand on the piezo stage controller. Tunable laser and photodetector in the rack behind, out of focus.', draw: function (ctx, w, h, u) { dTrace(ctx, w, h, u, { label: 'Measured transmission versus wavelength (illustrative)', x: 'wavelength', y: 'transmission', f: function (x) { return 0.55 + 0.12 * Math.cos(x * 26) - 0.1 * x; } }); } },
        { t: 8, cast: ['Gary'], caption: 'The transmission has a ripple with a 24 nm period. Gary reports it as an intrinsic device ripple, and fits it out.', shot: 'Over-the-shoulder to a monitor showing a report document; a plot captioned "intrinsic device ripple" is highlighted in yellow. The send button is clicked. Red stamp.', draw: function (ctx, w, h, u) { dTrace(ctx, w, h, u, { label: 'Reported as: intrinsic device ripple', x: 'wavelength', y: 'transmission', f: function (x) { return 0.55 + 0.12 * Math.cos(x * 26) - 0.1 * x; }, stamp: 'VIOLATION' }); } },
        { t: 16, cast: ['Alex'], caption: 'Alex backs the fiber off by 5 micrometers. The period changes. The chip did not move.', shot: 'Split screen. Left: Alex\'s gloved hand turning the stage micrometer a fraction of a turn. Right: the live trace, ripple period visibly shrinking. A ruler graphic marks the period before and after.', draw: function (ctx, w, h, u) { dTrace(ctx, w, h, u, { label: 'Gap 50 um (blue) and 55 um (amber)', x: 'wavelength', y: 'transmission', fs: [function (x) { return 0.55 + 0.12 * Math.cos(x * 26); }, function (x) { return 0.55 + 0.12 * Math.cos(x * 26 * 55 / 50); }] }); } },
        { t: 23, cast: ['Pat Reynolds'], caption: 'Replay: period to length. 24 nm at 1550 nm in air means a 50 micrometer gap: the fiber tip and the facet.', shot: 'Host beside a large graphic of the formula lambda squared over two L, with the fiber tip and the chip facet highlighted as the two mirrors of the cavity. A drop of index-matching gel fills the gap. Green check stamp.', draw: function (ctx, w, h, u) { dChip(ctx, w, h, u, { refl: [0.04, 0.96], hotFar: true, ruler: true, label: 'Period 24 nm in air: L = lambda^2 / (2 x 24 nm) = 50 um', stamp: 'COMPLIANT', stampColor: 'ok' }); } }
      ],
      narration: [DISCLAIM, 'Gary edge-couples a flat-cleaved fiber to a silicon chip, and sweeps a tunable laser across the band.', 'The transmission has a ripple. Its period is twenty-four nanometers.', 'Gary reports it as an intrinsic device ripple. He fits it, and subtracts it.', 'Alex backs the fiber off by five micrometers. The period changes. The chip did not move.', 'Let us replay that. In air, the ripple period is lambda squared over two times the gap.', 'At fifteen fifty nanometers, a period of twenty-four nanometers means a gap of fifty micrometers. That is the gap between the fiber tip and the facet.', 'Back off to fifty-five micrometers, and the period becomes about twenty-two nanometers. The ripple belongs to the gap.', 'Every interface reflects. Every reflection is a cavity.'] },
    { type: 'chat', title: 'Incident 2: How It Unfolded', app: 'email', meta: { header: 'Inbox', subject: 'Final report: waveguide transmission (intrinsic ripple characterized)', participants: [GARY, BY, MGR] },
      messages: [
        { from: GARY, text: 'Attached is the final report. The waveguide has an intrinsic ripple with a 24 nm period. I fit it and subtracted it.', side: 'right' },
        { from: BY, text: 'In air, 24 nm at 1550 nm is lambda squared over 2L, so L is 50 um. What is your fiber gap?', side: 'left' },
        { from: GARY, text: 'About 50 um. Coincidence.', side: 'right' },
        { from: BY, text: 'A flat-cleaved fiber end reflects about 3.3 percent. A cleaved silicon facet is about 31 percent. That is a cavity.', side: 'left' },
        { from: GARY, text: 'The simulation shows no ripple, so it cannot be the setup.', side: 'right' },
        { from: BY, text: 'Your simulation has perfectly matched layers where the facets are. It was told there are no reflections.', side: 'left' },
        { from: MGR, text: 'Can we call it an intrinsic feature and file a patent?', side: 'left' }
      ], narration: isThisOk('Gary reported the cavity between his fiber tip and the chip facet as an intrinsic device ripple, and subtracted it.') },
    { type: 'slide', layout: 'bullets', title: 'Incident 2: The Numbers',
      bullets: ['Air-gap etalon period: d-lambda = lambda^2 / (2 L). At 1550 nm, L = 50 um gives 24 nm; L = 55 um gives 21.8 nm', 'Flat silica end, n = 1.444 (Malitson, 1965): R = ((n - 1) / (n + 1))^2 = 3.3 percent, -14.8 dB', 'Cleaved silicon facet, n = 3.48 (Li, 1980): R = 31 percent (plane-wave Fresnel estimate)', 'Plane-wave peak-to-valley bound: 20 log10[(1 + r) / (1 - r)] with r = sqrt(R1 R2) = 0.10, about 1.8 dB. Diffraction in the gap returns less light, so the real ripple is smaller', 'Diagnostic: change the gap. If the period follows the gap, the ripple belongs to the gap'],
      callout: 'If the period moves when the fiber moves, the ripple is not the device.',
      narration: ['Let us check the numbers together.', 'In air, the ripple period is lambda squared over two times the gap. Fifty micrometers gives twenty-four nanometers. Fifty-five micrometers gives about twenty-two.', 'A flat end of fused silica, with an index of about one point four four four, reflects about three point three percent, or minus fifteen decibels.', 'A cleaved silicon facet, with an index of about three point four eight, reflects about thirty-one percent, as a plane-wave estimate.', 'Together they can make a ripple of up to about one point eight decibels peak to valley. Diffraction in the gap makes the real ripple smaller.', 'The diagnostic is simple. Change the gap. If the period follows the gap, the ripple belongs to the gap.', 'Every interface reflects. Every reflection is a cavity.'] },
    feltTabs('The customer received a report describing the gap between Gary\'s fiber and the chip as a property of the chip', [
      { label: 'Ashamed', body: '<p>Gary felt ashamed. That is a normal response. Shame is a normal response to a reflection, including your own.</p><p>This is a judgment-free zone. The incident has still been logged.</p>', narration: ['Gary felt ashamed.', 'That is a normal response. Shame is a normal response to a reflection, including your own.', 'This is a judgment-free zone. The incident has still been logged.'] },
      { label: 'Reflective', body: '<p>Gary felt reflective. About 3.3 percent reflective, at the fiber end, and about 31 percent at the facet.</p><p>Reflection is encouraged in people. In optical paths, please angle it.</p>', narration: ['Gary felt reflective.', 'About three point three percent reflective at the fiber end, and about thirty-one percent at the facet.', 'Reflection is encouraged in people. In optical paths, please angle it.'] },
      { label: 'Isolated', body: '<p>Gary felt isolated. Unlike his laser, which was not.</p><p>If you feel isolated after an incident, the Employee Assistance Program is available. If your laser is not isolated, please refer to Chapter 5.</p>', narration: ['Gary felt isolated.', 'Unlike his laser, which was not.', 'If you feel isolated after an incident, the Employee Assistance Program is available. If your laser is not isolated, please refer to Chapter Five.'] },
      { label: 'Logged', body: '<p>The incident has been logged. Being logged is not a punishment. It is a record. Records are how the next engineer finds the second mirror faster.</p>', narration: ['The incident has been logged.', 'Being logged is not a punishment. It is a record.', 'Records are how the next engineer finds the second mirror faster.', 'In Chapter Three, you will classify this incident. A flat fiber end, outside the chip, on the path Gary connected. That is External Feedback.', MANTRA] }
    ]),
    tester('You are Gary. The report with the intrinsic ripple went to the customer yesterday. Alex has just shown that the ripple period changes when the fiber gap changes. What would you do?', [
      OK('Log it, tell the report owner and the customer, re-measure with index-matching gel or an angled fiber, and issue a corrected report'),
      W('Re-run until the spectrum looks clean: nudge the fiber until a ripple peak lands on the wavelength in the specification table', 'Nudging the fiber changes the gap, which moves the ripple, which is the proof that the ripple belongs to the gap. Choosing the gap that happens to put a peak on the specification wavelength does not remove the cavity. It hides it at one wavelength and keeps it at every other wavelength. A measurement that depends on which run you kept is not a measurement of the device.'),
      W('Blame the simulation tool for not predicting the ripple', 'The simulation did exactly what it was told. A perfectly matched layer absorbs all outgoing light by construction, and a matched port reflects nothing by construction. Your model had no fiber, no air gap, and no facet reflection, so it could not show a ripple from them. If you want a simulation to predict a parasitic cavity, you must include both mirrors. The reflection in the lab came from the flat fiber end and the cleaved facet, fifty micrometers apart.'),
      W('Blame the tunable laser sweep: its output power must ripple', 'A tunable laser\'s output power can vary with wavelength, which is why transmission is normalized to a reference measurement of the laser alone. But the tunable laser does not know where your fiber is. The period changed when only the gap changed, from twenty-four nanometers at fifty micrometers to about twenty-two at fifty-five. That is lambda squared over two L, in air.'),
      W('Blame the fiber cleaver for making the cleave too flat', 'The cleaver did exactly what a standard cleaver is designed to do. It made a flat end, perpendicular to the fiber axis. A flat end of fused silica reflects about three point three percent straight back. If you need that reflection steered away from the guided mode, use an angle cleave, an angle-polished fiber, or index-matching gel. The cleaver followed the procedure. The procedure needed an angle.'),
      W('Say the index-matching gel must have dried out', 'The setup log shows no index-matching gel in this measurement. Gel that was never applied cannot dry out. Applied properly, a gel with an index close to that of glass would have largely removed the glass-to-air reflection at the fiber end, and with it most of this cavity. That is a good idea for the re-measurement.'),
      W('Smooth the spectrum with a moving average one ripple period wide', 'A moving average one period wide removes the ripple from the plot. It also removes any real feature narrower than twenty-four nanometers, and it leaves the cavity in the setup. Smoothing is a choice about the picture. The cavity is a fact about the bench.')
    ], 'Log it, correct the report, and remove the second mirror.'),
    bystander('You are Alex. You found the cause. Gary seems embarrassed and has asked you not to mention it to anyone. What do you do?', [
      OK('Use the approved phrase with Gary, offer to help re-measure, and make sure the correction is logged and sent to the customer'),
      WB('Say nothing. It is his report', 'It is his report, and it is now the customer\'s data. A wrong root cause in a customer report will be designed around by people who never met Gary. Respecting a colleague means helping him correct it, not helping him keep it.'),
      WB('Mention it at lunch, in a general way, so he gets the hint', 'A hint at lunch is not a correction. The report stays wrong, and nothing is logged. Speak to the owner directly, with the approved phrase.'),
      WB('Reply-all to the customer with the formula, lambda squared over 2L', 'The physics is correct, and we appreciate your enthusiasm for the formula. But reply-all is not a speak-up channel. Raise it with the report owner first, so the correction goes out once, from the owner, with the re-measured data.'),
      WB('Wait for the customer to notice', 'The customer may not notice. They may design a compensation filter for a ripple that exists only on Gary\'s bench. Waiting moves the reflection downstream, to a place where it costs more.')
    ], 'Speak up kindly, help fix it, log it.')
  ];

  /* ---------- Incident 3 (Chapter 4): The Demo Restart ---------- */
  var INC3 = [
    { type: 'video', title: 'Incident Review 3: The Demo Restart', duration: 34,
      keyframes: [
        { t: 0, cast: ['Gary'], caption: 'The live customer demo. The laser is tuned to the customer\'s channel. The isolator is still in Gary\'s bag from Friday.', shot: 'Customer conference room, wide. Demo rack with the laser, a wavemeter and a spectrum display on a large monitor. Gary from behind, presenting with a clicker. Customer engineers as out-of-focus silhouettes. A parts bag with a small cylindrical isolator visible in an open laptop bag on the floor.', draw: function (ctx, w, h, u) { dComb(ctx, w, h, u * 0.4, { label: 'Lasing mode (blue) under the filter (amber)' }); } },
        { t: 8, cast: ['Gary'], caption: 'Midway through, the wavemeter jumps. The customer\'s receiver loses lock.', shot: 'Close on the wavemeter display: the reading jumps to a new value. Cut to the customer receiver status panel turning from green to red. Gary freezes mid-sentence.', draw: function (ctx, w, h, u) { dTrace(ctx, w, h, u, { label: 'Lasing frequency versus time: a hop', x: 'time', y: 'frequency', f: function (x) { return x < 0.55 ? 0.45 + 0.05 * x : 0.75 + 0.05 * x; } }); } },
        { t: 15, cast: ['Gary'], caption: 'Gary power-cycles the laser. It comes back on the right channel. He says "minor software glitch" and nothing else.', shot: 'Close on a gloved hand flipping the laser driver output off and on. The wavemeter returns to the original value. Gary turns to the room, palms up, smiling. Red stamp.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: '"Minor software glitch"', sub: 'Laser power-cycled. Nothing logged.', people: ['Gary'], stamp: 'VIOLATION' }); } },
        { t: 22, cast: ['Gary', 'Morgan'], caption: 'Afterwards, Morgan: "The laser is retaliating. And frankly, so will I if this happens again."', shot: 'Corridor outside the conference room, medium two-shot from behind both figures. Morgan points at Gary\'s laptop bag. Gary\'s shoulders drop.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: '"The laser is retaliating. So will I."', sub: 'Retaliation by managers is also prohibited', people: ['Gary', 'Morgan'], stamp: 'VIOLATION' }); } },
        { t: 28, cast: ['Pat Reynolds'], caption: 'Replay: log the hop, the temperature, the current and the sweep direction. Put the isolator back. Retaliation is prohibited, by lasers and by managers.', shot: 'Host in the empty conference room beside the demo rack, isolator back in line, a lab log open on the table with a new entry. Green check stamp.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: 'Hop logged. Isolator restored. Nobody threatened.', sub: 'Temperature, current, sweep direction, setup', people: ['Pat Reynolds'], stamp: 'COMPLIANT', stampColor: 'ok' }); } }
      ],
      narration: [DISCLAIM, 'The live customer demo. The laser is tuned to the customer\'s channel.', 'The isolator is still in Gary\'s bag from Friday.', 'Midway through, the wavemeter jumps. The laser has hopped. The customer\'s receiver loses lock.', 'Gary power-cycles the laser. It comes back on the right channel.', 'Gary says, minor software glitch. He says nothing else.', 'Afterwards, Morgan says, the laser is retaliating. And frankly, so will I if this happens again.', 'Let us replay that.', 'Restarting a laser picks a mode again from a fresh start. It does not remove the reflector, and it does not remove the hop point. The next temperature drift brings the laser back to it.', 'Log the hop, the temperature, the current, and the sweep direction. Put the isolator back.', 'And retaliation is prohibited. By lasers, and by managers.', 'Every cavity hops.'] },
    { type: 'chat', title: 'Incident 3: How It Unfolded', app: 'phone', meta: { header: 'Messages', participants: [MGR, GARY] },
      messages: [
        { from: MGR, text: 'What happened in there', side: 'left' },
        { from: GARY, text: 'Laser hopped. I restarted it. I think they did not notice', side: 'right' },
        { from: MGR, text: 'Good. Keep it out of the log. The log goes to the customer', side: 'left' },
        { from: GARY, text: 'The isolator is still out from Friday', side: 'right' },
        { from: MGR, text: 'The laser is retaliating. If it happens again, so will I', side: 'left' },
        { from: GARY, text: 'ok', side: 'right' }
      ], narration: isThisOk('Morgan told Gary to keep a mode hop out of the log, and threatened him if it happens again.').concat(['A manager threatening an engineer over a mode hop is retaliation.', 'Retaliation against people is prohibited. So is keeping a hop out of the log.']) },
    felt('The laser hopped in front of the customer, Gary covered it with a story about software, and his manager threatened him',
      ['Gary also felt afraid to report. That is normal too', 'Retaliation against Gary for reporting is prohibited'],
      ['Gary also felt afraid to report. That is normal too.', 'Fear of retaliation is the most common reason people do not speak up. That is why retaliation is prohibited.', 'The laser is allowed to hop. It is a laser. Morgan is not allowed to retaliate. Morgan is a manager.', 'For the record, this incident is classified as Quid Pro Quo Feedback, followed by Retaliation by the laser. The manager\'s threat is handled under a separate policy, with the same zero tolerance.']),
    tester('You are Gary. The hop is not in the log, Morgan told you to keep it out, and the customer wants a second demo next week. What would you do?', [
      OK('Log the hop with temperature, current and sweep direction, put the isolator back, map the hop points in both sweep directions, and report the instruction to hide it through the speak-up channel'),
      W('Restart the laser before every demo, so it always starts in a good mode', 'A restart picks a mode again, starting from wherever the laser comes up. Because hop points show hysteresis, the restart decides which side of the hysteresis loop you start on. It does not remove the loop. The reflector on the customer board is still there, the isolator is still in your bag, and the next temperature drift will carry the laser back to the hop point, possibly in the middle of the next demo.'),
      W('Blame the temperature controller', 'The temperature controller held the temperature at its sensor, at the setpoint you gave it. That is all it was asked to do. It does not regulate the length of fiber between the laser and the open connector on the customer board, whose optical length changes with the room temperature and moves the feedback phase. Without the isolator, that external cavity adds its own modes to the competition, and one of them won.'),
      W('Blame the previous shift for leaving the laser at a bad operating point', 'An operating point is good or bad only relative to a hop map, and nobody has measured the hop map of this laser with this setup. The previous shift set a current and a temperature. You removed the isolator and connected a reflector. A hop map, measured in both directions, is how you choose an operating point with margin.'),
      W('Tell the customer it was a software glitch', 'The wavemeter showed the optical frequency jump. Software does not change the optical frequency of a laser. A jump in optical frequency is a mode hop, and calling it a glitch withholds the one fact the customer needs to keep their receiver locked. It is also untrue, which is a separate policy.'),
      W('Blame building vibration', 'Vibration can shake the fibers and modulate the phase of the light coming back from a reflector. With feedback present, that can trigger hops sooner. But vibration needs a reflector to work with, and an isolator would have kept that light out of the laser. Vibration is an accomplice. The open connector, with no isolator, is the root cause.'),
      W('Explain that the laser was just warming up', 'Warming up moves the temperature of the laser, which moves the cavity modes and the filter at different rates. That is exactly how a laser walks into a hop point. A laser that hops while warming up will hop again whenever the customer\'s system warms up. Warming up is the trigger, not the excuse.')
    ], 'Log it, fix it, and report the retaliation.'),
    bystander('You are Alex. You were in the room. You saw the wavemeter jump and heard "minor software glitch". Afterwards you heard Morgan say the laser is retaliating, and so will he. What do you do?', [
      OK('Speak up with the approved phrase, log what you saw, and report the threat through the non-retaliation channel'),
      WB('Nothing. It is between Gary and Morgan', 'A threat over a reported problem is everyone\'s business, because it teaches the whole team to stop reporting problems. And the mode hop is the customer\'s business. You saw both. Witnesses are part of the speak-up process.'),
      WB('Tell it at lunch as a funny story about a retaliating laser', 'It is a funny story. The laser part is physics. The manager part is a policy violation, and telling it at lunch spreads it without reporting it. Report first. Tell the story after the incident is closed, with names removed.'),
      WB('Interrupt the next demo to explain Lorentz reciprocity to the customer', 'We appreciate your command of Chapter Five. But this incident is a mode hop from unisolated feedback, not a reciprocity question, and a customer demo is not a speak-up channel. Speak to the owner and log it.'),
      WB('Wait for the annual engagement survey', 'The annual survey closes in eleven months. The second demo is next week. Speak up on the time scale of the problem, not the time scale of the survey.')
    ], 'Speak up, log it, and report retaliation.')
  ];

  /* ---------- Incident 4 (Chapter 5): Installed Backwards ---------- */
  var INC4 = [
    { type: 'video', title: 'Incident Review 4: Installed Backwards', duration: 34,
      keyframes: [
        { t: 0, cast: ['Gary'], caption: 'Gary installs a new inline fiber isolator. The arrow on the housing points at the laser.', shot: 'Still (Blender render): butterfly laser on a finned mount, yellow pigtail through an inline isolator spliced in backwards (splice sleeves on both sides), its printed arrow pointing toward the laser; capped FC/APC and FC/PC connectors in the foreground. Planned live shot: gloved hands mating the isolator, slow push-in on the arrow.', image: { src: 'media/render/butterfly_isolator_reversed_bench.jpg' }, draw: function (ctx, w, h, u) { dIso(ctx, w, h, u, { rev: true, gary: true, label: 'Isolator installed with the arrow pointing at the laser' }); } },
        { t: 8, cast: ['Gary'], caption: 'The output power drops by about the isolation ratio. Gary files a return request against the laser.', shot: 'Over-the-shoulder to a laptop: a vendor return form, reason field "laser output dead". Beside it, the power meter reading near the noise floor. Red stamp.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: 'Return request: "laser output dead"', sub: 'Filed against the laser vendor', people: ['Gary'], stamp: 'VIOLATION' }); } },
        { t: 15, cast: ['Gary'], caption: 'Downstream, an optical amplifier emits spontaneous emission backward. The reversed isolator passes it straight into the laser.', shot: 'Animated cutaway of the link: laser, reversed isolator, fiber, an amplifier module. A faint red glow leaves the amplifier input traveling backward, passes the isolator in its low-loss direction and lands on the laser facet.', draw: function (ctx, w, h, u) { dIso(ctx, w, h, u, { rev: true, right: 'amplifier', label: 'Backward light from downstream passes the reversed isolator' }); } },
        { t: 22, cast: ['Gary'], caption: 'On the free-space bench, Gary rotates a polarization-dependent isolator by 90 degrees to fit the mount. Nothing comes out.', shot: 'Optical table, medium close. A free-space isolator in a rotation mount, engraved input polarizer axis now horizontal while the laser polarization is vertical. Gary\'s gloved hand tightens the setscrew. The detector card behind it is dark.', draw: function (ctx, w, h, u) { dIso(ctx, w, h, u, { rot: true, label: 'Input polarizer crossed with the laser polarization' }); } },
        { t: 28, cast: ['Pat Reynolds'], caption: 'Replay: the arrow points away from the laser, and the input polarizer is aligned to the laser polarization. Then measure forward power before you close the setup.', shot: 'Still (Blender render): the same bench with the isolator arrow pointing away from the laser. Planned live shot: host beside both setups, a printed checklist with two ticked lines, direction and rotation.', image: { src: 'media/render/butterfly_isolator_fc_bench.jpg' }, draw: function (ctx, w, h, u) { dIso(ctx, w, h, u, { label: 'Arrow away from the laser. Polarizer aligned.', stamp: 'COMPLIANT', stampColor: 'ok' }); } }
      ],
      narration: [DISCLAIM, 'Gary installs a new fiber isolator. The arrow on the housing points at the laser.', 'The output power drops by about the isolation ratio. Gary files a return request against the laser.', 'Here is the subtle part. For the laser\'s own light reflected downstream, a reversed isolator attenuates the round trip just as much as a correct one. Forward loss times backward loss is the same product either way.', 'What the reversed isolator does is block almost all of the laser\'s output, and pass any light that starts downstream, such as amplified spontaneous emission traveling backward out of an amplifier, straight into the laser.', 'Meanwhile, on the free-space bench, Gary rotates a polarization-dependent isolator by ninety degrees to fit the mount.', 'Its input polarizer is now crossed with the laser polarization. By the law of Malus, the transmission goes as cosine squared of the angle. At ninety degrees, nothing comes out.', 'Let us replay that. The arrow points away from the laser. The input polarizer is aligned with the laser polarization.', 'Reciprocity is not optional. Isolate, angle, and verify.'] },
    { type: 'chat', title: 'Incident 4: How It Unfolded', app: 'thread', meta: { header: '#lab-help', participants: [GARY, BY, MGR] },
      messages: [
        { from: GARY, text: 'Power dropped after the isolator swap. Laser must be dead. Return request filed', side: 'right' },
        { from: BY, text: 'Which way does the arrow point', side: 'left' },
        { from: GARY, text: 'Toward the laser. But reciprocity says light goes both ways the same, so the direction should not matter, right?', side: 'right' },
        { from: BY, text: 'Reciprocity is exactly what an isolator breaks. That is its entire job', side: 'left' },
        { from: GARY, text: 'My link simulation shows no difference either way. I modeled it as a symmetric two-port', side: 'right' },
        { from: BY, text: 'A symmetric two-port is a reciprocal device. You modeled a patch cord', side: 'left' },
        { from: MGR, text: 'Can we rotate it 90 degrees as a compromise', side: 'left' }
      ], narration: isThisOk('Gary installed the isolator backwards, and believes reciprocity means direction does not matter.') },
    { type: 'slide', layout: 'policy', title: 'Policy 4.3: Isolator Orientation',
      bullets: ['A Faraday isolator is non-reciprocal: forward and backward transmission differ by design (Jalas et al., 2013)', 'Polarization-dependent isolator: input polarizer, 45 degree Faraday rotator, output polarizer at 45 degrees. Backward light is rotated a further 45 degrees in the same sense, reaches 90 degrees, and is blocked (Saleh and Teich, 1991)', 'Reversed (180 degrees): the laser output is attenuated by about the isolation ratio; light that starts downstream passes into the laser with only the insertion loss', 'For the laser\'s own reflection, the round trip is forward transmission times backward transmission: the same product in either orientation', 'Rotated about its axis: a polarization-dependent isolator transmits as cos^2 of the angle between laser polarization and input polarizer (Malus): 0 at 90 degrees, 0.5 at 45 degrees. A polarization-independent fiber isolator does not care about rotation. It cares about direction'],
      callout: 'Direction is not angle. Check both.',
      narration: ['Policy section four point three. Isolator orientation.', 'A Faraday isolator is non-reciprocal by design.', 'In a polarization-dependent isolator, light passes an input polarizer, is rotated by forty-five degrees in a Faraday rotator, and leaves through an output polarizer set at forty-five degrees.', 'Light coming backward is rotated another forty-five degrees in the same sense. It arrives at ninety degrees to the input polarizer, and is blocked. That is non-reciprocity.', 'If you install the isolator backwards, the laser output is attenuated by about the isolation ratio, and light that starts downstream gets into the laser with only the insertion loss.', 'For the laser\'s own reflection, the round trip is the forward transmission times the backward transmission. That product is the same in either orientation. So a reversed isolator does not protect you more, or less, from your own echo. It just takes your power, and lets strangers in.', 'If you rotate a polarization-dependent isolator about its axis, the transmission follows the law of Malus. Cosine squared. Zero at ninety degrees, one half at forty-five.', 'A polarization-independent fiber isolator does not care about rotation. It cares about direction.', 'Direction is not angle. Check both.'] },
    felt('The laser vendor returned the laser marked "no fault found", with a photo of the isolator arrow',
      ['The vendor\'s photo was circled in red'],
      ['The vendor\'s photo was circled in red. Red circles are a normal part of the return process.', 'Gary will be asked to re-read Policy section four point three. Twice. Once in each direction.', 'For the record, the light from the amplifier is External Feedback, admitted by an isolator installed backwards. The lost power is classified as embarrassing.']),
    tester('You are Gary. The vendor has returned the laser marked "no fault found". You have just noticed that the arrow on the isolator points at the laser. What would you do?', [
      OK('Turn the isolator around, re-measure output power, log the orientation error, tell the laser owner, and close the vendor return with the true cause'),
      W('Quietly turn the isolator around and say the vendor must have fixed something', 'The vendor found no fault because there was no fault in the laser. Saying they fixed something puts a false root cause into two companies\' records. The next engineer who sees low power will also send a good laser back. The true cause is one line in the log: isolator installed backwards.'),
      W('Blame the arrow on the housing for being ambiguous', 'The arrow on an inline isolator marks the direction of forward propagation, from the input port to the output port. It was printed for exactly this moment. The procedure is to check the arrow against the drawing, and then to measure forward power through the isolator before closing the setup. Both steps would have caught it.'),
      W('Blame the patch cord from the shared drawer', 'Patch cords from the shared drawer deserve suspicion. Mating an angle-polished connector to a flat one, for example, leaves a gap and a tilted interface, and adds loss. Swap the cord and check. But a backwards isolator attenuates by roughly its isolation ratio, which is far more than a mismatched connector is designed to explain, and the isolator is the part you changed.'),
      W('Blame the link simulation, which showed no difference between the two orientations', 'The simulation did exactly what it was told. You modeled the isolator as a symmetric two-port, and a symmetric scattering matrix describes a reciprocal device. A reciprocal device has no preferred direction, so the model could not tell the orientations apart. The entire function of an isolator is an asymmetric scattering matrix, as Jalas and coworkers put it. Model the device you installed.'),
      W('Rotate the isolator by 90 degrees and try again', 'Direction is not angle. A polarization-independent fiber isolator transmits the same for any rotation about its axis, so rotating it does nothing. A polarization-dependent isolator rotated by ninety degrees crosses its input polarizer with the laser polarization, and by the law of Malus, cosine squared of ninety degrees is zero. Turning it around end to end is the fix.'),
      W('Turn the laser current up to make up for the lost power', 'The light is not lost in the laser. It is being blocked by the isolator, which is doing its job in the wrong direction. Raising the current pushes the laser to an operating point nobody characterized, and almost all of the extra light is blocked as well.')
    ], 'Turn it around, re-measure, and log the true cause.'),
    bystander('You are Alex. Gary has posted a photo of his setup in the lab channel. The isolator arrow points at the laser. The vendor return has already shipped. What do you do?', [
      OK('Speak up now with the approved phrase, point to the arrow, promptly and privately, so the return and the log can be corrected'),
      WB('Add a laughing reaction to the photo and move on', 'A reaction is not a correction. It tells Gary that something is funny, and it does not tell him what. It also tells the rest of the channel. Speak to the owner directly, and point to the arrow.'),
      WB('Say nothing. The vendor will find no fault and he will work it out', 'He may work it out after a second return, or a third. Each return costs time, shipping, and the vendor\'s goodwill. You can save all of it with one sentence today.'),
      WB('Mention it at lunch', 'The return has already shipped. Lunch is later. Speak up on the time scale of the problem.'),
      WB('Turn the isolator around yourself after hours', 'The physics is right. The process is not. Gary\'s records and the vendor return now describe a setup that no longer exists, and nobody knows it changed. Untracked changes to an optical path are how future incidents begin.')
    ], 'One sentence, today, to the owner.')
  ];

  /* ---------- Incident 5 (Chapter 6): Two Modes and a Record ---------- */
  var INC5 = [
    { type: 'video', title: 'Incident Review 5: Two Modes and a Record', duration: 30,
      keyframes: [
        { t: 0, cast: ['Gary'], caption: 'Gary sweeps a high-Q ring. Every resonance has two dips.', shot: 'Dark lab, probe station, top-down through the microscope onto a ring resonator chip with lensed fibers on both sides. Monitor in the foreground: a transmission sweep with a doublet at each resonance. Gary from behind, leaning in.', draw: function (ctx, w, h, u) { dRing(ctx, w, h, u, { gary: true, speck: true, ccw: true, label: 'A scatterer couples clockwise and counter-clockwise light' }); } },
        { t: 7, cast: ['Gary'], caption: 'Gary reports the pair as two transverse mode families: "a dual-mode ring".', shot: 'Slide on a monitor titled "Dual-mode ring", two arrows labelling the two dips "mode 1" and "mode 2". Red stamp.', draw: function (ctx, w, h, u) { dTrace(ctx, w, h, u, { label: 'Reported as: two mode families', x: 'detuning', y: 'transmission', f: function (x) { return 1 - 0.75 * lor(x, 0.42, 0.04) - 0.75 * lor(x, 0.58, 0.04); }, stamp: 'VIOLATION' }); } },
        { t: 14, cast: ['Gary'], caption: 'For the record intrinsic Q, Gary fits one shallow dip and picks the over-coupled formula, because it gives the bigger number.', shot: 'Screen insert: a fitting tool dialog with a toggle "under-coupled / over-coupled". The cursor clicks over-coupled; the intrinsic Q field triples. Gary types "record" into the slide title. Red stamp.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: 'Intrinsic Q: record (over-coupled branch)', sub: 'Branch chosen because it gives the larger number', people: ['Gary'], stamp: 'VIOLATION' }); } },
        { t: 21, cast: ['Alex', 'Pat Reynolds'], caption: 'Replay: light comes back out of the input port exactly at the doublet. Two independent mode families would not do that, and would walk apart across the spectrum.', shot: 'Same bench; Alex adds a circulator at the input and a second detector on the reflection port. The monitor shows a reflection peak aligned with each doublet. Then a wide sweep: the splitting stays pinned to every resonance. Green check stamp.', draw: function (ctx, w, h, u) { dRing(ctx, w, h, u, { standing: true, ccw: true, speck: true, label: 'Backscatter: one mode pair, split, and reflecting', stamp: 'COMPLIANT', stampColor: 'ok' }); } }
      ],
      narration: [DISCLAIM, 'Gary sweeps a high Q ring. Every resonance has two dips.', 'Gary reports the pair as two mode families, and calls it a dual-mode ring.', 'For the record intrinsic Q, Gary fits one of the shallow dips, and picks the over-coupled formula, because it gives the bigger number.', 'Let us replay that.', 'Alex looks at the light coming back out of the input port. There is a reflection peak, exactly at each doublet.', 'Two independent mode families would not send light back. And because they have different group indices, they would have different free spectral ranges, and walk apart across the spectrum.', 'A backscatter doublet stays pinned to every resonance, and it reflects. Every interface reflects. Even a speck is an interface.'] },
    { type: 'chat', title: 'Incident 5: How It Unfolded', app: 'email', meta: { header: 'Inbox', subject: 'Record intrinsic Q on the new lot (dual-mode ring)', participants: [GARY, BY, MGR] },
      messages: [
        { from: GARY, text: 'Every resonance shows two modes. Fitting one dip, the intrinsic Q is a record, assuming over-coupling.', side: 'right' },
        { from: BY, text: 'Why over-coupled?', side: 'left' },
        { from: GARY, text: 'It gives the larger number. The fitting tool offers both.', side: 'right' },
        { from: BY, text: 'It offers both because transmission alone cannot tell them apart. And there is light coming back out of the input port at every doublet. That is one mode pair, split by backscatter.', side: 'left' },
        { from: MGR, text: 'Record Q and two modes. Put both in the abstract.', side: 'left' }
      ], narration: isThisOk('Gary chose the coupling regime that gives the larger Q, and called a backscatter doublet two modes.') },
    { type: 'slide', layout: 'bullets', title: 'Incident 5: The Numbers',
      bullets: ['Backscatter splitting of high-Q modes: Weiss et al., 1995; modal coupling model: Kippenberg, Spillane and Vahala, 2002', 'Single-mode extraction: Qi = 2 QL / (1 + sqrt(T0)) if under-coupled, 2 QL / (1 - sqrt(T0)) if over-coupled. The all-pass transmission is symmetric in loss and coupling (Yariv, 2000), so |T| alone cannot choose the branch', 'Coupled-mode model of this chapter, critically coupled before splitting, doublet well resolved: each dip bottoms at T0 = 0.25, and about 25 percent of the light comes back out of the input port at each dip', 'With T0 = 0.25 and each dip about one loaded linewidth wide: under-coupled branch gives Qi = 1.33 QL, over-coupled branch gives Qi = 4 QL. The true value in this model is 2 QL', 'Two transverse mode families have different group indices, hence different FSR = c / (2 pi R ng): their spacing changes from resonance to resonance'],
      callout: 'The bigger number is not a coupling regime.',
      narration: ['Let us check the numbers together.', 'Backscatter splitting of high Q modes was reported by Weiss and coworkers in nineteen ninety-five, and modeled as coupling between counter-propagating waves by Kippenberg, Spillane, and Vahala in two thousand two.', 'For a single resonance, you can get the intrinsic Q from the loaded Q and the dip depth. But there are two formulas, one for under-coupling and one for over-coupling.', 'The transmission of a ring on a bus is symmetric in the loss and the coupling, as Yariv wrote down in two thousand. So the transmission magnitude alone cannot tell you which formula applies.', 'In the coupled mode model of this chapter, take a ring that was critically coupled before splitting. Once the doublet is well resolved, each dip bottoms out at one quarter transmission, and about one quarter of the light comes back out of the input port.', 'Plug that into the two formulas. The under-coupled branch gives one point three three times the loaded Q. The over-coupled branch gives four times. The true value in this model is two times.', 'Gary chose four. Because it was bigger.', 'And two real mode families have different group indices, so their free spectral ranges differ, and their spacing changes from resonance to resonance. A doublet does not walk. It reflects.'] },
    feltTabs('Gary\'s record Q and his dual-mode ring were in the draft abstract, and both were one ring with a speck on it', [
      { label: 'Ashamed', body: '<p>Gary felt ashamed. That is a normal response. Shame is a normal response to a reflection, including your own.</p><p>This is a judgment-free zone. The incident has still been logged.</p>', narration: ['Gary felt ashamed.', 'That is a normal response. Shame is a normal response to a reflection, including your own.', 'This is a judgment-free zone. The incident has still been logged.'] },
      { label: 'Split', body: '<p>Gary felt split: one part wanted to report the record, one part knew the branch was chosen for its size. Like his resonance, the two parts are coupled.</p><p>Fit both. Report both. Choose with evidence.</p>', narration: ['Gary felt split.', 'One part of Gary wanted to report the record. The other part knew the branch was chosen for its size.', 'Like his resonance, the two parts are coupled. Fit both. Report both. Choose with evidence.'] },
      { label: 'Over-coupled', body: '<p>Gary felt over-coupled to the result. Strong coupling to a result broadens judgment the way strong coupling to a bus broadens a resonance.</p>', narration: ['Gary felt over-coupled to the result.', 'Strong coupling to a result broadens judgment, the way strong coupling to a bus broadens a resonance.'] },
      { label: 'Logged', body: '<p>The incident has been logged. Being logged is not a punishment. It is a record. Records are how the next engineer fits a doublet the first time.</p>', narration: ['The incident has been logged.', 'Being logged is not a punishment. It is a record.', 'Records are how the next engineer fits a doublet the first time.', 'For the record, a speck on your own ring is Intracavity, or Domestic, Feedback. It lives in the ring.', MANTRA] }
    ]),
    tester('You are Gary. The record Q is in the draft abstract, due tonight. The reflection port shows light at every doublet. What would you do?', [
      OK('Fit a coupled-mode doublet model, settle the coupling regime with independent evidence, report the intrinsic linewidth and the backscatter rate separately, and correct the abstract'),
      W('Keep the over-coupled branch: the fitting tool offered it', 'The fitting tool did exactly what it was told. It offered both branches because, as Yariv showed in two thousand, the transmission of a ring on a bus is symmetric in the loss and the coupling, so the transmission magnitude cannot choose between them. The choice has to come from independent evidence, for example a series of rings with different bus gaps, or a phase-sensitive measurement. In the model of this chapter, the over-coupled branch reports twice the true intrinsic Q.'),
      W('Blame the tunable laser sweep for making two dips', 'A tunable laser sweep can distort a trace if it hops or sweeps non-linearly, which is why sweeps are calibrated against a wavelength reference. But a sweep artifact does not send light back out of the input port exactly at the resonance, and it does not repeat the same splitting at every resonance. The reflection port says backscatter. Repeat the sweep in both directions, and with a reference, and see if the doublet stays.'),
      W('Blame the mode solver, which found a second mode family', 'The mode solver did exactly what it was told. It found the transverse modes that the waveguide cross-section supports. Whether the second dip belongs to that family is a separate question, and the answer is in the free spectral range. A second family has a different group index, so its resonances drift relative to the first family across the spectrum. Your doublets stay pinned, with the same small splitting, and they reflect. That is backscatter.'),
      W('Report it as two modes and let the reviewers decide', 'Reviewers can only judge what you show them. If the abstract says two modes and does not show the reflection port, the reviewers will agree with you, and the error moves into the literature. The data to decide is already on your bench.'),
      W('Average many sweeps until the two dips merge into one', 'Averaging a stable doublet does not merge it. If the dips do merge in the average, it is because the resonance drifted between sweeps, and the averaged dip is then broader than either real dip, so the Q you extract is lower and still wrong. The instrument averaged. The physics did not.')
    ], 'Fit the doublet, report both numbers, choose the regime with evidence.'),
    bystander('You are Alex, a co-author. The abstract deadline is tonight. Gary says the record Q is "basically right". What do you do?', [
      OK('Use the approved phrase with Gary and the corresponding author today, and offer to run the doublet fit before the deadline'),
      WB('Nothing. Record Q values are good for everyone', 'A record that is a factor of two too high is good for nobody, including the next group that tries to reproduce it. You are a co-author. Your name will be on the number.'),
      WB('Ask to be removed from the author list, quietly', 'Removing yourself protects you. It does not correct the abstract. The approved process is to raise the concern with the authors, and help fix it.'),
      WB('Leave an anonymous sticky note on the probe station that says "doublet"', 'The note is technically correct, and we admire its brevity. But an anonymous note cannot answer questions, cannot run the fit, and will not be read before tonight. Speak up in person.'),
      WB('Mention it at lunch next week', 'The deadline is tonight. Lunch next week is after the abstract is submitted. Speak up on the time scale of the problem.')
    ], 'Speak up before the deadline, and help fix it.')
  ];

  /* ---------- Incident 6 (Chapter 7): Passed by Averaging ---------- */
  var INC6 = [
    { type: 'video', title: 'Incident Review 6: Passed by Averaging', duration: 34,
      keyframes: [
        { t: 0, cast: ['Gary'], caption: 'Outgoing inspection. Gary screens lasers with a swept optical spectrum analyzer and a power meter with a long averaging time.', shot: 'Inspection station, wide, cool fluorescent light. A rack with a spectrum analyzer, a power meter and a wavemeter; a tray of butterfly lasers in anti-static foam; inspection sheets on a clipboard. Gary from behind in a lab coat, gloved hand on the analyzer start-sweep key.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: 'Outgoing inspection', sub: 'One sweep at a time. Long averaging time.', people: ['Gary'] }); } },
        { t: 8, cast: ['Gary'], caption: 'One laser hops between two adjacent modes. Most sweeps show only one of them. The power meter shows a steady average.', shot: 'Split screen. Left: fast photodiode trace on an oscilloscope switching between two levels. Right: the analyzer trace showing a single peak and the power meter showing a calm number. The oscilloscope is not connected in the inspection procedure; it is shown only to the viewer.', draw: function (ctx, w, h, u) { dTrace(ctx, w, h, u, { label: 'Fast intensity (blue) versus slow power meter (amber)', x: 'time', y: 'signal', fs: [telegraph, function () { return 0.58; }] }); } },
        { t: 15, cast: ['Gary'], caption: 'Some runs show a second peak. Gary re-runs until one looks clean, keeps that one, and the laser passes.', shot: 'Close on the inspection sheet: "Run 9 of 9: PASS" written in, runs 1 to 8 not recorded. A stack of printouts goes into the recycling bin. Red stamp.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: 'Run 9 of 9: PASS', sub: 'Runs 1 to 8: not attached', people: ['Gary'], stamp: 'VIOLATION' }); } },
        { t: 22, cast: ['Gary', 'Morgan'], caption: 'Two weeks later, the customer\'s coherent receiver finds the hops in the first hour.', shot: 'Conference room, a customer complaint email on the wall screen with a fast intensity trace attached. Morgan and Gary seen from behind.', draw: function (ctx, w, h, u) { dTrace(ctx, w, h, u, { label: 'Customer measurement: lasing frequency versus time', x: 'time', y: 'frequency', f: function (x) { return Math.sin(x * 23) > 0.4 ? 0.7 : 0.4; }, stamp: 'VIOLATION' }); } },
        { t: 28, cast: ['Pat Reynolds'], caption: 'Replay: a fast photodiode, a resolution bandwidth narrower than the mode spacing, sweeps in both directions, and every run kept.', shot: 'Same station, now with a fast photodiode and oscilloscope wired into the procedure, the analyzer set to a narrow resolution, and nine printouts stapled to the sheet. Green check stamp.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: 'Fast detector. Enough resolution. Every run kept.', sub: 'Then decide', people: ['Pat Reynolds'], stamp: 'COMPLIANT', stampColor: 'ok' }); } }
      ],
      narration: [DISCLAIM, 'Outgoing inspection. Gary screens lasers with a swept optical spectrum analyzer, and a power meter with a long averaging time.', 'One laser hops back and forth between two adjacent modes.', 'The analyzer can resolve the two modes. But it sweeps. It looks at each wavelength at a different moment, so a mode only appears if the laser is in it when the sweep passes that wavelength.', 'The power meter averages the two intensity levels into one calm number.', 'So most sweeps show one peak, some show two, and some show one peak that is lower than it should be.', 'Gary re-runs until one run looks clean. He keeps that one. The laser passes.', 'Two weeks later, the customer finds the hops in the first hour.', 'Let us replay that. A fast photodiode. A resolution bandwidth narrower than the mode spacing. Sweeps in both directions. Every run kept.', 'Isolate, angle, and verify. Verify is the part with the instruments.'] },
    { type: 'chat', title: 'Incident 6: How It Unfolded', app: 'thread', meta: { header: '#outgoing-inspection', participants: [GARY, BY, MGR] },
      messages: [
        { from: GARY, text: 'Last laser in the tray passed on run 9', side: 'right' },
        { from: BY, text: 'What happened on runs 1 to 8', side: 'left' },
        { from: GARY, text: 'Some had a second peak. Analyzer was probably warming up', side: 'right' },
        { from: BY, text: 'Runs 1 to 8 are the data. Run 9 is the run where the hop happened while the analyzer was looking somewhere else', side: 'left' },
        { from: MGR, text: 'We need the yield this quarter. A pass is a pass', side: 'left' }
      ], narration: isThisOk('Gary kept only the run that looked clean, and the laser passed.') },
    felt('The customer found the hops in the first hour, and attached a fast photodiode trace to the complaint',
      ['The customer\'s trace was very clear'],
      ['The customer\'s trace was very clear. Clarity is a gift, even when it arrives as a complaint.', 'Gary\'s inspection procedure will now include a fast photodiode. Gary will now include all runs.', 'For the record, measurement time traded for averaging is Quid Pro Quo Feedback. And the hops the customer found are Retaliation, delivered late.']),
    tester('You are Gary. The laser passed on run 9 and is packed for shipping. Several of runs 1 to 8 show a second peak. What would you do?', [
      OK('Hold the shipment, attach every run, re-test with a fast photodiode and a resolution bandwidth narrower than the mode spacing, log it, and tell the laser owner'),
      W('Blame the optical spectrum analyzer for showing a second peak on some sweeps', 'The analyzer did exactly what it was set to do. At each point of its sweep, it reports the power within its resolution bandwidth at that moment. If the laser hops during the sweep, some sweeps catch both modes and some catch one. The second peak on some runs is the analyzer working. A clean run is the run where the hop happened while it was looking elsewhere.'),
      W('Trust the power meter, which was steady on every run', 'The power meter did exactly what it was set to do. It averaged the power over its averaging time. A laser switching between two intensity levels faster than that gives a steady average. A steady average proves the average is steady. Use a fast photodiode to see what the laser is doing.'),
      W('Say the analyzer was warming up during runs 1 to 8', 'An instrument warming up drifts slowly in calibration. It does not create a second laser line one mode spacing away and then remove it on the next sweep. And if anything was warming up, it was the laser, which moves its modes and its filter at different rates, and that is exactly how a laser walks into a hop point.'),
      W('Widen the resolution bandwidth so the two peaks merge, then run it once', 'With a resolution bandwidth wider than the mode spacing, two modes appear as one peak. That is not a laser with one mode. That is an instrument told not to tell you. Set the resolution narrower than the mode spacing, so the analyzer can show what is there.'),
      W('Blame the calibration from the previous shift', 'A calibration error shifts or scales a reading. It does not make a second peak appear on some sweeps and not on others. The previous shift calibrated the instrument. The laser hopped.'),
      W('Re-run until the spectrum looks clean, since a clean run proves the laser can be single-mode', 'A clean run proves the laser was single-mode during that sweep. The other runs prove it was not single-mode during theirs. Keeping only the clean run is selecting the answer, not measuring it. The customer\'s receiver will see every run.')
    ], 'Hold it, keep every run, and measure fast.'),
    bystander('You are Alex, the second signature on the inspection sheet. Only run 9 is attached. What do you do?', [
      OK('Do not sign yet. Use the approved phrase, ask for all runs, and log the request'),
      WB('Sign. Gary is the one who ran it', 'The second signature exists so that someone other than the tester looks at the data. If you sign without looking, the sheet has two signatures and one check. Your signature is a measurement too.'),
      WB('Sign, and mention it at lunch', 'Once you sign, the laser ships. Lunch does not recall shipments. Ask before you sign.'),
      WB('Sign with a very small question mark next to your name', 'A very small question mark is below the resolution bandwidth of everyone who reads the sheet. Ask the question at full size.'),
      WB('Refuse to sign, and do not say why', 'Refusing is a start, but without a reason, nobody knows what to fix, and it looks like a personal conflict instead of a measurement concern. Say what you need: all runs, and a fast detector.')
    ], 'Ask before you sign.')
  ];

  /* insert each incident review before the chapter takeaways */
  function addIncident(ch, arr) {
    var i = 0; while (i < ch.steps.length && ch.steps[i].type !== 'takeaways') i++;
    Array.prototype.splice.apply(ch.steps, [i, 0].concat(arr));
  }
  addIncident(ch1, INC1); addIncident(ch2, INC2); addIncident(ch3, INC3);
  addIncident(ch4, INC4); addIncident(ch5, INC5); addIncident(ch6, INC6);

  /* =================================================================== */
  /* CHAPTER 3 (new): Classifying Unsolicited Feedback                      */
  /* Parody of harassment-training taxonomies. Physics per category:        */
  /* Marinins 2022, Bao 2024, Fresnel with Malitson / Li indices,           */
  /* De Koninck 2025, Saleh and Teich (Faraday rotation is wavelength       */
  /* dependent), Lapointe 2025 (example figures), Tkach 1986, Lenstra 1985, */
  /* N reflectors -> N (N - 1) / 2 two-mirror cavities.                     */
  /* =================================================================== */
  var CAT = {
    domestic: { name: 'Intracavity (Domestic) Feedback', def: 'unsolicited feedback from a reflector inside your own chip or package, such as a facet, an on-chip grating coupler, a waveguide crossing, or backscatter in a ring' },
    external: { name: 'External Feedback', def: 'unsolicited feedback from a reflector outside your package, on the optical path you connected: connectors, patch cords, fiber ends, lenses, detectors, or the device under test' },
    instrument: { name: 'Instrument-Originated Feedback', def: 'unsolicited feedback from the measurement equipment itself: an instrument input, a detector window, a power meter head, or a measurement path with no isolator' },
    thirdparty: { name: 'Third-Party Feedback', def: 'unsolicited feedback enabled by a supplier or vendor part that does not do what you assumed: an isolator specified for another band, flat-polished patch cords where angled ones were needed, or a pigtailed laser with no internal isolator' },
    intragroup: { name: 'Intra-Group Feedback', def: 'unsolicited feedback from resources shared inside your own group: the drawer of unlabeled patch cords, the setup the previous student left, an open port on a shared splitter' },
    crossinst: { name: 'Cross-Institutional Feedback', def: 'unsolicited feedback that arrives through a collaboration: a chip, module or measurement from another institution whose reflections nobody specified or wrote down' },
    qpq: { name: 'Quid Pro Quo Feedback', def: 'unsolicited feedback accepted in exchange for something else: isolation traded for insertion loss, an angled facet traded for an easier alignment, or measurement time traded for averaging' },
    hostile: { name: 'Hostile Optical Environment', def: 'a setup in which reflections are so widespread that no single reflector is to blame: flat connectors, flat-cleaved fibers, uncoated optics and open ports throughout' },
    retaliation: { name: 'Retaliation', def: 'the response of a laser to unsolicited feedback: linewidth narrowing or broadening, mode hops, and coherence collapse' }
  };
  var CAT_ORDER = ['domestic', 'external', 'instrument', 'thirdparty', 'intragroup', 'crossinst', 'qpq', 'hostile', 'retaliation'];
  function catSlide(n, key, example, signs, owner, extraNarr) {
    var c = CAT[key];
    return { type: 'slide', layout: 'policy', title: 'Category ' + n + ' of 9: ' + c.name,
      bullets: ['Definition: ' + c.def.charAt(0).toUpperCase() + c.def.slice(1)].concat(['Example: ' + example[0]]).concat(signs.map(function (s) { return 'Warning sign: ' + s[0]; })),
      callout: 'Owner: ' + owner,
      narration: ['Category ' + ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'][n - 1] + ' of nine. ' + c.name + '.', c.name + ' is defined as ' + c.def + '.', 'For example. ' + example[1]].concat(['Warning signs include the following.']).concat(signs.map(function (s) { return s[1]; })).concat(extraNarr || []).concat(['Report ' + c.name + ' to the owner: ' + owner + '.', 'Remember. ' + MANTRA]) };
  }
  function classify(scenario, right, wrongs, ask) {
    var opts = [OK(CAT[right].name)];
    wrongs.forEach(function (w) {
      opts.push({ text: CAT[w[0]].name, correct: false, why: '' + CAT[w[0]].name + ' is defined as ' + CAT[w[0]].def + '. ' + w[1] + ' Read the scenario again and ask two questions. Where is the reflector? And who put it there? An incident may belong to more than one category. For training purposes, please select the primary category.' });
    });
    return { type: 'check', kicker: 'Which category is this?', question: scenario + ' ' + (ask || 'Which category of unsolicited feedback is the source of this incident?'), options: opts,
      correctSay: 'This is ' + CAT[right].name + '.', narration: ['Classification exercise.', 'Read the scenario, and identify the primary category of unsolicited feedback.'] };
  }

  var chCat = {
    id: 'chcat', title: 'Classifying Unsolicited Feedback', subtitle: 'Nine categories, one mantra, zero tolerance',
    objectives: ['Name the nine categories of unsolicited feedback', 'Classify an incident by its primary category', 'Report each category to its owner'],
    steps: [
      objectives('Classifying Unsolicited Feedback', [
        ['Name the nine categories of unsolicited feedback recognized by Policy PHC-1550.4', 'Name the nine categories of unsolicited feedback recognized by Policy P H C fifteen fifty point four.'],
        'Classify an incident by where the reflector is and who put it there',
        'Recognize the warning signs of each category',
        'Report each category to its owner, including Retaliation'
      ]),
      { type: 'video', title: 'Cold Open: Gary\'s Bench, Classified', duration: 30,
        keyframes: [
          { t: 0, cast: ['Gary'], caption: 'Gary\'s bench. Every reflector on it belongs to a category. Gary does not know any of them.', shot: 'Optical bench, slow lateral dolly from left to right at bench height, 35 mm lens. Laser in a butterfly package, a chip on a stage, patch cords from an open drawer, a spectrum analyzer, a box with another institution\'s shipping label. Gary from behind, hands on hips.', draw: function (ctx, w, h, u) { dChip(ctx, w, h, u, { refl: [0.15, 0.35, 0.55, 0.75, 0.97], gary: true, label: 'Five reflectors, five owners' }); } },
          { t: 8, cast: [], caption: 'Category labels appear one by one: Intracavity. External. Instrument-Originated. Third-Party. Intra-Group. Cross-Institutional.', shot: 'Same dolly, repeated. As the camera passes each item, a printed policy tag on a lanyard drops into frame beside it: on the chip, on the patch cord, on the analyzer input, on an isolator box, on the drawer, on the shipping box. Tags sway gently.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: 'Intracavity. External. Instrument. Third-Party.', sub: 'Intra-Group. Cross-Institutional.' }); } },
          { t: 16, cast: ['Gary', 'Morgan'], caption: 'Morgan offers a trade: take out the isolator, get the insertion loss back. Quid Pro Quo.', shot: 'Two-shot from behind. Morgan holds out a small cylindrical isolator in one hand and a printed link budget in the other, as if weighing them. A tag reading Quid Pro Quo drops between them.', draw: function (ctx, w, h, u) { dCard(ctx, w, h, u, { title: 'Quid Pro Quo', sub: 'Isolation, in exchange for insertion loss', people: ['Gary', 'Morgan'], stamp: 'VIOLATION' }); } },
          { t: 23, cast: ['Pat Reynolds'], caption: 'When nothing is angled and everything reflects: a Hostile Optical Environment. What the laser does next: Retaliation.', shot: 'Crane up to a top-down view of the whole bench. Every flat surface flashes in turn; arcs connect every pair. The laser at the left edge glows red and its spectrum on the monitor broadens. Host steps into frame. Green check stamp once the arcs fade.', draw: function (ctx, w, h, u) { dChip(ctx, w, h, u, { refl: [0.12, 0.3, 0.5, 0.7, 0.88, 0.97], hotFar: true, ruler: true, label: 'Six reflectors: fifteen cavities', stamp: 'COMPLIANT', stampColor: 'ok' }); } }
        ],
        narration: ['This is Gary\'s bench.', 'Every reflector on it belongs to a category. Gary does not know any of them.', 'Intracavity. External. Instrument-Originated. Third-Party. Intra-Group. Cross-Institutional.', 'Morgan offers a trade. Take out the isolator, and get the insertion loss back. That is Quid Pro Quo.', 'When nothing is angled and everything reflects, that is a Hostile Optical Environment.', 'And what the laser does next is called Retaliation.', 'By the end of this chapter, you will be able to name all nine. You will hear them more than once. That is intentional.'] },
      { type: 'slide', layout: 'host', title: 'Why We Classify',
        bullets: ['Every reflection is a cavity, but not every cavity has the same owner', 'Classification decides who fixes it and who must be told', 'Two questions: where is the reflector, and who put it there?', 'An incident may belong to more than one category. For training purposes, select the primary category', 'Rule: classify the source, unless the question asks about the laser\'s response. The response is always Retaliation, classified separately', 'Nine categories. All mandatory'],
        callout: MANTRA,
        narration: ['Why do we classify unsolicited feedback?', 'Every reflection is a cavity. But not every cavity has the same owner.', 'Classification decides who fixes it, and who must be told.', 'To classify an incident, ask two questions. Where is the reflector? And who put it there?', 'An incident may belong to more than one category. For training purposes, please select the primary category.', 'Classify the source, unless the question asks about the laser\'s response. The response is always Retaliation, and it is classified separately.', 'There are nine categories. All nine are mandatory.', MANTRA] },

      /* 1 Intracavity */
      catSlide(1, 'domestic',
        ['A far-end silicon nitride grating coupler on the same chip returned about 4 percent (-14 dB) into flip-chip lasers (Marinins et al., 2022); facet reflections made an amplifier lase (Bao et al., 2024)', 'A far-end grating coupler on the same chip returned about four percent of the light into flip-chip bonded lasers, as Marinins and coworkers found. And facet reflections made an erbium-doped amplifier lase, as Bao and coworkers reported.'],
        [['The symptom survives every change outside the package', 'The symptom survives every change you make outside the package.'], ['The ripple period implies a length that fits on your chip, L = c / (2 n FSR)', 'The ripple period implies a length that fits on your own chip.'], ['Every unit of the same design shows it', 'Every unit of the same design shows it. It is not personal. It is in the layout.']],
        'the chip or package designer'),
      { type: 'chat', title: 'Category 1 in the Workplace', app: 'phone', meta: { header: 'Messages', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'Replaced every patch cord in the lab. The L-I ripple is still there', side: 'right' },
          { from: BY, text: 'Did the period change?', side: 'left' },
          { from: GARY, text: 'Not at all. Same on all six units', side: 'right' },
          { from: BY, text: 'Then it lives on the chip. Measure the distance to every reflector on the layout', side: 'left' }
        ], narration: isThisOk('Gary replaced every patch cord in the lab to fix a ripple that lives on his chip.') },
      classify('A laser flip-chip bonded on a silicon photonics chip shows a light versus current ripple. It does not change when any fiber or instrument is swapped, and it appears on every unit of the design. The ripple period matches the distance from the laser to an on-chip grating coupler.', 'domestic', [
        ['external', 'Swapping every fiber and instrument did not change the ripple, so the reflector is not on the path you connected.'],
        ['intragroup', 'Nothing shared by the group was involved. Every unit of the design shows it, including units nobody in the group has touched.'],
        ['hostile', 'One reflector, at a length that fits on the chip, explains the ripple. A Hostile Optical Environment has reflectors everywhere and many ripple periods at once.']
      ]),

      /* 2 External */
      catSlide(2, 'external',
        ['A flat end of fused silica, n = 1.444 (Malitson, 1965), reflects about 3.3 percent (-14.8 dB) back along the fiber', 'A flat end of fused silica, with an index of about one point four four four, reflects about three point three percent of the light back along the fiber. A lensed fiber, a lens, a detector, and the device under test all have surfaces.'],
        [['The symptom changes when you move, bend or disconnect a fiber', 'The symptom changes when you move, bend, or disconnect a fiber.'], ['The implied length matches a fiber or free-space distance on your bench', 'The implied cavity length matches a fiber or free-space distance on your bench.'], ['It improves with index-matching gel, an angled connector, or an isolator', 'It improves with index-matching gel, an angled connector, or an isolator in the right direction.']],
        'the owner of the setup'),
      { type: 'chat', title: 'Category 2 in the Workplace', app: 'thread', meta: { header: '#test-bench-2', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'The laser power jumps every time I move the lensed fiber in front of the device under test', side: 'right' },
          { from: BY, text: 'The device facet is a mirror too. You are building a cavity with your alignment stage', side: 'left' },
          { from: GARY, text: 'I am aligning, not building', side: 'right' },
          { from: BY, text: 'The light does not know the difference', side: 'left' }
        ], narration: isThisOk('Gary does not consider the device under test a reflector.') },
      classify('The output of a fiber-coupled laser jumps whenever Gary moves a lensed fiber in front of the facet of the device under test. The laser has no isolator.', 'external', [
        ['domestic', 'The reflector here is the facet of the device under test and the lensed fiber, both outside the laser package. Intracavity Feedback lives inside your own chip or package.'],
        ['instrument', 'The lensed fiber and the device under test are not measurement instruments. They are the optical path you connected.'],
        ['retaliation', 'The power jumps are the laser\'s response, and yes, that response is Retaliation. But this question asks for the source. The rule: classify the source, unless the question asks about the laser\'s response. Retaliation is classified separately, and both are reported.']
      ]),
      tester('You are the tester. A ripple on your laser does not change when you swap fibers, connectors or instruments, and it appears on every unit. What would you do?', [
        OK('Classify it as Intracavity Feedback, compute the implied length from the period, find the matching on-chip reflector, and report it to the chip design owner'),
        W('Keep replacing patch cords until it goes away', 'You have already replaced the patch cords, and the period did not change. Replacing them again tests the same hypothesis a second time. The period is a ruler: length equals the speed of light over two times the group index times the free spectral range. Use it to find the reflector on the chip.'),
        W('Blame the patch cords from the shared drawer anyway, since they are unlabeled', 'The shared drawer deserves suspicion in general, and it has its own category. But in this incident the symptom survived every swap. Suspicion is not evidence. The period is evidence.'),
        W('Classify it as External Feedback, so it becomes the setup owner\'s problem', 'Classification is not a way to assign a problem to someone else. It is a way to assign a problem to the person who can fix it. A reflector on the chip can only be fixed in the layout.'),
        W('Blame the layout simulation, which showed no reflections', 'The layout simulation did exactly what it was told. If each component was simulated with matched ports or absorbing boundaries, no component could reflect into another, so no parasitic cavity could appear. The chip, unfortunately, was not built with matched ports. Simulate the reflection of each component, and the cavity between them.')
      ], 'Intracavity Feedback goes to the chip design owner.'),

      /* 3 Instrument-Originated */
      catSlide(3, 'instrument',
        ['Without an isolator in the measurement path, parasitic reflections lead to an overestimate of laser linewidth (De Koninck et al., 2025). A flat detector window or instrument input reflects a few percent per surface (Fresnel)', 'Without an isolator in the measurement path, parasitic reflections lead to an overestimate of the laser linewidth, as De Koninck and coworkers state. A flat, uncoated detector window, or an instrument input, reflects a few percent at each surface.'],
        [['The result depends on which instrument is connected', 'The result depends on which instrument is connected.'], ['The symptom appears only during measurement', 'The symptom appears only while you are measuring. Which is, unfortunately, the only time you look.'], ['A tunable source misbehaves only with one device under test connected. Your source is a laser too', 'A tunable source misbehaves only when one particular device under test is connected. Your source is a laser too, and the device under test reflects into it.']],
        'the owner of the measurement setup, and the instrument custodian'),
      { type: 'chat', title: 'Category 3 in the Workplace', app: 'email', meta: { header: 'Inbox', subject: 'Linewidth depends on which setup we use', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'Same laser, two linewidth setups, two different answers. One of the setups must be broken.', side: 'right' },
          { from: BY, text: 'Does one of them have an isolator at the input and the other not?', side: 'left' },
          { from: GARY, text: 'Yes. The broken one has no isolator.', side: 'right' },
          { from: BY, text: 'Then it is not broken. It is measuring the laser plus its own reflections.', side: 'left' }
        ], narration: isThisOk('Gary believes a setup with no input isolator is broken, rather than participating.') },
      classify('The same laser reads a broader linewidth on the setup without an input isolator than on the setup with one. Nothing else differs.', 'instrument', [
        ['thirdparty', 'No supplier part failed to meet an assumption. The difference is an isolator that the measurement setup does not have.'],
        ['external', 'The reflections do come from outside the laser, but the defining feature is that they come from the measurement equipment itself, and they appear only while measuring.'],
        ['hostile', 'One identifiable difference, an isolator, explains the result. A Hostile Optical Environment has no single reflector to blame.']
      ]),

      /* 4 Third-Party */
      catSlide(4, 'thirdparty',
        ['The Faraday rotation angle depends on wavelength (Saleh and Teich, 1991), so an isolator built for one band does not rotate by 45 degrees in another, and its isolation falls. A pigtailed laser may ship with no internal isolator', 'The Faraday rotation angle depends on wavelength. An isolator built for one band does not rotate by exactly forty-five degrees in another band, so its isolation falls. And a pigtailed laser may ship with no internal isolator at all.'],
        [['The part number was not checked against your wavelength band', 'The part number was not checked against your wavelength band.'], ['Nobody has read the line on the datasheet about isolation', 'Nobody has read the line on the datasheet about isolation. Or the footnote.'], ['Patch cords arrived with flat end faces where angle-polished ones were ordered', 'Patch cords arrived with flat end faces, where angle-polished ones were ordered.']],
        'whoever specified and received the part, and then the supplier',
        ['Please note. The supplier did not install the part. Third-Party Feedback is enabled by the supplier and completed by us.']),
      { type: 'chat', title: 'Category 4 in the Workplace', app: 'thread', meta: { header: '#purchasing', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Isolators arrived. Installed one. Feedback is still there', side: 'right' },
          { from: BY, text: 'What band is it for', side: 'left' },
          { from: GARY, text: 'The label says 1310. We run at 1550. An isolator is an isolator', side: 'right' },
          { from: BY, text: 'The Faraday rotation depends on wavelength. At 1550 it is not 45 degrees', side: 'left' },
          { from: MGR, text: 'They were cheaper', side: 'left' }
        ], narration: isThisOk('Gary installed an isolator built for a different wavelength band.') },
      classify('An isolator labelled for the 1310 nm band is installed in a 1550 nm link. Backward light now reaches the laser.', 'thirdparty', [
        ['qpq', 'Nobody traded isolation for anything on purpose. The part was assumed to work at a wavelength it was not built for.'],
        ['instrument', 'The isolator is not measurement equipment. It is a purchased component on the optical path.'],
        ['domestic', 'The isolator is outside the laser package, and it is a supplier part, not your own chip.']
      ]),
      bystander('You are the bystander. While helping on Gary\'s bench, you notice the isolator in his 1550 nm link is labelled for 1310 nm. He is about to start a week-long measurement. What do you do?', [
        OK('Use the approved phrase now, show him the label, and help him log it and find an isolator for the right band'),
        WB('Say nothing. Isolators are isolators', 'Isolators are not all the same. The Faraday rotation angle depends on wavelength, so an isolator built for one band does not rotate by forty-five degrees in another, and its isolation falls. A week of measurement will record the laser plus its feedback.'),
        WB('Mention it at the end of the week, when the data is in', 'At the end of the week, the data is in, and it is a week of data with feedback. Speak up before the measurement, on the time scale of the problem.'),
        WB('Blame purchasing, in the group channel', 'Purchasing bought what was ordered, and the price is not the root cause. The part was received and installed without checking the band. Third-Party Feedback is enabled by the supplier and completed by us. Speak to the bench owner first.'),
        WB('Swap the isolator yourself while Gary is at lunch', 'The physics is right. The process is not. Gary\'s log now describes a different setup than the one on the bench. Untracked changes to an optical path are how future incidents begin.')
      ], 'Speak up before the measurement starts.'),

      /* 5 Intra-Group */
      catSlide(5, 'intragroup',
        ['An open, flat-polished port on a shared splitter reflects about 3.3 percent of what reaches it (Fresnel, n = 1.444) back through the splitter toward every laser on that bench', 'An open, flat-polished port on a shared splitter reflects about three point three percent of the light that reaches it, back through the splitter, toward every laser connected to it.'],
        [['Nobody knows who connected it', 'Nobody knows who connected it.'], ['The setup "has always been like that"', 'The setup has always been like that.'], ['Patch cords in the shared drawer have no labels, or labels in a previous student\'s handwriting', 'The patch cords in the shared drawer have no labels, or labels in the handwriting of a student who graduated.']],
        'the group lab manager, and every member of the group'),
      { type: 'chat', title: 'Category 5 in the Workplace', app: 'thread', meta: { header: '#group-lab', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Who owns the 1x4 splitter on bench 3? Two ports are open', side: 'right' },
          { from: BY, text: 'The previous student set it up. It has always been like that', side: 'left' },
          { from: GARY, text: 'Every laser on bench 3 is a little noisy', side: 'right' },
          { from: MGR, text: 'Bench 3 has always been a little noisy', side: 'left' }
        ], narration: isThisOk('Two open ports on a shared splitter have been reflecting into every laser on the bench, and everyone has adapted.') },
      classify('A shared splitter on the group\'s test bench has two ports left open with flat connectors. It was set up by a student who has since graduated. Every laser on that bench is a little noisier than elsewhere.', 'intragroup', [
        ['external', 'The reflectors are outside each laser package, yes. But the defining feature is that the resource is shared inside the group and nobody owns it. Primary category, please.'],
        ['hostile', 'Two identifiable open ports explain it. A Hostile Optical Environment has reflectors everywhere, with no single one to blame.'],
        ['crossinst', 'The student was in your own group. Graduation does not make someone another institution, although it sometimes feels that way.']
      ]),

      /* 6 Cross-Institutional */
      catSlide(6, 'crossinst',
        ['A partner\'s chip arrives with flat, uncoated facets although the email said they were angled. A cleaved silicon facet, n = 3.48 (Li, 1980), reflects about 31 percent as a plane-wave estimate', 'A partner\'s chip arrives with flat, uncoated facets, although the email said they would be angled. A cleaved silicon facet reflects about thirty-one percent, as a plane-wave estimate.'],
        [['The specification is an email that says "should be fine"', 'The specification is an email that says, should be fine.'], ['Nobody wrote down the return loss', 'Nobody wrote down the return loss.'], ['"The other group already measured it"', 'The other group already measured it. With their setup. Which you have not seen.']],
        'both principal investigators, in writing'),
      { type: 'chat', title: 'Category 6 in the Workplace', app: 'email', meta: { header: 'Inbox', subject: 'RE: Chips shipped (angled facets as discussed)', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'Chips from the partner university arrived. Under the microscope the facets look flat and uncoated.', side: 'right' },
          { from: BY, text: 'The email said angled facets. Is that in the agreement?', side: 'left' },
          { from: GARY, text: 'The agreement says "standard facets". Nobody defined standard.', side: 'right' },
          { from: BY, text: 'Then measure the return loss, write it down, and send it to both sides.', side: 'left' }
        ], narration: isThisOk('A collaboration specified "standard facets" and nobody defined the return loss.') },
      classify('Chips from a partner university arrive with flat, uncoated facets, although an email promised angled facets. The collaboration agreement never mentions return loss.', 'crossinst', [
        ['thirdparty', 'A partner university is not a supplier with a datasheet. The defining feature is a collaboration in which the reflections were never specified.'],
        ['domestic', 'The facets are on a chip from another institution. It will become part of your package, but the specification gap is between institutions.'],
        ['qpq', 'Nobody offered a trade. The angled facets were promised and simply did not arrive.']
      ]),
      tester('You are the tester. The partner\'s chips have flat, uncoated facets, and your laser is edge-coupled to them with no isolator. Your partner\'s student says "should be fine". What would you do?', [
        OK('Measure the return loss, log it, tell both principal investigators in writing, add an isolator or angled interface, and put return loss into the collaboration specification'),
        W('Say nothing. It is their chip', 'It is their chip, and it is your laser. A cleaved silicon facet reflects about thirty-one percent as a plane-wave estimate, and that light enters your laser if nothing stops it. Whose chip it is does not change where the light goes.'),
        W('Blame the dicing saw', 'A dicing saw cuts where and how it was told. If the facets were supposed to be angled, the angle had to be specified in the process, by someone. The tool did what it was told. Nobody told it about the angle.'),
        W('Publish anyway, with the phrase "facets as received"', '"As received" describes the facets. It does not describe their reflection, or what it did to your laser. Readers cannot correct for a number you did not measure.'),
        W('Angle-polish the facets yourself and do not tell the partner', 'The physics improves. The collaboration does not. The partner\'s records now describe a different chip than the one you measured, and the next shipment will arrive flat again.'),
        W('Trust the other group\'s measurement', 'The other group measured with their setup, at their wavelength, with their isolation. Return loss is a property of the interface in your setup. Measure it where it matters.')
      ], 'Measure it, write it down, and tell both sides.'),

      /* 7 Quid Pro Quo */
      catSlide(7, 'qpq',
        ['Figures reported for one integrated isolator: more than 25 dB isolation, less than 1.5 dB insertion loss (Lapointe et al., 2025). Removing such an isolator wins back at most 1.5 dB and gives up at least 25 dB of protection', 'Here are example figures for one integrated isolator: more than twenty-five decibels of isolation, and less than one and a half decibels of insertion loss, as reported by Lapointe and coworkers. Removing such an isolator wins back at most one and a half decibels, and gives up at least twenty-five.'],
        [['Any sentence that begins "If you just remove the isolator"', 'Any sentence that begins, if you just remove the isolator.'], ['"We will put it back after the demo"', 'We will put it back after the demo.'], ['Averaging time increased to make a measurement faster to pass', 'Averaging time increased to make a measurement easier to pass. Averaging is time traded for truth.']],
        'the person who proposed the trade, and their manager',
        ['Please note. A Quid Pro Quo offer does not need to be explicit. A link budget that only closes without the isolator is an offer.']),
      { type: 'chat', title: 'Category 7 in the Workplace', app: 'phone', meta: { header: 'Messages', participants: [MGR, GARY] },
        messages: [
          { from: MGR, text: 'If you take the isolator out you get the insertion loss back and we close the link budget', side: 'left' },
          { from: GARY, text: 'And the reflections?', side: 'right' },
          { from: MGR, text: 'We will put it back after the demo', side: 'left' },
          { from: MGR, text: 'Also bump the averaging on the noise test, it passes faster that way', side: 'left' }
        ], narration: isThisOk('Morgan offered insertion loss in exchange for isolation, and a passing result in exchange for averaging.') },
      classify('Morgan tells Gary that if he removes the isolator, the link budget closes, and that the isolator can go back in after the demo.', 'qpq', [
        ['retaliation', 'Retaliation is the laser\'s response, and it may well follow. But the defining feature here is the offer: isolation in exchange for insertion loss.'],
        ['intragroup', 'Morgan is in Gary\'s group, yes. But the defining feature is not a shared resource. It is a trade.'],
        ['thirdparty', 'The isolator works as specified. Nobody is proposing to replace it with a worse part. They are proposing to remove it.']
      ]),

      /* 8 Hostile Optical Environment */
      catSlide(8, 'hostile',
        ['Every pair of reflectors on a path is a cavity: N reflectors make N (N - 1) / 2 of them. Six flat surfaces make fifteen cavities, each with its own period c / (2 n L)', 'Every pair of reflectors on a path is a cavity. N reflectors make N times N minus one over two of them. Six flat surfaces make fifteen cavities, each with its own ripple period.'],
        [['Ripples at several periods at once', 'Ripples at several periods at once.'], ['The symptom moves when anything on the bench is touched', 'The symptom moves when anything on the bench is touched. Including the bench.'], ['Nothing on the bench is angled, coated, gelled or isolated', 'Nothing on the bench is angled, coated, gelled, or isolated.']],
        'everyone who uses the bench, coordinated by the lab manager',
        ['A Hostile Optical Environment is not one bad reflector. It is a culture.']),
      { type: 'chat', title: 'Category 8 in the Workplace', app: 'call', meta: { header: 'Video call', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'I see ripples at five different periods. Which one is the device?', side: 'right' },
          { from: BY, text: 'Possibly none. Flat cleaves, flat connectors, an uncoated lens, a flat detector window and an open splitter port. That is a lot of pairs', side: 'left' },
          { from: MGR, text: 'Every bench is like that', side: 'left' },
          { from: BY, text: 'That is the problem', side: 'left' }
        ], narration: isThisOk('Gary is working on a bench where every surface is flat and nothing is angled, and his manager says every bench is like that.') },
      classify('On Gary\'s bench: flat-cleaved fibers, flat connectors, an uncoated lens, a flat detector window and an open splitter port. The spectrum shows ripples at several periods at once, and they all move when someone leans on the table.', 'hostile', [
        ['external', 'Each reflector is External, yes. But with reflectors everywhere and ripples at several periods, there is no single reflector to blame. That is the defining feature of another category.'],
        ['intragroup', 'Some of these may be shared resources. But the defining feature is that everything reflects, and no single reflector explains it.'],
        ['instrument', 'One of the reflectors is a detector window. One. There are at least five.']
      ]),
      bystander('You are the bystander. You have just joined a bench where every surface is flat and nothing is angled. Everyone has adapted. What do you do?', [
        OK('Use the approved phrase with the lab manager, list the reflectors, and propose fixing them one at a time with the bench owners'),
        WB('Adapt. Everyone else has', 'Adapting to a Hostile Optical Environment means every measurement on the bench carries a mix of cavities, and nobody can tell which ripple belongs to which device. Adaptation is not mitigation.'),
        WB('Work only at night, when the bench is quieter', 'Reflections do not keep office hours. A flat connector reflects the same three point three percent at midnight. You may see less vibration at night, but vibration only moves the phase of reflections that are already there.'),
        WB('Increase the averaging until the ripples go away', 'Averaging can hide a ripple finer than your resolution, or a fluctuation faster than your averaging time. It does not remove a single reflector. It removes your ability to see them.'),
        WB('Quietly move to another bench', 'Moving protects your measurement and leaves the bench as it is for the next person. Speak up, so that the bench gets fixed, not just vacated.')
      ], 'Name it, list it, and fix it together.'),

      /* 9 Retaliation */
      catSlide(9, 'retaliation',
        ['Depending on strength and distance, feedback narrows or broadens the line, causes hops between external-cavity modes, or drives coherence collapse (Tkach and Chraplyvy, 1986; Lenstra et al., 1985)', 'Depending on the strength of the feedback and the distance to the reflector, the laser narrows or broadens its line, hops between external cavity modes, or falls into coherence collapse, as classified by Tkach and Chraplyvy, and described by Lenstra and coworkers.'],
        [['Mode hops after a change in the optical path', 'Mode hops after a change in the optical path.'], ['A periodic ripple in the light versus current curve', 'A periodic ripple in the light versus current curve.'], ['A dramatically broadened line that does not improve when you move the reflector', 'A dramatically broadened line that does not improve when you move the reflector.']],
        'the owner of the reflector that caused it. Never the laser',
        ['Retaliation is classified separately from the feedback that caused it. Both are reported.', 'Zero tolerance for Retaliation means zero tolerance for the conditions that cause it. We do not punish the laser.']),
      { type: 'chat', title: 'Category 9 in the Workplace', app: 'thread', meta: { header: '#all-hands', participants: [MGR, GARY, BY] },
        messages: [
          { from: MGR, text: 'The laser on bench 2 retaliated again. Please replace it', side: 'left' },
          { from: BY, text: 'It is responding to feedback. A new laser on the same path will respond the same way', side: 'left' },
          { from: GARY, text: 'Should I file the replacement under Retaliation?', side: 'right' },
          { from: BY, text: 'File it under the category of the reflector. Then file the Retaliation as well', side: 'left' }
        ], narration: isThisOk('Morgan wants to replace a laser for retaliating, instead of removing the feedback it is responding to.') },
      classify('After a flat connector is added close to the laser, the line broadens dramatically and the laser becomes noisy. Moving the connector a little does not help.', 'retaliation', [
        ['external', 'The connector is indeed External Feedback, and it is reported as the source. But this question asks about the laser\'s response, not the source: a dramatic broadening that does not improve when the reflector moves. Responses have their own category.'],
        ['hostile', 'One connector was added. A Hostile Optical Environment has reflectors everywhere. And in any case, this question asks about the laser\'s response, not the source.'],
        ['qpq', 'Nobody traded anything. And this question asks about the laser\'s response, not about how the reflector got there.']
      ], 'Which category describes the laser\'s response?'),
      tester('You are the tester. Your laser\'s line has broadened dramatically since a connector was added nearby, and Morgan says to replace the laser because it keeps retaliating. What would you do?', [
        OK('Treat the laser\'s response as a symptom: find and remove the feedback, log both the reflector and the Retaliation, and keep the laser'),
        W('Replace the laser, since it retaliated', 'The new laser will sit on the same optical path, in front of the same connector, and it will respond in the same way. Retaliation is the response, not the cause. Zero tolerance for Retaliation means zero tolerance for the conditions that cause it. We do not punish the laser.'),
        W('Raise the current to overpower the reflection', 'Feedback effects depend on the fraction of light that returns, not on the absolute power. More current sends more light to the connector, and more comes back. The ratio is the same.'),
        W('Move the connector a little further away', 'In the regime called coherence collapse, the broadening is largely insensitive to the distance to the reflector, as Tkach and Chraplyvy found. A small move will not change the regime. Remove, angle, or isolate the reflector.'),
        W('Report the laser to Human Resources', 'Human Resources does not have jurisdiction over lasers. Report the reflector to its owner, and the Retaliation to the setup log.'),
        W('Blame the laser driver for noisy current', 'A noisy current source can add noise to a laser, and it is worth checking. But the line broadened when the connector was added, and nothing changed in the driver. The driver did exactly what it did yesterday. The optical path is what changed.')
      ], 'Remove the cause. Keep the laser. Log both.'),

      { type: 'takeaways', title: 'Know Your Categories',
        items: CAT_ORDER.map(function (k) { return CAT[k].name + ': ' + CAT[k].def; }).concat(['Ask two questions: where is the reflector, and who put it there?', MANTRA]),
        narration: ['Let us review. Know your categories.'].concat(CAT_ORDER.map(function (k, i) { return 'Category ' + ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'][i] + '. ' + CAT[k].name + '.'; })).concat(['To classify an incident, ask two questions. Where is the reflector? And who put it there?', 'And one more time, together. ' + MANTRA, 'Click Next to continue.']) },
      ack('I can name the nine categories of unsolicited feedback, and I will classify, report, and log each incident by its primary category.'),
      refs('marinins', 'bao', 'malitson', 'li1980', 'dekoninck', 'saleh', 'lapointe', 'tkach', 'lenstra')
    ]
  };

  /* =================================================================== */
  /* FINAL EXAM POOL (22)                                                  */
  /* =================================================================== */
  function q(question, right, wrongs, say, spokenQ) {
    var opts = [{ text: right, correct: true, why: '' }];
    wrongs.forEach(function (w) { opts.push({ text: w[0], correct: false, why: w[1] }); });
    return { type: 'check', question: question, options: opts, correctSay: say || '', narration: [spokenQ || question] };
  }
  var pool = [
    q('Which of the following best describes optical feedback?', 'Light that re-enters a laser cavity after leaving it, adding a delayed field to the cavity field',
      [['Verbal comments about a laser\'s performance', 'Verbal comments are welcome under our speak-up culture, but they are not optical feedback. Optical feedback is light that re-enters the laser cavity after leaving it. It adds a delayed field with its own phase, and that changes the laser.'],
       ['Electrical feedback from the current driver only', 'Electrical control loops are a different topic. Optical feedback is light itself returning into the laser cavity, from facets, couplers, fiber tips, or any other interface.']]),
    q('Flip-chip InP lasers on a silicon photonics chip, with silicon nitride waveguides and grating couplers, show a light versus current ripple. In the published case, what was the dominant reflector?', 'A grating coupler at the far end of the waveguide, returning about 4 percent',
      [['The sub-micron gap between the laser and the chip', 'The gap was the obvious suspect because it is the closest interface. Marinins and coworkers traced the ripple to about four percent, minus fourteen decibels, of feedback from a silicon nitride grating coupler at the far end of the waveguide.'],
       ['The laser\'s own grating', 'The laser\'s own grating is designed feedback inside the cavity. The ripple came from external feedback: a far-end grating coupler returning about four percent.']], null, 'Flip-chip indium phosphide lasers on a silicon photonics chip, with silicon nitride waveguides and grating couplers, show a light versus current ripple. In the published case, what was the dominant reflector?'),
    q('What is the free spectral range of a parasitic cavity between two reflectors spaced L apart, with group index n?', 'c / (2 n L)',
      [['c / (n L)', 'The light has to make a round trip between the two reflectors, so the optical path is two times n times L. The free spectral range is c divided by two n L.'],
       ['2 n L / c', 'Two n L divided by c is the round-trip time, in seconds. The free spectral range is its inverse: c divided by two n L.']]),
    q('Two reflectors have power reflectances R1 and R2. What sets the depth of the resulting ripple?', 'The product R1 times R2, through r = sqrt(R1 R2)',
      [['Only the larger of the two reflectances', 'Both reflectances matter. The ripple comes from interference between light that bounces between the two reflectors and light that does not, and its contrast depends on the square root of R one times R two. A strong reflector paired with a weak one still makes a weak ripple, and two weak ones still make a ripple with an exact period.'],
       ['The spacing between the reflectors', 'The spacing sets the period of the ripple, not its depth. The depth is set by the square root of R one times R two.']], null, 'Two reflectors have power reflectances R one and R two. What sets the depth of the resulting ripple?'),
    q('A high-gain waveguide amplifier begins emitting a narrow line at its fluorescence peak. What is the most likely cause?', 'Facet reflections forming a parasitic cavity, causing parasitic lasing',
      [['The amplifier has entered a new low-noise regime', 'A narrow line at the gain peak from an amplifier is a warning, not a feature. Facet reflections plus enough gain make a laser. Bao and coworkers saw parasitic lasing at fifteen thirty-one point five nanometers in an erbium-doped thin film lithium niobate amplifier.']]),
    q('Why do sweep-up and sweep-down measurements of a laser often show mode hops at different temperatures?', 'Hysteresis: the laser stays in its current mode past the point where another mode would win from a cold start',
      [['Thermometers are less accurate when cooling', 'The hop points differ systematically by direction, which is the signature of hysteresis in the laser, not an error in the thermometer. Record the sweep direction.'],
       ['Mode hops are random', 'Mode hops in a stable setup are deterministic, set by where another mode starts to win the gain competition plus a hysteresis margin.']]),
    q('A datasheet quotes a wide tuning range. What should you check before planning a continuous sweep?', 'The mode-hop-free tuning range, which can be much smaller',
      [['Nothing; tuning range means continuous tuning', 'Total tuning range is usually reached through a staircase of mode hops. Continuous, mode-hop-free tuning is bounded by about one cavity free spectral range unless the cavity phase follows the filter. Read the figure, not only the abstract.']]),
    q('What does an optical isolator have to break?', 'Lorentz reciprocity',
      [['Energy conservation', 'No device breaks energy conservation. An isolator has to break Lorentz reciprocity, which requires a magneto-optic material with a bias field or time modulation. Nonlinearity helps only within a power window, limited by dynamic reciprocity. Jalas and coworkers explained this in two thousand thirteen.'],
       ['Mode orthogonality', 'Converting between orthogonal modes does not make a device an isolator. A linear, time-invariant, non-magnetic mode converter is still reciprocal. An isolator must break Lorentz reciprocity.']]),
    q('A static silicon mode converter shows high forward and low backward transmission on a single-mode detector. What happens to a reflection of its forward output?', 'It returns to the source in the input mode, with the same transmission as the forward path',
      [['It is blocked, because the backward transmission is low', 'The low backward number was measured for a backward input in one particular mode. A reflection of the forward output comes back in the mode that left, and in a reciprocal device that light reaches the source with the same transmission as the forward path.']]),
    q('An output waveguide is tilted at the facet. How should the fiber be aligned?', 'To the refracted beam, which leaves at a larger angle than the waveguide tilt',
      [['Along the waveguide direction', 'The beam refracts at the facet. Going from a higher index to a lower index, Snell\'s law makes the exit angle larger than the internal angle. Align the fiber to the refracted beam.']]),
    q('Why might reflections from a III-V to silicon taper matter, even if its insertion loss is excellent?', 'Feedback-sensitive circuits are cited as needing taper reflections below about -50 dB, and tapers are usually optimized only for loss',
      [['They do not matter if insertion loss is low', 'Insertion loss and return loss are different quantities. Kurczveil and coworkers cite a requirement below minus fifty decibels for feedback-sensitive circuits, and measured passive tapers below minus forty-one decibels. Budget both.']], null, 'Why might reflections from a three-five to silicon taper matter, even if its insertion loss is excellent?'),
    q('A high-Q ring shows a doublet. What does a single-Lorentzian fit do?', 'It misreports Q, because the apparent width includes the splitting',
      [['It reports Q accurately and conservatively', 'The single Lorentzian width includes the splitting from backscatter coupling, so it is not a measure of loss. Fit a coupled-mode doublet model.']]),
    q('What is one published source of resonance splitting in rings written by electron-beam lithography?', 'Writing discontinuities (jumps and pattern segmentation) along the circumference',
      [['The color of the resist', 'Hausmann and coworkers found that discontinuities in electron beam writing along the ring circumference produced split resonances, and continuous writing removed them.']]),
    q('Why can a thermally locked ring suddenly snap away from its lock?', 'Past a point, self-heating flips from negative to positive feedback; the balance point must be calibrated with a hot-to-cold sweep',
      [['Thermal locks are unconditionally stable', 'Kramnik and coworkers describe thermally locked rings sitting on the edge of instability. Past a point, self-heating switches from stabilizing to destabilizing, and the resonance snaps to shorter wavelength.']]),
    q('A wavemeter shows a steady wavelength. Can you conclude the laser is not hopping?', 'No. A hop faster than the update rate can be averaged or missed; check with a fast detector',
      [['Yes. A steady wavemeter proves single-mode operation', 'Every instrument integrates. A wavemeter reports one number per update. A laser hopping back and forth faster than that can produce a steady-looking average. Check the intensity on a fast photodiode, and the spectrum with enough resolution.']])
,
    q('An inline isolator is installed with its arrow pointing at the laser. What happens?', 'The laser output is attenuated by about the isolation ratio, and light that starts downstream reaches the laser with only the insertion loss',
      [['Nothing; by reciprocity, an isolator works the same in both directions', 'An isolator is the one component that must break Lorentz reciprocity. That is its entire job. Installed backwards, it blocks the laser output and passes light that starts downstream into the laser. Direction is not angle.'],
       ['Reflections of the laser light are now blocked twice as well', 'For the laser\'s own reflection, the round trip is forward transmission times backward transmission, which is the same product in either orientation. The reversed isolator does not protect you more. It takes your output power, and lets light from downstream sources in.']], 'The arrow points away from the laser.'),
    q('A ripple in an edge-coupled transmission measurement has a 24 nanometer period at 1550 nanometers, and the period changes when the fiber gap changes. What is the most likely cause?', 'A parasitic cavity between the flat fiber end and the chip facet, about 50 micrometers apart',
      [['An intrinsic ripple of the waveguide', 'An intrinsic ripple of the waveguide would not change when only the fiber moves. In air, the period is lambda squared over two times the gap. At fifteen fifty nanometers, a twenty-four nanometer period means a gap of fifty micrometers. The ripple belongs to the gap.'],
       ['Power ripple of the tunable laser', 'Transmission is normalized to a reference measurement of the laser, and in any case the laser does not know where your fiber is. The period followed the gap.']], 'The ripple belongs to the gap.'),
    q('A laser hops during a demo. Gary power-cycles it, and it comes back on the right channel. Is the problem solved?', 'No. A restart re-selects a mode; the hop point and the feedback that caused it are still there',
      [['Yes. The laser is back on the right channel', 'A restart decides which side of the hysteresis loop the laser starts on. It does not remove the hop point, or the reflector. The next temperature drift brings the laser back to it. Log it and fix the feedback.']], 'Restart is not repair.'),
    q('A ring resonance shows a doublet, and light comes back out of the input port at the doublet. What is the most likely explanation?', 'Backscatter couples the clockwise and counter-clockwise modes into two split standing waves',
      [['Two independent transverse mode families', 'Two independent mode families would not send light back out of the input port, and because their group indices differ, their spacing would change from resonance to resonance. A doublet that stays pinned and reflects is backscatter.']], 'One mode pair, split by backscatter.'),
    q('An optical spectrum analyzer with a resolution bandwidth wider than a laser\'s mode spacing shows one peak. Can you conclude the laser is single-mode?', 'No. Two modes closer than the resolution bandwidth are displayed as one peak',
      [['Yes. One peak means one mode', 'The analyzer reports the power within its resolution bandwidth. Two modes inside one resolution bandwidth appear as one peak. Set the resolution narrower than the mode spacing, and add a fast photodiode.']], 'The instrument was told not to tell you.'),
    q('Morgan suggests removing the isolator so the link budget closes, and putting it back after the demo. Which category of unsolicited feedback is the source of this incident?', 'Quid Pro Quo Feedback',
      [['Third-Party Feedback', 'Third-Party Feedback is enabled by a supplier part that does not do what you assumed. Here the isolator works as specified. The defining feature is a trade: isolation, in exchange for insertion loss.'],
       ['Retaliation', 'Retaliation is the laser\'s response, and it may well follow. But the defining feature here is the offer.']], 'Quid Pro Quo Feedback.'),
    q('A shared splitter on the group bench has open, flat-polished ports, set up by a student who has graduated. Which category of unsolicited feedback is the source of this incident?', 'Intra-Group Feedback',
      [['Cross-Institutional Feedback', 'The student was in your own group. Graduation does not make someone another institution. A shared resource inside your own group that nobody owns is Intra-Group Feedback.'],
       ['Hostile Optical Environment', 'Two identifiable open ports explain it. A Hostile Optical Environment has reflectors everywhere, with no single one to blame.']], 'Intra-Group Feedback.')
  ];

  /* =================================================================== */
  /* WIDGETS                                                               */
  /* =================================================================== */
  var STYLE = '.fbw{color:var(--ink,#e3e6ea);font-size:13px;line-height:1.35}.fbw *{border-radius:0;box-sizing:border-box}' +
    '.fbw-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px 18px;margin:8px 0}' +
    '.fbw-ctl label{display:flex;justify-content:space-between;font-size:12px;color:var(--ink-2,#8b939c)}.fbw-ctl label b{color:var(--ink,#e3e6ea);font-weight:600}' +
    '.fbw-ctl input[type=range]{width:100%}.fbw-row{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0}' +
    '.fbw-btn{background:var(--panel,#1d2228);color:var(--ink,#e3e6ea);border:1px solid var(--line,#3a414a);padding:4px 10px;cursor:pointer;font:inherit}' +
    '.fbw-btn:hover,.fbw-btn.on{border-color:var(--accent,#4ea1ff);color:var(--accent,#4ea1ff)}' +
    '.fbw-read{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:3px 14px;font-size:12px;font-variant-numeric:tabular-nums;margin:6px 0}' +
    '.fbw-read span{color:var(--ink-2,#8b939c)}.fbw-read b{color:var(--accent,#4ea1ff);font-weight:600}' +
    '.fbw-tag{font-size:11px;color:var(--ink-2,#8b939c);letter-spacing:.05em;text-transform:uppercase;margin:4px 0}' +
    '.fbw-hint{outline:2px solid var(--accent,#4ea1ff);outline-offset:3px;animation:fbwP 1.1s ease-in-out infinite}' +
    '@keyframes fbwP{0%,100%{outline-color:var(--accent,#4ea1ff)}50%{outline-color:transparent}}' +
    '.fbw-hand{display:inline-block;margin-left:8px;font-size:11px;font-weight:600;color:var(--accent,#4ea1ff);animation:fbwB .9s ease-in-out infinite}' +
    '@keyframes fbwB{0%,100%{transform:translateX(0)}50%{transform:translateX(-6px)}}' +
    '.fbw input[type=number]{width:90px;background:var(--panel-2,#14171b);color:var(--ink,#e3e6ea);border:1px solid var(--line,#3a414a);padding:3px 5px;font:inherit}';

  function shell(root) {
    var wrap = document.createElement('div'); wrap.className = 'fbw';
    var st = document.createElement('style'); st.textContent = STYLE; wrap.appendChild(st);
    root.appendChild(wrap); return wrap;
  }
  function el(tag, cls, html, parent) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; if (parent) parent.appendChild(e); return e; }
  function slider(parent, lab, min, max, step, val, fmt, on) {
    var box = el('div', 'fbw-ctl', null, parent);
    var l = el('label', null, '<span>' + lab + '</span><b></b>', box);
    var i = el('input', null, null, box); i.type = 'range'; i.min = min; i.max = max; i.step = step; i.value = val;
    var b = l.querySelector('b'); var upd = function () { b.textContent = fmt(+i.value); };
    i.addEventListener('input', function () { upd(); on(+i.value); }); upd();
    return { input: i, box: box, set: function (v) { i.value = v; upd(); } };
  }
  function hint(target, label) {
    target.classList.add('fbw-hint');
    var h = el('span', 'fbw-hand', '← ' + (label || 'start here'));
    if (target.tagName === 'BUTTON') target.insertAdjacentElement('afterend', h); else { var lb = target.querySelector('label span'); (lb || target).appendChild(h); }
    return function () { target.classList.remove('fbw-hint'); if (h.parentNode) h.parentNode.removeChild(h); };
  }
  function canvas(parent, hpx) {
    var cv = el('canvas', null, null, parent); cv.style.width = '100%'; cv.style.height = hpx + 'px'; cv.style.display = 'block';
    return function () {
      var dpr = window.devicePixelRatio || 1, w = Math.max(260, parent.clientWidth || 600);
      cv.width = Math.round(w * dpr); cv.height = Math.round(hpx * dpr);
      var ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return { ctx: ctx, w: w, h: hpx };
    };
  }
  function ticks(a, b, n) {
    var span = b - a; if (!(span > 0)) return [a];
    var st = Math.pow(10, Math.floor(Math.log10(span / n))), e = span / n / st;
    st *= e >= 5 ? 5 : e >= 2 ? 2 : 1;
    var out = []; for (var v = Math.ceil(a / st) * st; v <= b + st * 1e-6; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : v); return out;
  }
  function fmtN(v) { var a = Math.abs(v); if (a === 0) return '0'; if (a >= 1000 || a < 0.01) return v.toExponential(1); return (+v.toPrecision(3)).toString(); }
  /* axes: returns mapping functions */
  function axes(ctx, w, h, xr, yr, xl, yl, c, pad) {
    pad = pad || { l: 58, r: 12, t: 10, b: 36 };
    var L = pad.l, Rr = w - pad.r, T = pad.t, B = h - pad.b;
    var X = function (x) { return L + (x - xr[0]) / (xr[1] - xr[0]) * (Rr - L); };
    var Y = function (y) { return B - (y - yr[0]) / (yr[1] - yr[0]) * (B - T); };
    ctx.fillStyle = c.bg; ctx.fillRect(L, T, Rr - L, B - T);
    ctx.strokeStyle = c.border; ctx.lineWidth = 1; ctx.strokeRect(L, T, Rr - L, B - T);
    ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = c.muted;
    ticks(xr[0], xr[1], 6).forEach(function (t) { var px = X(t); ctx.globalAlpha = 0.25; ctx.beginPath(); ctx.moveTo(px, T); ctx.lineTo(px, B); ctx.stroke(); ctx.globalAlpha = 1; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(fmtN(t), px, B + 4); });
    ticks(yr[0], yr[1], 4).forEach(function (t) { var py = Y(t); ctx.globalAlpha = 0.25; ctx.beginPath(); ctx.moveTo(L, py); ctx.lineTo(Rr, py); ctx.stroke(); ctx.globalAlpha = 1; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(fmtN(t), L - 4, py); });
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(xl, (L + Rr) / 2, h - 2);
    ctx.save(); ctx.translate(11, (T + B) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText(yl, 0, 0); ctx.restore();
    return { X: X, Y: Y, L: L, R: Rr, T: T, B: B };
  }
  function apiCols(api) {
    var g = function (n, fb) { try { var v = api.css(n); return (v && String(v).trim()) || fb; } catch (e) { return fb; } };
    return { bg: g('--panel-2', '#14171b'), panel: g('--panel', '#1d2228'), fg: g('--ink', '#e3e6ea'), muted: g('--ink-2', '#8b939c'), accent: g('--accent', '#4ea1ff'), bad: g('--bad', '#e5534b'), ok: g('--ok', '#3fb950'), warn: g('--warn', '#d29922'), border: g('--line', '#3a414a') };
  }

  var widgets = {};

  /* ---- (a) Parasitic Fabry-Perot: lossless Airy transmission, FSR = c/(2 n L) ---- */
  widgets.fabryPerot = function (root, params, api) {
    var CL = 299792458, LAM = 1550e-9;
    var s = { r1: -14, r2: -30, logL: Math.log10(3), ng: 4.0, span: 200 }; // L in mm (log10), span in GHz
    var touched = {}, done = false;
    var wrap = shell(root);
    el('div', 'fbw-tag', 'Lossless two-mirror etalon. T = (1 - R1)(1 - R2) / (1 + R1 R2 - 2 sqrt(R1 R2) cos(2 pi f / FSR)). Frequency offset is relative to an arbitrary transmission peak.', wrap);
    var fit = canvas(wrap, 240);
    var rd = el('div', 'fbw-read', null, wrap);
    var g = el('div', 'fbw-grid', null, wrap);
    var unh = [];
    function mark(k) { touched[k] = 1; unh.forEach(function (f) { f(); }); unh = []; if (!done && Object.keys(touched).length >= 2) { done = true; api.complete(); } }
    var sR1 = slider(g, 'Reflector 1, R1', -60, -3, 1, s.r1, function (v) { return v + ' dB'; }, function (v) { s.r1 = v; mark('r1'); draw(); });
    slider(g, 'Reflector 2, R2', -60, -3, 1, s.r2, function (v) { return v + ' dB'; }, function (v) { s.r2 = v; mark('r2'); draw(); });
    var sL = slider(g, 'Spacing L', -2, 3, 0.01, s.logL, function (v) { var mm = Math.pow(10, v); return mm < 1 ? (mm * 1000).toPrecision(3) + ' um' : mm.toPrecision(3) + ' mm'; }, function (v) { s.logL = v; mark('L'); draw(); });
    slider(g, 'Group index n', 1, 4.5, 0.01, s.ng, function (v) { return v.toFixed(2); }, function (v) { s.ng = v; mark('ng'); draw(); });
    var row = el('div', 'fbw-row', '<span class="fbw-tag">Plot span</span>', wrap);
    [20, 200, 2000, 20000].forEach(function (sp) {
      var b = el('button', 'fbw-btn' + (sp === s.span ? ' on' : ''), sp >= 1000 ? (sp / 1000) + ' THz' : sp + ' GHz', row); b.type = 'button';
      b.addEventListener('click', function () { s.span = sp; Array.prototype.forEach.call(row.querySelectorAll('button'), function (x) { x.classList.remove('on'); }); b.classList.add('on'); mark('span'); draw(); });
    });
    var inv = el('div', 'fbw-row', '<span class="fbw-tag">Ripple-to-length calculator</span> observed period <input type="number" min="0" step="any" value="100"> pm at 1550 nm, with the group index above, implies spacing <b class="fbw-invout"></b>', wrap);
    var invIn = inv.querySelector('input'), invOut = inv.querySelector('.fbw-invout');
    invOut.style.color = 'var(--accent,#4ea1ff)';
    invIn.addEventListener('input', function () { mark('inv'); draw(); });
    unh.push(hint(sR1.box, 'drag'));
    unh.push(hint(sL.box, 'then drag'));

    function draw() {
      var o = fit(), ctx = o.ctx, w = o.w, h = o.h, c = apiCols(api);
      ctx.fillStyle = c.panel; ctx.fillRect(0, 0, w, h);
      var R1 = Math.pow(10, s.r1 / 10), R2 = Math.pow(10, s.r2 / 10), r = Math.sqrt(R1 * R2);
      var L = Math.pow(10, s.logL) * 1e-3, fsr = CL / (2 * s.ng * L), fsrG = fsr / 1e9;
      var Tmax = (1 - R1) * (1 - R2) / Math.pow(1 - r, 2), Tmin = (1 - R1) * (1 - R2) / Math.pow(1 + r, 2);
      var dbMax = 10 * Math.log10(Tmax), dbMin = 10 * Math.log10(Tmin), mid = (dbMax + dbMin) / 2, half = Math.max((dbMax - dbMin) / 2 * 1.25, 0.005);
      var A = axes(ctx, w, h, [-s.span / 2, s.span / 2], [mid - half, mid + half], 'frequency offset (GHz)', 'T (dB)', c);
      var N = 1400, under = fsrG < s.span / 250;
      if (under) {
        ctx.fillStyle = c.accent; ctx.globalAlpha = 0.35; ctx.fillRect(A.L, A.Y(dbMax), A.R - A.L, A.Y(dbMin) - A.Y(dbMax)); ctx.globalAlpha = 1;
        ctx.fillStyle = c.fg; ctx.font = '12px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('Ripple period finer than this plot can sample: shown as its envelope. An instrument this coarse averages it.', (A.L + A.R) / 2, A.T + 14);
      } else {
        ctx.strokeStyle = c.accent; ctx.lineWidth = 1.6; ctx.beginPath();
        for (var i = 0; i < N; i++) {
          var f = -s.span / 2 + s.span * i / (N - 1), ph = 2 * Math.PI * f / fsrG;
          var T = (1 - R1) * (1 - R2) / (1 + R1 * R2 - 2 * r * Math.cos(ph)), y = A.Y(10 * Math.log10(T));
          if (i) ctx.lineTo(A.X(f), y); else ctx.moveTo(A.X(f), y);
        }
        ctx.stroke();
      }
      var r1 = Math.sqrt(R1), r2 = Math.sqrt(R2), Rmax = Math.pow(r1 + r2, 2) / Math.pow(1 + r1 * r2, 2);
      var dlam = LAM * LAM * fsr / CL;
      var pp = 20 * Math.log10((1 + r) / (1 - r));
      rd.innerHTML = '<div><span>FSR </span><b>' + fmtN(fsrG) + ' GHz</b></div><div><span>Period at 1550 nm </span><b>' + fmtN(dlam * 1e12) + ' pm</b></div>' +
        '<div><span>Ripple peak-to-valley </span><b>' + fmtN(pp) + ' dB</b></div><div><span>sqrt(R1 R2) </span><b>' + fmtN(r) + '</b></div>' +
        '<div><span>Worst-case reflection back </span><b>' + (10 * Math.log10(Rmax)).toFixed(1) + ' dB</b></div>';
      var pv = parseFloat(invIn.value);
      invOut.textContent = pv > 0 ? (function () { var Lm = LAM * LAM / (2 * s.ng * pv * 1e-12); return Lm < 1e-3 ? (Lm * 1e6).toPrecision(3) + ' um' : Lm < 1 ? (Lm * 1e3).toPrecision(3) + ' mm' : Lm.toPrecision(3) + ' m'; })() : 'N/A';
    }
    var onR = function () { draw(); }; window.addEventListener('resize', onR);
    api.onCleanup(function () { window.removeEventListener('resize', onR); });
    draw();
  };

  /* ---- (b) Mode hop with hysteresis: illustrative, normalized ---- */
  widgets.modeHop = function (root, params, api) {
    var TMAX = 10, AC = 0.25;
    var s = { af: 0.8, h: 0.2, T: 0, m: 0, up: [], down: [], hopsUp: [], hopsDown: [], didUp: false, didDown: false, busy: false };
    var done = false, raf = 0;
    var wrap = shell(root);
    el('div', 'fbw-tag', 'Illustrative model. Normalized units: frequency in cavity free spectral ranges, temperature in arbitrary units. The laser stays in its mode until the filter peak is more than (0.5 + margin) away, then jumps to the mode nearest the filter.', wrap);
    var fit1 = canvas(wrap, 150), fit2 = canvas(wrap, 220);
    var rd = el('div', 'fbw-read', null, wrap);
    var row = el('div', 'fbw-row', null, wrap);
    var bUp = el('button', 'fbw-btn', 'Sweep up', row); bUp.type = 'button';
    var bDn = el('button', 'fbw-btn', 'Sweep down', row); bDn.type = 'button';
    var bClr = el('button', 'fbw-btn', 'Clear traces', row); bClr.type = 'button';
    var g = el('div', 'fbw-grid', null, wrap);
    var sT = slider(g, 'Temperature (manual)', 0, TMAX, 0.01, 0, function (v) { return v.toFixed(2); }, function (v) { moveTo(v, null); draw(); });
    slider(g, 'Filter tuning rate / comb tuning rate', 1.4, 8, 0.1, s.af / AC, function (v) { return v.toFixed(1) + 'x'; }, function (v) { s.af = v * AC; reset(); draw(); });
    slider(g, 'Hysteresis margin', 0, 0.45, 0.01, s.h, function (v) { return v.toFixed(2) + ' FSR'; }, function (v) { s.h = v; reset(); draw(); });
    var unUp = hint(bUp, 'start here'), unDn = null;

    function nuF(T) { return s.af * T; }
    function nearest(T) { return Math.round(nuF(T) - AC * T); }
    function stepT(T, rec, hops) {
      var d = s.m + AC * T - nuF(T);
      if (Math.abs(d) > 0.5 + s.h) { s.m = nearest(T); if (hops) hops.push(T); }
      s.T = T; if (rec) rec.push([T, s.m + AC * T]);
    }
    function moveTo(T, rec, hops) { var n = Math.max(1, Math.ceil(Math.abs(T - s.T) / 0.005)), T0 = s.T; for (var i = 1; i <= n; i++) stepT(T0 + (T - T0) * i / n, rec, hops); }
    function reset() { s.up = []; s.down = []; s.hopsUp = []; s.hopsDown = []; s.T = 0; s.m = nearest(0); sT.set(0); }
    function sweep(dir) {
      if (s.busy) return; s.busy = true;
      var rec, hops;
      if (dir > 0) { s.up = []; s.hopsUp = []; rec = s.up; hops = s.hopsUp; s.T = 0; s.m = nearest(0); }
      else { s.down = []; s.hopsDown = []; rec = s.down; hops = s.hopsDown; if (Math.abs(s.T - TMAX) > 1e-9) { s.T = TMAX; s.m = nearest(TMAX); } }
      rec.push([s.T, s.m + AC * s.T]);
      var t0 = null, from = s.T, to = dir > 0 ? TMAX : 0, dur = 3500;
      function frame(ts) {
        if (t0 === null) t0 = ts; var k = Math.min(1, (ts - t0) / dur);
        moveTo(from + (to - from) * k, rec, hops); sT.set(s.T); draw();
        if (k < 1) raf = requestAnimationFrame(frame);
        else {
          s.busy = false;
          if (dir > 0) { s.didUp = true; if (unUp) { unUp(); unUp = null; } if (!s.didDown && !unDn) unDn = hint(bDn, 'now sweep down'); }
          else { s.didDown = true; if (unDn) { unDn(); unDn = null; } }
          if (s.didUp && s.didDown && !done) { done = true; api.complete(); }
        }
      }
      raf = requestAnimationFrame(frame);
    }
    bUp.addEventListener('click', function () { sweep(1); });
    bDn.addEventListener('click', function () { sweep(-1); });
    bClr.addEventListener('click', function () { if (!s.busy) { reset(); draw(); } });

    function draw() {
      var c = apiCols(api);
      var ymax = Math.max(s.af, AC) * TMAX + 1.2, ymin = -1.2;
      /* spectrum panel */
      var o = fit1(), ctx = o.ctx; ctx.fillStyle = c.panel; ctx.fillRect(0, 0, o.w, o.h);
      var A = axes(ctx, o.w, o.h, [ymin, ymax], [0, 1.1], 'optical frequency (cavity FSR units)', 'gain', c);
      var fc = nuF(s.T), wd = 0.9;
      ctx.strokeStyle = c.warn; ctx.lineWidth = 1.6; ctx.beginPath();
      for (var i = 0; i <= 400; i++) { var x = ymin + (ymax - ymin) * i / 400, yv = lor(x, fc, wd); if (i) ctx.lineTo(A.X(x), A.Y(yv)); else ctx.moveTo(A.X(x), A.Y(yv)); }
      ctx.stroke();
      for (var m = Math.floor(ymin - AC * s.T) - 1; m <= Math.ceil(ymax - AC * s.T) + 1; m++) {
        var nu = m + AC * s.T; if (nu < ymin || nu > ymax) continue;
        var las = m === s.m; ctx.strokeStyle = las ? c.accent : c.muted; ctx.lineWidth = las ? 3.5 : 1.2;
        ctx.beginPath(); ctx.moveTo(A.X(nu), A.Y(0)); ctx.lineTo(A.X(nu), A.Y(las ? 1.05 : 0.12 + 0.3 * lor(nu, fc, wd))); ctx.stroke();
      }
      /* trajectory panel */
      var o2 = fit2(), c2 = o2.ctx; c2.fillStyle = c.panel; c2.fillRect(0, 0, o2.w, o2.h);
      var B2 = axes(c2, o2.w, o2.h, [0, TMAX], [ymin, ymax], 'temperature (arbitrary units)', 'lasing frequency (FSR units)', c);
      c2.strokeStyle = c.warn; c2.setLineDash([4, 4]); c2.lineWidth = 1; c2.beginPath(); c2.moveTo(B2.X(0), B2.Y(0)); c2.lineTo(B2.X(TMAX), B2.Y(nuF(TMAX))); c2.stroke(); c2.setLineDash([]);
      function trace(arr, col) {
        if (!arr.length) return; c2.strokeStyle = col; c2.lineWidth = 2; c2.beginPath();
        for (var j = 0; j < arr.length; j++) { var p = arr[j], px = B2.X(p[0]), py = B2.Y(p[1]); if (j && Math.abs(p[1] - arr[j - 1][1]) > 0.5) { c2.stroke(); c2.beginPath(); c2.moveTo(px, py); } else if (j) c2.lineTo(px, py); else c2.moveTo(px, py); }
        c2.stroke();
      }
      trace(s.up, c.accent); trace(s.down, c.bad);
      c2.fillStyle = c.fg; c2.beginPath(); c2.arc(B2.X(s.T), B2.Y(s.m + AC * s.T), 4, 0, 7); c2.fill();
      c2.font = '11px system-ui, sans-serif'; c2.textAlign = 'left'; c2.textBaseline = 'top';
      c2.fillStyle = c.accent; c2.fillText('sweep up', B2.L + 6, B2.T + 4); c2.fillStyle = c.bad; c2.fillText('sweep down', B2.L + 70, B2.T + 4); c2.fillStyle = c.warn; c2.fillText('filter peak', B2.L + 150, B2.T + 4);
      var f2 = function (a) { return a.length ? a.slice(0, 5).map(function (t) { return t.toFixed(2); }).join(', ') + (a.length > 5 ? ' ...' : '') : 'none yet'; };
      rd.innerHTML = '<div><span>Temperature </span><b>' + s.T.toFixed(2) + '</b></div><div><span>Lasing mode index </span><b>' + s.m + '</b></div>' +
        '<div><span>Hops going up at T = </span><b>' + f2(s.hopsUp) + '</b></div><div><span>Hops going down at T = </span><b>' + f2(s.hopsDown.slice().reverse()) + '</b></div>';
    }
    var onR = function () { draw(); }; window.addEventListener('resize', onR);
    api.onCleanup(function () { cancelAnimationFrame(raf); window.removeEventListener('resize', onR); });
    reset(); draw();
  };

  /* ---- (c) Backscatter doublet: coupled-mode theory, normalized to total linewidth kappa ----
     a_cw, a_ccw with decay kappa/2, backscatter coupling beta, bus coupling kappa_e = eta * kappa.
     D = kappa/2 - i Delta;  t = 1 - kappa_e D / (D^2 + beta^2);  r = i beta kappa_e / (D^2 + beta^2). */
  widgets.doublet = function (root, params, api) {
    var s = { b: 0, eta: 0.5 }, done = false;
    var wrap = shell(root);
    el('div', 'fbw-tag', 'Coupled-mode model. Detuning and backscatter rate are normalized to the total (loaded) linewidth kappa; eta = bus coupling rate / kappa (0.5 = critical coupling without backscatter).', wrap);
    var fit = canvas(wrap, 250);
    var rd = el('div', 'fbw-read', null, wrap);
    var g = el('div', 'fbw-grid', null, wrap);
    var un = null;
    var sB = slider(g, 'Backscatter coupling beta / kappa', 0, 2, 0.01, s.b, function (v) { return v.toFixed(2); }, function (v) { s.b = v; if (un) { un(); un = null; } if (!done && v >= 0.6) { done = true; api.complete(); } draw(); });
    slider(g, 'Bus coupling eta = kappa_e / kappa', 0.05, 0.95, 0.01, s.eta, function (v) { return v.toFixed(2); }, function (v) { s.eta = v; draw(); });
    un = hint(sB.box, 'drag right');
    function resp(x) {
      // D = 0.5 - i x ; D^2 = 0.25 - x^2 - i x ; den = D^2 + b^2
      var dr = 0.25 - x * x + s.b * s.b, di = -x, den = dr * dr + di * di;
      // kappa_e * D / den
      var nr = s.eta * 0.5, ni = -s.eta * x;
      var qr = (nr * dr + ni * di) / den, qi = (ni * dr - nr * di) / den;
      var tr = 1 - qr, ti = -qi;
      var rr2 = (s.b * s.eta) * (s.b * s.eta) / den;
      return [tr * tr + ti * ti, rr2];
    }
    function draw() {
      var o = fit(), ctx = o.ctx, c = apiCols(api); ctx.fillStyle = c.panel; ctx.fillRect(0, 0, o.w, o.h);
      var A = axes(ctx, o.w, o.h, [-4, 4], [0, 1.05], 'detuning / kappa', 'power fraction', c);
      var N = 1601, xs = [], ts = [], rs = [], tmin = 2, rmax = 0;
      for (var i = 0; i < N; i++) { var x = -4 + 8 * i / (N - 1), v = resp(x); xs.push(x); ts.push(v[0]); rs.push(v[1]); if (v[0] < tmin) tmin = v[0]; if (v[1] > rmax) rmax = v[1]; }
      [[ts, c.accent], [rs, c.bad]].forEach(function (p) { ctx.strokeStyle = p[1]; ctx.lineWidth = 2; ctx.beginPath(); for (var j = 0; j < N; j++) { if (j) ctx.lineTo(A.X(xs[j]), A.Y(p[0][j])); else ctx.moveTo(A.X(xs[j]), A.Y(p[0][j])); } ctx.stroke(); });
      var lvl = (1 + tmin) / 2, lo = null, hi = null;
      for (var k = 0; k < N; k++) { if (ts[k] <= lvl) { if (lo === null) lo = xs[k]; hi = xs[k]; } }
      var width = lo === null ? NaN : hi - lo;
      if (lo !== null) { ctx.strokeStyle = c.warn; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(A.X(lo), A.Y(lvl)); ctx.lineTo(A.X(hi), A.Y(lvl)); ctx.stroke(); ctx.setLineDash([]); }
      ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillStyle = c.accent; ctx.fillText('transmission |t|^2', A.L + 6, A.T + 4); ctx.fillStyle = c.bad; ctx.fillText('reflected toward laser |r|^2', A.L + 6, A.T + 18); ctx.fillStyle = c.warn; ctx.fillText('full width at half depth', A.L + 6, A.T + 32);
      rd.innerHTML = '<div><span>Supermode splitting </span><b>' + (2 * s.b).toFixed(2) + ' kappa</b></div>' +
        '<div><span>Width at half depth </span><b>' + (isFinite(width) ? width.toFixed(2) + ' kappa' : 'N/A') + '</b></div>' +
        '<div><span>Half-depth-width Q estimate / true loaded Q </span><b>' + (isFinite(width) && width > 0 ? (1 / width).toFixed(2) : 'N/A') + '</b></div>' +
        '<div><span>Peak reflection toward laser </span><b>' + (rmax > 0 ? (10 * Math.log10(rmax)).toFixed(1) + ' dB' : 'none') + '</b></div>';
    }
    var onR = function () { draw(); }; window.addEventListener('resize', onR);
    api.onCleanup(function () { window.removeEventListener('resize', onR); });
    draw();
  };

  /* =================================================================== */
  /* MEDIA (Phase B): real photos from media/fb and media/shared.           */
  /* Registry: media/credits_fb.js, media/credits_fe.js (shared).            */
  /* Spot targets: x, y are the TOP-LEFT of the box (plan gave centers).     */
  /* Blender renders (media/credits_render.js): butterfly bench correct and  */
  /* reversed (Incident Review 1 and 4 stills, reciprocity spot step),       */
  /* FC/APC vs FC/PC ferrules (Module 1), SiN microring (Module 6).          */
  /* =================================================================== */
  var FB = 'media/fb/', SH = 'media/shared/', RD = 'media/render/';
  function findStep(ch, key) {
    for (var i = 0; i < ch.steps.length; i++) { var st = ch.steps[i]; if (st.title === key || (st.question && st.question.indexOf(key) === 0)) return i; }
    throw new Error('media anchor not found: ' + key);
  }
  function after(ch, key, steps) { var i = findStep(ch, key); Array.prototype.splice.apply(ch.steps, [i + 1, 0].concat(steps)); }
  function patch(ch, key, fields) { var st = ch.steps[findStep(ch, key)]; Object.keys(fields).forEach(function (k) { st[k] = fields[k]; }); }
  function photo(title, file, caption, narration, kb) { var st = { type: 'photo', title: title, image: { src: file }, caption: caption, narration: narration }; if (kb) st.kb = kb; return st; }

  /* ---- Module 0 ---- */
  after(ch0, 'Feedback, Reflection and Zero Tolerance for Retaliation in the Optical Workplace', [
    photo('Our Workplace', FB + 'cleanroom_fiber_team.jpg', 'Technicians route optical fiber in a cleanroom.',
      ['Our people work with light every day.', 'Light, in turn, works with our people. This course keeps that relationship professional.'], [0.62, 0.45, 1.0, 0.60, 0.42, 1.12])
  ]);
  after(ch0, 'Course Enforcement Features', [
    photo('Automation Is Not an Exemption', FB + 'cleanroom_robot_cell.jpg', 'An industrial robot cell in a cleanroom.',
      ['Automation does not exempt anyone from this training.', 'The robot has already completed it. It scored one hundred percent.'])
  ]);
  patch(ch0, 'Speak-Up Procedure for Optical Incidents', { layout: 'split', image: { src: FB + 'meeting_room_empty.jpg' }, kicker: 'Policy section 1.1' });

  /* ---- Module 1 ---- */
  after(ch1, 'Cold Open: The Ripple', [
    photo('Lasers Live Among Surfaces', FB + 'tisa_laser_table.jpg', 'A titanium-sapphire laser with green pump light on an optical table.',
      ['Lasers live among surfaces.', 'Every surface is a potential correspondent. Every interface reflects.'])
  ]);
  after(ch1, 'Scenario 2: The Flat Cleave', [
    photo('The Angle Is the Point', FB + 'fiber_apc_patchcord.jpg', 'A fiber patch cord with FC connectors; per the uploader, both ends are angle polished.',
      ['An angle-polished end sends its reflection out of the fiber core.', 'A flat end sends it straight back. The angle is the point.']),
    photo('Angled and Flat', RD + 'fc_apc_vs_pc_ferrules.jpg', 'Rendered illustration: two FC connectors on an anti-static mat, dust caps off. The green-boot connector (upper) has an angle-polished ferrule tip; the blue-boot connector (lower) has a flat, slightly domed tip.',
      ['In this render, the green connector is angle polished. The blue one is flat.', 'Angled sends the reflection into the cladding. Flat sends it home. Please do not mate green with blue.'], [0.5, 0.5, 1.0, 0.8, 0.5, 1.3]),
    { type: 'spot', title: 'Spot the Unterminated Ends', image: { src: FB + 'spot_patch_panel.jpg' },
      prompt: 'This distribution frame passed its last audit. Find five places where a fiber ends in air.',
      targets: [
        { x: 0.285, y: 0.745, w: 0.080, h: 0.110, label: 'Unmated connector, end face toward you', why: 'An open connector is a glass-to-air interface. A flat polished silica end reflects about three point three percent straight back down the fiber. An angled end steers the reflection out of the core. Either way, an uncapped end collects dust, and dust is a reflector nobody budgeted.' },
        { x: 0.190, y: 0.435, w: 0.050, h: 0.170, label: 'Hanging connector', why: 'Unmated and uncapped. If this fiber is live, its end is the far mirror of a cavity that includes everything back to the transmitter.' },
        { x: 0.300, y: 0.110, w: 0.090, h: 0.170, label: 'Two connectors parked in mid-air', why: 'A connector that is not mated is an interface waiting to become a cavity. Park unused ends in a capped parking slot, not in the cable spaghetti.' },
        { x: 0.240, y: 0.375, w: 0.050, h: 0.120, label: 'Connector with no dust cap', why: 'The angle polish only helps while the end face is clean and undamaged. A scratch or a particle scatters light back into the core.' },
        { x: 0.120, y: 0.615, w: 0.170, h: 0.120, label: 'Cluster of loose connectors', why: 'Several more unmated ends. Each one is its own interface. Every interface reflects.' }
      ],
      missWhy: 'Not that spot. There are more loose ends in this frame than the audit asks for. Look for connectors hanging free, not plugged into an adapter.',
      narration: ['Spot the unterminated ends.', 'This frame passed its last audit. Click every place where a fiber ends in air.'] }
  ]);
  patch(ch1, 'A flip-chip InP laser on a silicon photonics chip', { image: { src: FB + 'fiber_stripped_singlemode.jpg' } });

  /* ---- Module 2 ---- */
  after(ch2, 'Cold Open: The Wiggles', [
    photo('A Cavity, Invited', FB + 'fp_sodium_rings.jpg', 'Fabry-Perot interference rings of the sodium D line.',
      ['Two parallel reflecting surfaces, and rings.', 'This is what a cavity looks like when it is invited. Yours will not look this tidy.'])
  ]);
  after(ch2, 'Interactive: Parasitic Fabry-Perot Ripple', [
    { type: 'hotspots', title: 'Every Interface on This Table', style: 'numbered', image: { src: FB + 'hotspot_optics_table.jpg' },
      prompt: 'This is a working table in a cold-atom laboratory. Open each marker to review the reflectors in plain sight.',
      spots: [
        { x: 0.51, y: 0.43, label: 'Large round optic', body: '<p>Whatever this optic is meant to transmit, each of its surfaces also reflects. For glass with an index near 1.5, Fresnel gives about 4 percent at normal incidence. An anti-reflection coating reduces that. It does not remove it.</p>', narration: ['Each surface of this optic reflects. A coating reduces that. It does not remove it.'] },
        { x: 0.79, y: 0.73, label: 'Large glass cube', body: '<p>A cube has parallel faces. Two parallel faces are an etalon, and at normal incidence the cube also sends a reflection straight back toward the source. A small tilt sends that reflection somewhere harmless.</p>', narration: ['Two parallel faces are an etalon. A small tilt sends the reflection somewhere harmless.'] },
        { x: 0.445, y: 0.695, label: 'Rotation mount', body: '<p>Rotation mounts usually carry a wave plate or polarizer: a thin plate with two parallel faces. A plate of thickness L and group index n is a low-finesse Fabry-Perot with free spectral range c / (2 n L).</p>', narration: ['A thin plate with two parallel faces is a Fabry-Perot, whether or not it was ordered as one.'] },
        { x: 0.14, y: 0.715, label: 'Small glass cube', body: '<p>Smaller does not mean exempt. Every pair of parallel faces is a cavity, and a table holds dozens of them.</p>', narration: ['Smaller does not mean exempt.'] },
        { x: 0.82, y: 0.65, label: 'Kinematic mount', body: '<p>The two adjustment knobs set the angle of the optic. Angle is your main control over where a reflection goes: into a beam block, not back into the laser.</p>', narration: ['Angle is your main control over where a reflection goes. Isolate, angle, and verify.'] },
        { x: 0.69, y: 0.72, label: 'Ring-mounted lens', body: '<p>Curved surfaces reflect too. A lens surface near a focus can return light almost exactly along the incoming axis.</p>', narration: ['Curved surfaces reflect too. Every interface reflects.'] }
      ],
      narration: ['Every interface on this table reflects. Open each marker.'] }
  ]);
  after(ch2, 'Scenario 2: The Accidental Laser', [
    photo('Scheduled Reflectors', FB + 'mirror_mounts.jpg', 'Kinematic mirror mounts on a breadboard.',
      ['Mirrors are the reflectors we schedule.', 'All other reflectors are walk-ins.'])
  ]);
  patch(ch2, 'A measured spectrum shows a ripple with a period of 0.1 nm', { image: { src: FB + 'fp_scanning_interferometer.jpg' } });

  /* ---- Module 3: Classifying ---- */
  patch(chCat, 'Why We Classify', { layout: 'split', image: { src: SH + 'open_plan_office.jpg' }, kicker: 'Classification' });
  after(chCat, 'Why We Classify', [
    { type: 'cards', title: 'Meet the Categories', style: 'grid', mode: 'explore',
      prompt: 'Open each card to meet one category of unsolicited feedback. All nine are mandatory.',
      cards: [
        { image: { src: FB + 'siph_wafer_300mm.jpg' }, label: CAT.domestic.name, body: 'Inside your own chip or package. Owner: the chip or package designer.' },
        { image: { src: FB + 'fiber_apc_patchcord.jpg' }, label: CAT.external.name, body: 'On the path you connected: connectors, patch cords, fiber ends, lenses, the device under test. Owner: the setup owner.' },
        { image: { src: FB + 'power_meter_head.jpg' }, label: CAT.instrument.name, body: 'From the measurement equipment itself. Owner: the measurement setup owner.' },
        { image: { src: FB + 'isolator_free_space.jpg' }, label: CAT.thirdparty.name, body: 'Enabled by a purchased part that does not do what you assumed. Completed by us.' },
        { image: { src: FB + 'spot_patch_panel.jpg' }, label: CAT.intragroup.name, body: 'From resources shared inside your group that nobody owns.' },
        { image: { src: SH + 'meeting_room_table.jpg' }, label: CAT.crossinst.name, body: 'Arrives through a collaboration in which nobody wrote down the return loss.' },
        { image: { src: SH + 'conference_room_modern.jpg' }, label: CAT.qpq.name, body: 'Isolation traded for insertion loss. Truth traded for averaging.' },
        { image: { src: FB + 'lab_bench_cables.jpg' }, label: CAT.hostile.name, body: 'Everything reflects, nothing is angled, and no single reflector is to blame.' },
        { image: { src: FB + 'laser_diode_needle.jpg' }, label: CAT.retaliation.name, body: 'The laser\'s response. Never the laser\'s fault.' }
      ],
      narration: ['Open each card to meet the nine categories. You will meet them again, one at a time.'] }
  ]);

  /* ---- Module 4: Mode Hop Awareness ---- */
  after(ch3, 'Cold Open: The Thermometer Ticket', [
    photo('Small Enough to Lose', FB + 'laser_diode_needle.jpg', 'A tunable diode laser chip in the eye of a sewing needle.',
      ['Small enough to lose.', 'Sensitive enough to notice your fiber. Every cavity hops.'])
  ]);
  after(ch3, 'Interactive: Mode Hop Map', [
    { type: 'hotspots', title: 'Anatomy of a Mode Hop Risk', style: 'pins', image: { src: FB + 'hotspot_littrow_lasers.jpg' },
      prompt: 'Per the photographer, the two blue units in front are diode lasers with Littrow resonators, in a setup for cooling and trapping rubidium atoms. Open each marker.',
      spots: [
        { x: 0.33, y: 0.66, label: 'Littrow laser', body: '<p>Inside, a grating at the Littrow angle sends its first diffraction order back into the diode. This is strong feedback by design, with consent: the external cavity and the grating choose the lasing mode.</p>', narration: ['A grating sends light back into the diode on purpose. Feedback with consent.'] },
        { x: 0.65, y: 0.66, label: 'Second laser', body: '<p>Mode-hop-free tuning of a grating-tuned external-cavity laser needs the grating angle and the cavity length to change together. Turn only one, and the laser jumps to the next external-cavity mode.</p>', narration: ['Turn only the grating, and the laser jumps to the next mode. Every cavity hops.'] },
        { x: 0.70, y: 0.53, label: 'Rotation mount', body: '<p>Downstream optics reflect too. Light returned from them re-enters the external cavity and adds a third mirror. This is why an optical isolator is commonly placed directly after such a laser.</p>', narration: ['Downstream optics add a third mirror. Isolate, angle, and verify.'] },
        { x: 0.25, y: 0.59, label: 'Steering mirror', body: '<p>Its angle decides whether a stray reflection walks off, or lines up with the laser again.</p>', narration: ['A steering mirror decides where a stray reflection goes.'] },
        { x: 0.85, y: 0.16, label: 'Video monitor', body: '<p>A video monitor updates at its frame rate. A hop that lasts microseconds, or a hop and a hop back, will not appear on it. See Chapter 7.</p>', narration: ['A monitor updates at its frame rate. A fast hop will not appear on it.'] }
      ],
      narration: ['Two lasers with gratings. Several opportunities to hop. Open each marker.'] }
  ]);
  patch(ch3, 'Without synchronized tuning of the cavity phase', { image: { src: FB + 'laser_diode_to_can.jpg' } });
  patch(ch3, 'You are Gary. The hop is not in the log', { image: { src: FB + 'trade_fair_hall.jpg' } });

  /* ---- Module 5: Reciprocity ---- */
  after(ch4, 'Cold Open: The Diode', [
    photo('A Real Isolator', FB + 'isolator_on_table.jpg', 'A free-space optical isolator on an optical table.',
      ['A real optical isolator. It contains a magnet.', 'That is not a coincidence. Reciprocity is not optional.'])
  ]);
  after(ch4, 'Isolation Facts and Myths', [
    { type: 'cards', title: 'Which of These Is an Isolator?', style: 'flip', mode: 'choose',
      prompt: 'Choose the component that passes light in one direction and blocks the return path.',
      cards: [
        { image: { src: FB + 'card_isolator.jpg' }, label: 'Faraday isolator', correct: true, body: 'A static magnetic field rotates the polarization in the same sense in both directions; between two polarizers, forward and backward transmission differ.', why: '' },
        { image: { src: FB + 'card_fp_interferometer.jpg' }, label: 'Fabry-Perot interferometer', correct: false, body: 'Two mirrors. Linear, static, non-magnetic.', why: 'A Fabry-Perot is linear, static and non-magnetic, so it is reciprocal. Its transmission is the same in both directions. It is also two mirrors, which makes it a source of feedback, not a cure for it.' },
        { image: { src: FB + 'card_beam_dump.jpg' }, label: 'Beam dump', correct: false, body: 'Absorbs whatever arrives.', why: 'A beam dump absorbs whatever arrives, from either direction. It is the right place to send a reflection you have already steered away. It does not decide direction.' },
        { image: { src: FB + 'card_iris.jpg' }, label: 'Iris aperture', correct: false, body: 'Clips the beam.', why: 'An iris clips the beam the same way in both directions. It can block a reflection that walked off axis, but light returning along the same path passes the same hole.' }
      ],
      correctSay: 'The one with the magnet. A magnet is how a passive, linear device gets permission to be non-reciprocal.',
      narration: ['Only one of these components is an isolator. Choose it.'] }
  ]);
  after(ch4, 'Scenario 3: The Straight Arrow', [
    photo('The Arrow Is a Commitment', FB + 'isolator_free_space.jpg', 'A free-space Faraday isolator mounted on rails on a breadboard.',
      ['The arrow on an isolator is a commitment.', 'Install accordingly. Direction is not angle.']),
    { type: 'spot', title: 'Inspect This Bench', image: { src: RD + 'butterfly_isolator_reversed_closeup.jpg' },
      prompt: 'Rule: the arrow printed on an isolator body must point away from the laser, the way the light travels. The laser mount is at the upper right. Exactly one component in this view has a required direction, and it is installed backwards. Find it.',
      targets: [
        { x: 0.13, y: 0.48, w: 0.38, h: 0.23, label: 'Isolator, arrow pointing at the laser', why: 'The printed arrow on the isolator points toward the laser. The laser output is now attenuated by about the isolation ratio, and light that starts downstream reaches the laser with only the insertion loss. Direction is not angle. Turn it around, and log it.' }
      ],
      missWhy: 'Not that one. Fibers, splice sleeves and mounts work in both directions. Find the one component with an arrow on it, and compare the arrow with the direction to the laser.',
      narration: ['Inspect this bench.', 'Exactly one component here has a required direction. It is installed backwards. Find it.'] }
  ]);
  patch(ch4, 'You are Gary. The vendor has returned the laser', { image: { src: FB + 'card_isolator.jpg' } });

  /* ---- Module 6: Backscatter ---- */
  after(ch5, 'Cold Open: The Split Personality', [
    photo('Every Die Has an Opinion', FB + 'siph_wafer_300mm.jpg', 'A 300 mm silicon photonics wafer.',
      ['Every die on this wafer is a collection of interfaces.', 'Each one has an opinion about direction. Every interface reflects.'])
  ]);
  patch(ch5, 'Why Resonances Split', { layout: 'split', image: { src: FB + 'microdisk_sem.jpg' }, heading: 'Why Resonances Split', kicker: 'Electron micrograph: a silica microdisk resonator' });
  after(ch5, 'Why Resonances Split', [
    photo('A Resonator Awaiting Light', RD + 'sin_microring_lensed_fiber.jpg', 'Rendered illustration (not a real micrograph): a lensed fiber at the edge of a silicon nitride chip, a bus waveguide, a ring resonator of 100 micrometer radius, and a drop waveguide.',
      ['A high Q resonator, awaiting light.', 'Some of that light will be returned to the sender. This is not a courtesy.'], [0.2, 0.8, 1.8, 0.69, 0.75, 1.8])
  ]);

  /* ---- Module 7: Instruments ---- */
  after(ch6, 'Cold Open: The Steady Number', [
    photo('Somewhere, an Instrument Is Averaging', FB + 'lab_bench_cables.jpg', 'A crowded atomic physics laboratory with optics, cables and cooling hoses.',
      ['Somewhere on this table, an instrument is averaging.', 'It will not tell you which one.'])
  ]);
  patch(ch6, 'Every Instrument Integrates', { layout: 'split', image: { src: FB + 'scope_flat_trace.jpg' }, heading: 'Every Instrument Integrates', kicker: 'Calm is a display setting' });
  after(ch6, 'Scenario 1: The Calm Wavemeter', [
    photo('The Head Integrates', FB + 'power_meter_head.jpg', 'An optical power meter sensor head on a post.',
      ['A power meter head reports the average power.', 'Including the average of a laser that hops between two modes. Verify is the part with the instruments.'])
  ]);

  /* vary the position of the correct option (deterministic rotation) */
  var nByLen = {};
  function rotate(st) { if (st.type !== 'check') return; var L = st.options.length; nByLen[L] = (nByLen[L] || 0) + 1; var k = nByLen[L] % L; st.options = st.options.slice(k).concat(st.options.slice(0, k)); }
  [ch0, ch1, ch2, chCat, ch3, ch4, ch5, ch6].forEach(function (ch) { ch.steps.forEach(rotate); });
  pool.forEach(rotate);

  /* =================================================================== */
  window.LMS_COURSES = window.LMS_COURSES || {};
  window.LMS_COURSES.feedback = {
    id: 'feedback', code: 'FB-1550',
    title: 'Feedback, Reflection and Zero Tolerance for Retaliation in the Optical Workplace',
    subtitle: 'Annual speak-up and non-retaliation training for lasers, cavities, and the people who couple them',
    duration: 'approximately 155 minutes',
    version: '2026.4', effective: '2026-12-01',
    policy: 'Policy PHC-1550.4: Speak-Up, Non-Retaliation and Return Loss Standard',
    mantra: MANTRA, mantraShort: MSHORT,
    host: { name: 'Pat Reynolds', role: 'Director, Office of Photonic Conduct' },
    employer: 'Intracavity Photonics, Inc.',
    chapters: [ch0, ch1, ch2, chCat, ch3, ch4, ch5, ch6],
    exam: { passPct: 80, count: 10, pool: pool },
    references: Object.keys(R).map(function (k) { return R[k]; }),
    speech: [],
    widgets: widgets
  };
})();
