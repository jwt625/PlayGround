/* Course: Ferroelectric Workplace Conduct and Crystal Orientation Awareness (FE-0042)
   Data + widgets only. No DOM access at load time. */
(function () {
  'use strict';

  var MANTRA = "Know your axes. Watch your charge. Scan both ways. Trust no symmetry plane.";
  var M = MANTRA;

  /* ---------- runtime helpers (only called by the engine at runtime) ---------- */

  function palette(api, root) {
    function g(n, fb) {
      var v = '';
      try { v = api && api.css ? api.css(n) : ''; } catch (e) { v = ''; }
      v = v && String(v).trim();
      return v || fb;
    }
    var inherited = '#cfd6de';
    try { inherited = getComputedStyle(root).color || inherited; } catch (e) { /* ignore */ }
    return {
      fg: g('--fg', g('--text', inherited)),
      muted: g('--muted', 'rgba(140,150,160,0.9)'),
      grid: g('--border', 'rgba(140,150,160,0.35)'),
      accent: g('--accent', '#2f7de1'),
      danger: g('--danger', g('--bad', '#d9534f')),
      ok: g('--ok', g('--good', '#3a9d5d')),
      warn: g('--warn', '#d39b2a'),
      panel: g('--panel', 'rgba(127,127,127,0.08)')
    };
  }

  function ensureStyle(accent) {
    if (document.getElementById('fe-widget-style')) return;
    var s = document.createElement('style');
    s.id = 'fe-widget-style';
    s.textContent =
      '@keyframes fePulse{0%{box-shadow:0 0 0 0 ' + accent + '}70%{box-shadow:0 0 0 9px rgba(0,0,0,0)}100%{box-shadow:0 0 0 0 rgba(0,0,0,0)}}' +
      '@keyframes feBob{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}' +
      '.fe-w{font:inherit;color:inherit;display:flex;flex-direction:column;gap:8px}' +
      '.fe-row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}' +
      '.fe-w button{font:inherit;color:inherit;background:transparent;border:1px solid currentColor;border-radius:0;padding:5px 10px;cursor:pointer}' +
      '.fe-w button.on{background:' + accent + ';border-color:' + accent + ';color:#fff}' +
      '.fe-w input[type=range]{accent-color:' + accent + ';min-width:160px;flex:1}' +
      '.fe-w label{display:flex;gap:6px;align-items:center;flex-wrap:wrap}' +
      '.fe-hint{animation:fePulse 1.3s infinite;outline:2px solid ' + accent + ';outline-offset:2px}' +
      '.fe-hand{display:inline-flex;align-items:center;gap:4px;color:' + accent + ';font-weight:600;animation:feBob 0.9s infinite;white-space:nowrap}' +
      '.fe-note{opacity:0.8;font-size:0.9em}' +
      '.fe-tag{display:inline-block;border:1px solid currentColor;padding:1px 6px;font-size:0.8em;opacity:0.85}' +
      '.fe-read{font-variant-numeric:tabular-nums}';
    document.head.appendChild(s);
  }

  function hand(text) {
    var d = document.createElement('span');
    d.className = 'fe-hand';
    d.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1v10M3 7l5 6 5-6" fill="none" stroke="currentColor" stroke-width="2"/></svg>' + text;
    return d;
  }

  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (html != null) e.innerHTML = html;
    return e;
  }

  function makeCanvas(root, aspect) {
    var w = Math.max(320, root.clientWidth || 640);
    var h = Math.round(w * aspect);
    var dpr = (window.devicePixelRatio || 1);
    var c = document.createElement('canvas');
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    c.style.width = '100%'; c.style.maxWidth = w + 'px'; c.style.height = 'auto'; c.style.display = 'block';
    var ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { c: c, ctx: ctx, w: w, h: h };
  }

  function cleanup(api, fn) { if (api && typeof api.onCleanup === 'function') api.onCleanup(fn); }
  function say(api, t) { try { if (api && api.say) return api.say(t); } catch (e) { /* ignore */ } return null; }

  /* ---------- animatic helpers for video keyframes ---------- */

  // code-drawn keyframe from the shared drawing library (anims/fe_draw.js): scene name, keyframe index
  function feDraw(name, k) {
    return function (ctx, w, h, u) {
      var F = window.LMS_DRAW && window.LMS_DRAW.fe;
      if (F && F.scenes && F.scenes[name]) F.scenes[name](ctx, w, h, u, k || 0);
    };
  }

  /* ---------- shared step builders ---------- */

  function tts(t) {
    return String(t).replace(/\bTE\b/g, 'T E').replace(/\bTM\b/g, 'T M').replace(/\bn_o\b/g, 'n o').replace(/\bn_e\b/g, 'n e').replace(/\bRF\b/g, 'R F').replace(/\bDC\b/g, 'D C');
  }


  function objectives(title, items, extra) {
    var n = ["Welcome to this chapter. " + title + "."];
    n.push("By the end of this chapter, you will be able to do the following.");
    items.forEach(function (it, i) { n.push("Objective " + (i + 1) + ". " + tts(it)); });
    n.push("Let's repeat those objectives, because repetition is how compliance happens.");
    items.forEach(function (it) { n.push(tts(it)); });
    n.push("And as always, remember our mantra. " + M);
    if (extra) n = n.concat(extra);
    return { type: 'slide', layout: 'bullets', title: 'Learning objectives', bullets: items, callout: M, narration: n };
  }

  // Eight categories of orientation misconduct (taught in the Sources of Orientation Misconduct chapter)
  var CATS = [
    'Tool-Initiated', 'Software-Facilitated', 'Third-Party', 'Peer-to-Peer',
    'Cross-Institutional', 'Domestic', 'Quid Pro Quo', 'Hostile Crystal Environment'
  ];
  var CATS_SAY = "Tool-initiated. Software-facilitated. Third-party. Peer-to-peer. Cross-institutional. Domestic. Quid pro quo. And hostile crystal environment.";
  var PHRASE_O = 'Pause, please. Orientation check.';
  var PHRASE_C = 'Pause, please. Charge check.';
  var RESPOND = 'Stop. Mark. Log. Speak up.';

  // Static widget lines: every line a widget may speak, registered in course.speech.
  // Keep each identical to the literal passed to the widget's speech call.

  function takeaways(items, withCats) {
    var extra = withCats ? [
      "And remember the eight categories of orientation misconduct. " + CATS_SAY,
      "Whatever the category, the response is the same. " + RESPOND
    ] : [];
    return {
      type: 'takeaways', title: 'Key takeaways', items: items.concat(withCats ? ['Eight categories, one response: ' + RESPOND, M] : [M]),
      narration: ["Let's review the key takeaways."].concat(items.map(tts)).concat(extra).concat([
        "One more time, together. " + M,
        "If you only remember one thing from this chapter, remember all four things in the mantra.",
        "Click Next to continue."
      ])
    };
  }

  // Insert steps just before the chapter's takeaways step
  function insertBeforeTakeaways(steps, extra) {
    for (var i = 0; i < steps.length; i++) {
      if (steps[i].type === 'takeaways') { Array.prototype.splice.apply(steps, [i, 0].concat(extra)); return steps; }
    }
    return steps.concat(extra);
  }

  // "How do you think Gary felt?" check, HR voice
  function feelCheck(context) {
    return {
      type: 'check', kicker: 'Reflection',
      question: 'How do you think Gary felt when ' + context + '?',
      narration: ["Let's pause and reflect.", "How do you think Gary felt when " + context + "?"],
      options: [
        { text: 'Ashamed', correct: true, why: '' },
        { text: 'Fine. Gary does not have feelings about crystal axes', correct: false, why: "Everyone at Five Sigma Photonics has feelings about crystal axes, including Gary, and including you. Gary felt ashamed. Shame is a normal response to an orientation incident. It is not a disciplinary outcome, and it is not a performance review. The Office of Photonic Conduct is a judgment-free zone. The incident has still been logged, because a feeling is not a corrective action. Know your axes." },
        { text: 'Relieved, because now he can blame the equipment', correct: false, why: "Relief is understandable, but it is not what Gary felt, and it would not help. The equipment did exactly what it was told. Every tool in this building turns a wafer or a die to exactly the angle in its recipe, or to wherever it happens to stop, and none of them read the traveler. The orientation reference is lost at a hand-off between people, not inside a tool. Gary felt ashamed. Shame is normal. The Office of Photonic Conduct is a judgment-free zone. The incident has been logged." },
        { text: 'It is not appropriate to speculate about how Gary felt', correct: false, why: "In most workplace contexts, that would be a thoughtful answer. In this training, it is not the answer we are looking for. Gary felt ashamed, and he told us so, in a voluntary self-assessment. Shame is a normal response to an orientation incident. The Office of Photonic Conduct is a judgment-free zone. And the incident has still been logged." }
      ],
      correctSay: "Gary felt ashamed. Shame is normal. The Office of Photonic Conduct is a judgment-free zone. The incident has still been logged."
    };
  }

  function ack(what) {
    return {
      type: 'ack',
      text: "I acknowledge that I have completed this chapter on " + what + " in full, without skipping, muting, or opening another tab. I understand that the crystal does not care about my intentions. I will " + M.charAt(0).toLowerCase() + M.slice(1) + " I understand that failure to comply may result in unwelcome charge accumulation, silent coordinate errors, and a mandatory repeat of this course.",
      narration: ["Please read the attestation and tick the box to confirm.", "Your attestation is recorded and may be reviewed by the Office of Photonic Conduct."]
    };
  }

  function refs(items) {
    return { type: 'refs', items: items, narration: ["The following sources were used in this chapter. All of them are real. Please read them during your own time, which is also company time."] };
  }

  /* ---------- reference list ---------- */

  var R = {
    zelmon: { cite: 'Zelmon et al., Journal of the Optical Society of America B, 1997 (infrared dispersion of undoped and MgO-doped lithium niobate)', doi: '10.1364/JOSAB.14.003319' },
    weis: { cite: 'Weis and Gaylord, Applied Physics A, 1985 (lithium niobate: summary of physical properties and crystal structure)', doi: '10.1007/BF00614817' },
    ieee176: { cite: 'IEEE Standard on Piezoelectricity, ANSI/IEEE Std 176-1987', doi: '10.1109/IEEESTD.1988.79638' },
    wangLT: { cite: 'Wang et al., Nature, 2024 (lithium tantalate photonic integrated circuits for volume manufacturing)', doi: '10.1038/s41586-024-07369-1' },
    wangPJ2020: { cite: 'Wang, Chen, Dai and Liu, IEEE Photonics Journal, 2020 (polarization coupling of x-cut thin film lithium niobate based waveguides)', doi: '10.1109/JPHOT.2020.2995317' },
    liu2024: { cite: 'Liu et al., Optics Express, 2024 (sharp bend and large free spectral range ring resonator on thin-film lithium niobate)', doi: '10.1364/OE.515173' },
    liu2023: { cite: 'Liu et al., Light: Advanced Manufacturing, 2023 (ultra-compact lithium niobate photonic chip for wavelength-division-multiplexing transmitters)', doi: '10.37188/lam.2023.013' },
    xie2026: { cite: 'Xie et al., Optics Express, 2026 (reflective-path-enabled folded arrayed waveguide grating on thin-film lithium niobate)', doi: '10.1364/OE.588499' },
    luke2020: { cite: 'Luke et al., Optics Express, 2020 (wafer-scale low-loss lithium niobate photonic integrated circuits)', doi: '10.1364/OE.401959' },
    xu2021: { cite: 'Xu et al., Optics Express, 2021 (mitigating photorefractive effect in thin-film lithium niobate microring resonators)', doi: '10.1364/OE.418877' },
    liang2017: { cite: 'Liang et al., Optica, 2017 (high-quality lithium niobate photonic crystal nanocavities)', doi: '10.1364/OPTICA.4.001251' },
    li2019: { cite: 'Li et al., Laser and Photonics Reviews, 2019 (high-Q 2D lithium niobate photonic crystal slab nanoresonators)', doi: '10.1002/lpor.201800228' },
    lu2019: { cite: 'Lu et al., Optica, 2019 (periodically poled thin-film lithium niobate microring resonators)', doi: '10.1364/OPTICA.6.001455' },
    shams2022: { cite: 'Shams-Ansari et al., APL Photonics, 2022 (reduced material loss in thin-film lithium niobate waveguides)', doi: '10.1063/5.0095146' },
    shen2024: { cite: 'Shen et al., Nature Photonics, 2024 (photonic link from single-flux-quantum circuits to room temperature)', doi: '10.1038/s41566-023-01370-2' },
    zhang2026: { cite: 'Zhang et al., arXiv, 2026 (reconfigurable single-ring photonic molecule on lithium niobate)', url: 'https://arxiv.org/abs/2606.06637' },
    luo2018: { cite: 'Luo et al., Optica, 2018 (highly tunable efficient second-harmonic generation in a lithium niobate nanophotonic waveguide)', doi: '10.1364/OPTICA.5.001006' },
    weigel2018: { cite: 'Weigel et al., Optics Express, 2018 (bonded thin film lithium niobate modulator on a silicon photonics platform)', doi: '10.1364/OE.26.023728' },
    naranjo2005: { cite: 'Naranjo, Gimzewski and Putterman, Nature, 2005 (observation of nuclear fusion driven by a pyroelectric crystal)', doi: '10.1038/nature03575' },
    bulmer1986: { cite: 'Bulmer et al., Applied Physics Letters, 1986 (pyroelectric effects in LiNbO3 channel-waveguide devices)', doi: '10.1063/1.96640' },
    xu2022: { cite: 'Xu et al., Nature, 2022 (femtosecond laser writing of lithium niobate ferroelectric nanodomains)', doi: '10.1038/s41586-022-05042-z' },
    popescu2013: { cite: 'Popescu et al., Journal of Applied Physics, 2013 (interferometric measurement of the pyroelectric coefficient in lithium niobate)', doi: '10.1063/1.4788696' },
    volk2008: { cite: 'Volk and Wohlecke, Lithium Niobate: Defects, Photorefraction and Ferroelectric Switching, Springer, 2008 (Table 7.1, coercive field)', doi: '10.1007/978-3-540-70766-0' },
    maeder2022: { cite: 'Maeder et al., Optics Letters, 2022 (high-bandwidth thermo-optic phase shifters for lithium niobate on insulator photonic integrated circuits)', doi: '10.1364/OL.469358' },
    zheng2024: { cite: 'Zheng et al., Laser and Photonics Reviews, 2024 (photonic neural network fabricated on thin-film lithium niobate)', doi: '10.1002/lpor.202400565' },
    liu2026: { cite: 'Liu et al., ACS Photonics, 2026 (visible and near-infrared thin-film lithium niobate modulators with thermo-optic bias)', doi: '10.1021/acsphotonics.6c00668' },
    yeh2026: { cite: 'Yeh et al., Nano Letters, 2026 (hysteresis and trap emission in dc-biased integrated lithium niobate electro-optic modulators)', doi: '10.1021/acs.nanolett.6c03406' },
    holzgrafe2024: { cite: 'Holzgrafe et al., Optics Express, 2024 (relaxation of the electro-optic response in thin-film lithium niobate modulators)', doi: '10.1364/OE.507536' },
    valdez2023: { cite: 'Valdez et al., Optics Express, 2023 (integrated O- and C-band silicon-lithium niobate Mach-Zehnder modulators with 100 GHz bandwidth)', doi: '10.1364/OE.480519' },
    arab2022: { cite: 'Arab Juneghani et al., Advanced Photonics Research, 2022 (thin-film lithium niobate optical modulators with an extrapolated bandwidth of 170 GHz)', doi: '10.1002/adpr.202200216' },
    boynton2020: { cite: 'Boynton et al., Optics Express, 2020 (a heterogeneously integrated silicon photonic/lithium niobate travelling wave electro-optic modulator)', doi: '10.1364/OE.28.001868' },
    zhu2021: { cite: 'Zhu et al., Advances in Optics and Photonics, 2021 (integrated photonics on thin-film lithium niobate)', doi: '10.1364/AOP.411024' },
    bordui1999: { cite: 'Bordui et al., Journal of Applied Physics, 1999 (chemically reduced lithium niobate single crystals: processing, properties and improved surface acoustic wave device fabrication and performance)', doi: '10.1063/1.369775' },
    chen2025: { cite: 'Chen et al., CrystEngComm, 2025 (black lithium niobate single crystal: influence of point defects on its microwave properties)', doi: '10.1039/D5CE00220F' },
    patent7309392: { cite: 'US Patent 7,309,392 B2, 2007 (lithium niobate substrate and method of producing the same), background section', url: 'https://patents.google.com/patent/US7309392B2/en' },
    sones2002: { cite: 'Sones et al., Journal of Materials Chemistry, 2002 (differential etch rates in z-cut LiNbO3 for variable HF/HNO3 concentrations)', doi: '10.1039/b106279b' }
  };

  /* ======================= CHAPTER 0 ======================= */

  var ch0 = {
    id: 'welcome', title: 'Welcome and Course Navigation',
    subtitle: 'How to complete this mandatory course',
    objectives: ['Navigate the course player', 'Understand course enforcement rules', 'Recite the mantra'],
    steps: [
      {
        type: 'slide', layout: 'title', title: 'Ferroelectric Workplace Conduct and Crystal Orientation Awareness',
        bullets: ['Course FE-0042', 'Annual mandatory refresher', 'Five Sigma Photonics, Inc.'],
        callout: M,
        narration: [
          "Welcome to Ferroelectric Workplace Conduct and Crystal Orientation Awareness.",
          "This is course F E zero zero four two. It is mandatory for all employees who work with, near, or emotionally adjacent to lithium niobate.",
          "This course takes approximately one hundred and eighty minutes. Most employees finish in approximately one hundred and eighty minutes.",
          "Before we begin, here is the mantra you will hear many, many times. " + M
        ]
      },
      {
        type: 'slide', layout: 'host', title: 'A message from the Office of Photonic Conduct',
        bullets: ['Pat Reynolds', 'Director, Office of Photonic Conduct', 'Five Sigma Photonics, Inc.'],
        narration: [
          "Hi. I'm Pat Reynolds, Director of the Office of Photonic Conduct here at Five Sigma Photonics.",
          "At Five Sigma Photonics, we believe every crystal deserves a workplace where its axes are known, its charge is respected, and its symmetry is never assumed.",
          "Lithium niobate is a wonderful colleague. It is electro-optic, piezoelectric, pyroelectric, photorefractive, and anisotropic. It brings its whole self to work.",
          "That also means it brings its whole self to your measurement, your simulation, and your hotplate.",
          "In this course, we will walk through seven areas of conduct. Each one comes from real incidents and real papers.",
          "Together, we can build a culture of " + M.toLowerCase()
        ]
      },
      {
        type: 'slide', layout: 'policy', title: 'Course enforcement rules',
        bullets: [
          'Narration cannot be skipped, sped up, or scrubbed',
          'Video pauses when this tab loses focus',
          'Next unlocks only after narration and required interaction finish',
          'Knowledge checks allow unlimited retries; wrong answers are explained in full',
          'Final assessment: 80 percent to pass, drawn from a question pool',
          'Completion is recorded under Policy PHC-17.3'
        ],
        callout: 'Leaving this tab pauses the course. The crystal is also paused. Neither of you will forget.',
        narration: [
          "Let's review how this course works.",
          "Narration cannot be skipped, sped up, or scrubbed. This is for your own protection.",
          "If you switch to another tab or window, the course will pause, and it will wait for you. It is patient. It has nowhere else to be.",
          "The Next button unlocks only after the narration ends and you complete the required interaction on each page.",
          "Knowledge checks allow unlimited retries. If you choose a wrong answer, we will explain, at length, why it is wrong. Then you may try again.",
          "The final assessment draws questions from a pool. You need eighty percent to pass.",
          "Your completion is recorded under Policy P H C seventeen point three, Ferroelectric Conduct and Crystal Orientation."
        ]
      },
      {
        type: 'slide', layout: 'bullets', title: 'Our mantra',
        bullets: ['Know your axes.', 'Watch your charge.', 'Scan both ways.', 'Trust no symmetry plane.'],
        callout: M,
        narration: [
          "Every chapter in this course connects back to one mantra.",
          "Know your axes. Every cut, every Euler angle, every tensor has a frame. Write it down.",
          "Watch your charge. Light, heat, and D C bias all move charge in this crystal, and moved charge moves your index.",
          "Scan both ways. A resonance measured in one direction only is a resonance you have not met.",
          "Trust no symmetry plane. The crystal is trigonal. Your mesh is not entitled to mirrors the crystal does not have.",
          "Let's say it together. " + M,
          "Thank you. You will be hearing it again."
        ]
      },
      {
        type: 'reveal', title: 'Meet the people in this course',
        prompt: 'Open each profile to continue.',
        tabs: [
          { label: 'Gary', body: '<p>Gary is a dedicated device engineer. Gary moves fast. Gary has never once checked which edge of the chip is the z axis.</p><p>In this course, Gary will make every mistake so that you do not have to.</p>', narration: ["This is Gary. Gary is a dedicated engineer who moves fast. Gary will make every mistake in this course, so that you do not have to."] },
          { label: 'Gary\'s manager', body: '<p>Gary\'s manager wants results by Friday. The manager approves things that look like results.</p>', narration: ["This is Gary's manager. Gary's manager wants results by Friday, and approves things that look like results."] },
          { label: 'Dana Okafor', body: '<p>Dana Okafor is Gary\'s labmate. Dana has seen this before. When something looks wrong, Dana speaks up.</p><p>Be like Dana.</p>', narration: ["This is Dana Okafor, Gary's labmate. Dana has seen this before. When something looks wrong, Dana speaks up. Be like Dana."] },
          { label: 'The crystal', body: '<p>Congruent lithium niobate. Point group 3m. Optic axis: z. Negative uniaxial: the extraordinary index is lower than the ordinary index.</p><p>The crystal does not attend meetings.</p>', narration: ["And this is the crystal. Congruent lithium niobate, point group three m. It is negative uniaxial, which means the extraordinary index is lower than the ordinary index. The crystal does not attend meetings, but it remembers everything."] }
        ]
      },
      {
        type: 'check',
        question: 'Practice question. Which of the following is the course mantra?',
        options: [
          { text: M, correct: true, why: '' },
          { text: 'Move fast. Bake hot. Scan once. Mirror everything.', correct: false, why: "That is not the mantra. That is, in fact, a summary of everything Gary did last quarter. Moving fast is fine, but baking hot without a discharge path builds up pyroelectric charge. Scanning once hides photorefractive and thermal distortion of your resonance. And mirroring everything imposes symmetries that a trigonal crystal does not have. The correct mantra is: " + M },
          { text: 'Inspect. Clean. Inspect. Connect.', correct: false, why: "That is a very good mantra, but it belongs to a different course about fiber connectors. In this course, we focus on ferroelectric crystals. The correct mantra is: " + M }
        ],
        correctSay: "Wonderful. You will hear it many more times."
      },
      ack('course navigation and enforcement')
    ]
  };

  /* ======================= CHAPTER 1 ======================= */

  var ch1 = {
    id: 'orientation', title: 'Crystal Orientation Awareness',
    subtitle: 'Cuts, frames, and Euler angles in the workplace',
    objectives: [
      'Name the cut of your wafer and the in-plane direction of every waveguide relative to the crystal axes',
      'Distinguish rotating the crystal from rotating the coordinates',
      'Recognize that plus 45 degrees and minus 45 degrees are not the same in a trigonal crystal',
      'Report tensor notation conventions before they report you',
      'Respond to an orientation incident with the approved speak-up phrase'
    ],
    steps: []
  };
  ch1.steps = [
    objectives(ch1.title, ch1.objectives),
    {
      type: 'video', title: 'Cold open: Which edge is z?', duration: 26,
      narration: [
        "Let's watch a typical Monday morning, and judge Gary's conduct.",
        "Gary picks up a freshly diced chip. It is square. It has no flat. It has no mark.",
        "Gary rotates it a few times, decides it is fine, and loads it.",
        "Three months of optimization later, Gary learns the model was set up in a different crystal frame than the device.",
        "Is this okay? No. This is not okay. " + M
      ],
      keyframes: [
        { t: 0, shot: 'Medium shot, cleanroom bench under yellow lithography light. Camera at bench height, 35 mm. Gary in a bunny suit holds a square 5 mm chip in tweezers above a gel-pack. The chip has no flat, no notch, no marking. Slow push-in.', caption: 'A square chip. No flat. No mark.', cast: ['Gary'], draw: feDraw('whichz', 0) },
        { t: 7, shot: 'Extreme close-up of the chip in tweezers, macro lens. Overlay of crystal axis arrows (y and z) drawn on the chip surface; each time Gary blinks (cut to his eyes), the arrows jump by 90 degrees.', caption: 'The axes move every time nobody is looking.', cast: ['Gary'], draw: feDraw('whichz', 1) },
        { t: 14, shot: 'Over-the-shoulder at Gary\'s desk, monitor shows a finite element model tree with a single dropdown labeled Coordinate system, set to the wrong one. A calendar on the wall flips three months. Gary\'s coffee mugs multiply.', caption: 'Three months of optimization, in the wrong frame.', cast: ['Gary'], draw: feDraw('whichz', 2) },
        { t: 21, shot: 'Freeze frame on Gary\'s face lit by the monitor. Red stamp graphic slams in: NOT OKAY. Hold 2 seconds.', caption: 'Is this okay?', cast: ['Gary'], draw: feDraw('whichz', 3) }
      ]
    },
    {
      type: 'slide', layout: 'host', title: 'Let\'s take a closer look',
      bullets: [
        'The cut names the crystal axis normal to the wafer surface',
        'x-cut: x normal, y and z in the plane; the optic axis lies in-plane',
        'z-cut: z normal; the optic axis points out of the plane',
        'Every in-plane waveguide angle must be stated relative to a crystal axis'
      ],
      callout: 'Square chips of anisotropic material without a mark are an orientation incident waiting to happen.',
      narration: [
        "Let's take a closer look.",
        "The name of a cut tells you which crystal axis is normal to the wafer surface.",
        "In x-cut lithium niobate, the x axis is normal to the surface. The y axis and the z axis, the optic axis, lie in the plane of the chip.",
        "In z-cut, the z axis is normal to the surface, and the optic axis points out of the plane.",
        "So on x-cut, the direction your waveguide runs in the plane changes the physics. On z-cut, it changes much less for the in-plane field.",
        "This is why every waveguide angle must be stated relative to a crystal axis, not relative to the chip edge, and not relative to your intuition.",
        "A practical tip from the lab: do not use square chips of an anisotropic material unless they are marked. Four identical edges means four chances to be wrong.",
        M
      ]
    },
    {
      type: 'reveal', title: 'Four orientation hazards',
      prompt: 'Open each hazard to continue.',
      tabs: [
        { label: 'Rotate the crystal or the room?', body: '<p>Rotating the crystal by an angle is the same as rotating the coordinate system by the opposite angle. Solvers differ in which one they mean.</p><p>Some finite element packages define Euler angles (z-x-z) as a rotation of the coordinate system. A thesis may define its rotated tensor as a rotation of the crystal. Same magnitude, opposite sign.</p><p>Component transformation is the inverse (the transpose, for a rotation) of the basis-vector transformation.</p>', narration: ["Rotating the crystal by an angle is the same as rotating the coordinates by the opposite angle.", "Solvers and theses do not always say which one they mean. Some finite element tools define Euler angles as a rotation of the coordinate system.", "If you copy a rotated tensor from a thesis that rotated the crystal, you can be off by the sign of the angle. Same magnitude. Opposite sign. Different device."] },
        { label: 'Plus 45 is not minus 45', body: '<p>Lithium niobate is trigonal, point group 3m. Its only mirror planes contain the z axis and are perpendicular to the x axis family.</p><p>On x-cut, rotating a waveguide by plus 45 degrees or minus 45 degrees from y is related by a mirror that is not a symmetry of the crystal. Photoelastic, piezoelectric, and electro-optic couplings can differ (for example the elastic $c_{14}$ and photoelastic $p_{14}$, $p_{41}$ terms, which flip sign under that mirror).</p><p>Logged incident: Euler angles $(-\\pi/4, \\pi/2, \\pi/2)$ used where $(\\pi/4, \\pi/2, \\pi/2)$ was intended.</p>', narration: ["Lithium niobate is trigonal, point group three m. Its mirror planes contain the z axis and are perpendicular to the x axis.", "So on x-cut, plus forty five degrees and minus forty five degrees from the y axis are not related by a symmetry of the crystal.", "Tensor terms that flip sign under that mirror, such as the elastic and photoelastic one four components, make the two directions behave differently.", "One logged incident: an Euler angle of minus pi over four was used where plus pi over four was intended. The model solved. Nothing errored. That is the problem."] },
        { label: 'Tests at 0 and 90 degrees', body: '<p>At 0 and 90 degrees, many sign and transpose errors cancel. The rotated tensor looks correct. The bug waits for 45 degrees.</p><p>Always validate a rotation at a non-special angle, against an independent calculation.</p><p>Also check that the angle label on a plot is the physical angle. In one logged case, a solver Euler angle of 135 degrees was the physical 45 degree device, and the plot said 45.</p>', narration: ["At zero and ninety degrees, many sign and transpose errors cancel, so your rotation code passes every test you wrote.", "The bug waits for forty five degrees.", "Always validate rotation code at a non-special angle against an independent calculation.", "Also check that the angle in your plot title is the physical angle. In one logged case, a solver angle of one hundred thirty five degrees corresponded to the physical forty five degree device."] },
        { label: 'Notation and constants', body: '<p>Voigt notation: the elastic stiffness carries no factor of 2, but piezoelectric strain coefficients on shear components do in the common contracted convention. Mixing conventions corrupts rotated tensors.</p><p>Solver scripting may expect a symmetric 6 by 6 matrix as an upper triangle, column by column, not row-major.</p><p>Clamped (constant strain, superscript $S$, as in $r^S$) and unclamped (constant stress, superscript $T$, as in $r^T$) constants differ. Low-frequency measurements see unclamped values; RF sees clamped. See Weis and Gaylord, 1985, for both tables.</p><p>Tensor signs depend on the axis convention (for example IEEE Std 176-1987 versus the 1949 IRE rules), and published lithium niobate tables do not all use the same frame. Check that two sources use the same one before mixing coefficients; mixing frames flips signs.</p>', narration: ["Tensor notation has its own hazards.", "In Voigt notation, the elastic stiffness has no factor of two, but piezoelectric strain coefficients on shear components usually do. Mixing them corrupts rotated tensors.", "Some solvers expect a symmetric six by six matrix entered as an upper triangle, column by column. Pasting a row-major list gives a valid-looking but wrong stiffness.", "Clamped and unclamped constants differ. Low frequency measurements see unclamped values. R F sees clamped values. Weis and Gaylord tabulate both.", "And never combine coefficients from two papers without checking that they use the same axis sign convention. Know your axes."] }
      ]
    },
    {
      type: 'chat', title: 'Scenario 1: The symmetric crystal', app: 'phone',
      meta: { header: 'Gary', participants: ['Gary', 'Dana'] },
      messages: [
        {"from": "Gary", "text": "quick q. I only simulated the minus 45 deg arm. plus 45 is the mirror image, so I was going to flip that result for the layout", "side": "left"},
        {"from": "Dana", "text": "which cut", "side": "right"},
        {"from": "Gary", "text": "x-cut. the geometry is mirror symmetric", "side": "left"},
        {"from": "Dana", "text": "the geometry is. the crystal is not. 3m has no mirror that maps +45 to -45 on x-cut, the 14 and 41 terms change sign", "side": "right"},
        {"from": "Dana", "text": "run both. and run one at 30 deg to check your rotation code", "side": "right"},
        {"from": "Gary", "text": "the rotation code passes the regression tests though", "side": "left"},
        {"from": "Dana", "text": "the regression tests are at 0 and 90", "side": "right"},
        {"from": "Gary", "text": "ok. queuing plus 45 and 30 tonight", "side": "left"}
      ],
      narration: [
        "Read these texts, and judge whether Gary's plan is proper.",
        "Gary planned to mirror one simulated angle instead of running the other one, because the geometry is symmetric. Is this okay?",
        "No. Dana did the right thing. Dana spoke up, named the missing symmetry, and asked for a test at a non-special angle.",
        "Know your axes. Trust no symmetry plane."
      ]
    },
    {
      type: 'chat', title: 'Scenario 2: The inherited template', app: 'email',
      meta: {"header": "Inbox", "subject": "Coupler optimization results, final (v7)", "participants": ["Gary", "Manager", "Dana"]},
      messages: [
        {"from": "Gary", "text": "Hi all,\n\nAttached are the optimized coupler results for the tapeout. The model converges, the mode profiles look clean, and the couplings are in the range we expected.\n\nBest,\nGary", "side": "left", "to": ["Manager", "Dana Okafor"], "date": "2026-11-09 16:12", "subject": "Coupler optimization results, final (v7)", "attachments": [{"name": "coupler_optimization_v7.pdf", "size": "3.1 MB"}]},
        {"from": "Manager", "text": "Hi Gary,\n\nThese look excellent. Unless anyone objects, let's freeze them for the tapeout on the 20th.\n\nThanks", "side": "left", "to": ["Gary"], "cc": ["Dana Okafor"], "date": "2026-11-09 17:40", "subject": "RE: Coupler optimization results, final (v7)"},
        {"from": "Dana", "text": "Hi both,\n\nBefore we freeze: which material frame is the model in? The template it was built from was set up for y-cut, and the wafers for this tapeout are x-cut.\n\nThanks,\nDana", "side": "right", "to": ["Gary", "Manager"], "date": "2026-11-10 08:55", "subject": "RE: RE: Coupler optimization results, final (v7)"},
        {"from": "Gary", "text": "Hi Dana,\n\nThe model converges, the mode profiles look right, and the numbers match what we expect from last year. I would rather not reopen it this close to tapeout unless we see an actual problem.\n\nGary", "side": "left", "to": ["Dana Okafor"], "cc": ["Manager"], "date": "2026-11-10 09:20", "subject": "RE: RE: RE: Coupler optimization results, final (v7)"},
        {"from": "Dana", "text": "Hi Gary,\n\nAll of those checks pass for the wrong cut too. Nothing in the solver compares the material frame with the wafer. One run with the frame set explicitly would settle it.\n\nDana", "side": "right", "to": ["Gary"], "cc": ["Manager"], "date": "2026-11-10 09:41", "subject": "RE: RE: RE: Coupler optimization results, final (v7)"},
        {"from": "Manager", "text": "How long would that take?", "side": "left", "to": ["Gary", "Dana Okafor"], "date": "2026-11-10 10:02", "subject": "RE: RE: RE: Coupler optimization results, final (v7)"},
        {"from": "Gary", "text": "A spot check is a day. If it changes the optimum, a full re-optimization is about three months.", "side": "left", "to": ["Manager", "Dana Okafor"], "date": "2026-11-10 10:15", "subject": "RE: RE: RE: Coupler optimization results, final (v7)"}
      ],
      narration: [
        "Read this email thread, and judge whether Gary's reply is proper.",
        "Gary's model inherited a material frame from an old template, and every check he relies on would pass either way. Is this okay?",
        "No. A wrong frame is not an error. It is a different, valid device that you did not intend to make.",
        "Make the material frame a named, reviewed parameter, and print it on every results slide.",
        "Know your axes."
      ]
    },
    {
      type: 'chat', title: 'Scenario 3: The animation', app: 'thread',
      meta: { header: '#figures', participants: ['Gary', 'Dana', 'Manager'] },
      messages: [
        {"from": "Gary", "text": "poling animation for the talk is done. I started from the review paper figure, but it is z-cut: z vertical, domains flip up and down through the film", "side": "left"},
        {"from": "Gary", "text": "ours are x-cut, so I redrew it. z in the plane, the polarization flips left and right, between the electrodes. cut labeled in the corner", "side": "left"},
        {"from": "Manager", "text": "the review figure is prettier. it's a schematic for a twenty minute talk, could we just use theirs and say it is illustrative", "side": "left"},
        {"from": "Gary", "text": "illustrative of z-cut. people copy talk figures into their own decks, so I'd rather show the cut we actually have", "side": "left"},
        {"from": "Dana", "text": "checked it against our layout. field and polarization along z, in the plane. good to go", "side": "right"}
      ],
      narration: [
        "Read this thread, and judge whether Gary's response is proper.",
        "Gary made a poling animation for a talk, and the manager would rather reuse a prettier figure. Is this okay?",
        "Yes. Gary drew the cut he actually has, labeled it, and kept it when the shortcut was offered. Even figures and animations have a cut.",
        "Know your axes."
      ]
    },
    {
      type: 'widget', title: 'Interactive: Spot the sign', widget: 'eulerSign', params: {},
      instructions: 'Rotate the crystal on an x-cut die. The left panel rotates the crystal as intended; the right panel uses the same matrix as a coordinate rotation by mistake. Check 0 or 90 degrees, then an angle in between, and compare the effective electro-optic coefficient of the two.',
      narration: ["In this activity, you will rotate a lithium niobate crystal on an x-cut die, and compute its electro-optic response two ways: by rotating the crystal, and by using the same matrix as a coordinate rotation by mistake.", "Move the angle slider. Notice where the two agree, and where they do not."]
    },
    {
      type: 'check',
      question: 'Gary verified his tensor rotation code at 0 and 90 degrees and it matched. He now trusts it at 45 degrees. What is the best next step?',
      options: [
        { text: 'Validate at a non-special angle, such as 30 or 45 degrees, against an independent calculation, and state whether the angle rotates the crystal or the coordinates', correct: true, why: '' },
        { text: 'Nothing. Two passing tests are sufficient', correct: false, why: "At zero and ninety degrees, many common rotation bugs cancel. For example, using the basis-vector transformation where the component transformation was needed, which is its inverse, or transposing a matrix, often gives the same answer at those special angles. Sign errors in the angle also hide there, because a rotation by plus ninety and minus ninety can map the axes onto equivalent positions for the components you checked. The bug appears at a general angle like thirty or forty five degrees. So the passing tests did not test the thing that is most likely to be wrong. Please validate at a non-special angle against an independent calculation, and write down whether your angle rotates the crystal or the coordinates. Know your axes." },
        { text: 'Flip the sign of the angle, just in case', correct: false, why: "Flipping the sign without knowing which convention is in use converts an unknown error into a different unknown error. Rotating the crystal by an angle equals rotating the coordinates by the opposite angle, so the correct sign depends entirely on the convention of the solver and of the source you are comparing against. In a trigonal crystal like lithium niobate, plus and minus forty five degrees on x-cut are not equivalent, so the sign matters. Determine the convention, then validate at a non-special angle." }
      ],
      correctSay: "A non-special angle and a stated convention. Excellent conduct."
    },
    {
      type: 'check',
      question: 'Which statement about x-cut thin-film lithium niobate is correct?',
      options: [
        { text: 'The optic axis z lies in the plane of the film, so the in-plane propagation direction changes what the TE mode sees', correct: true, why: '' },
        { text: 'The optic axis is normal to the film, so all in-plane directions are equivalent', correct: false, why: "That describes z-cut, where the z axis, the optic axis, is normal to the film. In x-cut, the x axis is normal to the film, and both the y axis and the z axis lie in the plane. Because the optic axis is in the plane, a transverse electric mode, polarized in the plane, sees an index that depends on the direction of propagation. Along y, it is polarized along z and sees mostly the extraordinary index. Along z, it is polarized along y and sees the ordinary index." },
        { text: 'x-cut and y-cut are the same thing, because x and y are both ordinary axes', correct: false, why: "The x and y axes are both ordinary axes for the index, so for a simple index calculation they look alike. But lithium niobate is trigonal, point group three m, and the x and y axes are not equivalent for the electro-optic, piezoelectric, and photoelastic tensors. In a logged example, the effective photoelastic coefficient at the same in-plane angle differed between x-cut and y-cut. The cut also changes etch behavior and sidewall angles." }
      ],
      correctSay: "Correct. On x-cut, direction matters."
    },
    takeaways([
      'The cut names the axis normal to the wafer; on x-cut the optic axis lies in-plane',
      'Rotating the crystal equals rotating the coordinates by the opposite angle; state which one you mean',
      'Plus 45 and minus 45 degrees are not equivalent on x-cut lithium niobate',
      'Validate rotations at a non-special angle',
      'Label the cut and the frame on every model, figure, and chip',
      'Only the first layer sees the flat; check it at the aligner, every wafer'
    ]),
    ack('crystal orientation awareness'),
    refs([R.weis, R.ieee176, R.zelmon])
  ];

  /* ======================= CHAPTER 2 ======================= */

  var ch2 = {
    id: 'bends', title: 'Respecting Boundaries: Mode Conversion in Bends',
    subtitle: 'Anisotropy, crossings, and why gentler is not always safer',
    objectives: [
      'Compute the bulk index seen by TE and TM light as an x-cut waveguide turns',
      'Explain why TE and TM effective indices can cross at an intermediate angle',
      'Recognize that a larger bend radius can increase mode conversion on x-cut',
      'Report anisotropy-driven phase errors in curved and angled devices',
      'Verify launch polarization at the chip, not from memory'
    ],
    steps: []
  };
  ch2.steps = [
    objectives(ch2.title, ch2.objectives),
    {
      type: 'video', title: 'Cold open: The gentle bend', duration: 24,
      narration: [
        "Watch Gary's plan for his ring, and judge whether it is okay.",
        "Gary's x-cut ring has extra loss and a polarization problem.",
        "Gary has a plan. Make the bend bigger and gentler. Everyone knows gentle bends are safer.",
        "The new ring is enormous. The light goes in as T E and comes out mostly as T M.",
        "Is this okay? No. On x-cut, the crystal changes what the mode sees as the waveguide turns. " + M
      ],
      keyframes: [
        { t: 0, shot: 'Top-down orthographic camera over an x-cut chip on a probe station chuck. A small ring resonator glows faintly. A light pulse (blue particle trail) travels around the ring. Crystal axis gizmo in the corner: y to the right, z up.', caption: 'A ring on x-cut. Light goes around.', cast: [], draw: feDraw('gentle', 0) },
        { t: 6, shot: 'Medium shot of Gary at the layout workstation, dragging a ring radius handle outward with obvious satisfaction. The ring on the monitor grows to fill the screen. Sticky note on the monitor: GENTLE = SAFE.', caption: 'Make it bigger. Bigger is gentler.', cast: ['Gary'], draw: feDraw('gentle', 1) },
        { t: 12, shot: 'Top-down again on the big ring. The pulse changes color from blue to red at a fixed azimuth (about one quarter of the way around), marked by a striped toll-booth barrier labeled POLARIZATION CHECKPOINT.', caption: 'At one angle, TE meets TM.', cast: [], draw: feDraw('gentle', 2) },
        { t: 19, shot: 'Freeze on the red pulse exiting the bus waveguide. Stamp graphic: NOT OKAY.', caption: 'Is this okay?', cast: [], draw: feDraw('gentle', 3) }
      ]
    },
    {
      type: 'slide', layout: 'host', title: 'Let\'s take a closer look',
      bullets: [
        'x-cut: TM is polarized along x and sees the ordinary index at every in-plane angle',
        'TE is polarized in-plane; its bulk index moves between the extraordinary and ordinary values as the guide turns',
        'Index ellipse: $\\dfrac{1}{n_{\\mathrm{TE}}^2} = \\dfrac{\\cos^2\\theta}{n_e^2} + \\dfrac{\\sin^2\\theta}{n_o^2}$, $\\theta$ measured from the crystal y axis',
        'At 1550 nm: $n_o \\approx 2.211$, $n_e \\approx 2.138$ (Zelmon et al., 1997)'
      ],
      callout: 'Waveguide geometry lowers TE and TM effective indices by different amounts, which can create a crossing.',
      narration: [
        "Let's take a closer look.",
        "On x-cut, the transverse magnetic mode is polarized mostly along x, the film normal. That is an ordinary direction, so it sees the ordinary index at every in-plane angle.",
        "The transverse electric mode is polarized in the plane. When the waveguide runs along y, it is polarized along z and sees the extraordinary index. When the waveguide runs along z, it is polarized along y and sees the ordinary index.",
        "In between, the bulk index follows the index ellipse. One over n squared equals cosine squared theta over n e squared, plus sine squared theta over n o squared, with theta measured from the crystal y axis.",
        "At fifteen fifty nanometers, Zelmon and colleagues give an ordinary index of about two point two one one, and an extraordinary index of about two point one three eight. Lithium niobate is negative uniaxial.",
        "In bulk, the two curves only touch along z. In a real waveguide, geometry lowers the T E and T M effective indices by different amounts. Then the curves can cross at an intermediate angle, and a bend that sweeps through that angle passes a hybridization point.",
        "Watch your angles. " + M
      ]
    },
    {
      type: 'widget', title: 'Interactive: Rotate the x-cut die', widget: 'xcutIndex', params: {},
      instructions: 'Drag the propagation angle slider across a wide range and watch the effective indices of the two lowest modes, computed with a full-vector anisotropic mode solver for a real rib profile. The curve color shows how TE-like each mode is. Then find the angle where the two curves come closest; zoom in around the cursor if it helps, and watch the field maps trade polarization.',
      narration: [
        "In this activity, you will rotate a rib waveguide on an x-cut die. The curves are not a sketch. They come from a full vector mode solver with the complete rotated permittivity tensor, for an eight hundred nanometer film with a four hundred nanometer etch and slanted sidewalls.",
        "The color of each curve shows how T E like that mode is. Sweep the angle, then find where the two curves come closest. There, the modes stop being T E or T M, and the field maps trade places."
      ]
    },
    {
      type: 'reveal', title: 'What the literature reports',
      prompt: 'Open each finding to continue.',
      tabs: [
        { label: 'A crossing at about 25 degrees', body: '<p>Wang et al. (Nature, 2024) report that in x-cut lithium niobate the fundamental TE and TM modes cross at a propagation angle of about 25 degrees for the geometry they studied, so a bend that sweeps all angles passes a hybridization point.</p><p>Lithium tantalate of the same dimensions shows no crossing; its birefringence is much smaller.</p><p>In their simulation (figure), the angle is measured from the crystal y axis; the arrows mark the two lithium niobate crossings.</p>', narration: ["Wang and colleagues, in Nature in twenty twenty four, report that in x-cut lithium niobate, the fundamental T E and T M modes cross at a propagation angle of about twenty five degrees for the geometry they studied.", "Lithium tantalate with the same dimensions showed no crossing, because its birefringence is much smaller.", "The angle depends on geometry. The existence of a crossing is the point."], media: { src: 'media/lit/fe_wang2024_fig2g.png', alt: 'Simulated effective index of the fundamental TE and TM modes versus angle from the crystal y axis for x-cut lithium niobate (red) and lithium tantalate (blue); arrows mark the lithium niobate crossings', caption: 'Wang et al., Nature 2024, Fig. 2g.', href: 'https://doi.org/10.1038/s41586-024-07369-1' } },
        { label: 'Gentler can be worse', body: '<p>Liu et al. (Optics Express, 2024), citing earlier work, note that on x-cut thin-film lithium niobate, bends hybridize modes and larger bend radii can increase mode conversion and loss.</p><p>The earlier work, Wang et al. (IEEE Photonics Journal, 2020), explains it with the anticrossing: a slow passage is adiabatic, so the light follows the hybrid branch and leaves in the other polarization. As at any anticrossing, a fast passage can jump across the gap instead.</p><p>In their simulated 90 degree bends (figure), raising the radius to 100 µm did not suppress TE to TM coupling at 1.31 µm, where the modes hybridize, but did at 1.55 µm, where they do not.</p>', narration: ["Liu and colleagues, in Optics Express in twenty twenty four, citing earlier work, note that on x-cut thin film lithium niobate, larger bend radii can increase mode conversion and loss.", "That earlier work, by Wang and colleagues in twenty twenty, explains it with the anticrossing. A slow passage is adiabatic. The light follows the hybrid branch and leaves in the other polarization. As at any anticrossing, a fast passage can jump across instead.", "In their simulated bends, a bigger radius did not reduce the conversion at the wavelength where the modes hybridize.", "So gentle is not automatically safe. Respect the boundary, but also respect the crossing."], media: { src: 'media/lit/fe_wang2020_fig6bc.png', alt: 'Simulated transmission through a 90 degree bend versus radius on oxide-clad x-cut thin-film lithium niobate at 1.31 and 1.55 micrometers: TE to TE, TE to TM and total', caption: 'Wang et al., IEEE Photonics J. 2020, Fig. 6(b,c).', href: 'https://doi.org/10.1109/JPHOT.2020.2995317' } },
        { label: 'Sidewalls are involved', body: '<p>Liu et al. (Light: Advanced Manufacturing, 2023) write that on x-cut, non-negligible hybridization might exist in sharp bends and adiabatic tapers, owing to lithium niobate anisotropy and angled sidewalls.</p><p>Etched lithium niobate ridges have slanted sidewalls; in their chip, about 30 degrees from vertical (cross section in the figure). Coupling the fundamental TE and TM modes needs both mirror symmetries of the cross section broken. Up-down is already broken by substrate and cladding, and the slant adds to it; left-right is broken by the off-diagonal permittivity of a rotated x-cut guide, or by a bend.</p>', narration: ["Liu and colleagues, in twenty twenty three, write that hybridization might exist in sharp bends and tapers, owing to anisotropy and angled sidewalls.", "Coupling the fundamental T E and T M modes needs both mirror symmetries of the cross section broken. Substrate, cladding and a slanted sidewall break up and down. A rotated crystal axis or a bend breaks left and right.", "Trust no symmetry plane, especially one your etch did not give you."], media: { src: 'media/lit/fe_liu2023_fig3e.jpg', alt: 'Electron micrograph of the cross section of etched lithium niobate ridges with slanted sidewalls, etch depths of about 200 and 180 nanometers', caption: 'Liu et al., Light Adv. Manuf. 2023, Fig. 3e.', href: 'https://doi.org/10.37188/lam.2023.013' } },
        { label: 'Arrays of angles', body: '<p>Arrayed waveguide gratings route arms at different angles. On x-cut, each angle sees a different index, which adds phase errors. Xie et al. (Optics Express, 2026) identify this in-plane anisotropy as an obstacle for narrow-spacing arrayed waveguide gratings on the platform.</p><p>Their fix (figure) folds the grating so that the length steps between arms sit in straight sections that all run in one direction, with loop reflectors sending the light back.</p>', narration: ["Any device with many arms at different angles inherits this problem.", "Xie and colleagues, in twenty twenty six, identify in-plane anisotropy as a source of phase errors in arrayed waveguide gratings on x-cut thin film lithium niobate.", "Every arm has an angle. Every angle has an index. Know your axes."], media: { src: 'media/lit/fe_xie2026_awg_folded.svg', alt: 'Schematic: conventional arrayed waveguide grating with arms turning through many angles, and a folded design with length steps in parallel straight sections ending in loop reflectors', caption: 'Xie et al., Opt. Express 2026, Fig. 1. Redrawn, illustrative.', href: 'https://doi.org/10.1364/OE.588499' } }
      ]
    },
    {
      type: 'chat', title: 'Scenario 1: The bigger ring', app: 'call',
      meta: { header: 'Weekly sync (video call)', participants: ['Gary', 'Manager', 'Dana'] },
      messages: [
        {"from": "Manager", "text": "Gary, update on the ring loss?", "side": "left"},
        {"from": "Gary", "text": "Tripled the radius. The bends are gentler now, so the bend loss should come down.", "side": "right"},
        {"from": "Dana", "text": "Did you check the TE and TM effective indices versus angle for our film and etch? If they cross, a bigger radius can convert more, not less.", "side": "left"},
        {"from": "Gary", "text": "The ring is rotationally symmetric, so I assumed the loss is the same at every angle.", "side": "right"},
        {"from": "Dana", "text": "On x-cut, every angle is different. That is the whole problem.", "side": "left"},
        {"from": "Manager", "text": "Can we get the angle sweep by Friday?", "side": "left"}
      ],
      narration: [
        "Watch this weekly sync, and judge whether Gary's fix is proper.",
        "Gary tripled the ring radius to reduce bend loss on x-cut. Is this okay?",
        "Not without checking the crossing. A round ring sweeps every propagation angle, and on x-cut every angle is different.",
        "Scan the angles. " + M
      ]
    },
    {
      type: 'chat', title: 'Scenario 2: The polarization blame', app: 'phone',
      meta: { header: 'Gary', participants: ['Gary', 'Manager'] },
      messages: [
        {"from": "Gary", "text": "output has a TM component. first suspect was the input fiber, it is an old patch cord", "side": "left"},
        {"from": "Manager", "text": "so swap the cord?", "side": "right"},
        {"from": "Gary", "text": "checked the input first, twice, with a polarizer. clean TE. the cord is cleared", "side": "left"},
        {"from": "Manager", "text": "then what is it", "side": "right"},
        {"from": "Gary", "text": "made on chip. on x-cut a bend through the TE-TM crossing angle or a taper with slanted sidewalls can convert. checking the bend angles and the taper next", "side": "left"}
      ],
      narration: [
        "Read these messages, and judge whether Gary's troubleshooting is proper.",
        "Gary checks his input before blaming it, then looks for the T M on chip. Is this okay?",
        "Yes. On x-cut, a bend through the crossing angle or a taper with slanted sidewalls can convert T E to T M on chip.",
        "The fiber is innocent until proven guilty, and Gary checked before accusing it."
      ]
    },
    {
      type: 'chat', title: 'Scenario 3: The routing shortcut', app: 'thread',
      meta: { header: '#layout-review', participants: ['Gary', 'Dana'] },
      messages: [
        { from: 'Gary', text: 'routed the two arms of the interferometer at different angles to save area. on x-cut the TE index depends on the angle, so matched length is not matched phase', side: 'left' },
        { from: 'Gary', text: 'computed the TE effective index versus angle for our guide and set the arm lengths for equal phase at 1550. plot attached', side: 'left' },
        { from: 'Dana', text: 'equal phase at one wavelength. the group index changes with angle too, so the balance moves across the band', side: 'right' },
        { from: 'Gary', text: 'swept it. the residual imbalance over the band is in the second plot, and the arm angles are on the layout', side: 'left' },
        { from: 'Dana', text: 'good. that is the review I wanted', side: 'right' }
      ],
      narration: [
        "Read this layout review thread, and judge whether Gary's routing is proper.",
        "Gary routed two interferometer arms at different angles on x-cut, to save area. Is this okay?",
        "Yes. Gary did not assume that matched length means matched phase. In bulk, the T E index moves by about zero point zero seven between the extraordinary and ordinary values. Gary computed it for his waveguide.",
        "Know your axes."
      ]
    },
    {
      type: 'check',
      question: 'On x-cut lithium niobate, which polarization sees an index that changes with in-plane propagation angle?',
      options: [
        { text: 'TE, polarized in the plane, because its field direction rotates between the z and y crystal axes as the guide turns', correct: true, why: '' },
        { text: 'TM, because it is the one polarized along the optic axis', correct: false, why: "On x-cut, the TM mode is polarized mainly along the film normal, which is the crystal x axis. The x axis is an ordinary direction, so TM sees the ordinary index at every in-plane propagation angle. The optic axis, z, lies in the plane. It is the TE mode, polarized in the plane, whose field direction rotates between z and y as the waveguide turns, so it is TE that moves between the extraordinary and ordinary index." },
        { text: 'Neither. Uniaxial crystals are isotropic in the plane', correct: false, why: "A uniaxial crystal is isotropic only in the plane perpendicular to its optic axis. That is the case for z-cut, where the optic axis is normal to the film. On x-cut, the optic axis lies in the film, so the plane of the film is not isotropic. The TE mode sees the extraordinary index when it propagates along y and the ordinary index when it propagates along z." }
      ],
      correctSay: "That's the one. TE turns, the index turns."
    },
    {
      type: 'check',
      question: 'Gary\'s x-cut ring shows TE to TM conversion. He proposes doubling the bend radius. What should a compliant engineer say?',
      options: [
        { text: 'Check whether TE and TM effective indices cross at some propagation angle for this geometry before changing the radius; on x-cut, a larger bend radius has been reported to increase conversion and loss', correct: true, why: '' },
        { text: 'Good idea. Larger radius always reduces conversion and loss', correct: false, why: "On isotropic platforms, larger bend radius usually means lower radiation loss and less mode mixing. On x-cut lithium niobate, the effective indices of TE and TM depend on propagation angle, and for some geometries they cross. Near a crossing, slanted sidewalls and bending couple the two. Liu and colleagues, in Optics Express in twenty twenty four, citing earlier work, note that larger bend radii can increase mode conversion and loss on x-cut thin film lithium niobate. The earlier work explains it: a slower passage through the crossing is more adiabatic, so the light follows the hybridized mode and converts. So check the crossing first." },
        { text: 'Switch to a fiber with better polarization maintenance', correct: false, why: "If the input is already clean TE, the conversion happens on the chip, not in the fiber. On x-cut, a bend that sweeps through the angle where TE and TM effective indices cross, combined with slanted sidewalls, can convert polarization on chip. Changing the fiber does not change the crystal." }
      ],
      correctSay: "Correct. Check the crossing before you change the radius."
    },
    takeaways([
      'On x-cut, TM sees the ordinary index at every in-plane angle; TE moves between the extraordinary and ordinary index',
      'Waveguide geometry can make TE and TM effective indices cross at an intermediate angle',
      'A larger bend radius can increase TE to TM conversion on x-cut',
      'Matched length at different angles is not matched phase',
      'TM on x-cut uses a smaller electro-optic coefficient: about three times the V pi; verify the launch with a polarizer'
    ], true),
    ack('mode conversion in bends'),
    refs([R.zelmon, R.wangLT, R.liu2024, R.wangPJ2020, R.liu2023, R.xie2026, R.weis])
  ];

  /* ======================= CHAPTER 3 ======================= */

  var ch3 = {
    id: 'photorefractive', title: 'Photorefractive Conduct',
    subtitle: 'When the light you send writes back',
    objectives: [
      'Distinguish the photorefractive blue shift from the thermal red shift of lithium niobate resonances',
      'Choose scan direction and power so a measured linewidth is not artificially narrow',
      'Recognize cladding and process choices that worsen photorefraction',
      'Understand that optical damage in lithium niobate is usually reversible, and usually not damage'
    ],
    steps: []
  };
  ch3.steps = [
    objectives(ch3.title, ch3.objectives),
    {
      type: 'video', title: 'Cold open: The record Q', duration: 26,
      narration: [
        "Watch this measurement, and judge whether Gary's result is okay.",
        "Gary measures a lithium niobate ring. He sweeps the laser once, in one direction, at a comfortable power.",
        "The resonance dip is beautifully narrow. Gary calculates a record quality factor and sends it to his manager.",
        "Dana repeats the sweep in the other direction. The dip is much wider.",
        "Is this okay? No. One direction is not a measurement. Scan both ways. " + M
      ],
      keyframes: [
        { t: 0, shot: 'Close-up of a tunable laser front panel and an oscilloscope in a dim optics lab. Gary presses SWEEP once. A single trace appears with a deep, narrow dip.', caption: 'One sweep. One direction.', cast: ['Gary'], draw: feDraw('recordq', 0) },
        { t: 7, shot: 'Over-the-shoulder on Gary typing in a chat window: RECORD Q. Manager in a small video-call tile gives two thumbs up. Confetti particles fall only inside the manager tile.', caption: 'A record quality factor.', cast: ['Gary', 'Manager'], draw: feDraw('recordq', 1) },
        { t: 14, shot: 'Dana leans in, presses REVERSE on the laser. Scope shows the same resonance now broad and lopsided. Split-screen comparison of the two traces.', caption: 'The other direction.', cast: ['Dana'], draw: feDraw('recordq', 2) },
        { t: 21, shot: 'Freeze on the split screen. Stamp: NOT OKAY.', caption: 'Is this okay?', cast: [], draw: feDraw('recordq', 3) }
      ]
    },
    {
      type: 'slide', layout: 'host', title: 'Let\'s take a closer look',
      bullets: [
        'Photorefraction: light excites carriers, they drift and get trapped, the space-charge field changes the index through the electro-optic effect',
        'In lithium niobate resonators it blue-shifts resonances with power',
        'Absorptive heating red-shifts them, on a different time scale',
        'Scanning against the shift snaps the dip narrow; scanning with it drags the dip wide'
      ],
      callout: 'Zero tolerance for unscanned directions.',
      narration: [
        "Let's take a closer look.",
        "In the photorefractive effect, light excites carriers. They move, get trapped, and build a space-charge field. Through the electro-optic effect, that field changes the refractive index.",
        "In lithium niobate resonators, this usually shifts the resonance to shorter wavelength as power increases. That is a blue shift.",
        "Absorptive heating does the opposite. It shifts the resonance to longer wavelength. That is a red shift. And it has a different time constant.",
        "When the resonance moves away from the laser as you sweep, it gets dragged along, and the dip looks wide. When it moves toward the laser, it snaps past, and the dip looks narrow.",
        "A narrow dip from the wrong direction inflates your quality factor. Luke and colleagues, in twenty twenty, scanned from long to short wavelength at under twenty microwatts to follow the blue shift.",
        "Scan both ways. Watch your charge. " + M
      ]
    },
    {
      type: 'widget', title: 'Interactive: Resonance sweep', widget: 'prSweep', params: {},
      instructions: 'Pick a mechanism and a power, then sweep in both directions. Change the sweep time by at least a factor of 100 and sweep both ways again. Compare each trace with the dashed cold resonance, and watch the shift history below the plot.',
      narration: ["In this activity, you will sweep a laser across a ring resonance in both directions.", "The thermal shift responds in about ten microseconds. The photorefractive shift builds and relaxes over tens of milliseconds to tens of seconds, and the crystal remembers the previous sweep.", "Sweep both ways, then sweep again much faster or much slower. Two directions that agree are not yet proof of a cold resonance."]
    },
    {
      type: 'reveal', title: 'Photorefractive conduct in the literature',
      prompt: 'Open each finding to continue.',
      tabs: [
        { label: 'Cladding', body: '<p>Xu et al. (Optics Express, 2021): the standard oxide top cladding dramatically worsens the photorefractive effect in thin-film lithium niobate microrings. Removing it, plus heat treatment, suppressed the photorefraction-induced relative resonance shift by more than two orders of magnitude.</p><p>In their process sequence (figure), stripping the oxide alone left the shift even larger; it fell below the detection limit only after the uncladded chip was re-annealed.</p><p>Shen et al. (Nature Photonics, 2024, supplement), citing that PECVD oxide cladding causes significant photorefraction, used spin-on, annealed HSQ cladding instead: in their HSQ-clad rings the photorefractive shift was not noticeable up to 0 dBm in the bus waveguide, a weak effect comparable with that of air-clad devices.</p>', narration: ["Xu and colleagues, in twenty twenty one, found that the standard oxide top cladding dramatically worsens photorefraction in thin film lithium niobate rings. Removing it, with heat treatment, suppressed the relative resonance shift by more than two orders of magnitude.", "Shen and colleagues, citing that plasma deposited oxide makes it worse, used annealed H S Q cladding instead. Their rings showed a weak photorefractive effect, comparable with air-clad devices.", "Cladding is not just an optical choice. It is a charge choice. Watch your charge."], media: { src: 'media/lit/fe_xu2021_fig5.png', alt: 'Maximum photorefractive frequency shift versus intracavity power after oxide coating, annealing, cladding removal and re-annealing; after re-annealing the shift is below the detection limit', caption: 'Xu et al., arXiv 2020, Fig. 5.', href: 'https://doi.org/10.48550/arXiv.2012.12671' } },
        { label: 'Small cavities, big shifts', body: '<p>Liang et al. (Optica, 2017): in lithium niobate photonic crystal nanocavities, the photorefractive tuning rate reached about 84 MHz per photon. In the same paper (figure), the blue shift saturated as power rose, was then quenched while the power was held at 41 µW, and came back after a few hours in the dark.</p><p>Li et al. (Laser and Photonics Reviews, 2019): at higher input power (2 mW), photorefraction was observed to quench in photonic crystal cavities. More power does not always mean more shift.</p>', narration: ["In small mode volumes, the effect is dramatic. Liang and colleagues reported a photorefractive tuning rate of about eighty four megahertz per photon in lithium niobate photonic crystal nanocavities.", "And it is not monotonic. Li and colleagues observed photorefraction quenching at higher input power in photonic crystal cavities.", "So a single power is not a characterization. Scan the power too."], media: { src: 'media/lit/fe_liang2017_sequence.svg', alt: 'Step diagram: resonance blue shift of about 55 picometers as input power rises from 330 nanowatts to 8 microwatts, saturation at 41 microwatts, return to the passive position after about 10 minutes at 41 microwatts, quenched afterwards, recovering after hours in the dark', caption: 'Liang et al., Optica 2017. Redrawn, illustrative.', href: 'https://doi.org/10.1364/OPTICA.4.001251' } },
        { label: 'Measuring loss honestly', body: '<p>Shams-Ansari et al. (APL Photonics, 2022) measured material loss with a pump-probe method on time scales shorter than the photorefractive response, to separate intrinsic loss from photorefractive artifacts.</p><p>First they timed the charge (figure): after the pump was switched off, the PECVD-oxide-clad sample A drifted with a time constant of about 100 seconds, while the annealed samples B and C showed no photorefractive drift within a minute.</p>', narration: ["If you want the true material loss, you have to outrun the charge.", "Shams-Ansari and colleagues used a pump-probe measurement faster than the photorefractive response to separate loss from photorefractive artifacts.", "Scan both ways, and when that is not enough, scan faster than the crystal can respond."], media: { src: 'media/lit/fe_shams2022_fig2.png', alt: 'Pump-probe setup and probe resonance shift versus time after the pump is switched off for three samples; the oxide-clad sample drifts over tens of seconds, the annealed samples do not', caption: 'Shams-Ansari et al., arXiv 2022, Fig. 2.', href: 'https://doi.org/10.48550/arXiv.2203.17133' } },
        { label: '"Damage" is a misnomer', body: '<p>Photorefractive optical damage is a reversible index change from trapped space charge, not physical damage. Light or heat can erase it.</p><p>In nonlinear devices it still matters: Lu et al. (Optica, 2019) attribute the early saturation of second-harmonic conversion in a poled microring to photorefraction, which shifts the pump and second-harmonic resonances by different amounts and creates a mismatch.</p><p>Zhang et al. (arXiv, 2026) even write a photorefractive grating on purpose to couple TE and TM modes, with hour-long lifetimes. Your chip remembers last Tuesday.</p><p>In their data (figure), the doublet splitting grows with each step of write power, then relaxes with a time constant of about 65 minutes once the pump is off, leaving a residual splitting.</p>', narration: ["The term optical damage is misleading. Photorefraction is usually a reversible index change from trapped charge. Light or heat can erase it.", "It still breaks things. In a poled microring, Lu and colleagues attribute an early saturation of the second harmonic to photorefraction, which shifts the pump and second harmonic resonances by different amounts.", "And it lasts. Zhang and colleagues wrote photorefractive gratings on purpose, with lifetimes of hours. Your chip remembers last Tuesday's experiment.", "Watch your charge."], media: { src: 'media/lit/fe_zhang2026_fig2de.png', alt: 'Transmission maps of a hybrid-mode doublet: the splitting grows with each step of write power, then relaxes over two hours after the pump is switched off', caption: 'Zhang et al., arXiv 2026, Fig. 2(d,e).', href: 'https://doi.org/10.48550/arXiv.2606.06637' } }
      ]
    },
    {
      type: 'chat', title: 'Scenario 1: The record', app: 'phone',
      meta: { header: 'Manager', participants: ['Gary', 'Manager'] },
      messages: [
        {"from": "Gary", "text": "Q on the new ring looks like a record. linewidth is tiny", "side": "right"},
        {"from": "Manager", "text": "great. which direction did you sweep", "side": "left"},
        {"from": "Gary", "text": "short to long. the other direction looks distorted, so I used the clean one", "side": "right"},
        {"from": "Manager", "text": "and what power", "side": "left"},
        {"from": "Gary", "text": "the usual 1 mW", "side": "right"}
      ],
      narration: [
        "Read these texts, and judge whether Gary's report is proper.",
        "Gary reports a record linewidth from the sweep direction that looked cleaner. Is this okay?",
        "No. A dip that snaps narrow in one direction is a property of your scan, not your resonator.",
        "Sweep both ways. Reduce power until the two directions agree. Report the direction and the power. " + M
      ]
    },
    {
      type: 'chat', title: 'Scenario 2: Protective cladding', app: 'email',
      meta: {"header": "Inbox", "subject": "Process change: adding oxide cladding to all rings", "participants": ["Gary", "Manager", "Dana"]},
      messages: [
        {"from": "Gary", "text": "Hi all,\n\nTo protect the devices, I added a deposited oxide top cladding to the whole wafer in the last run. It is standard practice on our other platforms.\n\nGary", "side": "left", "to": ["Manager", "Dana Okafor"], "date": "2026-11-05 13:10", "subject": "Process change: adding oxide cladding to all rings"},
        {"from": "Dana", "text": "Hi Gary,\n\nFor lithium niobate, deposited oxide cladding has been reported to make photorefraction much worse. Xu et al. 2021 saw more than two orders of magnitude less shift after removing it and heat treating. Paper attached.\n\nDana", "side": "right", "to": ["Gary"], "cc": ["Manager"], "date": "2026-11-05 14:02", "subject": "RE: Process change: adding oxide cladding to all rings", "attachments": [{"name": "xu2021_photorefraction_cladding.pdf", "size": "2.3 MB"}]},
        {"from": "Manager", "text": "So is the cladding protective or not?", "side": "left", "to": ["Dana Okafor", "Gary"], "date": "2026-11-05 14:20", "subject": "RE: RE: Process change: adding oxide cladding to all rings"},
        {"from": "Dana", "text": "It protects the surface. It does not protect the resonance. Could we measure both cladded and uncladded die, at several powers, both scan directions, before making it the default?", "side": "right", "to": ["Manager", "Gary"], "date": "2026-11-05 14:31", "subject": "RE: RE: Process change: adding oxide cladding to all rings"}
      ],
      narration: [
        "Read this email thread, and judge whether Gary's change is okay.",
        "Gary added deposited oxide cladding to every ring, by habit. Is this okay?",
        "Only after you measure it. In thin film lithium niobate, cladding choice is a photorefraction choice.",
        "Watch your charge."
      ]
    },
    {
      type: 'chat', title: 'Scenario 3: The lock that ran away', app: 'thread',
      meta: { header: '#lab-ops', participants: ['Gary', 'Dana', 'Manager'] },
      messages: [
        {"from": "Gary", "text": "thermal lock on the ring ran away overnight. I had set the lock side for a red shift, same as on silicon nitride", "side": "left"},
        {"from": "Gary", "text": "in LN there is also a photorefractive blue shift, slower than the thermal one. which wins depends on power and time, so I am measuring the shift sign and time constant at our power before I pick a side", "side": "left"},
        {"from": "Manager", "text": "can we lock on whichever side is stable today and characterize later", "side": "left"},
        {"from": "Gary", "text": "stable today is how it ran away last night. the measurement is one afternoon. I will log which side and why", "side": "left"},
        {"from": "Dana", "text": "agree. sweep both directions while you are at it", "side": "right"}
      ],
      narration: [
        "Read this lab operations thread, and judge whether Gary's response is proper.",
        "Gary's lock ran away overnight, and he chooses to measure before he relocks. Is this okay?",
        "Yes. Gary dropped a habit from other platforms and measured first. In lithium niobate, a blue photorefractive shift and a red thermal shift compete, with different time constants.",
        "Measure first. Scan both ways. " + M
      ]
    },
    {
      type: 'check',
      question: 'In a lithium niobate microring at moderate power, the photorefractive effect typically shifts the resonance in which direction, and what is the measurement consequence?',
      options: [
        { text: 'To shorter wavelength (blue shift); sweeping against the shift can make the dip look artificially narrow, so sweep both ways and at low power', correct: true, why: '' },
        { text: 'To longer wavelength (red shift), the same as heating, so nothing changes', correct: false, why: "Heating from absorption shifts the resonance to longer wavelength, a red shift. But in lithium niobate, the photorefractive effect usually shifts it to shorter wavelength, a blue shift, and on a different, usually slower, time scale. Because the two have opposite signs, habits from silicon or silicon nitride, where only the thermal red shift matters, do not transfer. Luke and colleagues swept from long to short wavelength, at very low power, to follow the blue shift." },
        { text: 'It does not shift; photorefraction only causes permanent damage', correct: false, why: "Despite the historical name optical damage, photorefraction is mostly a reversible index change. Light excites carriers, they get trapped, and the space-charge field changes the index through the electro-optic effect. That index change shifts the resonance, usually toward shorter wavelength in lithium niobate rings. Light or heat can erase it. So it shows up as a shift, a distortion of the line shape, and a dependence on scan direction." }
      ],
      correctSay: "Blue shift, and scan both ways."
    },
    {
      type: 'check',
      question: 'Gary wants to report the intrinsic loss of a new lithium niobate waveguide. Which approach is most compliant?',
      options: [
        { text: 'Sweep both directions at decreasing power until the linewidths agree, report direction and power, and consider a measurement faster than the photorefractive response', correct: true, why: '' },
        { text: 'Report the narrowest linewidth he ever observed', correct: false, why: "The narrowest observed linewidth is very often the one where the resonance snapped toward the laser during the sweep. That happens when you scan against the direction of a power-dependent shift. It inflates the quality factor and understates the loss. A compliant measurement sweeps both directions, reduces power until the directions agree, and reports both. When that is not enough, a pump-probe method faster than the photorefractive response, as Shams-Ansari and colleagues used, separates loss from charge effects." },
        { text: 'Use high power so the dip is easier to see', correct: false, why: "Higher power makes both the thermal and photorefractive shifts larger, so the line shape distorts more, and the apparent linewidth depends more on scan direction. In some small cavities, the behavior is not even monotonic in power. Lower power, both directions, and a stated power and direction give a measurement others can trust." }
      ],
      correctSay: "Excellent. Both directions, low power, stated clearly."
    },
    takeaways([
      'Photorefraction usually blue-shifts lithium niobate resonances; heating red-shifts them',
      'Sweep both directions and lower the power until they agree; report both',
      'Deposited oxide cladding can make photorefraction much worse',
      'Optical damage is mostly reversible charge, and it remembers'
    ], true),
    ack('photorefractive conduct'),
    refs([R.luke2020, R.xu2021, R.shen2024, R.liang2017, R.li2019, R.shams2022, R.lu2019, R.zhang2026])
  ];

  /* ======================= CHAPTER 4 ======================= */

  var ch4 = {
    id: 'pyroelectric', title: 'Pyroelectric Awareness',
    subtitle: 'Temperature change is charge',
    objectives: [
      'Estimate the unscreened pyroelectric field from a temperature change',
      'Identify fab, packaging, and cryogenic steps where temperature changes build charge',
      'Explain why z-cut and x-cut put the charge in different places',
      'Provide every crystal with a safe discharge path',
      'Never connect electronics to a chip that has just changed temperature with floating pads'
    ],
    steps: []
  };
  ch4.steps = [
    objectives(ch4.title, ch4.objectives),
    {
      type: 'video', title: 'Cold open: The hotplate', duration: 26,
      narration: [
        "Watch this routine bake, and judge whether it is okay.",
        "Gary puts a z-cut lithium niobate wafer on a hotplate to bake his resist.",
        "Gary did not apply any voltage. Gary is not worried about voltage.",
        "When the wafer comes off the hotplate, small sparks dance along the surface.",
        "Is this okay? No. Temperature change is charge. Watch your charge. " + M
      ],
      keyframes: [
        { t: 0, shot: 'Medium shot, cleanroom wet bench with a hotplate. Gary places a 100 mm z-cut wafer onto the hotplate with wafer tweezers. Hotplate display reads a bake temperature. Camera dolly left to right.', caption: 'A routine bake.', cast: ['Gary'], draw: feDraw('hotplate', 0) },
        { t: 7, shot: 'Close-up of the wafer surface: plus signs fade in on the top face and minus signs on the bottom face (graphic overlay), growing in number as the temperature rises.', caption: 'No voltage applied. Charge appears anyway.', cast: [], draw: feDraw('hotplate', 1) },
        { t: 14, shot: 'Gary lifts the wafer off the hotplate onto a cool metal surface. Small blue arcs flicker between the wafer edge and the tweezers. Gary does not notice.', caption: 'Cooling down. Sparks.', cast: ['Gary'], draw: feDraw('hotplate', 2) },
        { t: 21, shot: 'Freeze frame on an arc. Stamp: NOT OKAY. Then cut to the corrected take: same wafer wrapped in aluminum foil, slow ramp, no arcs, green stamp OKAY.', caption: 'Is this okay?', cast: ['Gary'], draw: feDraw('hotplate', 3) }
      ]
    },
    {
      type: 'slide', layout: 'host', title: 'Let\'s take a closer look',
      bullets: [
        'Pyroelectric: spontaneous polarization changes with temperature; bound surface charge appears on faces normal to z',
        'Unscreened field: $E = p\\,\\Delta T / (\\varepsilon_0 \\varepsilon_{33})$',
        'z-cut: charge on top and bottom faces, field overlaps the device; x-cut: charge on edges and sidewalls',
        'Screening by surface conduction and ambient takes time; until then, the field is real'
      ],
      callout: 'Unwelcome charge accumulation is never okay.',
      narration: [
        "Let's take a closer look.",
        "Lithium niobate has a spontaneous polarization along z. When its temperature changes, that polarization changes, and bound charge appears on faces normal to z.",
        "If nothing screens it, the field is p times delta T, divided by epsilon zero times epsilon three three.",
        "On z-cut, the charge sits on the top and bottom faces, right where your devices and resist are. On x-cut, it sits on the chip edges and sidewalls.",
        "Surface conduction and the environment eventually screen it. But eventually is not a process step.",
        "Watch your charge. " + M
      ]
    },
    {
      type: 'widget', title: 'Interactive: Pyroelectric field estimator', widget: 'pyroCalc', params: {},
      instructions: 'Set how far the hot plate is above room temperature, then change the ramp rate. Compare the peak field with the coercive field, and the top-face voltage with the spark threshold of air. Then try the thin film and the x-cut wafer.',
      narration: ["In this activity, you will heat a lithium niobate wafer on a grounded hot plate, hold it, and cool it down again.", "Move the temperature, then the ramp rate. A slow ramp only helps if the charge can leak away while you ramp.", "Then compare a z-cut wafer, a z-cut thin film, and an x-cut wafer, and compare the field with the field needed to flip domains."]
    },
    {
      type: 'reveal', title: 'Where temperature change happens at work',
      prompt: 'Open each workplace location to continue.',
      tabs: [
        { label: 'Hotplates and bakes', body: '<p>Bakes and cooldowns of z-cut wafers build large surface fields. Practitioners report fried devices and cracked substrates, and use conductive coatings or an aluminum-foil discharge path during processing.</p><p>Practitioners have also reported that heating z-cut lithium niobate exposed electron-beam resist without any beam (practitioner report, not independently verified).</p><p>Ramp slowly. Ground the wafer. Wrap it.</p>', narration: ["Every bake is a temperature change, and every temperature change is charge.", "Practitioners report fried devices and cracked substrates after hotplate steps, and use conductive coatings or aluminum foil as a discharge path.", "Some practitioners even report that heating a z-cut wafer exposed electron beam resist, with no beam involved. Treat that one as a practitioner report.", "Ramp slowly. Ground the wafer. Wrap it in foil. This is not a joke. This is the procedure."] },
        { label: 'Cryostats', body: '<p>Cooling to 4 kelvin is a temperature change of nearly 300 kelvin. The pyroelectric coefficient itself shrinks as the crystal cools, so the charge is less than the room-temperature coefficient times 296 kelvin, but it is not small.</p><p>Field notes from one lab: during a cooldown, the field from a lithium niobate chip discharged into a device on a different chip. Shielding electrodes saved those devices; transducers on the lithium niobate chip still failed.</p><p>The same lab saw released x-cut membranes collapse during cooldown. The suspected cause, flagged as a hypothesis, was pyroelectric charge plus electrostatic pull-in. Metal straps shorting the film to the substrate stopped it.</p>', narration: ["Cooling to four kelvin is a very large temperature change.", "In one lab's experience, during a cooldown, the field from a lithium niobate chip discharged into a device on a different chip nearby. Shielding electrodes saved those devices.", "The same lab saw released x-cut membranes collapse during cooldown. Their hypothesis, and it is labeled as a hypothesis, was pyroelectric charge plus electrostatic pull-in. Grounding straps from the film to the substrate stopped it.", "Your neighbor's chip is part of your workplace. Watch your charge."] },
        { label: 'Measurements', body: '<p>Bulmer et al. (Applied Physics Letters, 1986) reported pyroelectric instabilities in titanium-indiffused z-cut interferometers, while x-cut interferometers were stable.</p><p>Thermal tuning data can include a pyroelectric contribution. Luo et al. (Optica, 2018) attributed a measured second-harmonic thermal tuning slope larger than simulated to pyroelectric and thermal-expansion effects not in the model. Weigel et al. (Optics Express, 2018) list pyroelectric charge build-up among the concerns of etching lithium niobate.</p>', narration: ["In devices, pyroelectric charge looks like drift.", "Bulmer and colleagues, in nineteen eighty six, reported pyroelectric instabilities in z-cut interferometers, while x-cut interferometers were stable.", "So if your bias point follows the air conditioner, do not only blame the bias electronics. Watch your charge."] },
        { label: 'An extreme case', body: '<p>Naranjo, Gimzewski and Putterman (Nature, 2005) warmed a lithium tantalate crystal and used the resulting pyroelectric field, focused by a tungsten tip, to ionize and accelerate deuterium ions enough to drive nuclear fusion.</p><p>Your hotplate is not a fusion reactor. Please do not test this.</p>', narration: ["And here is how far this can go.", "Naranjo, Gimzewski, and Putterman, in Nature in two thousand five, warmed a lithium tantalate crystal and used its pyroelectric field to accelerate deuterium ions and drive nuclear fusion.", "Your hotplate is not a fusion reactor. Please treat it with the same respect anyway."] },
        { label: 'Heat writes domains', body: '<p>Xu et al. (Nature, 2022) wrote ferroelectric nanodomains in lithium niobate with femtosecond laser pulses, attributing the switching to the thermoelectric field from local heating rather than the optical field.</p><p>Local heating is not just heating. In a ferroelectric, it can be poling.</p>', narration: ["Xu and colleagues, in Nature in twenty twenty two, wrote ferroelectric domains in lithium niobate with femtosecond laser pulses, attributing the switching to the field from local heating, not the optical field itself.", "In a ferroelectric, heating can be poling. Watch your charge."] }
      ]
    },
    {
      type: 'chat', title: 'Scenario 1: The faster bake', app: 'phone',
      meta: { header: 'Gary', participants: ['Gary', 'Dana'] },
      messages: [
        {"from": "Gary", "text": "changed the bake recipe from slow ramp to place and remove. same final temperature, saves 20 min per wafer", "side": "left"},
        {"from": "Dana", "text": "which cut", "side": "right"},
        {"from": "Gary", "text": "z", "side": "left"},
        {"from": "Dana", "text": "that slow ramp is there for the pyroelectric charge. a fast temperature step leaves no time to screen", "side": "right"},
        {"from": "Gary", "text": "the resist does not care about the ramp, and nothing is biased during the bake", "side": "left"},
        {"from": "Dana", "text": "the wafer biases itself", "side": "right"}
      ],
      narration: [
        "Read these texts, and judge whether Gary's recipe change is proper.",
        "Gary removed a slow ramp to save time on z-cut wafers. Is this okay?",
        "No. A fast temperature step leaves no time for the charge to screen. Slow ramps and discharge paths exist for a reason.",
        "Watch your charge. " + M
      ]
    },
    {
      type: 'chat', title: 'Scenario 2: Kitchen foil', app: 'thread',
      meta: { header: '#cleanroom', participants: ['Gary', 'Manager', 'Dana'] },
      messages: [
        {"from": "Manager", "text": "why is there aluminum foil in the procurement request", "side": "left"},
        {"from": "Dana", "text": "discharge path for lithium niobate during bakes. it is in the procedure", "side": "right"},
        {"from": "Gary", "text": "I assumed it was for wrapping parts", "side": "left"},
        {"from": "Manager", "text": "approved. please label it", "side": "left"}
      ],
      narration: [
        "Read this cleanroom thread, and judge whether the purchase is okay.",
        "The foil in the cleanroom is a discharge path. Is this okay?",
        "Yes. This one is okay. Please label the foil.",
        "Watch your charge."
      ]
    },
    {
      type: 'chat', title: 'Scenario 3: The neighbor', app: 'email',
      meta: {"header": "Inbox", "subject": "Cryostat run 42: shared stage with a superconducting chip", "participants": ["Gary", "Manager", "Dana"]},
      messages: [
        {"from": "Gary", "text": "Hi all,\n\nCryostat run 42 plan: my lithium niobate chip shares the stage with a superconducting chip from the next group. A cooldown is a temperature change of almost 300 kelvin, and my chip does not need a bias to charge.\n\nSo: grounded shielding electrodes between the chips, every electrode and the backside metal of my chip grounded, and a slow ramp. The ramp only helps if the charge can leak away while it cools, which gets slower as it gets colder, so the shields and grounds do the real work. I have walked the other chip's owner through it, and they agreed.\n\nGary", "side": "left", "to": ["Manager", "Dana Okafor"], "date": "2026-11-12 18:22", "subject": "Cryostat run 42: shared stage with a superconducting chip", "attachments": [{"name": "run42_stage_layout.pdf", "size": "0.9 MB"}]},
        {"from": "Dana", "text": "Hi Gary,\n\nLooks right to me. Please add the grounding check to the run checklist, so it survives the next person who loads the stage.\n\nDana", "side": "right", "to": ["Gary"], "cc": ["Manager"], "date": "2026-11-13 08:47", "subject": "RE: Cryostat run 42: shared stage with a superconducting chip"},
        {"from": "Manager", "text": "Approved. Thanks for coordinating before the cooldown instead of after it.", "side": "left", "to": ["Gary", "Dana Okafor"], "date": "2026-11-13 09:30", "subject": "RE: RE: Cryostat run 42: shared stage with a superconducting chip"}
      ],
      narration: [
        "Read this email, and judge whether Gary's cryostat plan is proper.",
        "Gary's lithium niobate chip will share a cooldown with a neighbor's superconducting chip. Is this okay?",
        "Yes. Pyroelectric fields do not respect chip boundaries, and Gary planned for that. Shield, ground, and coordinate with your neighbors.",
        "Watch your charge. " + M
      ]
    },
    {
      type: 'check',
      question: 'Why is a fast hotplate step riskier for z-cut lithium niobate than for silicon?',
      options: [
        { text: 'Lithium niobate is pyroelectric: a temperature change creates bound surface charge on faces normal to z, and a fast step leaves no time for it to be screened', correct: true, why: '' },
        { text: 'Lithium niobate has a much higher thermal conductivity than silicon', correct: false, why: "Lithium niobate actually has a lower thermal conductivity than silicon, but that is not the main hazard. The main hazard is that lithium niobate is pyroelectric. Its spontaneous polarization along z changes with temperature, which leaves bound charge on faces normal to z. On z-cut, those are the top and bottom faces. If the change is fast, surface conduction cannot screen it in time, and the unscreened field can be large enough to cause sparks, damage devices, or crack substrates, according to practitioner reports." },
        { text: 'It is not riskier; without an applied voltage there is no field', correct: false, why: "In a pyroelectric crystal, no applied voltage is needed. A temperature change alone changes the spontaneous polarization and leaves bound surface charge. The unscreened field is p times delta T over epsilon zero epsilon three three. For lithium niobate, with literature values of the pyroelectric coefficient, a temperature change of tens of kelvin can give fields of kilovolts per millimeter. That is why slow ramps, grounding, and foil discharge paths are part of the procedure." }
      ],
      correctSay: "Temperature change is charge."
    },
    takeaways([
      'Temperature change in lithium niobate and lithium tantalate is charge',
      'The unscreened field is the pyroelectric coefficient times the temperature change over the permittivity, and it can approach the coercive field',
      'z-cut puts the charge on the device faces; x-cut on the edges',
      'Ramp slowly, ground, use a discharge path, and protect your neighbors in the cryostat',
      'Short the pads during thermal steps; discharge before connecting anything'
    ], true),
    ack('pyroelectric awareness'),
    refs([R.weis, R.popescu2013, R.volk2008, R.bulmer1986, R.luo2018, R.weigel2018, R.naranjo2005, R.xu2022, R.bordui1999, R.chen2025, R.patent7309392, R.zhu2021])
  ];

  /* ======================= CHAPTER 5 ======================= */

  var ch5 = {
    id: 'bias', title: 'Bias Point Retention',
    subtitle: 'Your DC is not your RF',
    objectives: [
      'Recognize DC bias drift in lithium niobate electro-optic devices',
      'Distinguish low-frequency V pi from RF V pi',
      'Identify V pi and bandwidth reporting conventions before comparing papers',
      'Choose a bias strategy that is honest about drift',
      'Report RF results under schedule pressure, and recognize retaliation'
    ],
    steps: []
  };
  ch5.steps = [
    objectives(ch5.title, ch5.objectives),
    {
      type: 'video', title: 'Cold open: Quadrature on Monday', duration: 24,
      narration: [
        "Watch Gary's Monday, and judge his response to the drift.",
        "Gary biases his lithium niobate modulator at quadrature with a D C voltage, and goes to lunch.",
        "When he returns, the bias point has rolled down the transfer curve.",
        "Gary increases the voltage. It drifts again. Gary increases the voltage again.",
        "Is this okay? No. Charge moves under D C bias. Watch your charge. " + M
      ],
      keyframes: [
        { t: 0, shot: 'Medium shot of a test rack: a voltage source, a modulator on a fiber stage, a photodetector readout. Gary sets the voltage, the readout sits at half power. He leaves frame toward lunch.', caption: 'Quadrature, set by DC voltage.', cast: ['Gary'], draw: feDraw('quadrature', 0) },
        { t: 6, shot: 'Time-lapse: wall clock spins one hour. A large ball on an oversized cosine-squared curve sculpture next to the rack rolls slowly downhill.', caption: 'The bias point leaves.', cast: [], draw: feDraw('quadrature', 1) },
        { t: 13, shot: 'Gary returns, cranks the voltage knob. The ball is pushed back, then rolls away again. Repeat three times, faster each cut.', caption: 'More voltage. More drift.', cast: ['Gary'], draw: feDraw('quadrature', 2) },
        { t: 19, shot: 'Freeze on Gary turning the knob. Stamp: NOT OKAY.', caption: 'Is this okay?', cast: ['Gary'], draw: feDraw('quadrature', 3) }
      ]
    },
    {
      type: 'slide', layout: 'host', title: 'Let\'s take a closer look',
      bullets: [
        'Under DC bias, charge redistributes in the stack (lithium niobate, cladding, interfaces) and screens the field in the crystal',
        'The field division starts capacitive and ends resistive; the bias point drifts between the two',
        'So V pi depends on frequency: low-frequency V pi can be very different from RF V pi',
        'Long DC bias can leave memory (hysteresis) that lasts weeks'
      ],
      callout: 'Please certify the frequency of your DC.',
      narration: [
        "Let's take a closer look.",
        "A modulator is a stack of layers: lithium niobate, cladding, buffer layers, interfaces, and metal. Each has a permittivity and a conductivity.",
        "When you step the voltage, the field divides between layers by capacitance. Over time, charge moves, and the field divides by conductance instead.",
        "The bias point drifts between those two states. The drift can go either way, depending on the stack.",
        "This also means V pi depends on frequency. Liu and colleagues, in twenty twenty six, measured V pi above forty volts at ten kilohertz and below in visible and near infrared thin film modulators, and extracted the high-frequency value with a first-order high-pass model.",
        "And charge has memory. Yeh and colleagues held thirty volts for thirty six hours. The electro-optic response dropped four and a half times, became diode-like, and took about three weeks to recover thermally.",
        "Watch your charge. " + M
      ]
    },
    {
      type: 'widget', title: 'Interactive: Bias point time trace', widget: 'biasDrift', params: {},
      instructions: 'Pick a device, then run at least two bias strategies. The time axis runs from 1 microsecond to 1 year. Then try another device and a different controller range.',
      narration: ["In this activity, you will set a modulator to quadrature and watch its bias point from one microsecond to one year.", "Charge moving in and around the crystal slowly cancels the applied field. Compare holding a constant voltage, a bias controller, and a heater, on thin film and on bulk crystal."]
    },
    {
      type: 'reveal', title: 'Bias conduct in the literature',
      prompt: 'Open each finding to continue.',
      tabs: [
        { label: 'Thermo-optic escape', body: '<p>Maeder et al. (Optics Letters, 2022): on the same chip, an electro-optic phase shifter drifted over minutes from DC drift, while a thermo-optic phase shifter stayed within 0.5 dB over 33.3 minutes.</p><p>Liu et al. (ACS Photonics, 2026) attribute most of an apparent heater drift to time-varying contact resistance between the probes and the heater pads. Without feedback, with the source voltage stepped to hold constant electrical power, the extinction ratio decayed from 25 to 21 dB over 100 minutes; under constant-current drive it showed no systematic drift over 95 minutes. With closed-loop thermo-optic bias control it held about 25 dB for 93 minutes (figure).</p>', narration: ["Maeder and colleagues compared phase shifters on the same chip. The electro-optic one drifted over minutes. The thermo-optic one stayed within half a dB over thirty three minutes.", "But heaters have their own conduct issues. Liu and colleagues attribute most of an apparent heater drift to changing probe contact resistance. Under constant current drive, the systematic drift was gone.", "The heater was mostly innocent. The probe needle did most of it."], media: { src: 'media/lit/fe_liu2026_fig5a.png', alt: 'Pulse extinction ratio and heater power over about 93 minutes under closed-loop thermo-optic bias control at the null point, then a bias-off period; inset shows a 117 picosecond pulse', caption: 'Liu et al., ACS Photonics 2026, Fig. 5a.', href: 'https://doi.org/10.1021/acsphotonics.6c00668' } },
        { label: 'Drift as a budget line', body: '<p>Zheng et al. (Laser and Photonics Reviews, 2024) budgeted a 1 kHz refresh of all phase-shifter voltages in a thin-film lithium niobate photonic neural network, specifically to handle bias drift. Their chip (figure) has 54 phase shifters.</p><p>Drift is not a lab nuisance. It is an energy and control cost.</p>', narration: ["Zheng and colleagues budgeted a one kilohertz refresh of every phase shifter voltage in a thin film lithium niobate photonic neural network, to handle bias drift.", "That is a mandatory re-attestation one thousand times per second. Drift belongs in the architecture, not in the footnotes."], media: { src: 'media/lit/fe_zheng2024_fig2.jpg', alt: 'Photograph of the packaged thin-film lithium niobate photonic neural network chip with fiber arrays and wire bonds, and a micrograph of its electrodes', caption: 'Zheng et al., arXiv 2024, Fig. 2.', href: 'https://doi.org/10.48550/arXiv.2402.16513' } },
        { label: 'No drift observed', body: '<p>Some modulators are set to quadrature by tuning the laser wavelength on an unbalanced interferometer instead of applying DC bias (for example, Valdez et al., Optics Express, 2023). That avoids drift; it does not measure it.</p><p>Read the methods before quoting a drift-free result.</p>', narration: ["Some papers set the operating point by tuning the laser wavelength on an unbalanced interferometer, instead of applying a D C bias.", "That avoids drift. It does not measure it. Before you quote a drift-free result, read the methods section. Watch your charge."], media: { src: 'media/lit/fe_valdez2023_quadrature.svg', alt: 'Computed transmission of an unbalanced Mach-Zehnder interferometer versus laser wavelength, with the quadrature point reached by tuning the laser and no DC bias', caption: 'Valdez et al., Opt. Express 2023. Computed, illustrative.', href: 'https://doi.org/10.1364/OE.480519' } },
        { label: 'Which V pi?', body: '<p>V pi and V pi L are quoted per arm, push-pull, single-arm, series push-pull, or differential. These differ by factors of 2.</p><p>A DC V pi is often measured with a kilohertz to megahertz triangle wave whose frequency is not stated.</p><p>Electro-optic S21 in dB is electrical dB: at minus 3 dB, the drive power must double; at minus 6 dB, V pi has doubled. The normalization frequency moves the 3 dB point.</p>', narration: ["Before you compare V pi between papers, ask which V pi.", "Per arm, push-pull, single arm, or differential conventions differ by factors of two.", "The frequency of a D C V pi measurement is often unstated.", "And electro-optic bandwidth is usually quoted in electrical dB. At minus three dB, drive power doubles. At minus six dB, V pi has doubled. Also check the normalization frequency.", "Know your axes, including the axes of your plots."], media: { src: 'media/lit/fe_vpi_electrical_db.svg', alt: 'Computed drive voltage and drive power multiples needed for the same phase versus electro-optic response in electrical dB: at minus 3 dB voltage times 1.41 and power times 2, at minus 6 dB voltage times 2 and power times 4', caption: 'Computed from the definition of electrical dB.' } },
        { label: 'Clamped and unclamped', body: '<p>Static and low-frequency measurements correspond to unclamped (constant stress) electro-optic coefficients; RF corresponds to clamped (constant strain). Weis and Gaylord (1985) tabulate both. At 633 nm, Weis and Gaylord list r33 of about 28.8 to 30.8 pm/V clamped versus 32.2 to 34.0 pm/V unclamped. Using one r33 for both regimes shifts predicted V pi L by several percent.</p>', narration: ["Low frequency and R F also see different electro-optic coefficients. Low frequency is unclamped. R F is clamped. The difference in r three three is several percent, and several percent of V pi is a real number on a datasheet.", "Watch your charge, and watch your clamping."], media: { src: 'media/lit/fe_weis1985_r33.svg', alt: 'Range bars of the r33 electro-optic coefficient of lithium niobate at 633 nanometers: clamped 28.8 to 30.8 picometers per volt, unclamped 32.2 to 34.0 picometers per volt', caption: 'Weis and Gaylord, Appl. Phys. A 1985. Redrawn.', href: 'https://doi.org/10.1007/BF00614817' } }
      ]
    },
    {
      type: 'chat', title: 'Scenario 1: The low V pi', app: 'email',
      meta: {"header": "Inbox", "subject": "Datasheet draft: V pi", "participants": ["Gary", "Manager", "Dana"]},
      messages: [
        {"from": "Gary", "text": "Hi all,\n\nV pi for the datasheet, measured with our standard triangle sweep on the scope. I have put it in the draft as the V pi.\n\nGary", "side": "left", "to": ["Manager", "Dana Okafor"], "date": "2026-11-06 11:20", "subject": "Datasheet draft: V pi", "attachments": [{"name": "datasheet_draft_v3.docx", "size": "1.8 MB"}]},
        {"from": "Dana", "text": "Hi Gary,\n\nWhat frequency was the triangle? Low-frequency V pi in thin film can differ a lot from RF V pi because of relaxation.\n\nDana", "side": "right", "to": ["Gary"], "cc": ["Manager"], "date": "2026-11-06 12:05", "subject": "RE: Datasheet draft: V pi"},
        {"from": "Gary", "text": "The standard 1 kHz, same as every lot.", "side": "left", "to": ["Dana Okafor"], "cc": ["Manager"], "date": "2026-11-06 12:30", "subject": "RE: RE: Datasheet draft: V pi"},
        {"from": "Dana", "text": "Then it is a 1 kHz V pi, not an RF V pi. Also: per arm or push-pull? And is the 3 dB bandwidth in electrical dB, normalized at what frequency?", "side": "right", "to": ["Gary"], "cc": ["Manager"], "date": "2026-11-06 12:44", "subject": "RE: RE: Datasheet draft: V pi"},
        {"from": "Manager", "text": "Please add a footnote. Several footnotes.", "side": "left", "to": ["Gary", "Dana Okafor"], "date": "2026-11-06 13:10", "subject": "RE: RE: RE: Datasheet draft: V pi"}
      ],
      narration: [
        "Read this email thread, and judge whether Gary's V pi entry is proper.",
        "Gary measured V pi at an unstated low frequency and labeled it as the V pi. Is this okay?",
        "No. State the frequency, the drive convention, and the bandwidth convention. Your D C is not your R F.",
        "Watch your charge. " + M
      ]
    },
    {
      type: 'chat', title: 'Scenario 2: The overnight hold', app: 'phone',
      meta: { header: 'Gary', participants: ['Gary', 'Manager'] },
      messages: [
        {"from": "Gary", "text": "left the modulator at 30 V over the weekend as a stability test", "side": "left"},
        {"from": "Manager", "text": "and?", "side": "right"},
        {"from": "Gary", "text": "the EO response is a lot smaller now and the curve looks rectified", "side": "left"},
        {"from": "Manager", "text": "will it come back", "side": "right"},
        {"from": "Gary", "text": "probably. I will give it a few days to relax and re-measure", "side": "left"}
      ],
      narration: [
        "Read this message exchange, and judge whether Gary's stability test is okay.",
        "Gary held a large D C bias for days. Is this okay?",
        "It is a valid experiment, if you planned it. Yeh and colleagues showed long D C bias can reduce the response, make it rectifying, and take weeks to recover.",
        "Know your history. Watch your charge."
      ]
    },
    {
      type: 'chat', title: 'Scenario 3: The drift-free claim', app: 'thread',
      meta: { header: '#paper-club', participants: ['Gary', 'Dana', 'Manager'] },
      messages: [
        { from: 'Gary', text: 'this paper shows no drift at all. before anyone copies their stack: they tuned the laser to quadrature, so no DC bias was applied', side: 'left' },
        { from: 'Gary', text: 'no DC bias, no DC drift test. the stack might be fine, but this paper does not show it', side: 'left' },
        { from: 'Manager', text: 'can we also not apply DC bias', side: 'left' },
        { from: 'Dana', text: 'not in a product with a fixed laser', side: 'right' },
        { from: 'Gary', text: 'agreed. I will put their stack through our DC hold test before anyone calls it drift-free', side: 'left' }
      ],
      narration: [
        "Read this paper club thread, and judge Gary's reading of the paper.",
        "Gary found a paper that reports no drift at all, and read how it was biased. Is this okay?",
        "Yes. Gary read the methods. Avoiding the test is not passing the test.",
        "Watch your charge. " + M
      ]
    },
    {
      type: 'check',
      question: 'Gary measures V pi with a 1 kHz triangle on a thin-film lithium niobate modulator and quotes it as the high-speed V pi. What is the issue?',
      options: [
        { text: 'Low-frequency V pi can differ strongly from RF V pi because charge relaxation screens the field at low frequency; state the frequency and measure at the frequency you claim', correct: true, why: '' },
        { text: 'There is no issue. The Pockels effect is instantaneous, so V pi is the same at all frequencies', correct: false, why: "The Pockels effect itself is very fast. But the field inside the lithium niobate is not the voltage you apply divided by the gap at all frequencies. At low frequency, charge moves in the cladding, interfaces, and contacts, and screens or redistributes the field. At high frequency, it cannot follow. Liu and colleagues measured V pi above forty volts at ten kilohertz and below in thin film devices, and extracted the high-frequency value separately. Relaxation can also enhance the response in some devices. So the frequency must be stated, and the claim must match the measurement." },
        { text: 'The issue is only that a triangle is used instead of a sine', correct: false, why: "The waveform shape is a minor point. The main issue is frequency. Low-frequency and RF V pi can be very different in thin film lithium niobate, because charge relaxation in the layer stack changes the field in the crystal at low frequency. Also, low-frequency measurements correspond to unclamped electro-optic coefficients and RF to clamped ones. Please state the frequency, and measure at the frequency you claim." }
      ],
      correctSay: "Your DC is not your RF."
    },
    takeaways([
      'DC bias drifts because charge redistributes in the layer stack',
      'Low-frequency V pi is not RF V pi; state the frequency',
      'Long DC bias can leave weeks of memory',
      'State V pi convention, bandwidth dB convention, and normalization frequency',
      'A traveling-wave modulator turned around runs counter-propagating: a few gigahertz of bandwidth instead of tens',
      'Zero tolerance for retaliation; protection does not depend on being right'
    ], true),
    ack('bias point retention'),
    refs([R.liu2026, R.yeh2026, R.holzgrafe2024, R.maeder2022, R.zheng2024, R.valdez2023, R.weis, R.volk2008, R.arab2022, R.boynton2020, R.sones2002, R.zhu2021])
  ];

  /* ======================= CHAPTER 6 ======================= */

  var ch6 = {
    id: 'simulation', title: 'Simulation Integrity',
    subtitle: 'Numbers that look physical are not physics',
    objectives: [
      'Use symmetry planes only where the crystal has the symmetry',
      'Interpret a quality factor that does not respond to the absorbing layer',
      'Know what a Voltage terminal and a Charge terminal mean in a piezoelectric eigenfrequency study',
      'Report every model with its frame, symmetry, and boundary conditions'
    ],
    steps: []
  };
  ch6.steps = [
    objectives(ch6.title, ch6.objectives),
    {
      type: 'video', title: 'Cold open: Q equals seven times ten to the thirteen', duration: 24,
      narration: [
        "Watch this simulation, and judge whether Gary's result is okay.",
        "Gary simulates a quarter of a lithium niobate nanobeam, with two symmetry planes, to save time.",
        "The mechanical quality factor is seven times ten to the thirteen. Gary is thrilled.",
        "Dana runs the full geometry, with no symmetry planes, in the same crystal frame. The quality factor is about two thousand two hundred.",
        "Is this okay? No. Trust no symmetry plane. " + M
      ],
      keyframes: [
        { t: 0, shot: 'Screen capture style shot of a finite element model: a nanobeam cut to one quarter, the missing three quarters hidden behind a stage curtain labeled TRUST ME. Gary at the keyboard.', caption: 'A quarter model, to save time.', cast: ['Gary'], draw: feDraw('qnano', 0) },
        { t: 6, shot: 'Huge analog Q meter on the office wall; the needle slams to the far right and sticks at 10^13. Gary raises both fists.', caption: 'Q of seven times ten to the thirteen.', cast: ['Gary'], draw: feDraw('qnano', 1) },
        { t: 12, shot: 'Dana pulls the curtain open: full nanobeam, no symmetry planes. Needle swings back down to about 10^3.', caption: 'Full geometry, same frame.', cast: ['Dana'], draw: feDraw('qnano', 2) },
        { t: 18, shot: 'Freeze on the curtain half open. Stamp: NOT OKAY.', caption: 'Is this okay?', cast: ['Gary', 'Dana'], draw: feDraw('qnano', 3) }
      ]
    },
    {
      type: 'slide', layout: 'host', title: 'Let\'s take a closer look',
      bullets: [
        'A symmetry plane forces the solution to be symmetric or antisymmetric about it',
        'An anisotropic, rotated crystal often lacks that symmetry; the plane removes the coupling that leaks energy',
        'Logged case: quarter model with symmetry, $Q$ from $2\\times10^{10}$ to $7.1\\times10^{13}$; full model, same frame, fundamental $Q \\approx 2200$',
        'If weakening the absorbing layer raises Q, the loss channel is not being captured'
      ],
      callout: 'Symmetry planes are a privilege the crystal must grant.',
      narration: [
        "Let's take a closer look.",
        "A symmetry plane tells the solver that the solution is symmetric, or antisymmetric, about that plane. It removes every solution that is neither.",
        "For an isotropic silicon device, that is usually fine. For lithium niobate in a rotated frame, the crystal often does not have that symmetry. The plane silently deletes the coupling that leaks energy.",
        "In one logged case, quarter models gave mechanical quality factors from two times ten to the ten up to seven times ten to the thirteen. The full model, in the same crystal frame, gave about two thousand two hundred for the fundamental mode.",
        "Another warning sign: in the same investigation, weakening the perfectly matched layer made the quality factor higher, not lower. When the quality factor does not respond to the absorber, the loss is not reaching the absorber.",
        "Trust no symmetry plane. " + M
      ]
    },
    {
      type: 'reveal', title: 'Simulation integrity hazards',
      prompt: 'Open each hazard to continue.',
      tabs: [
        { label: 'Absorbing layers', body: '<p>Logged case: weakening the perfectly matched layer raised Q. A quality factor that does not respond to the absorber means energy is not reaching the absorber, often because a symmetry or a boundary condition blocked the leak.</p><p>Sweep the absorber strength and thickness. Q should converge, not wander.</p>', narration: ["Sweep the absorber strength and thickness. A converged quality factor should be insensitive to reasonable changes.", "If a weaker absorber gives a higher quality factor, the loss channel is not being captured. Something upstream, often a symmetry plane, is blocking the leak."] },
        { label: 'Cyclic symmetry', body: '<p>Logged case: a cyclic symmetry condition behaved like a reflection and gave the wrong mode profile, even with copied face meshes. The full geometry gave the correct mode, in 55 minutes instead of 22.</p><p>Thirty-three extra minutes is cheaper than a wrong paper.</p>', narration: ["In one logged case, a cyclic symmetry condition behaved like a mirror and produced the wrong mode. The full geometry took fifty five minutes instead of twenty two, and gave the right answer.", "Thirty three extra minutes is cheaper than a wrong paper. Trust no symmetry plane."] },
        { label: 'Voltage versus Charge terminal', body: '<p>In a piezoelectric eigenfrequency study, a Voltage terminal is treated as a short circuit: the setpoint is ignored, and you get the resonance. A Charge terminal with zero charge is an open circuit, and you get the anti-resonance.</p><p>The coupling coefficient $k^2$ comes from the two frequencies. Two eigen runs, one short and one open.</p>', narration: ["In a piezoelectric eigenfrequency study, an electrode set to a Voltage terminal acts as a short circuit. The voltage value is ignored. You get the resonance.", "A Charge terminal with zero charge acts as an open circuit. You get the anti-resonance.", "Which one you need depends on what the electrode is connected to. The coupling coefficient comes from comparing the two runs.", "If you typed one volt and expected a biased device, the eigen solver did not tell you it ignored you."] },
        { label: 'Pipelines and bookkeeping', body: '<p>Logged cases: an extra selection step before a mirror-and-join dataset operation returned zero interaction integrals; a surface integral missed the hole sidewalls; an optimizer result was overwritten by an unchanged output filename; a sidewall taper was computed for one side instead of two.</p><p>A zero, a perfect number, or a suspiciously round improvement is a reason to look, not a reason to celebrate.</p>', narration: ["Not every problem is physics. In logged cases, one extra selection step before a dataset operation returned zero coupling. A surface integral skipped the hole sidewalls. An output file was overwritten because its name was not changed. A sidewall taper was computed for one side instead of two.", "A zero, a perfect number, or a suspiciously large improvement is a reason to look again.", "Know your axes. Trust no symmetry plane."] }
      ]
    },
    {
      type: 'chat', title: 'Scenario 1: The quarter model', app: 'call',
      meta: { header: 'Design review (video call)', participants: ['Gary', 'Manager', 'Dana'] },
      messages: [
        { from: 'Gary', text: 'Mechanical Q is seven times ten to the thirteen. Quarter model, two symmetry planes, very efficient.', side: 'right' },
        { from: 'Dana', text: 'Is the crystal symmetric about both planes in this frame?', side: 'left' },
        { from: 'Gary', text: 'The geometry is.', side: 'right' },
        { from: 'Dana', text: 'The geometry is not the crystal. Run the full model once in the same frame and compare.', side: 'left' },
        { from: 'Manager', text: 'If the full model agrees we can keep using the quarter model, right?', side: 'left' },
        { from: 'Dana', text: 'Right. Check once, then you have earned the shortcut.', side: 'left' }
      ],
      narration: [
        "Watch this design review meeting, and judge whether Gary's model is okay.",
        "Gary used symmetry planes because the geometry was symmetric. Is this okay?",
        "Only if the crystal, in your frame, has the same symmetry. Run the full model once to earn the shortcut.",
        "Trust no symmetry plane. " + M
      ]
    },
    {
      type: 'chat', title: 'Scenario 2: The ignored volt', app: 'phone',
      meta: { header: 'Gary', participants: ['Gary', 'Dana'] },
      messages: [
        { from: 'Gary', text: 'set the electrode to 1 V in the eigen study and the mode is the same at 0 V and 100 V. looked it up: a Voltage terminal in an eigen study is a short, the value is ignored', side: 'left' },
        { from: 'Gary', text: 'so that run is the short-circuit frequency. running a Charge terminal with zero charge for the open circuit. the difference gives k squared', side: 'left' },
        { from: 'Dana', text: 'right. which one goes in the deck', side: 'right' },
        { from: 'Gary', text: 'both, labeled short and open', side: 'left' }
      ],
      narration: [
        "Read these texts, and judge whether Gary's response is proper.",
        "Gary's eigen study ignored the voltage he typed in, and he works out why. Is this okay?",
        "Yes. Gary noticed that nothing changed, and found out why before reporting. In an eigen study, a Voltage terminal is a short circuit. A Charge terminal with zero charge is an open circuit.",
        "Know your boundaries."
      ]
    },
    {
      type: 'chat', title: 'Scenario 3: The template frame', app: 'thread',
      meta: { header: '#simulation', participants: ['Gary', 'Manager', 'Dana'] },
      messages: [
        {"from": "Gary", "text": "new results attached. started from last year's solver template, but last year was y-cut and this wafer is x-cut, so I reset the material frame", "side": "left"},
        {"from": "Dana", "text": "checked against anything", "side": "right"},
        {"from": "Gary", "text": "a known case first: in bulk, TE along y sees n_e and TE along z sees n_o, as it should. cut, Euler angles, and symmetry settings are on slide 1", "side": "left"},
        {"from": "Manager", "text": "good. please make that first slide standard for every results deck", "side": "left"}
      ],
      narration: [
        "Read this thread, and judge whether Gary's use of the template is proper.",
        "Gary reused last year's solver template for a new wafer. Is this okay?",
        "Yes. Gary checked the template's crystal frame against the wafer, and tested it on a known case. Put the cut, the Euler angles, and the symmetry settings on the first slide of every results deck.",
        M
      ]
    },
    {
      type: 'check',
      question: 'A quarter model of a rotated-frame lithium niobate resonator with two symmetry planes gives $Q = 10^{13}$. The full model, same frame, gives $Q \\approx 10^{3}$. Which is more trustworthy, and why?',
      narration: ["A quarter model of a rotated-frame lithium niobate resonator, with two symmetry planes, gives a quality factor of ten to the thirteen. The full model, in the same frame, gives about ten to the three. Which is more trustworthy, and why?"],
      options: [
        { text: 'The full model, because the symmetry planes impose a symmetry the rotated anisotropic crystal does not have, deleting the leaky coupling', correct: true, why: '' },
        { text: 'The quarter model, because it allows a finer mesh', correct: false, why: "A finer mesh is a fine goal, but it cannot fix a wrong boundary condition. A symmetry plane forces the solution to be symmetric or antisymmetric about that plane. In an anisotropic crystal like lithium niobate, especially in a rotated frame, the elastic and piezoelectric tensors often do not have that symmetry. The plane removes exactly the mixed solutions that couple to leaky modes, so the quality factor becomes absurdly high. Run the full model at least once to check." },
        { text: 'Both, they just describe different modes', correct: false, why: "They describe the same physical device, so they should agree on the same physical mode. When a quarter model gives ten to the thirteen and the full model gives about ten to the three, the quarter model has imposed a symmetry that the crystal does not have. That deletes the coupling that leaks energy. The full model is the reference until the symmetry is proven." }
      ],
      correctSay: "Trust no symmetry plane."
    },
    {
      type: 'check',
      question: 'Gary weakens the perfectly matched layer and the computed Q goes up. What does this most likely mean?',
      options: [
        { text: 'The loss channel is not reaching the absorber; something such as a symmetry plane or boundary condition is blocking the leak', correct: true, why: '' },
        { text: 'The absorber was too strong and reflecting; the new Q is the correct one', correct: false, why: "A too-strong absorber can reflect, so this reflex is reasonable. But if the quality factor is truly limited by radiation into the absorber, then weakening the absorber and getting a higher quality factor suggests the radiation is not being captured at all. In the logged case, the cause was symmetry planes in an anisotropic crystal, which blocked the leaky coupling. Sweep the absorber, and check the symmetry assumptions." },
        { text: 'It is a mesh refinement artifact and can be ignored', correct: false, why: "A trend you can reproduce by changing the absorber is not noise. If the quality factor does not respond to the absorber the way radiation loss should, the leak is not reaching it. Look for symmetry planes, boundary conditions, or crystal frames that suppress the coupling." }
      ],
      correctSay: "Correct. Follow the energy."
    },
    takeaways([
      'Symmetry planes must be granted by the crystal, not just the geometry',
      'Q that rises when the absorber weakens means the loss is not captured',
      'In piezo eigen studies, Voltage terminal is short, zero-charge Charge terminal is open',
      'Put cut, Euler angles, and symmetry settings on the first slide'
    ], true),
    ack('simulation integrity'),
    refs([{ cite: 'Logged cases in this chapter come from unpublished lab notes; there is no public source to cite.' }])
  ];

  /* ======================= INCIDENT SCENARIOS ======================= */
  // All incidents are fictional and take place at Five Sigma Photonics, Inc. Physics and cited numbers are real.

  function incidentSlide(title, bullets, narration) {
    return {
      type: 'slide', layout: 'policy', title: 'Incident report: ' + title, bullets: bullets,
      narration: ["We will now review an incident at Five Sigma Photonics."].concat(narration).concat([
        "As you watch, ask yourself two questions. Where was the orientation reference lost? And who could have spoken up?",
        M
      ])
    };
  }

  function feltSlide(what) {
    return {
      type: 'slide', layout: 'host', title: 'How do you think Gary felt?', hostImg: 'media/render/host_pat_empathy.jpg',
      hostCam: { idle: 'media/render/host_pat_cam_empathy_idle.mp4', talk: ['media/render/host_pat_cam_empathy_talk.mp4'] },
      bullets: ['Take a moment to consider how Gary felt when ' + String(what).replace(/\b(?:[A-Z] ){1,5}[A-Z]\b/g, function (m) { return m.replace(/ /g, ''); })],
      narration: [
        "Let's take a moment to think about Gary.",
        "How do you think Gary felt when " + what + "?"
      ],
      stages: [{
        prompt: 'See how Gary felt',
        bullets: ['Gary felt ashamed', 'Shame is a normal response to an orientation incident', 'The Office of Photonic Conduct is a judgment-free zone', 'The incident has still been logged', 'Feelings are valid. Logs are mandatory.'],
        narration: [
          "Gary felt ashamed.",
          "That is okay. Shame is a normal response to an orientation incident. It is not a disciplinary outcome.",
          "The Office of Photonic Conduct is a judgment-free zone.",
          "The incident has still been logged.",
          "Feelings are valid. Logs are mandatory. " + M
        ]
      }]
    };
  }

  var phraseSlide = {
    type: 'slide', layout: 'policy', title: 'Policy reminder: the approved speak-up phrases',
    bullets: [
      'Orientation concerns: "' + PHRASE_O + '"',
      'Charge concerns: "' + PHRASE_C + '"',
      'Say it before the irreversible step: expose, etch, bond, dice, plug in, power on',
      'Then: ' + RESPOND,
      'No employee will be disciplined for a good-faith orientation or charge check'
    ],
    callout: 'Zero tolerance for silent flats.',
    narration: [
      "Five Sigma Photonics has two approved speak-up phrases.",
      "For orientation concerns, say: " + PHRASE_O,
      "For charge concerns, say: " + PHRASE_C,
      "Say them before the irreversible step. Before you expose, etch, bond, dice, plug in, or power on.",
      "Then follow the four-step response. " + RESPOND,
      "Let's practice. " + PHRASE_O + " " + PHRASE_C,
      "No employee will be disciplined for a good-faith orientation or charge check. Not even Gary.",
      M
    ]
  };

  /* ---- (a) Chapter 1: the flat at three o'clock (90 degrees, lithography) ---- */

  var incFlat = [
    incidentSlide("The flat at three o'clock", [
      'Category: crystal orientation, 90 degrees',
      'Location: cleanroom, lithography bay',
      'Product: x-cut thin-film lithium niobate modulators, lot 7',
      "Persons involved: Gary, Dana Okafor (labmate), Gary's manager",
      'Status: logged'
    ], ["It concerns a ninety degree orientation error during lithography on x-cut lithium niobate."]),
    {
      type: 'video', title: "Incident video: The flat at three o'clock", duration: 34,
      narration: [
        "Watch the incident as it happened, and judge Gary's conduct.",
        "Gary carries a cassette of x-cut wafers from the spin coater to the mask aligner.",
        "A spin coater stops each wafer at whatever angle it stops. That is normal. The flat is the only reference that survives.",
        "The procedure says: flat toward the operator, at six o'clock. Gary places the wafer with the flat at three o'clock.",
        "For the first layer, there are no alignment marks yet. The mask is lined up with the flat. Gary lines it up with the edge in front of him.",
        "Every later layer aligns to the marks of the first layer. Every layer is aligned perfectly to every other layer. Together, they are aligned perfectly to the wrong crystal axis.",
        "Six weeks later, test reports that the modulators need about eight times the R F drive voltage.",
        "Is this okay? No. This is not okay. " + M
      ],
      keyframes: [
        { t: 0, shot: 'Cleanroom corridor under yellow lithography light, 35 mm lens, camera at chest height tracking alongside. Gary in a white bunny suit, hood, mask and nitrile gloves, seen from behind and in profile, carries a 100 mm wafer cassette away from a spin coater (bowl lid open, chuck still decelerating) toward a manual mask aligner. The wafers in the cassette sit at random rotations; their flats point in different directions.', caption: 'From the spin coater. Each wafer stops wherever it stops.', cast: ['Gary'], draw: feDraw('flat', 0) },
        { t: 7, shot: "Top-down macro of the mask aligner vacuum chuck with three alignment pins. A translucent 100 mm x-cut lithium niobate wafer, resist-coated, slides in under a gloved hand with its flat against the right-hand pin (three o'clock). In sharp focus in the foreground, a laminated card taped to the tool bezel: FLAT AT SIX O'CLOCK.", caption: "Procedure: flat at six o'clock. Gary: flat at three o'clock.", cast: ['Gary'], draw: feDraw('flat', 1) },
        { t: 14, shot: 'Microscope eyepiece view (circular vignette): straight vertical chrome waveguide lines on the mask slide over the wafer. Inset, on a clipboard: the lot traveler page with a hand sketch showing the waveguides perpendicular to the flat.', caption: 'First layer: aligned to the edge in front of him.', cast: [], draw: feDraw('flat', 2) },
        { t: 21, shot: 'Montage of later layers: electrode and gold pad layers land on first-layer alignment crosses, verniers read zero. A semi-transparent crystal axis gizmo fades in over the wafer: the waveguides run along z, the dashed green intended direction runs along y.', caption: 'Every layer aligned to every other layer.', cast: [], draw: feDraw('flat', 3) },
        { t: 28, shot: 'Test bay, six weeks later. A monitor shows a modulator transfer curve stretched across a far larger voltage range than the reference trace. Dana, in a lab coat, seen from behind, points at the screen. Freeze frame. Red stamp: NOT OKAY.', caption: 'Is this okay?', cast: ['Dana'], draw: feDraw('flat', 4) }
      ]
    },
    {
      type: 'chat', title: 'The thread: Lot 7 out of family', app: 'email',
      meta: {"header": "Inbox", "subject": "Lot 7 modulators: V pi out of family", "participants": ["Dana", "Gary", "Manager"]},
      messages: [
        {"from": "Dana", "text": "Hi all,\n\nLot 7 test summary attached. Low-frequency V pi is about 4.5 times the lot 6 value, and RF V pi is about 8 times. Same layout, same electrodes, same wafer supplier. Wafer map photo attached as well.\n\nDana", "side": "left", "to": ["Gary", "Manager"], "cc": ["Lithography process owner"], "date": "2026-11-16 09:12", "subject": "Lot 7 modulators: V pi out of family", "attachments": [{"name": "lot7_vpi_summary.xlsx", "size": "1.2 MB"}, {"name": "lot7_wafer_map.jpg", "size": "840 KB"}]},
        {"from": "Gary", "text": "Thanks Dana.\n\nLots 5 and 6 are both in family, so lot 7 looks like a one-off. The new supplier batch or the coater would be my first guesses; the coater was acting up that week.\n\nProposal: we exclude lot 7 from the review summary as non-representative and rerun the wafer next week.\n\nGary", "side": "right", "to": ["Dana Okafor", "Manager"], "cc": ["Lithography process owner"], "date": "2026-11-16 10:03", "subject": "RE: Lot 7 modulators: V pi out of family"},
        {"from": "Dana", "text": "Hi Gary,\n\nIncoming inspection confirmed x-cut, and the flats match the supplier datasheet. Under the microscope the waveguides run parallel to the primary flat; the traveler sketch has them perpendicular. It looks like the first layer went down 90 degrees from the flat, not a material problem.\n\nDana", "side": "left", "to": ["Gary", "Manager"], "cc": ["Lithography process owner"], "date": "2026-11-16 10:41", "subject": "RE: RE: Lot 7 modulators: V pi out of family"},
        {"from": "Manager", "text": "Thanks both.\n\nCan we keep Thursday's review focused on the representative lots? I would rather not open a failure analysis for one lot this close to the review. Gary, please fix the loading step and we will move on.", "side": "left", "to": ["Gary", "Dana Okafor"], "date": "2026-11-16 11:15", "subject": "RE: RE: Lot 7 modulators: V pi out of family"},
        {"from": "Gary", "text": "Will do. I will update the recipe card and clean up the lot 7 data in the shared folder so the review deck is not confusing.", "side": "right", "to": ["Manager", "Dana Okafor"], "date": "2026-11-16 11:32", "subject": "RE: RE: RE: Lot 7 modulators: V pi out of family"}
      ],
      narration: [
        "The incident unfolds over email. Judge whether the response is proper.",
        "The test result is clear. The first guesses are the supplier batch and the coater. Dana follows the evidence to the hand-off.",
        "Gary proposes excluding lot 7 as non-representative and cleaning up its data. The manager would rather not open a failure analysis this close to the review. Is this okay?",
        "No. Nobody here is acting in bad faith, and that is the problem. Lot 7 gets no failure analysis, no traveler entry, and no root cause on record, and its data is about to be removed instead of marked. The next lot goes through the same hand-off.",
        "Know your axes. " + RESPOND
      ]
    },
    feelCheck("the lot 7 test summary arrived in everyone's inbox"),
    {
      type: 'reveal', title: 'What went wrong, technically',
      prompt: 'Open each finding to continue.',
      tabs: [
        { label: 'The modulator lost r33', body: '<p>Designed: on x-cut, the waveguide runs along y. TE light is polarized along z, the electrodes put the field along z, and the index change is set by $n_e^3 r_{33}$.</p><p>Fabricated: the waveguide runs along z. TE light is polarized along y, the electrodes now put the field along y, and the index change is set by $n_o^3 r_{22}$.</p><p>$V_\\pi$ ratio, for equal overlap:</p><p>$$\\frac{V_\\pi(\\text{fabricated})}{V_\\pi(\\text{designed})} = \\frac{n_e^3\\, r_{33}}{n_o^3\\, r_{22}}$$</p><p>With $n_e = 2.138$ and $n_o = 2.211$ at 1550 nm (Zelmon et al., 1997) and the 633 nm coefficients compiled by Weis and Gaylord (1985):</p><p>RF (clamped): $r_{33} = 28.8$ to $30.8$ pm/V, $r_{22} = 3.4$ pm/V, ratio about 7.7 to 8.2.<br>Low frequency (unclamped): $r_{33} = 32.2$ to $34.0$ pm/V, $r_{22} = 6.7$ to $6.8$ pm/V, ratio about 4.3 to 4.6 (unclamped $r_{22}$ is composition sensitive, so treat this as about 4 to 5).</p><p>Your DC is not your RF, even when you are wrong.</p>', narration: [
          "On x-cut, the design runs the waveguide along y. T E light is polarized along z, the electrodes put the field along z, and the modulator uses r three three, the largest coefficient.",
          "With the waveguide along z, T E light is polarized along y, and the electrodes now put the field along y. The modulator uses r two two instead.",
          "For equal overlap, the V pi ratio is n e cubed times r three three, over n o cubed times r two two.",
          "With the indices from Zelmon and colleagues, and the coefficients compiled by Weis and Gaylord, that is about eight times at R F, where the clamped values apply, and about four and a half times at low frequency, where the unclamped values apply.",
          "Even the error has a different D C and R F value. Know your axes."
        ] },
        { label: 'The polarizations lost their material birefringence', body: '<p>Along y on x-cut, TE sees $n_e$ and TM sees $n_o$: $n_o - n_e \\approx 0.07$ of material birefringence adds to whatever separation the waveguide geometry gives.</p><p>Along z, the optic axis, TE and TM both see $n_o$ in bulk, so the material birefringence no longer contributes. Their separation now depends on the waveguide cross-section: film thickness, etch depth, width, and sidewall angle. For some profiles they are nearly degenerate; for others they stay apart.</p><p>And, as the bends chapter will show, on x-cut the TE-like and TM-like effective indices can cross as a bend sweeps the propagation direction. With the layout rotated 90 degrees, every bend sweeps a different range of crystal directions, so the crossings move. Every polarization number in the design was computed for the other axis.</p>', narration: [
          "Along y, T E sees the extraordinary index and T M sees the ordinary index. About zero point zero seven of material birefringence adds to whatever the waveguide geometry gives.",
          "Along z, both see the ordinary index in bulk. The separation now depends only on the waveguide cross-section: film thickness, etch depth, width, and sidewall angle. For some profiles, T E and T M are nearly degenerate.",
          "And, as the bends chapter will show, on x-cut the two can cross as a bend sweeps the propagation direction. Rotate the layout, and the crossings move. Know your axes."
        ] },
        { label: 'Why every layer agreed', body: '<p>Alignment marks carry orientation from layer to layer. They never carry it from the crystal. Only the first layer sees the flat.</p><p>A spin coater, a hotplate, and a wafer cassette all release a wafer at an arbitrary angle. That is expected. The flat check at the aligner is the step that restores the reference.</p><p>Before etch, a wrong first-layer exposure can usually be stripped and redone. After etch, it is permanent.</p>', narration: [
          "Alignment marks carry orientation from one layer to the next. They never carry it from the crystal. Only the first layer sees the flat.",
          "The spin coater stopped at an arbitrary angle, as spin coaters do. The flat check at the aligner exists to restore the reference.",
          "Before etch, a wrong first layer can usually be stripped and redone. After etch, it is permanent. Speaking up early is cheaper. " + RESPOND
        ] }
      ]
    },
    {
      type: 'check', kicker: 'What would you do?',
      question: "You are Gary. The lot 7 summary is in your inbox, the wafer map photo shows the waveguides parallel to the flat, and you remember loading the flat at three o'clock. What do you do?",
      narration: ["You are Gary.", "The test summary is in your inbox. You remember loading the flat at three o'clock.", "What would you do?"],
      options: [
        { text: "Reply to all: confirm the flat was loaded at three o'clock; log it in the lot traveler and the shared run sheet with the wafer IDs and the measured V pi at both frequencies; tell the process owner today; propose a flat check sign-off at the aligner", correct: true, why: '' },
        { text: "Fix the loading step for lot 8, and exclude lot 7 from the review summary as non-representative", correct: false, why: "Fixing the loading step is right, and excluding a lot from a summary sounds tidy. But lot 7 is not non-representative. It is an accurate measurement of a device with its waveguides along the crystal z axis, and it is the only evidence that the hand-off failed. Excluded without a failure analysis, it leaves no traveler entry and no root cause, and nobody can tell later whether lot 8 was fixed or just lucky. The record has to say what happened: the flat was loaded at three o'clock, here are the measurements, here is the corrective action. Exclude it from the device statistics if you like, but mark it, explain it, and keep it. Know your axes." },
        { text: "Write the root cause as \"spin coater or new supplier batch, under investigation\" and move on", correct: false, why: "Under investigation sounds careful. Here it is not, because the evidence is already in hand: incoming inspection confirmed the material, and the microscope shows the waveguides parallel to the flat. The spin coater did exactly what it always does: it held the wafer by vacuum, spun it, and stopped at whatever angle it stopped at. That is why the flat exists. A root cause of 'under investigation' that nobody investigates becomes the root cause of record, and it points the corrective action at the wrong place. The orientation was lost at the hand-off to the aligner. Write that down. Know your axes." },
        { text: "Open a maintenance ticket on the aligner chuck and prealigner, to rule out the tool first", correct: false, why: "Ruling out the tool first sounds methodical. But a vacuum chuck holds the wafer exactly where it was placed, and a prealigner turns it to the angle in its recipe, very reliably, including when the recipe angle is wrong. On a manual aligner, the operator seats the flat against the pins. Here the wafer was seated with the flat at three o'clock, and you remember doing it. The ticket would close as no fault found, the tool owner would lose an afternoon, and lot 7 would still have no record. Check the flat against the traveler at the aligner, and log what happened. Know your axes." },
        { text: "Ask the supplier to confirm their flat convention before anyone writes anything down", correct: false, why: "Confirming a convention with a supplier is often a good idea, and if this were a new supplier with an unread datasheet, it would be the first step. But incoming inspection already confirmed that these wafers match the datasheet. The flat was in the right place on the wafer. The wafer was in the wrong place on the chuck. Waiting for the supplier delays the record by days and implies a cause the evidence does not support. Log what you know now, and update it if anything changes. Know your axes." },
        { text: "Ask the simulation team to re-check the model, in case it overpredicted the electro-optic response", correct: false, why: "Re-checking a model is never wasted. But the model was right about the device in the design: a waveguide along y, with T E polarized along z, using r three three. The fabricated device is a different device, with its waveguide along z. Looking for a modeling error first sends the investigation away from the hand-off, and it delays the record. If you want the model to explain the measurement, set it up with the waveguide along z and confirm the r two two response. Then log the incident. Know your axes." },
        { text: "Clean up the lot 7 data in the shared folder so the review deck is not confusing", correct: false, why: "A clean review deck is a reasonable goal. Cleaning up the data is not the way to get it. Moved, renamed, or deleted files are how a failure disappears from the record, and the next person to see a lot with eight times the V pi will have nothing to compare it with. Keep the data where it is, label it with the actual orientation, and put a one-line explanation in the deck if it would confuse people. Marked data is not confusing. Missing data is. Know your axes." }
      ],
      correctSay: "You logged it, you told the process owner, and you proposed a fix to the procedure. " + RESPOND + " Know your axes."
    },
    {
      type: 'check', kicker: 'You are Dana, Gary\'s labmate',
      question: "You are next in line for the mask aligner. Gary's wafer is on the chuck with the flat at three o'clock. The card on the tool says six o'clock. Gary's hand is on the exposure button. What do you do?",
      narration: ["You are Dana.", "You are next in line for the mask aligner. The flat is at three o'clock. The card says six o'clock. Gary is about to expose.", "What would you do?"],
      options: [
        { text: 'Speak up now with the approved phrase, "' + PHRASE_O + '" Point to the flat and the card, and let Gary re-seat the wafer and note it in the traveler', correct: true, why: '' },
        { text: "Say nothing yet. Gary may be using a different card for this mask set", correct: false, why: "Giving a coworker the benefit of the doubt is kind, and sometimes a mask set does use a different loading convention. The way to find out is to ask, in one sentence, before the exposure. If Gary has a reason, he can explain it in ten seconds and you both learn something. If he does not, you saved a lot. After exposure, the resist can usually be stripped and redone, which takes an hour. After etch, the error is permanent. Use the approved phrase. Pause, please. Orientation check." },
        { text: "Mention it after the exposure, so you do not interrupt a timed step", correct: false, why: "Not interrupting a timed step is a good instinct in a lithography bay. But the exposure has not started, and the flat check takes seconds. After the exposure, the resist has to be stripped and the wafer re-coated, and after etch the error is permanent. The cheapest moment to speak up is before the button is pressed. Use the approved phrase now. Pause, please. Orientation check." },
        { text: "Re-seat the wafer for Gary while he checks the mask, to save time", correct: false, why: "Helping is good. Handling another person's wafer on their run, without telling them, is not. Gary may come back, find the wafer somewhere unexpected, and rotate it again, and the traveler will not record what happened. Corrections are made by the person running the lot, out loud, and written in the traveler. Use the approved phrase, and let Gary re-seat it. Pause, please. Orientation check." },
        { text: "Take a photo of the flat for the traveler and send it to Gary after the run", correct: false, why: "A photo for the traveler is genuinely useful evidence. Sending it after the run turns it into evidence of a lost lot instead of a saved one. The photo can wait; the exposure cannot. Speak up first, at the tool, with the approved phrase, and then take the photo for the traveler together. Pause, please. Orientation check." },
        { text: "Raise it with the process owner at the end of the week, so the loading procedure gets fixed properly", correct: false, why: "Fixing the procedure is the right long-term goal, and the process owner is the right person for it. But this wafer is on the chuck now. A procedure fix at the end of the week does not save this lot. Prevention first: speak up at the tool. Then, yes, raise the procedure with the process owner. Pause, please. Orientation check." }
      ],
      correctSay: PHRASE_O + " Five words before exposure cost nothing. After etch, they cost a lot. Speak up."
    },
    phraseSlide
  ];

  /* ---- (a2) Chapter 2: the rods at twelve o'clock (90 degrees, launch polarization) ---- */

  var incRods = [
    incidentSlide("The rods at twelve o'clock", [
      'Category: launch polarization, 90 degrees',
      'Location: modulator test bench 3',
      'Product: x-cut thin-film lithium niobate modulators, waveguides along y',
      'Persons involved: Gary, Dana Okafor (labmate), Gary\'s manager',
      'Status: logged'
    ], ["It concerns a ninety degree error in the launch polarization at the chip input."]),
    {
      type: 'video', title: "Incident video: The rods at twelve o'clock", duration: 30,
      narration: [
        "Watch the incident as it happened, and judge Gary's conduct at the bench.",
        "Gary replaces the polarization-maintaining patch cord that feeds the lensed fiber at the chip input. The old one was damaged.",
        "He looks at the new cord's end face in the fiber inspection scope, and sets the rotation of its mating sleeve by eye.",
        "He turns the two stress rods to vertical, because he remembers the slow axis as perpendicular to the rods.",
        "In this type of fiber, the slow axis runs along the line through the stress rods. The laser light is launched on the slow axis, so it leaves the fiber polarized vertically, out of the chip plane. That is T M.",
        "The modulator still works. On the oscilloscope, it just needs nearly three times the voltage.",
        "Is this okay? No. " + M
      ],
      keyframes: [
        { t: 0, shot: 'Optical test bench, 50 mm lens, shallow depth of field. Gloved hands unplug a polarization-maintaining patch cord from a rotatable mating sleeve that feeds the lensed fiber at the chip input, and plug in a new one. A small lithium niobate chip on a vacuum stage sits between two lensed fibers. Background out of focus: a rack with a tunable laser and a photodetector.', caption: 'The old patch cord: stress rods horizontal.', cast: ['Gary'], draw: feDraw('rods', 0) },
        { t: 7, shot: 'Screen insert: the fiber inspection scope image of the patch cord end face, a bright core between two dark circular stress rods. A gloved hand turns the mating sleeve rotation knob; the rods swing from horizontal to vertical. A sticky note on the monitor bezel, handwritten: SLOW AXIS PERPENDICULAR TO RODS (crossed out by a second hand later in the film).', caption: 'The new patch cord, set from memory: rods vertical.', cast: ['Gary'], draw: feDraw('rods', 1) },
        { t: 14, shot: 'Cutaway render of the chip cross-section: thin-film ridge on oxide, gold electrodes either side. A red double arrow for the optical field points straight up out of the film (TM) instead of sideways along the crystal z axis (TE).', caption: 'Light arrives polarized out of the plane: TM.', cast: [], draw: feDraw('rods', 2) },
        { t: 21, shot: 'Oscilloscope close-up: the modulator transfer curve, clean and high-contrast, but stretched across a much larger voltage sweep than the reference trace in grey. Freeze. Red stamp: NOT OKAY.', caption: 'Is this okay?', cast: [], draw: feDraw('rods', 3) }
      ]
    },
    {
      type: 'chat', title: 'The thread: The voltage went up overnight', app: 'thread',
      meta: { header: '#test-bench', participants: ['Gary', 'Dana', 'Manager'] },
      messages: [
        { from: 'Gary', text: 'V pi on the reference chip is 2.9 times what it was last week, same triangle sweep on the scope. same chip, same probes, same electrodes', side: 'left' },
        { from: 'Manager', text: 'did anything change', side: 'left' },
        { from: 'Gary', text: 'new input patch cord. rods set like always', side: 'left' },
        { from: 'Dana', text: 'which way are the rods', side: 'right' },
        { from: 'Gary', text: 'vertical. slow axis horizontal', side: 'left' },
        { from: 'Dana', text: 'in this fiber the slow axis is along the rods, and the laser is launched on the slow axis. vertical rods, vertical polarization. you are launching TM. on x-cut along y that uses r13, not r33', side: 'right' },
        { from: 'Gary', text: 'so the 14 chips I measured this week', side: 'left' },
        { from: 'Dana', text: 'TM. rotate the fiber 90 degrees, re-measure, and log which runs used which polarization', side: 'right' }
      ],
      narration: [
        "Read this test bench thread, and judge whether Gary's response is proper.",
        "Gary notices the V pi went up, but trusts his memory of the fiber. Is this okay?",
        "No. A clean transfer curve with the wrong voltage is the signature of the wrong electro-optic coefficient.",
        "Dana asks the one question that matters: which way are the rods?",
        "Know your axes. " + RESPOND
      ]
    },
    feltSlide("he realized fourteen chips had been measured in T M"),
    {
      type: 'reveal', title: 'What went wrong, technically',
      prompt: 'Open each finding to continue.',
      tabs: [
        { label: 'TM uses r13', body: '<p>x-cut, waveguide along y, electrodes put the field along z.</p><p>TE: polarized along z, index change set by $n_e^3 r_{33}$.<br>TM: polarized along x, index change set by $n_o^3 r_{13}$.</p><p>For equal overlap:</p><p>$$\\frac{V_\\pi(\\mathrm{TM})}{V_\\pi(\\mathrm{TE})} = \\frac{n_e^3\\, r_{33}}{n_o^3\\, r_{13}}$$</p><p>With $n_e = 2.138$, $n_o = 2.211$ (Zelmon et al., 1997) and the 633 nm values compiled by Weis and Gaylord (1985):</p><p>RF (clamped): $r_{13} = 7.7$ to $8.6$ pm/V, ratio about 3.2 to 3.4.<br>Low frequency (unclamped): $r_{13} = 10.0$ to $10.9$ pm/V, ratio about 2.8 to 2.9.<br>(Ratios from same-source pairs of coefficients.)</p><p>The TM mode also overlaps the in-plane field differently, so the real ratio is a measurement, not a division.</p>', narration: [
          "On x-cut, with the waveguide along y, the electrodes put the field along z.",
          "T E light is polarized along z and uses r three three. T M light is polarized along x, and uses r one three.",
          "For equal overlap, the V pi ratio is n e cubed r three three over n o cubed r one three. With the values compiled by Weis and Gaylord, that is about three point two to three point four at R F, and about two point eight to two point nine at low frequency.",
          "The T M mode also overlaps the field differently. So the real ratio is something you measure, not something you divide by. Know your axes."
        ] },
        { label: 'Why nobody noticed', body: '<p>The photodetector at the output does not care about polarization. The transfer curve looks clean, with good extinction. Only the voltage scale changed.</p><p>A polarization error does not look like an error. It looks like a worse device.</p>', narration: [
          "The output photodetector does not care about polarization. The transfer curve looked clean, with good extinction. Only the voltage scale changed.",
          "A polarization error does not look like an error. It looks like a worse device. That is why it survives for a week."
        ] },
        { label: 'Stress rods and slow axes', body: '<p>In polarization-maintaining fiber with two stress rods, the slow axis runs along the line through the rods. Linear light launched on the slow axis, or on the fast axis, exits polarized along that axis.</p><p>That is the only case the fiber maintains. Linear light launched at an angle to the axes splits between the slow and fast modes, which travel with different propagation constants. The output is then in general elliptical, and its state changes with fiber length, wavelength, temperature, and stress. A misaligned launch can look linear at the moment you check it and drift when the lab warms up.</p><p>The metric is the polarization extinction ratio: rotate a polarizer at the output and compare maximum and minimum power:</p><p>$$\\mathrm{PER} = 10\\log_{10}\\frac{P_{\\max}}{P_{\\min}}\\ \\mathrm{dB}$$</p><p>Do not set fiber rotation from memory. Measure the launch with a rotating polarizer and a power meter before coupling to the chip, and record the fiber serial number, the polarization direction, and the extinction ratio in the bench log.</p>', narration: [
          "In polarization-maintaining fiber with two stress rods, the slow axis runs along the line through the rods.",
          "The fiber maintains only what you launch on an axis. Linear light launched at an angle splits between the slow and fast modes, which travel with different propagation constants. The output is then in general elliptical, and it changes with length, wavelength, temperature, and stress.",
          "Do not set the fiber rotation from memory. Measure the launch with a rotating polarizer and a power meter before coupling to the chip. Record the fiber, the polarization direction, and the extinction ratio in the bench log.",
          "Know your axes, including the axes of your fiber."
        ] }
      ]
    },
    {
      type: 'check', kicker: 'What would you do?',
      question: 'You are Gary. You now know that this week\'s V pi values for 14 chips were taken with TM light, and two of them were already sent to a customer. What do you do?',
      narration: ["You are Gary.", "Fourteen chips were measured in T M this week. Two results already went to a customer.", "What would you do?"],
      options: [
        { text: 'Set the launch to TE and verify it with a polarizer; re-measure all 14 chips; mark the TM runs as TM in the bench log and the shared run sheet; tell the process owner and the account owner so the customer data can be corrected', correct: true, why: '' },
        { text: "Rotate the fiber, re-measure, and replace this week's numbers in the run sheet with the corrected ones", correct: false, why: "Re-measuring is exactly the right technical step. Replacing the numbers is not. Two results already left the building labeled as T E, and overwritten rows cannot tell anyone which values to correct, or how long the launch was wrong. Corrected data and overwritten data look the same in a run sheet, unless the old rows are kept and marked. Keep the T M runs, label them T M, add the T E re-measurements beside them, and tell the people who received the old numbers. Know your axes." },
        { text: 'Divide the TM V pi values by 2.9 and report them as TE', correct: false, why: "The factor of about two point nine is an estimate for equal mode overlap, using unclamped coefficients measured at six hundred thirty three nanometers, which apply to your low-frequency triangle sweep. The T M mode overlaps the in-plane electrode field differently from the T E mode, so the real ratio differs from chip to chip and design to design. At R F, where the clamped coefficients apply, the same estimate is about three point two to three point four, so even the constant depends on frequency. Dividing by a constant turns a measurement into a guess and labels the guess as a measurement. Re-measure in T E. Know your axes." },
        { text: "Ask the fiber vendor to confirm the slow-axis marking before changing anything", correct: false, why: "Confirming a marking with a vendor sounds careful. But the fiber did what this type of fiber does: its slow axis runs along the line through the stress rods. The rotation was set from memory, at the bench. Waiting for a vendor reply delays the correction of data that is already with a customer. Check the launch polarization with a polarizer now, and log it. Know your axes." },
        { text: 'Report the fiber rotation mount as faulty', correct: false, why: "The rotation mount held the angle it was set to, which is its only job. It cannot read stress rods, and it does not know which axis is slow. The rods were turned to vertical by hand, on purpose, from memory. A maintenance ticket would close as no fault found. Check the launch polarization with a polarizer before coupling, and record it. Know your axes." },
        { text: 'Blame the inspection scope: the end-face image must be mirrored', correct: false, why: "Some cameras and microscopes do mirror the image, and that is worth knowing. But a mirror flips left and right. It maps a vertical line of stress rods to a vertical line, and a horizontal line to a horizontal line. A mirror cannot turn a ninety degree error into a correct alignment, or the other way around. The rods really were vertical. The slow axis really was vertical. Know your axes." },
        { text: 'Blame the electrode gold: it must have degraded', correct: false, why: "Degraded electrodes would raise loss and change the R F response. They would not multiply the low-frequency V pi by about three on the same chip overnight while the transfer curve stays clean. A factor near three, appearing the day a fiber was changed, points at the launch polarization, and the electro-optic coefficient changing from r three three to r one three. Know your axes." }
      ],
      correctSay: "Verify, re-measure, label, and tell the people who received the old numbers. " + RESPOND
    },
    {
      type: 'check', kicker: 'You are Dana',
      question: 'You walk past test bench 3. The end-face image on the monitor shows the stress rods vertical. The bench card says horizontal. Gary is about to start a 14-chip overnight run and leave. What do you do?',
      narration: ["You are Dana.", "The stress rods are vertical. The bench card says horizontal. Gary is about to start an overnight run.", "What would you do?"],
      options: [
        { text: 'Speak up now: "' + PHRASE_O + '" Point to the rods and the bench card, and suggest a polarizer check before the run starts', correct: true, why: '' },
        { text: 'Say nothing. Maybe Gary knows something you do not', correct: false, why: "Gary may indeed know something you do not. The way to find out is to ask, before an overnight run that will produce fourteen chips of data in the wrong polarization. If Gary has a reason, he can explain it in ten seconds and you will both learn something. If he does not, you saved a night of bench time and a week of confusion. Use the approved phrase. " + PHRASE_O },
        { text: 'Leave a sticky note on the monitor and go home', correct: false, why: "A sticky note is better than nothing, but Gary is about to press start and leave. A note found tomorrow morning is a note about fourteen wrong measurements. Speak to him now, while the fix is a turn of a knob. Use the approved phrase. " + PHRASE_O },
        { text: 'Turn the rotation mount yourself after Gary leaves', correct: false, why: "An undocumented change to someone else's bench in the middle of their run makes the run uninterpretable. Half of the chips may be measured one way and half the other, and the bench log will say neither. Corrections are made by the person running the measurement, out loud, and recorded. Use the approved phrase now. " + PHRASE_O },
        { text: 'Tell the manager tomorrow that the overnight data is probably wrong', correct: false, why: "Telling the manager is a valid escalation when speaking to the person directly does not work. Here, the person is standing in front of you, and the run has not started. Prevention first. Use the approved phrase. " + PHRASE_O }
      ],
      correctSay: PHRASE_O + " One sentence saved a night of bench time. Know your axes."
    }
  ];
  /* ---- (c) Chapter 4: the hot plug (pyroelectric discharge into connected electronics) ---- */

  var incHotPlug = [
    incidentSlide('The hot plug', [
      'Category: pyroelectric discharge',
      'Location: package qualification lab',
      'Product: bulk z-cut lithium niobate phase modulator, 0.5 mm thick, in a test fixture on a thermoelectric cooler',
      'Equipment involved: one driver amplifier (former)',
      "Persons involved: Gary, Dana Okafor (labmate), Gary's manager",
      'Status: logged'
    ], ["It concerns a pyroelectric discharge from a bulk z-cut lithium niobate chip into a connected driver amplifier."]),
    {
      type: 'video', title: 'Incident video: The hot plug', duration: 32,
      narration: [
        "Watch the incident as it happened, and judge Gary's conduct.",
        "Gary is qualifying a test fixture. The bulk z-cut lithium niobate chip sits on a thermoelectric cooler, with its electrode pads wired to an R F connector.",
        "To protect the driver amplifier during the thermal step, Gary unplugs its cable. Now the electrodes are floating.",
        "Gary steps the cooler from twenty five to seventy degrees Celsius, at full speed. A temperature change of forty five kelvin.",
        "The crystal's polarization changes. Charge appears on the faces. With the pads floating, it has nowhere to go.",
        "Gary plugs the driver amplifier back in, to start the R F test. There is a small snap.",
        "The amplifier output stage is gone. Under the microscope, there is a burn mark across the electrode gap.",
        "Is this okay? No. Temperature change is charge. Watch your charge. " + M
      ],
      keyframes: [
        { t: 0, shot: 'Package qualification lab, bench height, 35 mm. A gold-plated test fixture on a thermoelectric cooler block with a heatsink and fan below. A 0.5 mm thick bulk lithium niobate chip, about 2 mm by 40 mm, is wire-bonded to a coaxial RF connector on the fixture. A gloved hand unscrews the cable from the connector and sets it on the bench; the cable leads to a boxed driver amplifier module.', caption: 'Cable unplugged "to protect the amplifier". Pads now floating.', cast: ['Gary'], draw: feDraw('hotplug', 0) },
        { t: 7, shot: 'Close-up of a temperature controller front panel: setpoint changes from 25.0 to 70.0, ramp rate field reads MAX. Rack-focus to the chip: a graphic overlay of minus signs accumulating along the top surface under the electrodes.', caption: 'Forty-five kelvin, at full speed.', cast: [], draw: feDraw('hotplug', 1) },
        { t: 14, shot: 'Macro, low angle, the connector and cable: the gloved hand brings the cable nut toward the fixture connector. A tiny blue-white arc jumps across the last fraction of a millimeter at the center pin. Single frame flash, then normal light.', caption: 'Plugged back in while charged.', cast: ['Gary'], draw: feDraw('hotplug', 2) },
        { t: 21, shot: 'The driver amplifier module on the bench, its status LED dark. A thin wisp of smoke (practical effect) rises from its vent. Cut to a microscope image of the chip: a dark burn track across the gap between signal and ground electrodes.', caption: 'One driver amplifier, former.', cast: [], draw: feDraw('hotplug', 3) },
        { t: 27, shot: 'Freeze on the smoke wisp. Red stamp: NOT OKAY. Then the corrected take: shorting cap on the connector during the step, slow ramp on the controller, cap removed only after the chip is back at room temperature and discharged; green stamp OKAY.', caption: 'Is this okay?', cast: [], draw: feDraw('hotplug', 4) }
      ]
    },
    {
      type: 'chat', title: 'The thread: Amplifier failure on fixture 2', app: 'email',
      meta: {"header": "Inbox", "subject": "Driver amplifier failure on fixture 2", "participants": ["Gary", "Dana", "Manager"]},
      messages: [
        {"from": "Gary", "text": "Hi all,\n\nThe driver amplifier on fixture 2 failed during thermal qualification. These amplifiers have failed before. I have swapped in a spare and logged it as an amplifier failure, cause unknown, so we keep the qualification on schedule.\n\nGary", "side": "left", "to": ["Manager", "Dana Okafor"], "date": "2026-11-18 15:05", "subject": "Driver amplifier failure on fixture 2", "attachments": [{"name": "fixture2_thermal_qual_log.csv", "size": "880 KB"}]},
        {"from": "Dana", "text": "Hi Gary,\n\nWas the chip lithium niobate, and were the pads connected during the temperature step?\n\nDana", "side": "right", "to": ["Gary"], "cc": ["Manager"], "date": "2026-11-18 15:40", "subject": "RE: Driver amplifier failure on fixture 2"},
        {"from": "Gary", "text": "Bulk z-cut. I unplugged the amplifier during the step to protect it, and plugged it back in at 70 C to start the RF test.", "side": "left", "to": ["Dana Okafor"], "cc": ["Manager"], "date": "2026-11-18 16:02", "subject": "RE: RE: Driver amplifier failure on fixture 2"},
        {"from": "Dana", "text": "Then the amplifier was protected during the step and exposed right after it. Floating pads on z-cut store the pyroelectric charge; plugging in at temperature discharged it through the amplifier output.", "side": "right", "to": ["Gary"], "cc": ["Manager"], "date": "2026-11-18 16:25", "subject": "RE: RE: RE: Driver amplifier failure on fixture 2"},
        {"from": "Manager", "text": "So not a random failure?", "side": "left", "to": ["Dana Okafor", "Gary"], "date": "2026-11-18 16:40", "subject": "RE: RE: RE: Driver amplifier failure on fixture 2"},
        {"from": "Dana", "text": "Not random. Shorting cap on the connector during every thermal step, slow ramp, connect only after discharge. I would also keep the spare in its box until the procedure is updated.", "side": "right", "to": ["Manager", "Gary"], "cc": ["Equipment owner, fixture 2"], "date": "2026-11-18 16:52", "subject": "RE: RE: RE: Driver amplifier failure on fixture 2"}
      ],
      narration: [
        "Read this email thread, and judge whether Gary's handling of the failure is proper.",
        "Gary logs the amplifier failure as cause unknown, swaps in a spare, and keeps the qualification on schedule. Is this okay?",
        "No. A failure with a mechanism is not random. Floating electrodes on z-cut lithium niobate store pyroelectric charge, and plugging in at temperature delivers it to whatever is on the other end of the cable.",
        "Watch your charge. " + RESPOND
      ]
    },
    feelCheck('the driver amplifier stopped working with a small snap'),
    {
      type: 'reveal', title: 'What went wrong, technically',
      prompt: 'Open each finding to continue.',
      tabs: [
        { label: 'How much charge', body: '<p>Bound surface charge density on the z faces: $\\sigma = p\\,\\Delta T$. Unscreened field: $E = p\\,\\Delta T / (\\varepsilon_0 \\varepsilon_{33})$, $\\varepsilon_{33} = 28$. Voltage across a slab of thickness $t$: $V = E\\,t$.</p><p>$\\Delta T = 45$ K, $t = 0.5$ mm, $p = 4.0\\times10^{-5}\\ \\mathrm{C\\,m^{-2}\\,K^{-1}}$ (Weis and Gaylord, 1985) to about $9.5\\times10^{-5}\\ \\mathrm{C\\,m^{-2}\\,K^{-1}}$ (Popescu et al., 2013):</p><p>$\\sigma = 1.8$ to $4.3$ nC per square millimeter; $E = 7.3$ to $17$ kV/mm; $V = 3.6$ to $8.6$ kV between a wide floating top electrode and the grounded fixture below.</p><p>Unscreened upper bound. Surface conduction, air, and partial discharges reduce it. In bulk lithium niobate, dark conduction is slow, so a large part can remain for a long time.</p>', narration: [
          "Let's estimate the charge. The bound charge density is the pyroelectric coefficient times the temperature change.",
          "For forty five kelvin, with the literature range of the coefficient, that is about two to four nanocoulombs per square millimeter.",
          "If nothing screens it, the field is that charge density over epsilon zero epsilon three three. Seven to seventeen kilovolts per millimeter.",
          "Across a half millimeter chip, between a wide floating top electrode and the grounded fixture below, that is an upper bound of about four to nine kilovolts.",
          "Screening reduces it over time. In bulk lithium niobate, dark conduction is slow. Watch your charge."
        ] },
        { label: 'Why it reached the amplifier', body: '<p>With the pads shorted or connected through a low-impedance path, pyroelectric charge drains continuously as a small current, $i = p\\,A\\,dT/dt$, while the temperature changes.</p><p>With the pads floating, it is stored. Connecting a cable later releases it in one event, through the connector center pin, into whatever is on the other end: an amplifier output, a probe tip, a transimpedance amplifier input.</p><p>Which face goes negative depends on which face is up: the coefficient is negative, so heating leaves the plus z face negative, and cooling leaves it positive (Weis and Gaylord, 1985). Either way, the energy is the same.</p>', narration: [
          "When the pads are shorted, or connected through a low impedance, the pyroelectric charge drains as a small current while the temperature changes.",
          "When the pads float, the charge is stored. Plugging in a cable later releases it all at once, into whatever is on the other end. An amplifier output. A probe tip. An amplifier input.",
          "The sign depends on which face is up. Heating leaves the plus z face negative, and cooling leaves it positive. The amplifier does not care about the sign.",
          "Disconnecting the amplifier to protect it did protect it, during the step. The danger moved to the moment of reconnection. Watch your charge."
        ] },
        { label: 'Industry already knows', body: '<p>Surface acoustic wave industry: patent background describes lithium niobate substrates charging up with temperature changes during processing, producing sparks that destroy the comb electrodes and can break the substrate (US Patent 7,309,392 B2, 2007; industry patent statement).</p><p>Bordui et al. (Journal of Applied Physics, 1999): chemically reduced lithium niobate wafers, with dc conductivity raised by more than five orders of magnitude, gave surface acoustic wave filters with increased resistance to pyroelectric charging during thermal cycling.</p><p>Chen et al. (CrystEngComm, 2025): chemical reduction ("black" lithium niobate) adds point defects that affect the material\'s microwave loss. Safety has a cost.</p><p>Zhu et al. (Advances in Optics and Photonics, 2021), from the authors\' own experience: on bulk lithium niobate, the pyroelectric effect tends to cause electrostatic discharge that damages superconducting nanowires under rapid temperature change; they did not see it on thin-film waveguides. Labeled as experience reported in a review, not a controlled study.</p>', narration: [
          "The surface acoustic wave industry learned this a long time ago.",
          "Patent background from that industry describes lithium niobate substrates charging up with temperature changes during processing, producing sparks that destroy the comb electrodes and can break the substrate. That is an industry statement in a patent, not a peer reviewed study.",
          "Bordui and colleagues, in nineteen ninety nine, chemically reduced lithium niobate wafers, raising the conductivity by more than five orders of magnitude, and made surface acoustic wave filters that resisted pyroelectric charging during thermal cycling. These darkened wafers are often called black lithium niobate. They trade optical transparency for safety, and Chen and colleagues report that the added point defects affect microwave loss.",
          "And Zhu and colleagues report, from their own experience, that on bulk lithium niobate, pyroelectric discharge damaged superconducting nanowires during rapid temperature changes, while thin film waveguides did not seem to have the problem. The substrate matters.",
          "Watch your charge."
        ] }
      ]
    },
    {
      type: 'check', kicker: 'What would you do?',
      question: 'You are Gary. You now understand that the amplifier failed from a pyroelectric discharge you caused. Fixture 3 is waiting for the same thermal step this afternoon. What do you do?',
      narration: ["You are Gary.", "You understand the failure now. Fixture three is waiting for the same thermal step.", "What would you do?"],
      options: [
        { text: 'Tag the failed amplifier, log the incident with the mechanism in the shared run sheet, tell the equipment owner and the process owner, and hold fixture 3 until the procedure has a shorting cap during thermal steps, a ramp limit, and a discharge check before reconnecting', correct: true, why: '' },
        { text: "Swap in the spare, log the failure as 'amplifier failure, cause unknown', and keep fixture 3 on schedule", correct: false, why: "Cause unknown sounds honest, and keeping a qualification on schedule matters. But the cause is not unknown anymore. You know the mechanism: floating electrodes on a bulk z-cut chip stored the pyroelectric charge from a forty five kelvin step, and reconnecting at temperature delivered it through the connector. Logged as unknown, the record sends the next engineer to the amplifier vendor, and running fixture three the same way will very likely deliver the same discharge to the spare. Log the mechanism, and fix the procedure before the next run. Watch your charge." },
        { text: 'Blame the amplifier vendor for an inadequate ESD rating', correct: false, why: "Every amplifier has a limit to the electrostatic discharge it survives, and the datasheet states the rating it was tested to. A charged lithium niobate chip with floating electrodes is not a person touching a pin. It is a large stored charge, released through a connector center pin straight into the output stage. No reasonable rating is designed for that. The amplifier was not inadequate. It was connected to a charged crystal. Blaming the supplier would classify this as third-party misconduct, when the cause is in the procedure. Watch your charge." },
        { text: 'Report the temperature controller for ramping too fast', correct: false, why: "The temperature controller ramped at maximum speed because the ramp rate was set to maximum. It did exactly what it was told. A slower ramp does help, because it gives surface conduction more time to screen the charge, but the controller is not the cause, and with floating pads even a slow ramp can leave charge on bulk lithium niobate, where dark conduction is slow. The cause is floating electrodes during a temperature change, followed by a hot reconnection. Set a ramp limit in the procedure, short the pads, and discharge before reconnecting. Watch your charge." },
        { text: 'Point out that you were wearing a grounded wrist strap the whole time', correct: false, why: "Wearing a wrist strap was correct, and thank you. But a wrist strap grounds you. It does not ground the crystal. Lithium niobate is an insulator, and charge on an insulating surface does not flow to ground just because a grounded person is nearby. The charge here was on the crystal faces under floating electrodes, and it left through the cable you connected. Insulators are handled with shorting paths, ionizers, and slow ramps. Watch your charge." },
        { text: "Rework the burned electrode so the chip can finish its test, and add a note later", correct: false, why: "Finishing the test is tempting, because the chip is otherwise fine. But bridging a burned electrode gap does not restore the device, and the burn track is the record of the discharge path. A note added later is a note that often never gets added. Photograph the damage, log it with the mechanism now, and keep the chip as evidence for the procedure change. Watch your charge." },
        { text: 'Blame the humidity in the lab', correct: false, why: "Humidity does affect how fast surface charge leaks away, and very dry air makes static problems worse. But humidity did not create the charge. A forty five kelvin step on a pyroelectric crystal did, and floating pads stored it. Changing the weather is not a corrective action. Short the pads, limit the ramp, and discharge before reconnecting. Watch your charge." }
      ],
      correctSay: "Tag it, log it with the mechanism, tell the owners, and fix the procedure before the next run. Watch your charge. " + RESPOND
    },
    {
      type: 'check', kicker: 'You are Dana',
      question: 'Fixture 3 has just finished a fast thermal step to 70 C. Its connector has no shorting cap. Gary is reaching for the driver amplifier cable. What do you do?',
      narration: ["You are Dana.", "The fixture just finished a thermal step. The connector has no shorting cap. Gary is reaching for the amplifier cable.", "What would you do?"],
      options: [
        { text: 'Speak up now: "' + PHRASE_C + '" Ask Gary to wait, let the chip return to room temperature, discharge the pads through a resistor to ground, and then connect', correct: true, why: '' },
        { text: "Say nothing. Gary has run this fixture many times and knows it", correct: false, why: "Experience is real, and Gary has run this fixture many times. But this step was faster than usual, the connector has no shorting cap, and the charge does not care about experience. Waiting a few minutes and discharging the pads costs nothing. After the snap, it costs an amplifier and a day. Use the approved phrase. Pause, please. Charge check." },
        { text: 'Switch off the amplifier power supply so it is safe', correct: false, why: "Switching off the supply feels safe, but an unpowered amplifier is not protected from an electrostatic discharge into its output. Discharge damage does not need the device to be powered. The stored charge on the crystal goes through the connector either way. The fix is to remove the charge before connecting, not to remove the power. Use the approved phrase. " + PHRASE_C },
        { text: 'Hand Gary a wrist strap', correct: false, why: "A wrist strap grounds Gary. It does not ground the crystal, which is an insulator with charge sitting on its faces under floating electrodes. When the cable connects, that charge goes through the connector, not through Gary. Wrist straps are part of the procedure, but they do not address this hazard. Use the approved phrase, and discharge the pads before connecting. " + PHRASE_C },
        { text: 'Mention it at the next safety meeting', correct: false, why: "The next safety meeting is a good place to discuss the procedure change. It is not a good time to stop this particular connection, which is happening in the next five seconds. Speak first, then bring it to the meeting. Use the approved phrase. " + PHRASE_C }
      ],
      correctSay: PHRASE_C + " Wait, discharge, then connect. Watch your charge."
    }
  ];

  /* ---- (b, d) Chapter 5: the chip that was turned around (180 degrees) and the Friday shipment ---- */

  var incTurned = [
    incidentSlide('The chip that was turned around', [
      'Category: crystal orientation, 180 degrees',
      'Location: RF probe station 1',
      'Product: x-cut thin-film lithium niobate traveling-wave modulators, 5 mm, on-chip termination',
      "Persons involved: Gary, Dana Okafor (labmate), Gary's manager",
      'Status: logged, and escalated'
    ], ["It concerns a one hundred eighty degree orientation error at an R F probe station, and what happened when the schedule became involved."]),
    {
      type: 'video', title: 'Incident video: The chip that was turned around', duration: 32,
      narration: [
        "Watch the incident as it happened. Then judge Gary's conduct.",
        "Gary takes a modulator chip from a shared gel-pak and places it on the probe station chuck. It is turned one hundred eighty degrees from the bench card.",
        "The lensed fiber is on the left, so the light goes in on the left, as always. That is the end the chip designer intended as the output. A Mach-Zehnder modulator does not mind which way the light goes.",
        "The R F pads at the other end end in an on-chip termination. So the R F probe can only land on the R F input pads. Today, those are on the right. Gary moves the probe arm.",
        "The light now travels left to right. The R F wave travels right to left.",
        "The D C V pi is perfect. The high-speed response falls off a cliff at a few gigahertz.",
        "Is this okay? No. Know your axes. Your D C is not your R F. " + M
      ],
      keyframes: [
        { t: 0, shot: 'Top-down macro of a gel-pak with a grid of identical rectangular modulator dies, no die marks visible from above. Gloved tweezers lift one die. Cut to the probe station chuck: the die placed with its label text upside down relative to a laminated bench card on the chuck edge.', caption: 'From the shared gel-pak. Turned around.', cast: ['Gary'], draw: feDraw('turned0') },
        { t: 8, shot: 'Probe station, 45 degree overhead: lensed fiber on a stage at the left facet. A GSG RF probe on a manipulator is unscrewed from the left side and remounted on the right, landing on pads next to the words RF IN, which are upside down in the frame. At the left end, a small on-chip thin-film resistor is visible.', caption: 'Light enters on the left. RF can only enter on the right.', cast: ['Gary'], draw: feDraw('turned1') },
        { t: 16, shot: 'Network analyzer screen: the electro-optic response trace drops smoothly, crosses the minus 3 dB line at a few GHz, and dives into a deep null. A second reference trace (grey) from last week stays flat to tens of GHz.', caption: 'DC V pi: perfect. High-speed response: no.', cast: [], draw: feDraw('turned2') },
        { t: 24, shot: 'Freeze on the null. Red stamp: NOT OKAY. Corrected take: die turned around, fiber and probe back to the bench card positions, RF and light both traveling left to right, response flat; green stamp OKAY.', caption: 'Is this okay?', cast: [], draw: feDraw('turned3') }
      ]
    },
    {
      type: 'chat', title: 'The thread: Chip 3 bandwidth', app: 'thread',
      meta: { header: '#rf-test', participants: ['Gary', 'Dana', 'Manager'] },
      messages: [
        {"from": "Gary", "text": "chip 3: EO response is -3 dB at about 6 GHz with a null near 13.5 GHz. DC V pi is fine. probably the network analyzer calibration, it was done yesterday", "side": "left"},
        {"from": "Dana", "text": "which end does the light go in", "side": "right"},
        {"from": "Gary", "text": "left. same as always", "side": "left"},
        {"from": "Dana", "text": "and the RF probe", "side": "right"},
        {"from": "Gary", "text": "right side today. that is where the RF in pads were", "side": "left"},
        {"from": "Dana", "text": "then the chip is turned around. light and RF counter-propagate. 6 GHz and a null at 13.5 is exactly what a 5 mm chip does backwards", "side": "right"},
        {"from": "Gary", "text": "ok. I will turn it around, re-measure, and update its row. no need to make a thing of it, it is one chip", "side": "left"},
        {"from": "Dana", "text": "turn it around and re-measure, yes. and log it. the rest of the lot came out of the same gel-pak", "side": "right"}
      ],
      narration: [
        "Read this test thread, and judge whether Gary's plan is proper.",
        "Gary's first guess is the network analyzer calibration. His second is to fix the one chip and update its row. Is this okay?",
        "No. The calibration did nothing wrong, and quiet fixes hide the other chips loaded the same way.",
        "Know your axes. " + RESPOND
      ]
    },
    feltSlide("he saw the null at thirteen and a half gigahertz on a chip he had loaded himself"),
    {
      type: 'reveal', title: 'What went wrong, technically',
      prompt: 'Open each finding to continue.',
      tabs: [
        { label: 'Direction is part of the design', body: '<p>Light entering at time $t_0$ reaches position $z$ at $t_0 + z\\,n_g/c$. The RF wave along the electrode has index $n_m$. Integrating the phase the light picks up along a length L (the time-of-flight treatment used by Boynton et al., 2020, p. 8), the response is</p><p>$$|H(f)| = \\left|\\frac{\\sin x}{x}\\right|, \\qquad x = \\frac{\\pi f L\\,(n_m \\mp n_g)}{c}$$</p><p>with the minus sign for co-propagation and the plus sign for counter-propagation (RF loss ignored).</p><p>A published thin-film design reports an RF effective index $n_m = 2.2$ and an optical group index $n_g = 2.23$ (Arab Juneghani et al., 2022, p. 3). For $L = 5$ mm:</p><p>Co-propagating: the velocity term alone reaches -3 dB (electrical) near 890 GHz, so RF loss sets the bandwidth instead.<br>Counter-propagating: -3 dB (electrical, $|H| = 0.71$) at about 6.0 GHz, first null at $c / [L\\,(n_m + n_g)]$, about 13.5 GHz.</p>', narration: [
          "A traveling wave modulator works because the R F wave and the light move together. Light that enters at one moment keeps seeing the same part of the R F wave all the way along the electrode.",
          "Follow the light through the device, and add up the phase it picks up. The response is a sinc function. Its argument is pi, times frequency, times length, times the index difference, over the speed of light.",
          "When the waves co-propagate, the index difference is n m minus n g. In a velocity matched design, that is almost zero. A published thin film design reports an R F index of two point two and an optical group index of two point two three.",
          "When the waves counter-propagate, the difference becomes a sum, about four point four.",
          "For a five millimeter device, the velocity term alone would allow hundreds of gigahertz when co-propagating, so R F loss sets the bandwidth instead. Counter-propagating, the response falls to minus three electrical dB at about six gigahertz, and to zero at about thirteen and a half gigahertz.",
          "The velocity match became a velocity sum. Know your axes."
        ] },
        { label: 'What did not change', body: '<p>Turning a finished chip 180 degrees in its own plane turns the electrodes and the crystal together. The electro-optic response relative to the chip\'s own pads is unchanged.</p><p>A Mach-Zehnder modulator at DC gives the same phase whichever way the light travels, so the DC V pi is unchanged.</p><p>That is why the DC test passed, and why the error survived until the RF test.</p>', narration: [
          "Turning a finished chip around in its own plane turns the electrodes and the crystal together. The electro-optic response, measured from the chip's own pads, does not change.",
          "At D C, the modulator gives the same phase whichever way the light travels. So the D C V pi was perfect.",
          "Only the relative direction of the R F wave and the light changed. That is why the D C test passed. Your D C is not your R F."
        ] },
        { label: 'When 180 degrees flips the sign', body: '<p>On x-cut, a half turn in the plane is a 180 degree rotation about x. It maps +y to -y and +z to -z. A tensor component changes sign if its indices contain y and z an odd number of times in total: $r_{33}$, $r_{13}$, $r_{22}$, $r_{51}$, and the pyroelectric coefficient all flip in any frame that did not turn with the crystal. The refractive indices do not.</p><p>It matters when something stayed fixed while the crystal turned:</p><p>A mask written on a wafer loaded 180 degrees from the flat: electrodes in the mask frame, crystal turned. The voltage-to-phase sign reverses; a bias controller set to lock on a rising slope finds the other quadrature, and data polarity inverts.</p><p>A z-cut wafer processed face down: the +z and -z faces swap. Hydrofluoric acid etches the two faces at very different rates (Sones et al., 2002), and the pyroelectric charge on heating has the opposite sign.</p><p>Thin film made by ion slicing: the implanted face of the donor crystal is bonded to the handle wafer (process: Zhu et al., 2021), so the exposed film surface is the opposite face from the donor\'s top face. This matters for z-cut; on x-cut, plus x and minus x are related by the crystal\'s mirror symmetry.</p>', narration: [
          "Now, precisely, when does a half turn flip a sign?",
          "On x-cut, a half turn in the plane maps plus y to minus y, and plus z to minus z. A coefficient changes sign if its indices contain y and z an odd number of times in total. So r three three, r one three, r two two, r five one, and the pyroelectric coefficient all flip in any frame that did not turn with the crystal. The refractive indices do not change.",
          "So it matters when something stayed fixed while the crystal turned. If a mask is written on a wafer loaded a half turn from its flat, the electrodes are in the mask frame and the crystal is turned. The voltage to phase sign reverses, a bias controller set for a rising slope finds the other quadrature, and the data comes out inverted.",
          "On z-cut, processing a wafer face down swaps the plus z and minus z faces. Hydrofluoric acid etches those two faces at very different rates, and heating charges them with opposite signs.",
          "And in thin film made by ion slicing, the implanted face of the donor crystal is bonded down, so the film's exposed surface is the opposite face. That matters for z-cut.",
          "Know your axes. Trust no symmetry plane."
        ] }
      ]
    },
    {
      type: 'check', kicker: 'What would you do?',
      question: 'You are Gary. You realize chip 3 was loaded turned around, and the other 11 chips of the lot came out of the same unmarked gel-pak. What do you do?',
      narration: ["You are Gary.", "Chip three was loaded turned around. The other eleven chips came out of the same unmarked gel-pak.", "What would you do?"],
      options: [
        { text: 'Turn chip 3 to match the bench card and re-measure; record the orientation of every chip already measured in the shared run sheet; re-test any chip whose orientation is unknown; mark the dies and the gel-pak; tell the process owner', correct: true, why: '' },
        { text: "Turn chip 3 around, re-measure, and update its row in the run sheet", correct: false, why: "Turning the chip around and re-measuring gives the right number for chip three. Updating one row does not tell anyone that chips were being loaded turned around. The other eleven chips came out of the same unmarked gel-pak, and some of their results may already be in the run sheet, measured backwards, with nothing to flag them. Re-measure chip three, mark the old row instead of replacing it, record the orientation of every chip, and tell the process owner. Know your axes." },
        { text: 'Re-calibrate the network analyzer and re-measure', correct: false, why: "A calibration corrects the instrument and cables up to the probe tips. It does not know which way the light goes on the chip. Re-calibrating will give you the same six gigahertz roll-off and the same null near thirteen and a half gigahertz, because those come from the light and the R F wave traveling in opposite directions through a five millimeter electrode. A calibration problem would also not produce a clean sinc null at exactly the frequency the counter-propagation formula predicts. The instrument did exactly what it was told. Know your axes." },
        { text: 'Replace the RF probe: it must be worn', correct: false, why: "A worn probe raises contact resistance and adds loss and ripple. It does not create a deep null at the speed of light divided by the device length times the sum of the R F and optical indices. That null is the signature of counter-propagation. A new probe would land on the same pads and measure the same physics. Know your axes." },
        { text: 'Report the probe station chuck theta stage for rotating the die', correct: false, why: "A theta stage rotates the chuck by the amount you command, and then holds it. It cannot turn a die end for end by itself, and a half turn is not a small alignment correction. The die was placed turned around when it came out of the gel-pak, and the chuck held it there faithfully. The reference was lost at the hand-off from the gel-pak to the chuck. Know your axes." },
        { text: 'Blame the gel-pak supplier: the dies must have rotated in transit', correct: false, why: "A gel-pak holds each die by surface tack, in the orientation it was placed. Dies do not turn end for end in a gel-pak. These dies were placed by a coworker, without die marks, in no recorded orientation. The orientation was never recorded, so it could not be lost in transit. That is a hand-off within the same group, which is peer-to-peer misconduct, not third-party. Mark the dies and record the orientation. Know your axes." },
        { text: "Report the DC V pi now, and add the RF numbers when the re-test is done", correct: false, why: "The D C V pi is correct, because turning a finished chip around does not change its D C response, and reporting what you trust is a reasonable instinct. But this is a high-speed modulator, and its R F response is what the customer is buying. Reporting only D C now, while you know the R F test was done backwards on some chips, lets the record say the lot is characterized when it is not. Report both, with the R F marked as pending re-test. Your D C is not your R F." }
      ],
      correctSay: "One chip fixed, eleven chips accounted for, and the dies marked. Know your axes. " + RESPOND
    },
    {
      type: 'chat', title: 'The thread: The Friday shipment', app: 'phone',
      meta: { header: 'Manager', participants: ['Manager', 'Dana'] },
      messages: [
        {"from": "Manager", "text": "customer demo is Friday. DC V pi is in spec across the lot. RF can follow with the next lot", "side": "left"},
        {"from": "Dana", "text": "we do not know which chips were measured backwards. re-testing RF on 12 chips is about a day", "side": "right"},
        {"from": "Manager", "text": "can we hold the RF numbers off the run sheet until they are re-measured? I do not want half-validated data in front of the customer", "side": "left"},
        {"from": "Manager", "text": "appreciate you being a team player on this one", "side": "left"}
      ],
      narration: [
        "Now the schedule enters the room.",
        "The manager suggests shipping on the D C data, and holding the R F numbers off the run sheet until they are re-measured. Then: appreciate you being a team player on this one.",
        "Pause here. Nothing in it is a threat. Notice what would be missing from the record, and what is being asked of you."
      ]
    },
    {
      type: 'check', kicker: 'What would you do?',
      question: 'You are Dana. Your manager suggests shipping on DC data and holding the RF numbers off the run sheet until they are re-measured. What do you do?',
      narration: ["You are Dana.", "Your manager suggests holding the R F numbers off the run sheet until they are re-measured.", "What would you do?"],
      options: [
        { text: 'Put the RF results and the orientation finding on the run sheet, reply in writing with the facts and a re-test plan with its time estimate, and if the pressure continues, use the Photonic Conduct Speak-Up Line', correct: true, why: '' },
        { text: "Agree to hold the RF numbers until the re-test, so half-validated data does not confuse the customer", correct: false, why: "Not confusing the customer is a fair concern. But holding known results off the run sheet means the record says the high-speed performance is fine, when you know some chips were measured backwards and their true response is unknown. Half-validated data is handled by marking it, not by hiding it. Put the R F numbers on the run sheet, marked as measured with an unknown orientation, add the re-test plan and its time estimate, and let the shipping decision be made with that in view. Speak up." },
        { text: 'Ship, but add a vague footnote: "RF performance under characterization"', correct: false, why: "A vague footnote is a way of saying something without saying it. The customer will read it as routine. You know something specific: some chips were measured with the light and the R F wave counter-propagating, which collapses the bandwidth to a few gigahertz, and which chips is unknown. Write that down plainly, with the re-test plan. Know your axes." },
        { text: 'Re-test only chip 3 in the right orientation and ship the lot with that one result', correct: false, why: "Chip three is the only chip whose orientation you are sure of. One correct result does not certify eleven chips whose orientation was never recorded. Shipping the lot under chip three's number would turn one measurement into twelve claims. Re-test every chip whose orientation is unknown, or state clearly which chips were not re-tested. Know your axes." },
        { text: "Raise it in the team channel, so everyone sees the issue before Friday", correct: false, why: "Visibility helps, and a team channel is less dramatic than going over anyone's head. But it moves a disagreement about one decision into a public thread, where the facts get lost and people take sides. The right sequence is: facts on the run sheet, a written reply to your manager with the re-test plan, and, if the pressure continues, the Photonic Conduct Speak-Up Line, which exists for exactly this. Speak up, in the right place." },
        { text: "Move the RF data to a separate 'to be validated' folder that is not linked from the run sheet", correct: false, why: "A separate folder for unvalidated data sounds organized. In practice it is a way for data to leave the record without anyone deciding that it should. The run sheet is where the next person looks. Keep the R F data on it, marked as measured with an unknown orientation, with the re-test plan beside it. Know your axes." }
      ],
      correctSay: "Facts on the run sheet, the plan in writing, and the Speak-Up Line if the pressure continues. Speak up."
    },
    {
      type: 'chat', title: 'The thread: The following Monday', app: 'phone',
      meta: { header: 'Manager', participants: ['Manager', 'Dana'] },
      messages: [
        {"from": "Manager", "text": "quick one before our 1:1. I have trimmed the customer review attendee list to keep it small, so you are off this one", "side": "left"},
        {"from": "Manager", "text": "also want to talk about tone. the run sheet note on Friday came across as escalating", "side": "left"},
        {"from": "Dana", "text": "ok. is this about the RF data on the run sheet", "side": "right"},
        {"from": "Manager", "text": "it is about how we communicate as a team", "side": "left"}
      ],
      narration: [
        "Dana put the R F results on the run sheet, marked as measured with an unknown orientation. The following Monday, Dana was left off the customer review, and a one-on-one was booked about tone.",
        "Consider what you have seen."
      ]
    },
    {
      type: 'check', kicker: 'Classify the behavior',
      question: "Which statement best describes the manager's Monday message?",
      narration: ["Which statement best describes the manager's Monday message?"],
      options: [
        { text: 'Retaliation: being left off a review and called in about tone, right after a good-faith report', correct: true, why: '' },
        { text: "Normal attendee management for a customer meeting", correct: false, why: "Attendee lists do get trimmed for legitimate reasons. But this change came right after Dana put the R F results on the run sheet, it removed Dana from the one meeting where those results matter, and it arrived together with a conversation about tone that refers to the run-sheet note. An adverse change that follows a report, and that would discourage a reasonable person from reporting again, is retaliation under Policy P H C seventeen point three, even when it is described as routine. Five Sigma Photonics has zero tolerance for retaliation." },
        { text: "Constructive feedback about communication style", correct: false, why: "Feedback on communication can be legitimate. But the only communication named here is an accurate entry on the run sheet, which the procedure requires. Calling a required report escalating, in the same breath as taking someone off a review, teaches people to stop writing things down. That is retaliation, however politely it is delivered. Five Sigma Photonics has zero tolerance for retaliation." },
        { text: 'Retaliation only if the coworker turns out to be right about the orientation', correct: false, why: "Protection does not depend on being right. A good-faith report is protected even if the investigation later finds a different cause. If people could only speak up when they were certain, nobody would speak up before the etch. In this case the coworker was right, but that is not why the report is protected. Five Sigma Photonics has zero tolerance for retaliation." }
      ],
      correctSay: "This is retaliation. Good-faith orientation and charge reports are protected, whether or not they turn out to be right. Five Sigma Photonics has zero tolerance for retaliation."
    },
    {
      type: 'slide', layout: 'policy', title: 'Zero tolerance for retaliation',
      bullets: [
        'Good-faith orientation and charge reports are protected activity under Policy PHC-17.3',
        'Retaliation includes removing access to reviews or authorship, reassigning tool time, and negative comments about tone that follow a report',
        'Protection does not depend on being right',
        'Report retaliation through the Photonic Conduct Speak-Up Line',
        'Linking goodwill, tool time, or credit to a skipped check is a separate violation: quid pro quo orientation'
      ],
      callout: 'The crystal does not retaliate. Neither do we.',
      narration: [
        "Five Sigma Photonics has zero tolerance for retaliation.",
        "Good-faith orientation and charge reports are protected activity under Policy P H C seventeen point three.",
        "Retaliation includes removing access to reviews or authorship, reassigning tool time, and negative comments about tone that follow a report. It is often polite.",
        "Protection does not depend on being right.",
        "If you experience or witness retaliation, report it through the Photonic Conduct Speak-Up Line.",
        "And remember Friday's message about being a team player. Linking goodwill, tool time, or credit to a skipped check is its own category. Quid pro quo orientation.",
        "The crystal does not retaliate. Neither do we. " + M
      ]
    }
  ];
  /* ======================= CHAPTER: SOURCES OF ORIENTATION MISCONDUCT ======================= */

  // Category definitions: name (on screen), spoken name, def (policy definition, shown and spoken)
  var CAT = {
    tool: { name: 'Tool-Initiated Orientation Misconduct', say: 'tool-initiated orientation misconduct',
      def: 'any change in the orientation of a wafer, die, or chip made by a piece of equipment, which no person re-verifies against the flat, a fiducial, or a die mark before the next step' },
    soft: { name: 'Software-Facilitated Orientation Misconduct', say: 'software-facilitated orientation misconduct',
      def: 'any orientation error introduced through design, layout, or simulation software, in which the software uses a crystal frame, rotation convention, or axis sign different from the one its user believes, and the user has not checked it against a known case' },
    third: { name: 'Third-Party Orientation Misconduct', say: 'third-party orientation misconduct',
      def: 'any orientation error that originates in a material, document, or convention supplied by an outside vendor, and that enters the workplace without incoming verification' },
    peer: { name: 'Peer-to-Peer Orientation Misconduct', say: 'peer-to-peer orientation misconduct',
      def: 'any orientation error that passes between members of the same lab, group, or shift through shared containers, templates, or hand-offs, where the orientation information does not travel with the object' },
    cross: { name: 'Cross-Institutional Orientation Misconduct', say: 'cross-institutional orientation misconduct',
      def: 'any orientation error that crosses an organizational boundary, such as a partner university, an external foundry, or a collaborator, when designs, dies, data, or figures cross without a written, agreed statement of the crystal frame' },
    dom: { name: 'Domestic Orientation Misconduct', say: 'domestic orientation misconduct',
      def: 'orientation misconduct within your own physical setup, bench, notebook, or records, in which you are both the source and the affected party' },
    qpq: { name: 'Quid Pro Quo Orientation', say: 'quid pro quo orientation',
      def: 'any exchange in which a benefit, such as tool time, a better queue position, co-authorship, or schedule relief, is offered, requested, or accepted in return for skipping, shortening, or not reporting an orientation or charge check' },
    hostile: { name: 'Hostile Crystal Environment', say: 'a hostile crystal environment',
      def: 'a workplace in which conditions are severe or pervasive enough that correct orientation and charge conduct is effectively impossible, regardless of individual intent' }
  };
  var CAT_ORDER = ['tool', 'soft', 'third', 'peer', 'cross', 'dom', 'qpq', 'hostile'];

  function catSteps(key, n, c) {
    var K = CAT[key];
    var slide = {
      type: 'slide', layout: 'policy', title: 'Category ' + n + ' of 8: ' + K.name,
      bullets: ['Definition: ' + K.def.charAt(0).toUpperCase() + K.def.slice(1) + '.'].concat(
        c.examples.map(function (e) { return 'Example: ' + e; })).concat(
        ['Warning signs: ' + c.warnings.join('; ')]),
      callout: c.callout,
      narration: [
        "Category " + n + " of eight. " + K.name + ".",
        "Under Policy P H C seventeen point three, " + K.say + " is defined as " + K.def + ".",
        "Let's hear that again. " + K.say.charAt(0).toUpperCase() + K.say.slice(1) + " is " + K.def + "."
      ].concat(c.examplesSay).concat([
        "Warning signs include the following. " + c.warningsSay.join(' '),
        "Whatever the category, the response is the same. " + RESPOND
      ])
    };
    var chat = c.chat;
    var opts = [{ text: K.name, correct: true, why: '' }];
    c.wrong.forEach(function (w) {
      var D = CAT[w[0]];
      opts.push({
        text: D.name, correct: false,
        why: D.say.charAt(0).toUpperCase() + D.say.slice(1) + " is " + D.def + ". " + w[1] +
          " This incident is " + K.say + ". Let's say the definition one more time. " + K.say.charAt(0).toUpperCase() + K.say.slice(1) + " is " + K.def +
          ". Classifying correctly sends the corrective action to the right place. Know your categories. " + RESPOND
      });
    });
    var classify = {
      type: 'check', kicker: 'Classify the incident',
      question: c.classifyQ + ' Which category of orientation misconduct is this?',
      narration: ["Classify the incident.", "Which category of orientation misconduct is this?"],
      options: opts,
      correctSay: "This is " + K.say + ". " + RESPOND
    };
    return [slide, chat, classify];
  }

  var chSrc = {
    id: 'sources', title: 'Sources of Orientation Misconduct',
    subtitle: 'Eight categories. One response.',
    objectives: [
      'Name the eight categories of orientation misconduct',
      'Classify an orientation incident by its source',
      'Recognize the warning signs of each category',
      'Respond to every category the same way: stop, mark, log, and speak up'
    ],
    steps: []
  };

  chSrc.steps = [objectives(chSrc.title, chSrc.objectives, [
    "In this chapter, you will learn the eight recognized sources of orientation misconduct. " + CATS_SAY,
    "You will hear them many times. That is intentional."
  ]), {
    type: 'video', title: 'Cold open: Many hands', duration: 32,
    narration: [
      "Watch one die's journey, and judge whether it is okay.",
      "This is one die. Before it reaches a test bench, it passes through many hands.",
      "A spin coater. A mask aligner. A mode solver. A wafer supplier. A shared gel-pak. A partner university. Your own fiber mount. A deal made under schedule pressure. And a lab where nobody has time to check.",
      "Each one can turn it. None of them writes it down unless someone does.",
      "Is this okay? No. " + M
    ],
    keyframes: [
      { t: 0, shot: 'Abstract set: a single rectangular lithium niobate die with a small red corner mark, resting on a white turntable against a dark grey seamless background, lit by one soft key light. Slow orbit.', caption: 'One die.', cast: [], draw: feDraw('hands', 0) },
      { t: 6, shot: 'Rapid sequence of real-scale props, each entering frame and touching the die once, each leaving it at a new angle: a spin coater chuck, a mask aligner vacuum chuck with pins, a monitor showing a mode solver material dialog, a sealed wafer box from a supplier, a shared gel-pak with unlabeled pockets.', caption: 'Tools, software, suppliers, coworkers.', cast: [], draw: feDraw('hands', 1) },
      { t: 14, shot: 'Continuing: a padded shipping box with a partner university return label (generic, no real names), a fiber rotation mount on Gary\'s bench, two gloved hands shaking over a printed tool reservation calendar, a cluttered shelf of unmarked die boxes in a lab with a crackling hotplate.', caption: 'Collaborators, yourself, deals, environments.', cast: ['Gary'], draw: feDraw('hands', 2) },
      { t: 24, shot: 'Return to the turntable: the die, now turned 90 degrees from its start, red mark in the wrong corner. Freeze. Red stamp: NOT OKAY.', caption: 'Is this okay?', cast: [], draw: feDraw('hands', 3) }
    ]
  }, {
    type: 'slide', layout: 'host', title: 'Orientation misconduct can come from anywhere',
    bullets: [
      'Orientation misconduct: any act or omission that makes a device\'s crystal axes differ from what its records say',
      'It may originate with a person, a tool, a program, a supplier, a partner, or an environment',
      'It is classified by source',
      'The response does not depend on the source: ' + RESPOND
    ],
    callout: 'Eight categories. One response.',
    narration: [
      "At Five Sigma Photonics, orientation misconduct means any act or omission that makes a device's crystal axes differ from what its records say.",
      "It may originate with a person, a tool, a program, a supplier, a partner institution, or an entire environment.",
      "We classify it by source, so that the corrective action goes to the right place.",
      "But the response does not depend on the source. " + RESPOND,
      "Here are the eight categories. " + CATS_SAY,
      "Know your categories. " + M
    ]
  }];

  var CAT_CONTENT = {
    tool: {
      examples: [
        'A spin coater stops each wafer at an arbitrary angle',
        'A prealigner on a mask aligner or stepper turns the wafer to the angle in its recipe, right or wrong',
        'A probe station chuck has a theta stage that holds whatever angle it was last set to',
        'A dicing saw cuts along the axes you program; dies stay on the tape in wafer orientation until something picks them up',
        'A pick-and-place tool sets each die down at the angle in its recipe'
      ],
      examplesSay: [
        "Examples. A spin coater stops each wafer at an arbitrary angle.",
        "A prealigner on a mask aligner or stepper turns the wafer to the angle in its recipe, whether the recipe is right or wrong.",
        "A probe station chuck has a theta stage that holds whatever angle it was last set to.",
        "A dicing saw cuts along the axes you program, and the dies stay on the tape in wafer orientation, until something picks them up.",
        "And a pick-and-place tool sets each die down at the angle in its recipe."
      ],
      warnings: ['"The tool takes care of orientation"', 'a recipe whose angle setting nobody has read', 'no flat check between tools', 'orientation is not a field in the traveler'],
      warningsSay: ["The phrase, the tool takes care of orientation.", "A recipe whose angle setting nobody has read.", "No flat check between tools.", "And orientation that is not a field in the traveler."],
      callout: 'Tools are not disciplined. Recipes and hand-offs are.',
      chat: {
        type: 'chat', title: 'Category 1 vignette: The prealigner', app: 'thread',
        meta: { header: '#equipment', participants: ['Gary', 'Dana'] },
        messages: [
          {"from": "Gary", "text": "the prealigner turned my wafer 90 degrees on the last run. opening a ticket so equipment can look at it", "side": "left"},
          {"from": "Dana", "text": "it found the flat and turned the wafer to the recipe angle. which recipe did you run", "side": "right"},
          {"from": "Gary", "text": "the default one. it is what everyone uses", "side": "left"},
          {"from": "Dana", "text": "the default recipe was written for a different product and a different mask. the tool did exactly what the recipe said", "side": "right"},
          {"from": "Gary", "text": "ok, so I will just switch recipes next time", "side": "left"},
          {"from": "Dana", "text": "switch recipes, and check the flat on the chuck before every first-layer exposure. put the recipe name in the traveler", "side": "right"}
        ],
        narration: ["A tool turned the wafer, exactly as its recipe said, and nobody checked the flat before exposure.", "Tools are not disciplined. Recipes and hand-offs are. Know your axes."]
      },
      classifyQ: 'A prealigner turned an x-cut wafer to the angle in a default recipe written for a different product. Nobody checked the flat before exposure.',
      wrong: [
        ['soft', 'In this incident, no crystal frame was set in design or simulation software. The wafer was physically turned by a tool, following its recipe, and nobody re-verified the flat.'],
        ['dom', 'In this incident, the rotation was applied by a shared tool, and the reference was lost between the tool and the next step. Domestic misconduct is reserved for errors inside your own setup.'],
        ['hostile', 'In this incident, a single tool event had a clear fix: a flat check on the chuck. A hostile crystal environment is severe or pervasive, not one recipe.']
      ]
    },
    soft: {
      examples: [
        'Euler angles that the solver applies as a rotation of the coordinates, entered as a rotation of the crystal',
        'An anisotropic material entered as a diagonal tensor in the simulation axes, with the crystal z placed on the wrong simulation axis',
        'A layout script whose zero angle is the chip edge, not the crystal y axis',
        'Rotation code tested only at 0 and 90 degrees'
      ],
      examplesSay: [
        "Examples. Euler angles that the solver applies as a rotation of the coordinates, entered as if they rotated the crystal.",
        "An anisotropic material entered as a diagonal tensor in the simulation's own axes, with the crystal z axis placed on the wrong simulation axis.",
        "A layout script whose zero angle is the chip edge, not the crystal y axis.",
        "And rotation code tested only at zero and ninety degrees."
      ],
      warnings: ['results that look physical', 'the material frame is not on the first slide', '"the solver knows the crystal"', 'a template inherited from a different cut'],
      warningsSay: ["Results that look physical.", "A material frame that is not on the first slide.", "The phrase, the solver knows the crystal.", "And a template inherited from a different cut."],
      callout: 'The solver does not know your wafer. It knows your input.',
      chat: {
        type: 'chat', title: 'Category 2 vignette: z is z', app: 'phone',
        meta: { header: 'Gary', participants: ['Gary', 'Dana'] },
        messages: [
          {"from": "Gary", "text": "my time-domain simulation says TE and TM never cross on our x-cut. bends are safe", "side": "left"},
          {"from": "Dana", "text": "how did you enter the crystal", "side": "right"},
          {"from": "Gary", "text": "diagonal anisotropic material. n_e on z", "side": "left"},
          {"from": "Dana", "text": "simulation z or crystal z? in your setup, simulation z is the film normal", "side": "right"},
          {"from": "Gary", "text": "the crystal axes line up with the simulation axes, that is the default", "side": "left"},
          {"from": "Dana", "text": "then your x-cut model is a z-cut model. the software solved it correctly. check it against a known case: bulk TE along y on x-cut should see n_e", "side": "right"}
        ],
        narration: ["Gary put the crystal z axis on the simulation z axis, which in his setup is the film normal. The software solved a z-cut device, correctly.", "The solver does not know your wafer. It knows your input. Know your axes."]
      },
      classifyQ: "Gary's time-domain simulation placed the crystal z axis on the simulation z axis, which in his setup is the film normal, so his x-cut model was a z-cut model. The software solved it correctly, and nobody checked it against a known case.",
      wrong: [
        ['tool', 'In this incident, no wafer or die was physically turned. The error lives in a crystal frame inside simulation software.'],
        ['third', 'In this incident, the software behaved as documented. A diagonal tensor in the simulation axes is a reasonable, documented input. Nothing supplied was wrong; it was not checked against a known case.'],
        ['dom', 'Gary did type it himself. But when the error lives in a software frame convention, the policy classifies it as software-facilitated, even when the user typed it. Domestic misconduct covers your own physical setup and records.']
      ]
    },
    third: {
      examples: [
        "Each supplier's datasheet defines which crystal direction the primary flat marks, and whether there is a secondary flat",
        'Rotated cuts, such as the 128 degree Y-cut used for surface acoustic wave wafers, are named differently from optical X-cut and Z-cut',
        'Tensor signs in supplied data may follow IEEE Std 176-1987 or an older convention',
        'In ion-sliced thin film, the implanted face of the donor crystal is bonded down, so the exposed film surface is the opposite polar face'
      ],
      examplesSay: [
        "Examples. Each supplier's datasheet defines which crystal direction the primary flat marks, and whether there is a secondary flat.",
        "Rotated cuts, such as the one hundred twenty eight degree Y-cut used for surface acoustic wave wafers, are named differently from optical X-cut and Z-cut.",
        "Tensor signs in supplied data may follow I triple E standard one seventy six, from nineteen eighty seven, or an older convention.",
        "And in thin film made by ion slicing, the implanted face of the donor crystal is bonded down, so the exposed film surface is the opposite polar face."
      ],
      warnings: ['the datasheet is not attached to the traveler', 'two suppliers, one mask', '"it says X-cut on the box"', 'nobody knows which face of the thin film is plus z'],
      warningsSay: ["A datasheet that is not attached to the traveler.", "Two suppliers, one mask.", "The phrase, it says X-cut on the box.", "And nobody knowing which face of the thin film is plus z."],
      callout: 'X-cut names the axis normal to the wafer. It does not name the flat.',
      chat: {
        type: 'chat', title: 'Category 3 vignette: Same cut, new supplier', app: 'email',
        meta: {"header": "Inbox", "subject": "Cost reduction: new x-cut wafer supplier", "participants": ["Gary", "Dana", "Manager"]},
        messages: [
          {"from": "Gary", "text": "Hi all,\n\nFor the cost reduction: we have switched to the new supplier's x-cut wafers starting with lot 11. Same cut, same mask, about 20 percent cheaper.\n\nGary", "side": "left", "to": ["Manager", "Dana Okafor"], "date": "2026-11-03 14:20", "subject": "Cost reduction: new x-cut wafer supplier", "attachments": [{"name": "supplier_quote_xcut_100mm.pdf", "size": "410 KB"}]},
          {"from": "Dana", "text": "Hi Gary,\n\nDid anyone compare the new datasheet's flat definition with the old one?\n\nDana", "side": "right", "to": ["Gary"], "cc": ["Manager"], "date": "2026-11-03 15:02", "subject": "RE: Cost reduction: new x-cut wafer supplier"},
          {"from": "Gary", "text": "Both datasheets say x-cut, 100 mm, same thickness. The flats are standard.", "side": "left", "to": ["Dana Okafor"], "cc": ["Manager"], "date": "2026-11-03 15:30", "subject": "RE: RE: Cost reduction: new x-cut wafer supplier"},
          {"from": "Dana", "text": "X-cut says which axis is normal to the wafer. It does not say which in-plane direction the flat marks. That is in the flat specification, and the new one is different from the old one.\n\nDana", "side": "right", "to": ["Gary"], "cc": ["Manager"], "date": "2026-11-03 16:05", "subject": "RE: RE: Cost reduction: new x-cut wafer supplier", "attachments": [{"name": "flat_spec_comparison.pdf", "size": "220 KB"}]},
          {"from": "Manager", "text": "Thanks Dana. Please attach both datasheets to the traveler and add an incoming flat check for lot 11 onward.", "side": "left", "to": ["Gary", "Dana Okafor"], "date": "2026-11-03 16:40", "subject": "RE: RE: Cost reduction: new x-cut wafer supplier"}
        ],
        narration: ["The new supplier's datasheet defines the flat at a different crystal direction. Nobody compared it with the old one, and the old mask was used.", "X-cut names the axis normal to the wafer. It does not name the flat. Know your axes."]
      },
      classifyQ: "A team switched to a new supplier's x-cut wafers. The new datasheet defines the primary flat at a different crystal direction than the old supplier's. Nobody compared them, and the old mask was used.",
      wrong: [
        ['cross', 'In this incident, the material came from a commercial supplier under its own datasheet, not from a collaborating institution exchanging designs or dies.'],
        ['tool', 'In this incident, no tool turned anything. Every tool placed the wafer exactly by its flat. The flat meant something different.'],
        ['peer', 'In this incident, the convention came from outside the group, in a supplied document. Peer-to-peer misconduct passes inside the group through shared containers and templates.']
      ]
    },
    peer: {
      examples: [
        'A shared gel-pak with unmarked dies placed in any orientation',
        'A run-sheet template with no orientation column',
        'The previous shift loaded the chip and left no note',
        "A simulation template copied from last year, in last year's cut"
      ],
      examplesSay: [
        "Examples. A shared gel-pak with unmarked dies, placed in any orientation.",
        "A run sheet template with no orientation column.",
        "A previous shift that loaded the chip and left no note.",
        "And a simulation template copied from last year, in last year's cut."
      ],
      warnings: ['"someone already loaded it"', 'unlabeled gel-pak pockets', 'templates inherited, never reviewed', 'orientation as tribal knowledge'],
      warningsSay: ["The phrase, someone already loaded it.", "Unlabeled gel-pak pockets.", "Templates inherited and never reviewed.", "And orientation kept as tribal knowledge."],
      callout: 'If the orientation does not travel with the die, the die travels alone.',
      chat: {
        type: 'chat', title: 'Category 4 vignette: The way I always put it', app: 'thread',
        meta: { header: '#lab-ops', participants: ['Gary', 'Previous shift'] },
        messages: [
          {"from": "Gary", "text": "who put the lot 9 dies in the shared gel-pak", "side": "left"},
          {"from": "Previous shift", "text": "me, last night. they are all good dies", "side": "right"},
          {"from": "Gary", "text": "which end is the input facet", "side": "left"},
          {"from": "Previous shift", "text": "same as last week, label toward the front", "side": "right"},
          {"from": "Gary", "text": "the label is on both ends on this mask", "side": "left"},
          {"from": "Previous shift", "text": "then input toward the front, I think", "side": "right"}
        ],
        narration: ["Orientation was kept in one person's head and placed in a shared gel-pak without marks or a note.", "If the orientation does not travel with the die, the die travels alone. Know your axes."]
      },
      classifyQ: 'A coworker on the previous shift placed dies in a shared gel-pak with no marks and no note. Gary loaded one turned around.',
      wrong: [
        ['dom', "In this incident, the orientation information was lost between two members of the same group, at a shared container. Gary did load it, but the reference was never passed to him."],
        ['tool', 'In this incident, the gel-pak held each die exactly where it was placed. A gel-pak is a container, not a tool that turns anything.'],
        ['qpq', 'In this incident, nothing was offered or exchanged for skipping a check. Nobody checked, and nobody was asked not to.']
      ]
    },
    cross: {
      examples: [
        'A partner foundry ships dies drawn in its own layout coordinates, with zero degrees along its own chip edge',
        'A collaborator sends tensor constants in a different sign convention',
        'Images taken through the substrate are mirror images of the top view, so plus angles become minus angles',
        'A thirty-message email thread in which nobody states which way plus z points'
      ],
      examplesSay: [
        "Examples. A partner foundry ships dies drawn in its own layout coordinates, with zero degrees along its own chip edge.",
        "A collaborator sends tensor constants in a different sign convention.",
        "Images taken through the substrate are mirror images of the top view, so plus angles become minus angles.",
        "And a thirty message email thread in which nobody states which way plus z points."
      ],
      warnings: ['"as discussed"', 'figures with no axis marker', 'two groups, two definitions of zero degrees', '"standard orientation"'],
      warningsSay: ["The phrase, as discussed.", "Figures with no axis marker.", "Two groups with two definitions of zero degrees.", "And the phrase, standard orientation."],
      callout: 'There is no standard orientation. There is only a written one.',
      chat: {
        type: 'chat', title: 'Category 5 vignette: Standard orientation', app: 'email',
        meta: {"header": "Inbox", "subject": "Dies shipped", "participants": ["Partner lab", "Gary", "Dana"]},
        messages: [
          {"from": "Partner lab", "text": "Hi Gary,\n\nThe dies shipped this morning, in standard orientation as discussed. Tracking number below.\n\nBest regards,\nPartner lab", "side": "left", "to": ["Gary"], "cc": ["Dana Okafor"], "date": "2026-10-28 08:30", "subject": "Dies shipped", "attachments": [{"name": "packing_list_batch3.pdf", "size": "96 KB"}]},
          {"from": "Gary", "text": "Thanks! One oddity: our plus 45 degree devices behave like our minus 45 degree simulations. Probably our model.\n\nGary", "side": "left", "to": ["Partner lab"], "cc": ["Dana Okafor"], "date": "2026-11-02 17:10", "subject": "RE: Dies shipped"},
          {"from": "Dana", "text": "Hi both,\n\nHow do you measure the angle on your side? We measure from crystal y toward z, viewed from the top.\n\nDana", "side": "right", "to": ["Partner lab"], "cc": ["Gary"], "date": "2026-11-03 09:05", "subject": "RE: RE: Dies shipped"},
          {"from": "Partner lab", "text": "From the chip edge, counterclockwise, viewed from the back. We image through the substrate.", "side": "left", "to": ["Dana Okafor"], "cc": ["Gary"], "date": "2026-11-03 13:40", "subject": "RE: RE: RE: Dies shipped"},
          {"from": "Dana", "text": "Counterclockwise from the back is clockwise from the top, so your plus 45 is our minus 45. On x-cut, those are different devices. Could we put a frame sketch in the shipping note from now on?", "side": "right", "to": ["Partner lab"], "cc": ["Gary"], "date": "2026-11-03 14:02", "subject": "RE: RE: RE: Dies shipped"},
          {"from": "Gary", "text": "Thanks, that explains it. I will relabel our data.", "side": "left", "to": ["Partner lab", "Dana Okafor"], "date": "2026-11-03 14:20", "subject": "RE: RE: RE: Dies shipped"}
        ],
        narration: ["Two institutions, two definitions of zero degrees, and one mirror. On x-cut, plus forty five and minus forty five degrees are different devices.", "There is no standard orientation. There is only a written one. Know your axes."]
      },
      classifyQ: 'A partner university shipped dies measured from its own chip edge, viewed through the substrate. The email thread said "standard orientation, as discussed". Plus 45 degrees became minus 45 degrees.',
      wrong: [
        ['third', 'In this incident, the dies and their convention came from a collaborating institution exchanging work, not from a commercial supplier under a datasheet.'],
        ['soft', 'In this incident, no software frame was wrong. Two institutions defined angles differently, and one of them viewed the chip from the back.'],
        ['peer', 'In this incident, the hand-off crossed an organizational boundary. Peer-to-peer misconduct stays inside one lab, group, or shift.']
      ]
    },
    dom: {
      examples: [
        'Your own fiber rotation mount, set from memory',
        'Your own notebook, where TE was written before you knew which way the chip faced',
        'Your own die mark, made on the wrong corner',
        'Your own script, which you wrote and therefore did not test'
      ],
      examplesSay: [
        "Examples. Your own fiber rotation mount, set from memory.",
        "Your own notebook, where T E was written down before you knew which way the chip faced.",
        "Your own die mark, made on the wrong corner.",
        "And your own script, which you wrote, and therefore did not test."
      ],
      warnings: ['"I always do it this way"', 'no second check, because there is no second person', 'notebook entries without a sketch', 'relief that nobody saw'],
      warningsSay: ["The phrase, I always do it this way.", "No second check, because there is no second person.", "Notebook entries without a sketch.", "And relief that nobody saw."],
      callout: 'Domestic misconduct has no witnesses. Log it anyway.',
      chat: {
        type: 'chat', title: 'Category 6 vignette: Notes to self', app: 'phone',
        meta: { header: 'Notes to self', participants: ['Gary'] },
        messages: [
          {"from": "Gary", "text": "note: chip 4 loaded, input on the left", "side": "right"},
          {"from": "Gary", "text": "correction: input on the right?", "side": "right"},
          {"from": "Gary", "text": "the mark is on the other corner. today's chip 4 data may be mirrored", "side": "right"},
          {"from": "Gary", "text": "logged in the run sheet: chip 4, today's data loaded mirrored, flagged, rows kept. re-measure tomorrow", "side": "right"}
        ],
        narration: ["Read Gary's notes to himself, and judge whether his response is proper.", "Gary found his own loading error, and nobody else saw it. Is this okay?", "Yes. Gary is the source, the victim, and the only witness. He logged it anyway, because domestic misconduct has no witnesses. Know your axes."]
      },
      classifyQ: 'Gary marked his own chip on the wrong corner, loaded it from his own notes, and was the only person who knew until he logged it.',
      wrong: [
        ['peer', 'In this incident, no coworker, shared container, or shared template was involved. Gary is the source and the affected party.'],
        ['tool', 'In this incident, no tool turned anything. Gary made the mark, read the mark, and loaded the chip.'],
        ['hostile', 'In this incident, one person made one error in one setup. A hostile crystal environment is severe or pervasive across a workplace.']
      ]
    },
    qpq: {
      examples: [
        '"Take my aligner slot if you sign off the flat check without the microscope"',
        '"Put me on the paper and I will not mention the frame"',
        '"We make Friday if nobody re-measures the RF response"',
        '"Skip the slow ramp and I will cover your cleanroom shift"'
      ],
      examplesSay: [
        "Examples. Take my aligner slot if you sign off the flat check without the microscope.",
        "Put me on the paper and I will not mention the frame.",
        "We make Friday if nobody re-measures the R F response.",
        "And, skip the slow ramp and I will cover your cleanroom shift."
      ],
      warnings: ['schedule offered in exchange for a check', '"just this once"', 'co-authorship mentioned in the same message as a measurement', 'requests to keep something off the run sheet'],
      warningsSay: ["Schedule offered in exchange for a check.", "The phrase, just this once.", "Co-authorship mentioned in the same message as a measurement.", "And requests to keep something off the run sheet."],
      callout: 'A check cannot be traded. It can only be done.',
      chat: {
        type: 'chat', title: 'Category 7 vignette: The stepper slot', app: 'phone',
        meta: { header: 'Gary', participants: ['Gary', 'Dana'] },
        messages: [
          {"from": "Gary", "text": "can you co-sign the lot 12 flat check? I looked at the wafers this morning, they are fine. happy to swap you my stepper slot tomorrow, I know you need it", "side": "left"},
          {"from": "Dana", "text": "did you check the flats against the traveler", "side": "right"},
          {"from": "Gary", "text": "visually, yes. they look right", "side": "left"},
          {"from": "Dana", "text": "thanks for the slot offer, but I cannot co-sign a check I did not see. I can do it with you now, it takes ten minutes", "side": "right"},
          {"from": "Gary", "text": "fair. 10 minutes it is", "side": "left"}
        ],
        narration: ["Gary offered tool time in the same message as a request to co-sign a check Dana did not see. Dana declined the trade and offered to do the check.", "A check cannot be traded. It can only be done. Know your axes."]
      },
      classifyQ: 'Gary offered his stepper slot to a coworker in exchange for signing off a flat check that was not performed.',
      wrong: [
        ['peer', 'Both people are in the same group, so it can look like peer-to-peer. But the defining feature here is the exchange: a benefit offered for a skipped check. That makes it quid pro quo, which the policy treats more seriously.'],
        ['hostile', 'In this incident, one specific offer was made for one specific check. A hostile crystal environment is a pervasive condition, not a single trade.'],
        ['dom', 'In this incident, a second person was asked to take part. Domestic misconduct involves only your own setup and your own records.']
      ]
    },
    hostile: {
      examples: [
        'Hotplates without ramp control, so every bake is a temperature step',
        'No shorting caps or clips, so electrodes float during every thermal cycle',
        'Dies stored in shared boxes with no markings',
        'Schedules with no time for a flat check'
      ],
      examplesSay: [
        "Examples. Hotplates without ramp control, so every bake is a temperature step.",
        "No shorting caps or clips, so electrodes float during every thermal cycle.",
        "Dies stored in shared boxes with no markings.",
        "And schedules with no time for a flat check."
      ],
      warnings: ['sparks described as normal', 'orientation errors joked about but never logged', '"you get used to it"', 'the aluminum foil is for lunch'],
      warningsSay: ["Sparks described as normal.", "Orientation errors joked about but never logged.", "The phrase, you get used to it.", "And aluminum foil that is for lunch."],
      callout: 'You should not get used to it.',
      chat: {
        type: 'chat', title: 'Category 8 vignette: You get used to it', app: 'thread',
        meta: { header: '#cleanroom', participants: ['New hire', 'Gary', 'Manager', 'Dana'] },
        messages: [
          {"from": "New hire", "text": "is it normal that the z-cut wafers crackle when they come off the hotplate", "side": "right"},
          {"from": "Gary", "text": "yes, that is lithium niobate. everyone sees it", "side": "left"},
          {"from": "New hire", "text": "and the dies in the white box, which way is up", "side": "right"},
          {"from": "Gary", "text": "they are all the same die, it does not really matter for screening", "side": "left"},
          {"from": "Manager", "text": "good questions. let us pick this up after tapeout", "side": "left"},
          {"from": "Dana", "text": "we said after tapeout for the last three tapeouts", "side": "left"}
        ],
        narration: ["Read this channel, and judge the answers the new hire gets.", "Crackling wafers, unmarked dies, and no time to fix either. Is this okay?", "No. When correct conduct is impossible for everyone, the environment is the incident. You should not get used to it. Watch your charge."]
      },
      classifyQ: 'In one lab, hotplates have no ramp control, wafers crackle when they come off, dies are stored unmarked, and new hires are told "you get used to it".',
      wrong: [
        ['tool', 'A hotplate without ramp control is part of it, but this incident is not one tool turning one wafer. It is many conditions, together, making correct conduct impossible.'],
        ['peer', 'Coworkers repeat "you get used to it", but the problem is not a single hand-off between two people. It is pervasive.'],
        ['dom', 'No single person here can fix it in their own setup. That is what makes it an environment.']
      ]
    }
  };

  // Action checks, one per two categories
  var srcAct = {
    afterSoft: {
      type: 'check', kicker: 'What would you do?',
      question: 'You are Gary. You discover that your "x-cut" simulations were z-cut simulations. The results are already in a design review deck sent to three groups. What do you do?',
      narration: ["You are Gary.", "Your x-cut simulations were z-cut simulations, and three groups have the deck.", "What would you do?"],
      options: [
        { text: 'Re-run in the stated frame; validate against a known case (bulk TE along y on x-cut sees $n_e$); send a correction to everyone who received the deck; log it; put the cut, frame, and Euler angles on the first slide from now on', correct: true, why: '' },
        { text: 'File a bug report: the solver default should be x-cut', correct: false, why: "The solver has no way to know what cut your wafer is. A diagonal anisotropic material entered in the simulation's own axes is a documented, reasonable input, and the solver solved exactly the device you described. A bug report would be closed as working as intended, and your deck would still be wrong in three groups' inboxes. Check every material frame against a known case, such as the bulk T E index along y on x-cut, which should be the extraordinary index. Then send the correction. Know your categories. Know your axes." },
        { text: "Replace the slides in the shared folder with corrected ones, marked v8", correct: false, why: "Versioning the deck is good practice. But three groups already downloaded version seven, and some may already be designing around a crossing that does not exist on your wafer. A new version in a folder is not a correction unless the people with the old one are told what changed and why. Send the corrected deck with a one-line explanation, and log it. Know your axes." },
        { text: 'Add a footnote: "Crystal frame details available on request"', correct: false, why: "A footnote that hides the frame is the opposite of the procedure. The frame belongs on the first slide, in plain words: the cut, the propagation axis, and the Euler angles, with their convention. And the existing results are wrong, so they need a correction, not a footnote. Know your axes." },
        { text: "Blame last year's template, which someone else made", correct: false, why: "An inherited template can contribute, and if it did, that is worth logging as peer-to-peer misconduct too. But you ran the simulation and you sent the results, and the check against a known case is the user's step. The fix is the same either way: re-run, validate, correct, log. Know your categories." },
        { text: "Cross-check with a second solver before telling anyone", correct: false, why: "A second solver is a good cross-check of numerics. It is not a check of your input. Given the same diagonal tensor with crystal z on the film normal, a second solver will solve the same z-cut device and agree with the first one, which will feel like confirmation. Check the frame against a known case, such as the bulk T E index along y on x-cut, and tell the three groups now, not after the cross-check. Know your axes." }
      ],
      correctSay: "Fix the frame, validate against a known case, and tell everyone who has the old deck. " + RESPOND
    },
    afterPeer: {
      type: 'check', kicker: 'You are Dana',
      question: 'You see a coworker taking dies from the shared, unmarked gel-pak and loading them into the probe station without looking for a die mark. What do you do?',
      narration: ["You are Dana.", "A coworker is loading dies from the shared unmarked gel-pak, without checking orientation.", "What would you do?"],
      options: [
        { text: 'Speak up now: "' + PHRASE_O + '" Help check the orientation of the loaded die, and propose marked dies and an orientation column in the run sheet', correct: true, why: '' },
        { text: "Say nothing. The coworker loads these dies every day and knows them", correct: false, why: "Daily experience is real. But these dies are unmarked and come from a shared gel-pak, so nobody can know their orientation from experience alone, including the person who loads them every day. Checking costs a minute now. Use the approved phrase. Pause, please. Orientation check." },
        { text: 'Come back at night and mark every die yourself', correct: false, why: "Your initiative is admirable. But marking dies whose orientation is already unknown records a guess as a fact. And doing it alone, at night, leaves no record of who decided what. Orientation is recovered by checking each die against a feature you can verify, together, and recording it. Speak up now. " + PHRASE_O },
        { text: "Post a general reminder about die marking in the lab channel", correct: false, why: "A general reminder is polite and avoids singling anyone out. But the die being loaded right now is still unmarked, and a reminder read tomorrow does not check it. Speak directly, kindly, and now, and then propose the marking fix for everyone. Pause, please. Orientation check." },
        { text: "Wait to see if their results look bad before saying anything", correct: false, why: "A modulator chip loaded turned around can give a perfect D C result and a quietly wrong R F result. Waiting for results to look bad means waiting for the wrong kind of result. Speak up before the measurement. " + PHRASE_O }
      ],
      correctSay: PHRASE_O + " And then fix the container, not just the die."
    },
    afterDom: {
      type: 'check', kicker: 'What would you do?',
      question: 'You are Gary. Months later, you find that your own notebook recorded angles as seen through the substrate, mirrored. A paper draft with your angle labels is circulating among co-authors. What do you do?',
      narration: ["You are Gary.", "Your notebook angles were mirrored, and a paper draft is circulating.", "What would you do?"],
      options: [
        { text: 'Tell the co-authors now, with the affected figures and the corrected labels; log it; add a viewing-direction sketch to every notebook page that records an angle', correct: true, why: '' },
        { text: 'Fix the labels in the next draft without mentioning why', correct: false, why: "Silent edits to angle labels in a paper draft look, to co-authors, like random changes, and some may reverse them. Others may have already used the old labels in their own analysis or simulations. Say what changed and why: the notebook recorded the mirrored view from the back side, so plus and minus angles were swapped. On x-cut, those are different devices. Know your axes." },
        { text: 'Blame the partner lab, since they image through the substrate too', correct: false, why: "The partner lab's convention may be a cross-institutional issue in general, but in this case the notebook is yours, and you recorded the mirrored view yourself. That is domestic misconduct. Assigning it to another institution would be inaccurate and would send the correction to people who did nothing wrong. Know your categories." },
        { text: "Rewrite the affected notebook pages neatly with the corrected angles", correct: false, why: "Neat records are good, and rewriting looks like a correction. But the original pages are the evidence of what was measured and how it was labeled, and that is exactly what your co-authors need to correct the paper. Leave the originals, add a dated correction note on each page, and tell the co-authors. Know your axes." },
        { text: 'Do nothing, since the effect might be small', correct: false, why: "On x-cut lithium niobate, plus forty five and minus forty five degrees are not related by a symmetry of the crystal, so swapped labels can attach the wrong device to the wrong measurement. Whether the effect is small is something you find out after you tell your co-authors, not before. Know your axes." }
      ],
      correctSay: "Domestic misconduct has no witnesses, so you become the reporter. " + RESPOND
    },
    afterHostile: {
      type: 'check', kicker: 'What would you do?',
      question: 'You are the new hire. In your first week you see crackling wafers off a hotplate with no ramp control, unmarked dies, and no shorting caps on the test fixtures. What do you do?',
      narration: ["You are the new hire.", "Crackling wafers, unmarked dies, floating electrodes, and a team that says you get used to it.", "What would you do?"],
      options: [
        { text: 'Log each observation, raise them with the process owner and your manager in writing with specific requests (ramp control, shorting caps, die marking, time for checks), and use the Photonic Conduct Speak-Up Line if nothing changes', correct: true, why: '' },
        { text: "Follow the existing habits for now. You are new, and the team knows the tools", correct: false, why: "Learning before you judge is a reasonable instinct for a new hire. But crackling wafers are pyroelectric discharges, unmarked dies are orientation incidents waiting to happen, and floating electrodes during thermal cycling are how driver amplifiers fail. You are new, which means you can still see it. Write down what you see, and raise it now. Watch your charge." },
        { text: 'Quietly bring your own ramp-controlled hotplate and shorting caps, and say nothing', correct: false, why: "Protecting your own work is understandable, and your devices will be safer. But everyone else's will not, and an unofficial tool in a shared lab is its own problem. A hostile crystal environment is fixed at the level of the environment: procedures, equipment, and time. Raise it in writing with specific requests." },
        { text: "Raise it informally with a senior colleague first, and see if it comes up again", correct: false, why: "Asking a senior colleague is a good first step for context. But in this lab the senior colleagues already say you get used to it, and seeing if it comes up again means waiting for the next spark. Informal conversations create no record and assign no owner. Write it down, with specific requests, and send it to the process owner and your manager. Watch your charge." },
        { text: "Wait until your ninety-day review to raise it, when you have more standing", correct: false, why: "Waiting until you have standing feels sensible. But every week until then is more bakes without ramp control and more fixtures without shorting caps. Good-faith reports are protected from your first day, whatever your tenure. Raise it now, in writing. Watch your charge." }
      ],
      correctSay: "Specific, written, and escalated if nothing changes. You should not get used to it. Watch your charge."
    }
  };

  CAT_ORDER.forEach(function (k, i) {
    chSrc.steps = chSrc.steps.concat(catSteps(k, i + 1, CAT_CONTENT[k]));
    if (k === 'soft') chSrc.steps.push(srcAct.afterSoft);
    if (k === 'peer') chSrc.steps.push(srcAct.afterPeer);
    if (k === 'dom') chSrc.steps.push(srcAct.afterDom);
    if (k === 'hostile') chSrc.steps.push(srcAct.afterHostile);
  });

  chSrc.steps = chSrc.steps.concat([
    {
      type: 'takeaways', title: 'Know your categories',
      items: CAT_ORDER.map(function (k) { return CAT[k].name; }).concat(['Classify by source; respond the same way: ' + RESPOND, M]),
      narration: [
        "Let's review the eight categories of orientation misconduct.",
        "One. Tool-initiated. A tool turned it, and nobody re-checked.",
        "Two. Software-facilitated. A program used a frame you did not check.",
        "Three. Third-party. A supplier's material or convention entered without verification.",
        "Four. Peer-to-peer. Orientation did not travel with the die inside the group.",
        "Five. Cross-institutional. Orientation did not travel with the die across a boundary.",
        "Six. Domestic. You, your bench, your notebook.",
        "Seven. Quid pro quo. A check traded for a benefit.",
        "Eight. Hostile crystal environment. Correct conduct made impossible.",
        "Once more, faster. " + CATS_SAY,
        "Eight categories. One response. " + RESPOND,
        "And, of course. " + M,
        "Click Next to continue."
      ]
    },
    ack('the sources of orientation misconduct'),
    refs([R.ieee176, R.zhu2021])
  ]);

  /* ---- wire incidents into their chapters ---- */
  insertBeforeTakeaways(ch1.steps, incFlat);
  insertBeforeTakeaways(ch2.steps, incRods);
  insertBeforeTakeaways(ch4.steps, incHotPlug);
  insertBeforeTakeaways(ch5.steps, incTurned);

  /* ======================= MEDIA (photos, drawn images, interactive scenes) ======================= */
  // Images are registered in media/credits_fe.js (photos and the drawn solver window).
  // Blender renders are registered in media/credits_render.js.

  function findStep(steps, key) {
    for (var i = 0; i < steps.length; i++) {
      var s = steps[i];
      if (s.title === key || s.question === key) return i;
    }
    throw new Error('Media anchor not found: ' + key);
  }
  function addAfter(steps, key, extra) {
    var i = findStep(steps, key);
    Array.prototype.splice.apply(steps, [i + 1, 0].concat(extra));
  }
  function addBefore(steps, key, extra) {
    var i = findStep(steps, key);
    Array.prototype.splice.apply(steps, [i, 0].concat(extra));
  }
  function photoStep(title, src, caption, narration, kb) {
    var st = { type: 'photo', title: title, image: { src: src }, caption: caption, narration: narration };
    if (kb) st.kb = kb;
    return st;
  }
  function splitStep(title, src, bullets, narration, callout) {
    var st = { type: 'slide', layout: 'split', title: title, heading: title, image: { src: src }, bullets: bullets, narration: narration };
    if (callout) st.callout = callout;
    return st;
  }
  function heroStep(title, src, subtitle, narration, kicker) {
    var st = { type: 'slide', layout: 'hero', title: title, image: { src: src }, subtitle: subtitle, narration: narration };
    if (kicker) st.kicker = kicker;
    return st;
  }

  /* ---- Module openers: a hero title card at the start of every module ---- */
  [
    [ch1, 1, 'media/render/ln_wafer_carrier.jpg', [0.44, 0.62, 1.15, 0.40, 0.70, 1.3], ["Module one. Crystal Orientation Awareness.", "Every wafer arrives with a flat. Every flat arrives with an opinion about your crystal axes."]],
    [chSrc, 2, 'media/fe/orient_hotspot_mask_aligners.jpg', [0.5, 0.4, 1.0, 0.45, 0.35, 1.2], ["Module two. Sources of Orientation Misconduct.", "Tools, software, suppliers, coworkers, partners, and you. Eight categories. One response."]],
    [ch2, 3, 'media/fe/opener_bends_drawn.jpg', [0.5, 0.35, 1.0, 0.5, 0.3, 1.12], ["Module three. Respecting Boundaries: Mode Conversion in Bends.", "On x-cut, every bend is also a change of crystal direction."]],
    [ch3, 4, 'media/fe/opener_photorefractive_drawn.jpg', [0.5, 0.3, 1.0, 0.45, 0.3, 1.12], ["Module four. Photorefractive Conduct.", "The light you send writes back. Scan both ways."]],
    [ch4, 5, 'media/render/ln_wafer_hotplate.jpg', [0.5, 0.4, 1.0, 0.58, 0.25, 1.4], ["Module five. Pyroelectric Awareness.", "No voltage has been applied. Charge is being generated anyway."]],
    [ch5, 6, 'media/render/tfln_probe_station_wide.jpg', [0.5, 0.5, 1.0, 0.5, 0.6, 1.3], ["Module six. Bias Point Retention.", "Your D C is not your R F. Please hold that thought, and your bias point."]],
    [ch6, 7, 'media/fe/sim_solver_window_drawn.jpg', [0.5, 0.4, 1.0, 0.6, 0.35, 1.25], ["Module seven. Simulation Integrity.", "Solved without errors is not the same as solved correctly."]]
  ].forEach(function (o) {
    var ch = o[0];
    ch.steps.unshift({
      type: 'slide', layout: 'hero', title: ch.title, heading: ch.title, kicker: 'Module ' + o[1],
      subtitle: ch.subtitle, image: { src: o[2], kb: o[3] }, narration: o[4]
    });
  });

  /* ---- Module 0: Welcome ---- */
  ch0.steps[0].image = { src: 'media/fe/welcome_litho_bay.jpg' };
  ch0.steps[0].subtitle = 'Annual mandatory refresher. Five Sigma Photonics, Inc.';
  addAfter(ch0.steps, 'Course enforcement rules', [
    photoStep('Everyone is responsible', 'media/fe/welcome_fiber_test_floor.jpg',
      'Photonics is a team activity. Orientation is everyone\'s responsibility.',
      ["Photonics is a team activity.", "Orientation is everyone's responsibility. " + M],
      [0.5, 0.5, 1.0, 0.6, 0.45, 1.15])
  ]);
  (function () {
    var i = findStep(ch0.steps, 'Our mantra');
    ch0.steps[i].layout = 'hero';
    ch0.steps[i].image = { src: 'media/shared/open_plan_office.jpg' };
    ch0.steps[i].subtitle = 'It applies at every desk, including the empty ones.';
  })();

  /* ---- Module 1: Crystal Orientation Awareness ---- */
  addAfter(ch1.steps, 'Let\'s take a closer look', [
    photoStep('A real wafer, a real flat', 'media/fe/orient_ln_wafer_zcut.jpg',
      'A z-cut lithium niobate wafer. Note the flat.',
      ["This is a lithium niobate wafer, labeled z-cut. Note the flat.", "The supplier specification states which crystal direction the flat marks. Read it before you need it. Know your axes."],
      [0.5, 0.55, 1.0, 0.45, 0.8, 1.35])
  ]);
  addAfter(ch1.steps, 'Four orientation hazards', [
    splitStep('The flat is the only reference', 'media/fe/orient_wafer_flat.jpg',
      ['Every die on this wafer knows where the flat is', 'After dicing, none of them will', 'Mark the die before it leaves the wafer'],
      ["Every die on this wafer knows where the flat is. After dicing, none of them will.", "Mark the die before it leaves the wafer. " + M])
  ]);
  // The incident "The flat at three o'clock" takes place in a lithography bay: tour one.
  (function () {
    var steps = ch1.steps, i = -1;
    for (var k = 0; k < steps.length; k++) if (steps[k].title === 'What went wrong, technically') { i = k; break; }
    if (i < 0) throw new Error('Media anchor not found: flat incident reveal');
    steps.splice(i + 1, 0, {
      type: 'hotspots', title: 'Tour the lithography bay', style: 'numbered',
      prompt: 'Open every numbered point to continue.',
      image: { src: 'media/fe/orient_hotspot_mask_aligners.jpg' },
      narration: ["This is a real lithography bay, with two mask aligners. Open every numbered point to continue."],
      spots: [
        { x: 0.27, y: 0.40, label: 'Alignment microscope', body: '<p>The aligner\'s microscope lines up marks on the mask with marks on the wafer. It aligns to the pattern you already made, not to the crystal. If layer one went down 90 degrees from the flat, every later layer lines up perfectly with the mistake.</p>', narration: ["The alignment microscope lines up the mask with marks you already made. It aligns to your pattern, not to the crystal.", "If layer one went down ninety degrees from the flat, every later layer lines up perfectly with the mistake."] },
        { x: 0.29, y: 0.47, label: 'Mask holder and wafer stage', body: '<p>A round wafer is placed with its flat against a reference. That is the only moment the crystal axes enter the process. A diced piece has no flat. Load it by its die mark, or do not load it.</p>', narration: ["This is the only moment the crystal axes enter the process: the flat against the reference.", "A diced piece has no flat. Load it by its die mark, or do not load it."] },
        { x: 0.60, y: 0.33, label: 'Second mask aligner', body: '<p>Two aligners, two loading habits. Before you move a job from one tool to the other, confirm which edge each one treats as down.</p>', narration: ["Two aligners, two loading habits. Before you move a job between them, confirm which edge each one treats as down."] },
        { x: 0.50, y: 0.24, label: 'Video monitor', body: '<p>A camera image can be rotated or mirrored relative to the stage. Check it against a known mark before you use it to judge a direction.</p>', narration: ["A camera image can be rotated or mirrored relative to the stage. Check it against a known mark before you trust it with a direction."] },
        { x: 0.80, y: 0.24, label: 'Inspection microscopes', body: '<p>Inspect after exposure and before etch. Check that the waveguides run in the intended direction relative to the flat. Resist can be stripped and redone. An etch cannot.</p>', narration: ["Inspect after exposure and before etch. Resist can be stripped and redone. An etch cannot."] },
        { x: 0.88, y: 0.06, label: 'Yellow light', body: '<p>The room lighting leaves out the short wavelengths that expose photoresist. It does not leave out orientation errors.</p>', narration: ["The yellow light leaves out the wavelengths that expose photoresist. It does not leave out orientation errors. Know your axes."] }
      ]
    });
  })();
  incFlat.forEach(function (st) { if (/^You are Dana/.test(st.kicker || '')) st.image = { src: 'media/fe/orient_microscope_operator.jpg' }; });
  // Renders (media/credits_render.js). The die cards use crops of the fixture renders (media/fe/card_die_fixture_*.jpg,
  // made from media/render/tfln_die_fixture_*.jpg, box x 440-1400, y 160-880); re-crop if the renders change.
  addAfter(ch1.steps, 'The flat is the only reference', [{
    type: 'cards', title: 'Which chip is oriented correctly?', style: 'flip', mode: 'choose',
    prompt: 'Three identical modulator dies in the same fixture. Which one has been loaded in accordance with the crystal orientation policy?',
    narration: ["Three identical dies. One fixture. The fixture is engraved with plus z, I N, O U T, and R F.", "Which die has been loaded in accordance with the crystal orientation policy?"],
    cards: [
      { image: { src: 'media/fe/card_die_fixture_90deg.jpg' }, label: 'Die A', correct: false,
        why: "Die A is turned ninety degrees. Turning a finished die does not change its crystal physics: the waveguide and the electrodes turn with the crystal. It changes everything around it. The polished facets now face the closed walls of the pocket, the I N and O U T slots line up with no waveguide at all, and the pads are not where the probes land. A ninety degree turn that changes the physics happens earlier, at the mask aligner." },
      { image: { src: 'media/fe/card_die_fixture_0deg.jpg' }, label: 'Die B', correct: true,
        why: "Die B. Both plus z arrows agree, the waveguide runs on the line between the I N and O U T slots, and the R F input pads sit at the I N end, as the layout says." },
      { image: { src: 'media/fe/card_die_fixture_180deg.jpg' }, label: 'Die C', correct: false,
        why: "Die C is turned one hundred eighty degrees. Its plus z arrow points against the fixture's. For a finished die, that alone does not change the electro-optic response measured from its own pads, and the D C V pi is the same. What changes is everything the fixture assumes. The waveguide no longer sits on the slot axis, so the fibers miss the facets. And the pad set the layout calls the R F input is now at the O U T end, so whether the R F wave travels with the light now depends on which pads the probe lands on, not on the record." }
    ],
    correctSay: "Die B. Both arrows agree, the waveguide sits on the slot axis, and the R F input is at the I N end. Know your axes."
  }]);
  incFlat.forEach(function (st) {
    if (st.type === 'video') st.keyframes[1].image = { src: 'media/render/ln_wafer_carrier.jpg', kb: [0.46, 0.5, 1.0, 0.36, 0.72, 1.5] };
  });

  /* ---- Sources of Orientation Misconduct ---- */
  (function () {
    var i = findStep(chSrc.steps, 'Orientation misconduct can come from anywhere');
    chSrc.steps[i].layout = 'hero';
    chSrc.steps[i].image = { src: 'media/shared/cube_farm.jpg' };
    chSrc.steps[i].subtitle = 'Eight categories. One response.';
    chSrc.steps[i].kicker = 'Policy PHC-17.3';
  })();
  var toolWhy = " It does exactly what it is told, and it does not know the crystal axes of your wafer.";
  addAfter(chSrc.steps, 'Category 1 vignette: The prealigner', [{
    type: 'cards', title: 'Which of these knows your crystal axes?', style: 'grid', mode: 'choose',
    prompt: 'Which of these knows the crystal axes of your wafer?',
    narration: ["Here are some tools you will meet at work.", "Which of these knows the crystal axes of your wafer?"],
    cards: [
      { image: { src: 'media/fe/card_tool_spin_coater.jpg' }, label: 'The spin coater', correct: false,
        why: "The spin coater holds a wafer on a vacuum chuck in whatever rotation it was placed, spins it, and stops wherever it stops. It does not record where the flat was." + toolWhy },
      { image: { src: 'media/fe/card_tool_mask_aligner.jpg' }, label: 'The mask aligner', correct: false,
        why: "The mask aligner registers the mask to marks on the wafer, or to the flat if you place the flat against the reference. A piece with no flat and no mark can go on in any of four rotations." + toolWhy },
      { image: { src: 'media/fe/card_tool_wafer_prober.jpg' }, label: 'The wafer prober', correct: false,
        why: "A prober squares the wafer to the die pattern it can see. It knows the street grid, not the crystal axes." + toolWhy },
      { image: { src: 'media/fe/card_tool_wafer_cassette.jpg' }, label: 'The wafer cassette', correct: false,
        why: "A cassette holds wafers in slots. Whether their flats line up depends on whoever loaded it." + toolWhy },
      { image: { src: 'media/fe/card_packaged_chip.jpg' }, label: 'The package', correct: false,
        why: "A die can be attached and bonded into a package in more than one rotation. After the lid goes on, the only record is the bonding diagram." + toolWhy },
      { image: { src: 'media/fe/welcome_cleanroom_corridor.jpg' }, label: 'None of them. You do.', correct: true,
        why: "None of these tools knows the crystal axes. You do, because you read the supplier specification, marked the die, and wrote it in the traveler." }
    ],
    correctSay: "None of them. You do. You read the specification, you mark the die, you write it in the traveler. Know your axes."
  }]);
  addAfter(chSrc.steps, 'Category 5 vignette: Standard orientation', [
    photoStep('Where conventions are assumed', 'media/shared/meeting_room_table.jpg',
      'Standard orientation is whatever was assumed in this room. Write it down before you leave it.',
      ["Standard orientation is whatever was assumed in this room.", "Write it down before you leave it. Know your axes."])
  ]);

  /* ---- Module 3: Bends ---- */
  addAfter(ch2.steps, 'Cold open: The gentle bend', [
    photoStep('Bends are a normal part of working life', 'media/fe/bends_fiber_coils.jpg',
      'On x-cut, every bend is also a change of crystal direction.',
      ["Bends are a normal part of working life.", "On x-cut, every bend is also a change of crystal direction. Know your axes."],
      [0.5, 0.5, 1.0, 0.55, 0.5, 1.2])
  ]);
  addBefore(ch2.steps, 'What the literature reports', [
    splitStep('The image does not show the axis', 'media/fe/bends_sem_inspection.jpg',
      ['The microscope shows the geometry', 'The crystal direction is not in the image', 'It is in your layout, relative to the flat'],
      ["The microscope shows the geometry. The crystal direction is not in the image.", "It is in your layout, relative to the flat. Know your axes."])
  ]);
  ch2.steps[findStep(ch2.steps, 'Gary\'s x-cut ring shows TE to TM conversion. He proposes doubling the bend radius. What should a compliant engineer say?')].image = { src: 'media/fe/bends_layout_review.jpg' };
  // Rods incident: rods horizontal (slow axis in the chip plane, TE) before, rods vertical (TM) after.
  incRods.forEach(function (st) {
    if (st.type !== 'video') return;
    st.keyframes[0].image = { src: 'media/render/pm_panda_endface_90deg.jpg' };
    st.keyframes[1].image = { src: 'media/render/pm_panda_endface_0deg.jpg' };
  });

  /* ---- Module 4: Photorefractive ---- */
  addAfter(ch3.steps, 'Cold open: The record Q', [
    photoStep('The light you send', 'media/fe/pr_green_laser.jpg',
      'A visible laser beam through cage optics.',
      ["Every beam you send into lithium niobate can leave charge behind.", "Scan both ways. Watch your charge."],
      [0.5, 0.5, 1.0, 0.7, 0.5, 1.3])
  ]);
  addBefore(ch3.steps, 'Interactive: Resonance sweep', [
    splitStep('It looks calm', 'media/fe/pr_ln_crystal_mount.jpg',
      ['A lithium niobate crystal in a mount', 'It looks calm', 'It keeps a record of the light you send through it'],
      ["A lithium niobate crystal in a mount. It looks calm.", "It keeps a record of the light you send through it. Watch your charge."])
  ]);
  addAfter(ch3.steps, 'Scenario 2: Protective cladding', [
    photoStep('Power levels are workplace conduct', 'media/fe/pr_optics_lab.jpg',
      'Every optical table is a workplace.',
      ["Every optical table is a workplace. Power levels are workplace conduct.", "Scan both ways."])
  ]);

  /* ---- Module 5: Pyroelectric ---- */
  (function () {
    var v = ch4.steps[findStep(ch4.steps, 'Cold open: The hotplate')];
    v.keyframes[0].image = { src: 'media/fe/pyro_hot_plate.jpg' };
    v.keyframes[1].image = { src: 'media/render/ln_wafer_hotplate.jpg', kb: [0.5, 0.5, 1.0, 0.58, 0.25, 1.6] };
  })();
  addAfter(ch4.steps, 'Interactive: Pyroelectric field estimator', [
    splitStep('Every thermal step', 'media/fe/pyro_furnaces.jpg',
      ['Every thermal step is a temperature change', 'Every temperature change is charge', 'Slow is a number. Ask for it.'],
      ["Every thermal step is a temperature change. Every temperature change is charge.", "Watch your charge."])
  ]);
  addAfter(ch4.steps, 'Where temperature change happens at work', [{
    type: 'hotspots', title: 'Inspect the resist bay', style: 'callout',
    prompt: 'Open every point to continue.',
    image: { src: 'media/fe/pyro_hotspot_resist_bay.jpg' },
    narration: ["This is a real photoresist bay. Open every point to continue."],
    spots: [
      { x: 0.53, y: 0.47, label: 'Spin coater, lid open', body: '<p>Spin coating happens at room temperature. The bake after it does not. On z-cut, every ramp up and every cooldown builds surface charge. Use the slow-ramp recipe and a conductive discharge path.</p>', narration: ["Spin coating happens at room temperature. The bake after it does not.", "On z-cut, every ramp and every cooldown builds surface charge. Use the slow ramp, and a discharge path."] },
      { x: 0.87, y: 0.78, label: 'Controller panel', body: '<p>Spin speed and acceleration are programmed and repeatable. Hold your bake ramps to the same standard.</p>', narration: ["Spin speed and acceleration are programmed and repeatable. Hold your bake ramps to the same standard."] },
      { x: 0.67, y: 0.53, label: 'Wash bottles', body: '<p>Do not rinse or quench a hot lithium niobate wafer. A fast temperature drop builds charge just like a fast rise, and thermal shock can crack the substrate.</p>', narration: ["Do not quench a hot lithium niobate wafer. A fast drop builds charge just like a fast rise."] },
      { x: 0.685, y: 0.16, label: 'Eye protection sign', body: '<p>Safety glasses protect you from solvents. Your devices need their own protection from charge. Both are required.</p>', narration: ["Safety glasses protect you. Your devices need their own protection, from charge."] },
      { x: 0.055, y: 0.52, label: 'Fire extinguisher', body: '<p>Charge from a heated wafer is handled with grounding and slow ramps, not an extinguisher. Know where it is anyway.</p>', narration: ["Charge from a heated wafer is handled with grounding and slow ramps, not an extinguisher. Know where it is anyway. Watch your charge."] }
    ]
  }]);
  addAfter(ch4.steps, 'Scenario 3: The neighbor', [{
    type: 'cards', title: 'Where your crystal changes temperature', style: 'flip', mode: 'explore',
    prompt: 'Turn over each card to see where your crystal changes temperature.',
    narration: ["Turn over each card to see where your crystal changes temperature."],
    cards: [
      { image: { src: 'media/fe/pyro_hot_plate.jpg' }, label: 'Hot plates and bakes', body: 'Resist bakes and cooldowns: often the fastest temperature change in a process flow.', narration: ["Hot plates and bakes. Often the fastest temperature change in the whole flow."] },
      { image: { src: 'media/fe/pyro_furnaces.jpg' }, label: 'Furnaces and anneals', body: 'Slow ramps in, slow ramps out. "Slow" is a number. Ask for it.', narration: ["Furnaces and anneals. Slow is a number. Ask for it."] },
      { image: { src: 'media/fe/bias_wire_bonder.jpg' }, label: 'Bonding', body: 'Many bonding processes heat the stage. Packaging is a thermal step too.', narration: ["Bonding. Many bonding processes heat the stage. Packaging is a thermal step too."] },
      { image: { src: 'media/fe/pyro_cryostat.jpg' }, label: 'Cryostats', body: 'Cooling to 4 kelvin or below is a temperature change of nearly 300 kelvin.', narration: ["Cryostats. Cooling to four kelvin or below is a temperature change of nearly three hundred kelvin. Watch your charge."] }
    ]
  }]);
  addAfter(ch4.steps, 'The thread: Amplifier failure on fixture 2', [
    photoStep('After', 'media/fe/pyro_burnt_board.jpg',
      'Representative image of a failed component, with a different failure mode. Not the fixture 2 amplifier, which declined to be photographed.',
      ["This is a representative image of a failed component.", "It is not the fixture two amplifier, which declined to be photographed. Watch your charge."])
  ]);
  incHotPlug.forEach(function (st) { if (/^You are Dana/.test(st.kicker || '')) st.image = { src: 'media/fe/pyro_esd_label.jpg' }; });
  (function () {
    var i = -1;
    for (var k = 0; k < ch4.steps.length; k++) if (ch4.steps[k].title === 'What went wrong, technically') { i = k; break; }
    if (i < 0) throw new Error('Media anchor not found: hot plug reveal');
    var ph = photoStep('What a discharge can look like', 'media/render/pyro_discharge_comb_micrograph.jpg',
      'Discharge damage on interdigitated electrodes.',
      ["This is what discharge damage on interdigitated electrodes can look like.", "Most of the fingers completed the cooldown. A few in the middle did not. Watch your charge."],
      [0.5, 0.5, 1.0, 0.49, 0.45, 1.6]);
    ph.image.alt = 'Rendered illustration of interdigitated gold electrodes with discharge damage: missing gold segments, a fused bridge between fingers, resolidified gold spheres, and a small cracked crater';
    ch4.steps.splice(i + 1, 0, ph);
  })();

  /* ---- Module 6: Bias ---- */
  (function () {
    var v = ch5.steps[findStep(ch5.steps, 'Cold open: Quadrature on Monday')];
    v.keyframes[0].image = { src: 'media/fe/bias_scope.jpg', kb: [0.45, 0.4, 1.0, 0.45, 0.35, 1.25] };
  })();
  addAfter(ch5.steps, 'Bias conduct in the literature', [
    photoStep('Packaging does not fix the bias', 'media/fe/bias_wire_bonder.jpg',
      'Packaging fixes the electrodes in place. It does not fix the bias point.',
      ["Packaging fixes the electrodes in place.", "It does not fix the bias point. Watch your charge."])
  ]);
  addAfter(ch5.steps, 'Scenario 2: The overnight hold', [{
    type: 'spot', title: 'A measurement bench, Monday morning',
    prompt: 'Find the five hazards on this measurement bench.',
    image: { src: 'media/fe/bias_spot_bench.jpg' },
    narration: ["This is a measurement bench. Find the five hazards."],
    // target boxes: x, y = top-left corner (engine hit test), w, h = size, all 0..1 of the image
    targets: [
      { x: 0.70, y: 0.52, w: 0.07, h: 0.18, label: 'Soldering iron out of its stand', why: "A soldering iron is out of its stand, lying on the notes. A heat source near a lithium niobate chip shifts its bias through thermo-optic and pyroelectric effects." },
      { x: 0.755, y: 0.41, w: 0.05, h: 0.14, label: 'Open drink can', why: "No food or drink at the test bench. A spill on a bias connection is an unplanned bias change." },
      { x: 0.423, y: 0.89, w: 0.07, h: 0.06, label: 'Bare board held only by clip leads', why: "The device under test is loose on the desk, held by clip leads. A bias point that moves when someone walks past is not a bias point." },
      { x: 0.71, y: 0.80, w: 0.13, h: 0.13, label: 'Loose sticky notes', why: "Bias voltages, times, and drift readings go into the lab notebook with the date and time. Sticky notes are not records." },
      { x: 0.47, y: 0.73, w: 0.22, h: 0.18, label: 'Tangle of test leads', why: "Crossing, unlabeled leads make it impossible to say which supply drives which electrode. Route them, label them, record them." }
    ],
    missWhy: "That is not one of the hazards we are looking for. It may still be a hazard. Please keep looking."
  }]);
  incTurned.forEach(function (st) { if (st.kicker === 'What would you do?' && /chip 3 was loaded turned around/.test(st.question)) st.image = { src: 'media/fe/card_tool_wafer_prober.jpg' }; });
  // Turned-around incident: the shared tray, and the bench as it should have looked.
  addAfter(ch5.steps, 'The thread: Chip 3 bandwidth', [{
    type: 'spot', title: 'The shared tray',
    prompt: 'Twelve dies were placed in this tray. Find the one that is not facing the same way as its colleagues.',
    image: { src: 'media/render/tfln_gel_tray_one_rotated.jpg' },
    narration: ["Twelve dies were placed in this tray. Eleven of them followed the procedure. Find the one that did not."],
    // target box: x, y = top-left corner, w, h = size (0..1 of the image)
    targets: [
      { x: 0.508, y: 0.471, w: 0.121, h: 0.198, label: 'The die that was turned', why: "This die is turned ninety degrees from its colleagues. A half turn would not even be visible from above, which is why dies are marked. Nothing in the tray will tell the next person. Mark the die, mark the tray, and record the orientation in the run sheet. Know your axes." }
    ],
    missWhy: "That die is fine. It is facing the same way as its colleagues. Keep looking."
  }]);
  (function () {
    var i = -1;
    for (var k = 0; k < ch5.steps.length; k++) if (ch5.steps[k].title === 'What went wrong, technically') { i = k; break; }
    if (i < 0) throw new Error('Media anchor not found: turned-around reveal');
    // Pins projected from world points through the render camera (tools/blender/project_points.py, probe station close view), checked by screenshot.
    ch5.steps.splice(i + 1, 0, {
      type: 'hotspots', title: 'How chip 3 should have looked', style: 'pins',
      prompt: 'A correctly oriented device under test. Open every pin to continue.',
      image: { src: 'media/render/tfln_probe_station_closeup.jpg' },
      narration: ["Here is a correctly oriented device under test. Every probe is where the layout said it would be. Open every pin."],
      spots: [
        { x: 0.39, y: 0.58, label: 'Fixture plus z engraving', body: '<p>The fixture is engraved with plus z. It is the reference for every die loaded here.</p>', narration: ["The fixture is engraved with plus z. It is the reference for every die loaded here."] },
        { x: 0.563, y: 0.478, label: 'Die plus z arrow', body: '<p>The die arrow agrees with the fixture arrow. Check it before the probes come down, not after the result looks strange.</p>', narration: ["The die arrow agrees with the fixture arrow. Check it before the probes come down."] },
        { x: 0.535, y: 0.635, label: 'Lensed fiber, input facet', body: '<p>Light enters at the end the layout calls the input, at the same end as the RF input.</p>', narration: ["Light enters at the end the layout calls the input, the same end as the R F input."] },
        { x: 0.485, y: 0.455, label: 'Probe on the RF input pads', body: '<p>The RF wave starts here and travels with the light. That is the velocity match you paid for.</p>', narration: ["The R F wave starts here and travels with the light. That is the velocity match you paid for."] },
        { x: 0.69, y: 0.5, label: 'Probe on the termination pads', body: '<p>The far end of the electrode, terminated through the second probe. It is not an input, even though the pads look identical.</p>', narration: ["These are the termination pads at the far end, terminated through the second probe. They look identical to the input pads. They are not an input."] },
        { x: 0.652, y: 0.4, label: 'Lensed fiber, output facet', body: '<p>Light leaves here, after traveling the whole electrode alongside the RF wave.</p>', narration: ["Light leaves here, after traveling the whole electrode alongside the R F wave. Know your axes."] }
      ]
    });
  })();

  /* ---- Module 7: Simulation ---- */
  addAfter(ch6.steps, 'Cold open: Q equals seven times ten to the thirteen', [
    photoStep('Results do not review themselves', 'media/fe/sim_screen_from_behind.jpg',
      'Results do not review themselves.',
      ["Results do not review themselves.", "Trust no symmetry plane."])
  ]);
  addAfter(ch6.steps, 'Let\'s take a closer look', [
    heroStep('The fabricated device', 'media/fe/sim_wafer_dies.jpg',
      'It has no symmetry planes you did not build.',
      ["This is a fabricated device. It has no symmetry planes you did not build. " + M]),
    {
      type: 'hotspots', title: 'Review the model before the result', style: 'pins',
      prompt: 'Open every pin on the solver window to continue. The quality factor is the logged case from this chapter.',
      image: { src: 'media/fe/sim_solver_window_drawn.jpg' },
      narration: ["Here is a solver window. Open every pin before you look at the result."],
      spots: [
        { x: 0.12, y: 0.247, label: 'Material frame', body: '<p>"Rotated frame" is not a frame. Which cut, which Euler angles, and does the solver rotate the crystal or the coordinates? Print it on the first slide.</p>', narration: ["Rotated frame is not a frame. State the cut, the Euler angles, and the rotation convention. Know your axes."] },
        { x: 0.651, y: 0.481, label: 'Symmetry planes', body: '<p>This is the mistake in this model. Each plane forces the solution to be symmetric or antisymmetric about it. The geometry is symmetric. Is the crystal, in this frame?</p>', narration: ["Each symmetry plane deletes every solution that is neither symmetric nor antisymmetric. The geometry is symmetric. The crystal may not be. Trust no symmetry plane."] },
        { x: 0.917, y: 0.37, label: 'Perfectly matched layer', body: '<p>It sits in the anchor, where the beam\'s energy leaks into the substrate, and that is where it belongs. Sweep its strength and thickness. If a weaker absorber gives a higher quality factor, the loss is not reaching the absorber.</p>', narration: ["The absorber sits in the anchor, where energy leaks into the substrate. That is where it belongs.", "Sweep it. If a weaker absorber gives a higher quality factor, the loss is not reaching it."] },
        { x: 0.512, y: 0.823, label: 'Quality factor', body: '<p>Seven times ten to the thirteen, from the quarter model. The full model in the same frame has not been run. In the logged case, it gave about 2200.</p>', narration: ["Seven times ten to the thirteen, from the quarter model. In the logged case, the full model gave about two thousand two hundred."] },
        { x: 0.804, y: 0.812, label: 'Solved without errors', body: '<p>Solvers report whether the equations were solved, not whether they were the right equations.</p>', narration: ["Solved without errors means the equations were solved. It does not mean they were the right equations. Trust no symmetry plane."] }
      ]
    }
  ]);
  addAfter(ch6.steps, 'Scenario 3: The template frame', [
    splitStep('Write it down', 'media/fe/sim_lab_notebook.jpg',
      ['The crystal frame', 'The rotation convention', 'The mesh and the boundary conditions', 'In ink'],
      ["Record the crystal frame, the rotation convention, the mesh, and the boundary conditions. In ink. Know your axes."])
  ]);


  /* ======================= EXAM POOL ======================= */

  function q(question, correct, wrongs, correctSay) {
    var opts = [{ text: correct, correct: true, why: '' }];
    wrongs.forEach(function (w) { opts.push({ text: w[0], correct: false, why: w[1] + " Remember: " + M }); });
    return { type: 'check', question: question, options: opts, correctSay: correctSay || '' };
  }

  var pool = [
    q('On x-cut lithium niobate at 1550 nm, a TE mode propagating along the crystal y axis is polarized along z. Which bulk index does it see?',
      'The extraordinary index, about 2.138 (Zelmon et al., 1997)',
      [['The ordinary index, about 2.211', "Propagating along y, the in-plane TE field points along z, the optic axis, so it sees the extraordinary index, about two point one three eight at fifteen fifty nanometers. The ordinary index, about two point two one one, is what TM sees on x-cut, and what TE sees when propagating along z."],
       ['The average of the two, because the film is thin', "Film thickness changes the effective index of the mode, but the bulk index seen by a field along z is the extraordinary index. For propagation along y on x-cut, TE is polarized along z and sees about two point one three eight at fifteen fifty nanometers."]]),
    q('Which polarization on x-cut sees the ordinary index at every in-plane propagation angle?',
      'TM, polarized along the film normal x',
      [['TE, polarized in the plane', "TE is polarized in the plane, which contains the optic axis z, so its bulk index changes with angle between the extraordinary and ordinary values. TM is polarized along x, an ordinary direction, at every in-plane angle."],
       ['Neither; both change with angle', "TM on x-cut is polarized along x, the film normal, which is always an ordinary direction. Its bulk index does not change with in-plane angle. Its effective index still depends on geometry."]]),
    q('On x-cut thin-film lithium niobate, what has been reported about increasing the bend radius, and what should you check first?',
      'A larger radius can increase TE to TM conversion and loss; check whether TE and TM effective indices cross at some propagation angle for your geometry',
      [['Nothing to check; a larger radius always reduces conversion', "That is true on isotropic platforms. On x-cut, Liu and colleagues, citing earlier work, note that larger bend radii can increase mode conversion and loss. The bend sweeps through angles where TE and TM can hybridize; the earlier work explains it: a slower passage through a crossing is more adiabatic and converts more."],
       ['Bend radius changes only radiation loss; polarization conversion can only come from the input fiber', "On x-cut, anisotropy and slanted sidewalls near a TE and TM crossing angle can convert polarization on chip, and a larger radius has been reported to increase that conversion. Check the crossing before blaming the fiber."]]),
    q('Rotating the crystal by +30 degrees is equivalent to which coordinate operation?',
      'Rotating the coordinate system by -30 degrees',
      [['Rotating the coordinate system by +30 degrees', "Rotating the object by an angle is equivalent to rotating the coordinate system by the opposite angle. Solvers and sources differ in which one they mean, which is a common source of sign errors."],
       ['There is no relation between the two', "They are directly related: rotating the crystal by an angle equals rotating the coordinates by the opposite angle. That is exactly why you must state which one you mean."]]),
    q('Why are tests of tensor rotation code at 0 and 90 degrees insufficient?',
      'Many sign and transpose errors cancel at special angles; the bug shows only at general angles',
      [['They are sufficient; rotation is linear', "Linearity does not protect you from using the inverse or the transpose of the correct matrix. At zero and ninety degrees, those errors often give the same result. Test at a general angle against an independent calculation."],
       ['Because 90 degrees is outside the valid range of Euler angles', "Ninety degrees is a perfectly valid angle. The issue is that special angles hide sign and transpose errors. Test at a general angle."]]),
    q('In lithium niobate resonators, the photorefractive effect typically shifts resonances in which direction with increasing power?',
      'Toward shorter wavelength (blue shift), opposite to thermal heating',
      [['Toward longer wavelength, the same as heating', "Heating red-shifts resonances. Photorefraction in lithium niobate rings usually blue-shifts them, on a different time scale. That is why scan direction and power must be reported."],
       ['It never shifts resonances', "Photorefraction changes the index through the electro-optic effect of a trapped space-charge field, which shifts resonances, usually toward shorter wavelength in lithium niobate rings."]]),
    q('Which practice protects a lithium niobate linewidth measurement from scan-direction artifacts?',
      'Sweep both directions at decreasing power until the results agree, and report direction and power',
      [['Sweep once, in the direction that gives the narrowest dip', "The narrowest dip often comes from scanning against a power-dependent shift, so the resonance snaps past the laser. That inflates the quality factor. Sweep both ways at low power."],
       ['Use maximum laser power for best signal to noise', "More power makes thermal and photorefractive shifts larger, so the line shape depends more on scan direction. Lower power, both directions."]]),
    q('According to Xu et al. (Optics Express, 2021), what happened to the photorefractive resonance shift in thin-film lithium niobate microrings when the oxide top cladding was removed and the device heat treated?',
      'It was suppressed by more than two orders of magnitude',
      [['It increased, because the cladding was protecting the surface', "Xu and colleagues found the oxide cladding dramatically worsens photorefraction. Removing it, together with heat treatment, suppressed the relative shift by more than two orders of magnitude."],
       ['Nothing changed; cladding is purely optical', "Cladding changes charge transport at the lithium niobate surface. Xu and colleagues found removing it, with heat treatment, suppressed the photorefractive shift by more than two orders of magnitude."]]),
    q('What is the unscreened pyroelectric field across a z-cut crystal after a temperature change dT?',
      'E = p dT / (eps0 eps33), with p the pyroelectric coefficient',
      [['Zero, unless a voltage is applied', "A pyroelectric crystal needs no applied voltage. The spontaneous polarization changes with temperature and leaves bound surface charge, giving a field of p times delta T over epsilon zero epsilon three three until it is screened."],
       ['E = p dT, independent of permittivity', "Bound surface charge density p times delta T produces a field reduced by the permittivity, so E equals p delta T over epsilon zero epsilon three three."]]),
    q('Why do practitioners wrap lithium niobate chips in aluminum foil or add conductive coatings during bakes?',
      'To provide a discharge path for pyroelectric surface charge created by the temperature change',
      [['To improve temperature uniformity only', "Foil may help uniformity a little, but the reason in lithium niobate processing is the pyroelectric charge. A conductive path lets it neutralize instead of sparking, damaging devices, or cracking substrates."],
       ['To block ultraviolet light from the cleanroom', "Yellow-light rooms already handle that. The foil is a discharge path for pyroelectric charge created by heating and cooling."]]),
    q('Bulmer et al. (1986) compared titanium-indiffused interferometers on z-cut and x-cut lithium niobate. What did they report?',
      'Pyroelectric instabilities on z-cut, while x-cut interferometers were stable',
      [['x-cut was unstable and z-cut was stable', "On z-cut, the pyroelectric charge sits on the top and bottom faces, where the field overlaps the waveguide. Bulmer and colleagues reported instabilities on z-cut and stable x-cut interferometers."],
       ['No difference between cuts', "The cut decides where the pyroelectric charge sits. Bulmer and colleagues reported instabilities on z-cut and stable behavior on x-cut."]]),
    q('Why can the V pi measured with a low-frequency triangle differ from the RF V pi in thin-film lithium niobate?',
      'Charge relaxation in the layer stack changes the field in the crystal at low frequency, and low frequency sees unclamped rather than clamped coefficients',
      [['It cannot differ; the Pockels effect is instantaneous', "The Pockels effect is fast, but the field in the crystal is not. At low frequency, charge moves in cladding and interfaces. Liu and colleagues measured V pi above forty volts at ten kilohertz and below in thin film devices."],
       ['Only because of cable losses at RF', "Cable and electrode losses matter at RF, but the low-frequency difference comes from charge relaxation in the stack. State the frequency of every V pi."]]),
    q('A paper reports no bias drift, but set quadrature by tuning the laser wavelength on an unbalanced interferometer. What can you conclude about DC drift?',
      'Nothing; no DC bias was applied, so drift under DC bias was not tested',
      [['The stack is drift-free', "Without a DC bias, there is no DC drift to observe. Avoiding the test is not passing it."],
       ['The stack drifts strongly', "You cannot conclude that either. The measurement did not apply DC bias, so it says nothing about drift under DC bias."]]),
    q('In a piezoelectric eigenfrequency study, what does an electrode set to a Voltage terminal represent?',
      'A short circuit; the voltage setpoint is ignored and the result is the resonance',
      [['A biased electrode at the set voltage', "An eigenfrequency study is linear and unforced. A Voltage terminal is treated as a fixed potential, effectively a short, and the setpoint is ignored. For an open circuit, use a Charge terminal with zero charge."],
       ['An open circuit, giving the anti-resonance', "That is the Charge terminal with zero charge. The Voltage terminal is the short circuit, giving the resonance."]]),
    q('A quarter model with symmetry planes of a lithium niobate resonator gives an extremely high Q, and weakening the absorbing layer raises Q further. What should you do first?',
      'Run the full model without symmetry planes in the same crystal frame, and sweep the absorber',
      [['Publish; both trends confirm a high Q', "A quality factor that rises when the absorber weakens means the loss is not reaching the absorber. In anisotropic crystals, symmetry planes can delete the leaky coupling. In one logged case, the full model gave about two thousand two hundred."],
       ['Add more symmetry planes to reduce compute time further', "Every symmetry plane removes solutions the crystal may need. Check with the full model first."]]),
    q('On an x-cut lithium niobate modulator with the waveguide along y and the electrode field along z, light is accidentally launched TM instead of TE. Which electro-optic coefficient now sets the response?',
      'r13, so V pi rises by roughly the ratio of n_e^3 r33 to n_o^3 r13, about 3 times at RF for equal overlap',
      [['r33, so nothing changes', "T E light on this device is polarized along z and uses r three three. T M light is polarized along x, the film normal, and with the field along z it uses r one three, which is about three to three and a half times smaller in the values compiled by Weis and Gaylord."],
       ['r22, so V pi rises about eight times', "r two two is what you get if the waveguide itself runs along z, for example after a mask was aligned ninety degrees off the flat. For a T M launch on a waveguide along y, the coefficient is r one three."]]),
    q('A traveling-wave modulator is loaded turned around, so the light and the RF wave counter-propagate. What happens to the velocity-mismatch term in the response sin(x)/x, x = pi f L (n_m -/+ n_g)/c?',
      'The difference n_m - n_g becomes the sum n_m + n_g, so the bandwidth collapses (about 6 GHz at -3 dB for a 5 mm device with n_m 2.2 and n_g 2.23)',
      [['Nothing changes, because the DC V pi is unchanged', "The D C V pi is indeed unchanged, because turning a finished chip around turns its electrodes with it. But at high frequency, what matters is whether the light keeps up with the R F wave. Counter-propagating, the index difference becomes a sum of about four point four, and the response of a five millimeter device falls to minus three electrical dB near six gigahertz."],
       ['The sign of the electro-optic response flips, so the bandwidth doubles', "Turning a finished chip around does not flip its electro-optic sign relative to its own pads, and a sign flip would not change bandwidth anyway. The change is the relative direction of light and R F, which turns a velocity match into a velocity sum."]]),
    q('A bulk z-cut lithium niobate chip went through a fast 45 K temperature step with its electrode pads floating. What is the safe next step before connecting a driver amplifier?',
      'Let it return to room temperature and discharge the pads through a resistor to ground before connecting; in future, short the pads during thermal steps',
      [['Connect immediately; the amplifier is unpowered, so it is safe', "Electrostatic discharge damage does not need the device to be powered. The pyroelectric charge stored under floating pads goes through the connector either way. The unscreened upper bound for forty five kelvin across half a millimeter is several kilovolts."],
       ['Connect immediately while wearing a grounded wrist strap', "A wrist strap grounds you, not the crystal. Lithium niobate is an insulator, and the charge on its faces leaves through the cable you connect. Discharge the pads first."]]),
    q('A partner lab images dies through the substrate and measures angles counterclockwise from its own chip edge. The email says "standard orientation, as discussed". Which category of orientation misconduct is this?',
      'Cross-Institutional Orientation Misconduct',
      [['Third-Party Orientation Misconduct', "Third-party misconduct originates with a commercial supplier's material or documentation. Here, dies and conventions crossed an organizational boundary between collaborators without a written, agreed frame. That is cross-institutional orientation misconduct."],
       ['Domestic Orientation Misconduct', "Domestic misconduct stays inside your own setup and records. Here, two institutions used two definitions of angle, and one of them viewed the chip from the back. That is cross-institutional orientation misconduct."]])
  ];

  // Spoken intros for the scenario questions added to the pool (question text has symbols that do not read aloud)
  (function () {
    var add = [
      ["On an x-cut modulator with the waveguide along y, light is accidentally launched T M instead of T E. Which electro-optic coefficient now sets the response?"],
      ["A traveling-wave modulator is loaded turned around, so the light and the R F wave travel in opposite directions. What happens to the velocity mismatch term in the response?"],
      ["A bulk z-cut chip went through a fast forty five kelvin temperature step with its pads floating. What is the safe next step before connecting a driver amplifier?"],
      ["A partner lab measures angles from its own chip edge, viewed through the substrate, and calls it standard orientation. Which category of orientation misconduct is this?"]
    ];
    for (var i = 0; i < add.length; i++) pool[pool.length - add.length + i].narration = add[i];
  })();

  /* ======================= WIDGETS ======================= */

  var widgets = {};

  /* xcutIndex, prSweep, biasDrift, pyroCalc and eulerSign live in widgets/*.js (mode data from tools/modesolver/); they register on window.LMS_WIDGETS and their spoken lines on LMS_WIDGET_SPEECH. */



  /* ---- chat platform skins (engine field `skin`; generic app looks, no product names in shown text) ---- */
  var CHAT_SKINS = {
    'Scenario 1: The symmetric crystal': 'imessage',
    'Scenario 2: The inherited template': 'mail',
    'Scenario 3: The animation': 'slack',
    'The thread: Lot 7 out of family': 'mail',
    'Category 1 vignette: The prealigner': 'teams',
    'Category 2 vignette: z is z': 'signal',
    'Category 3 vignette: Same cut, new supplier': 'mail',
    'Category 4 vignette: The way I always put it': 'slack',
    'Category 5 vignette: Standard orientation': 'mail',
    'Category 6 vignette: Notes to self': 'imessage',
    'Category 7 vignette: The stepper slot': 'whatsapp',
    'Category 8 vignette: You get used to it': 'teams',
    'Scenario 1: The bigger ring': 'meeting',
    'Scenario 2: The polarization blame': 'whatsapp',
    'Scenario 3: The routing shortcut': 'slack',
    'The thread: The voltage went up overnight': 'slack',
    'Scenario 1: The record': 'imessage',
    'Scenario 2: Protective cladding': 'mail',
    'Scenario 3: The lock that ran away': 'slack',
    'Scenario 1: The faster bake': 'signal',
    'Scenario 2: Kitchen foil': 'teams',
    'Scenario 3: The neighbor': 'mail',
    'The thread: Amplifier failure on fixture 2': 'mail',
    'Scenario 1: The low V pi': 'mail',
    'Scenario 2: The overnight hold': 'whatsapp',
    'Scenario 3: The drift-free claim': 'teams',
    'The thread: Chip 3 bandwidth': 'slack',
    'The thread: The Friday shipment': 'imessage',
    'The thread: The following Monday': 'imessage',
    'Scenario 1: The quarter model': 'meeting',
    'Scenario 2: The ignored volt': 'signal',
    'Scenario 3: The template frame': 'teams'
  };
  // Prompt under each chat (engine default: "Pause and reflect: Is this OK?"), where a different question fits better.
  var CHAT_ISOK = {
    'The thread: The Friday shipment': 'Pause and reflect: What would you do?',
    'The thread: The following Monday': 'Pause and reflect: Is this retaliation?'
  };
  CAT_ORDER.forEach(function (k, i) { CHAT_ISOK[CAT_CONTENT[k].chat.title] = 'Pause and reflect: Which category of orientation misconduct is this?'; });
  [ch0, ch1, chSrc, ch2, ch3, ch4, ch5, ch6].forEach(function (ch) {
    ch.steps.forEach(function (st) {
      if (st.type !== 'chat') return;
      var k = CHAT_SKINS[st.title];
      if (!k) throw new Error('Chat step without a skin: ' + st.title);
      st.skin = k;
      if (CHAT_ISOK[st.title]) st.isok = CHAT_ISOK[st.title];
    });
  });


  /* ======================= INTERACTION PASS: stages, tap-to-advance, reflect, sort ======================= */

  // "Is this okay?" moments: the narration sets the scene, a popup asks the question, and the answer opens as a stage.
  function joinCaps(t) { return String(t).replace(/\b(?:[A-Z] ){1,5}[A-Z]\b/g, function (m) { return m.replace(/ /g, ''); }); }
  var OK_DEFAULT = [
    { text: 'OK', say: 'Look again. Something is missing.' },
    { text: 'Not OK', correct: true, say: 'Here is why.' }
  ];
  // Scenarios where Gary acts properly: the correct answer is OK.
  function okYes(right, why) {
    return [
      { text: 'OK', correct: true, say: right },
      { text: 'Not OK', say: 'This one is okay. ' + why }
    ];
  }
  var ASK_SPECIAL = {
    'Scenario 2: Kitchen foil': [
      { text: 'OK', correct: true, say: 'This one really is okay.' },
      { text: 'Not OK', say: 'This one is actually fine. The foil is a discharge path.' }
    ],
    'Scenario 3: The animation': okYes('Gary drew the cut he actually has.', 'Gary drew the cut he actually has, and labeled it.'),
    'Category 6 vignette: Notes to self': okYes('Gary logged his own error.', 'Gary logged his own error, with no witnesses.'),
    'Scenario 2: The polarization blame': okYes('Gary checked the input before blaming it.', 'Gary checked the input before blaming it, then looked on chip.'),
    'Scenario 3: The routing shortcut': okYes('Gary computed the phase instead of assuming it.', 'Gary computed the phase instead of assuming it.'),
    'Scenario 3: The lock that ran away': okYes('Gary measured before he picked a side.', 'Gary measured the shift before he picked a lock side.'),
    'Scenario 3: The neighbor': okYes('Gary planned for the charge.', 'Gary shielded, grounded, and coordinated before the cooldown.'),
    'Scenario 3: The drift-free claim': okYes('Gary read the methods.', 'Gary read the methods before quoting the result.'),
    'Scenario 2: The ignored volt': okYes('Gary found out why before reporting.', 'Gary found out why the voltage was ignored, before reporting.'),
    'Scenario 3: The template frame': okYes('Gary checked the frame.', 'Gary checked the frame and tested a known case.'),
    'Scenario 2: The overnight hold': [
      { text: 'OK', say: 'Only if it was planned.' },
      { text: 'Not OK', say: 'It can be fine, if it was planned.' },
      { text: 'Only if it was a planned experiment', correct: true, say: 'Exactly. Planned, it is an experiment.' }
    ],
    'Scenario 1: The quarter model': [
      { text: 'OK', say: 'Not until the shortcut has been checked once.' },
      { text: 'Not OK', say: 'It can be fine, once it has been checked.' },
      { text: 'Only after one full-model check', correct: true, say: 'Exactly. Check once, then use the shortcut.' }
    ],
    'Scenario 2: Protective cladding': [
      { text: 'OK', say: 'Not before it has been measured.' },
      { text: 'Not OK', say: 'It can be fine, once it has been measured.' },
      { text: 'Only after measuring with and without it', correct: true, say: 'Exactly. Measure first.' }
    ]
  };
  var VIDEO_WHY = {
    'Cold open: Which edge is z?': 'A wrong frame does not raise an error. It builds a different device.',
    "Incident video: The flat at three o'clock": 'The flat was the only reference, and it was loaded at three o\'clock.',
    'Cold open: Many hands': 'Every hand turned it, and none of them wrote it down.',
    "Incident video: The rods at twelve o'clock": 'The rotation was set from memory, and nobody checked the launch polarization.'
  };
  function stripLead(t) { return String(t).replace(/^(?:No|Yes)\.\s+/, ''); }
  var answerPrompts = ['See the answer', 'Check the verdict', 'Reveal the outcome', 'See what the policy says'];
  var nAns = 0;
  [ch1, chSrc, ch2, ch3, ch4, ch5, ch6].forEach(function (ch) {
    ch.steps.forEach(function (st) {
      if (!st.narration || st.stages) return;
      if (st.type === 'chat') {
        var k = -1;
        for (var i = 0; i < st.narration.length; i++) if (/Is this okay\?$/.test(st.narration[i])) { k = i; break; }
        if (k < 0 || k === st.narration.length - 1) return;
        var rest = st.narration.slice(k + 1);
        rest[0] = stripLead(rest[0]);
        st.narration = st.narration.slice(0, k + 1);
        st.narration[k] = st.narration[k].replace(/\s*Is this okay\?$/, '');
        if (!st.narration[k]) st.narration.splice(k, 1);
        st.ask = { question: 'Is this OK?', after: 'messages', options: ASK_SPECIAL[st.title] || OK_DEFAULT };
        st.stages = [{ prompt: answerPrompts[nAns++ % answerPrompts.length], body: '<p>' + rest.map(joinCaps).join(' ') + '</p>', narration: rest }];
      } else if (st.type === 'video') {
        var last = st.narration[st.narration.length - 1];
        var m = /^Is this okay\? No\.\s*(?:This is not okay\.\s*)?([\s\S]*)$/.exec(last);
        if (!m) return;
        st.narration = st.narration.slice(0, -1);
        var why = (VIDEO_WHY[st.title] ? VIDEO_WHY[st.title] + ' ' : '') + m[1];
        st.ask = { question: 'Is this OK?', options: OK_DEFAULT };
        st.stages = [{ prompt: 'See why', body: '<p>' + joinCaps(why) + '</p>', narration: [why] }];
      }
    });
  });
  // The cold open of the Sources module follows the same pattern.
  (function () {
    var v = chSrc.steps[findStep(chSrc.steps, 'Cold open: Many hands')];
    var last = v.narration[v.narration.length - 1];
    var m = /^Is this okay\? No\.\s*([\s\S]*)$/.exec(last);
    if (m) {
      v.narration = v.narration.slice(0, -1);
      v.ask = { question: 'Is this OK?', options: OK_DEFAULT };
      v.stages = [{ prompt: 'See why', body: '<p>' + m[1] + '</p>', narration: [m[1]] }];
    }
  })();
  // "How do you think Gary felt?" slides: the popup asks, the stage answers.
  [ch2, ch5].forEach(function (ch) {
    ch.steps.forEach(function (st) {
      if (st.title !== 'How do you think Gary felt?' || !st.stages) return;
      st.narration = st.narration.slice(0, 1);
      st.ask = { question: 'How do you think Gary felt?', options: [
        { text: 'Ashamed', correct: true, say: 'Yes. Gary felt ashamed.' },
        { text: 'Fine', say: 'Gary did not feel fine.' },
        { text: 'Relieved', say: 'Gary did not feel relieved.' }
      ] };
    });
  });

  // Incident threads unfold one message at a time.
  ['The thread: Lot 7 out of family', 'The thread: The voltage went up overnight', 'The thread: Amplifier failure on fixture 2',
   'The thread: Chip 3 bandwidth', 'The thread: The Friday shipment', 'The thread: The following Monday', 'Category 6 vignette: Notes to self'
  ].forEach(function (t) {
    var hit = false;
    [ch1, chSrc, ch2, ch4, ch5].forEach(function (ch) { ch.steps.forEach(function (st) { if (st.type === 'chat' && st.title === t) { st.tapToAdvance = true; hit = true; } }); });
    if (!hit) throw new Error('tapToAdvance anchor not found: ' + t);
  });

  // Reflection moments (ungraded).
  ch0.steps.splice(ch0.steps.length - 1, 0, {
    type: 'reflect', title: 'Self-assessment', mode: 'scale',
    prompt: 'How confident are you in your orientation conduct?',
    scale: ['Not confident', 'Slightly confident', 'Moderately confident', 'Very confident', 'I have never made an orientation error'],
    narration: ["Before we begin, a short self-assessment. How confident are you in your orientation conduct?"],
    reveal: '<p>Thank you. Your confidence has been recorded and will be compared with your final assessment. Confidence is not a control. Know your axes.</p>',
    revealNarration: ["Confidence is not a control. Know your axes."]
  });
  addAfter(ch4.steps, 'After', [{
    type: 'reflect', title: 'A moment for the amplifier', mode: 'choice',
    prompt: 'Take a moment to consider how this made the driver amplifier feel.',
    choices: ['Betrayed', 'Overwhelmed', 'Grounded, too late', 'It is an amplifier'],
    narration: ["Take a moment to consider how this made the driver amplifier feel."],
    reveal: '<p>The amplifier did not feel anything. It is an amplifier. Gary, however, felt ashamed. Shame is normal. The incident has been logged. Watch your charge.</p>',
    revealNarration: ["The amplifier did not feel anything. It is an amplifier.", "Gary, however, felt ashamed. Shame is normal. The incident has been logged. Watch your charge."]
  }]);
  addAfter(ch5.steps, 'Zero tolerance for retaliation', [{
    type: 'reflect', title: 'Your experience', mode: 'text',
    prompt: 'Describe a time you felt pressure to skip, shorten, or not report a check. You do not have to name anyone.',
    narration: ["Take a moment to describe a time you felt pressure to skip, shorten, or not report a check. You do not have to name anyone."],
    reveal: '<p>Thank you for sharing. Your response stays in this browser. If the pressure was real, the Photonic Conduct Speak-Up Line is available, and good-faith reports are protected.</p>',
    revealNarration: ["Thank you for sharing. If the pressure was real, the Photonic Conduct Speak-Up Line is available. Speak up."]
  }]);
  chSrc.steps.splice(findStep(chSrc.steps, 'Know your categories'), 0, {
    type: 'reflect', title: 'Personal inventory', mode: 'choice',
    prompt: 'Which category of orientation misconduct have you most recently committed?',
    choices: CATS.concat(['None, to my knowledge']),
    narration: ["Which category of orientation misconduct have you most recently committed? Answer honestly. This reflection is ungraded."],
    reveal: '<p>There are no wrong answers here. If you chose "None, to my knowledge", please note that domestic orientation misconduct has no witnesses, including you. Stop. Mark. Log. Speak up.</p>',
    revealNarration: ["There are no wrong answers here.", "If you chose none, to my knowledge, please note that domestic orientation misconduct has no witnesses, including you."]
  });

  // Sort exercises in the taxonomy module.
  var whyCat = function (k, extra) { return "This one is " + CAT[k].say + ". " + extra; };
  addAfter(chSrc.steps, srcAct.afterSoft.question, [{
    type: 'sort', title: 'Who knows your crystal axes?',
    prompt: 'Sort each item. Which ones know the crystal axes of your wafer, and which ones do exactly what they are told?',
    narration: ["Sort each item. Which ones know the crystal axes of your wafer, and which ones do exactly what they are told?"],
    buckets: [{ id: 'told', label: 'Does exactly what it is told' }, { id: 'knows', label: 'Knows your crystal axes' }],
    items: [
      { text: 'The spin coater', bucket: 'told', why: "The spin coater stops the wafer wherever it stops. It does not know where the flat is." },
      { text: 'The aligner prealigner', bucket: 'told', why: "The prealigner turns the wafer to the angle in its recipe, right or wrong." },
      { text: 'The probe station theta stage', bucket: 'told', why: "The theta stage holds whatever angle it was set to." },
      { text: 'The pick-and-place recipe', bucket: 'told', why: "The pick-and-place tool sets each die down at the angle in its recipe." },
      { text: 'The mode solver', bucket: 'told', why: "The mode solver solves the frame you entered, not the wafer you bought." },
      { text: 'The shared gel-pak', bucket: 'told', why: "A gel-pak holds each die exactly as it was placed. It records nothing." },
      { text: 'A traveler entry you wrote at the tool', bucket: 'knows', why: "A traveler entry written at the tool, against the flat, is a record of the crystal axes. That is what it is for." },
      { text: 'You', bucket: 'knows', why: "You know the crystal axes, because you read the specification and checked the flat. Know your axes." }
    ]
  }]);
  addAfter(chSrc.steps, srcAct.afterPeer.question, [{
    type: 'sort', title: 'Classify by source, part one',
    prompt: 'Drag each incident into its category of orientation misconduct.',
    narration: ["Drag each incident into its category of orientation misconduct."],
    buckets: [{ id: 'tool', label: CATS[0] }, { id: 'soft', label: CATS[1] }, { id: 'third', label: CATS[2] }, { id: 'peer', label: CATS[3] }],
    items: [
      { text: 'A prealigner turns a wafer to a recipe angle written for another product', bucket: 'tool', why: whyCat('tool', "A tool applied the rotation, and nobody re-checked the flat.") },
      { text: 'A probe station chuck is left at the angle from the last user', bucket: 'tool', why: whyCat('tool', "The chuck held the angle it was last set to.") },
      { text: 'A simulated x-cut device has its crystal z on the film normal', bucket: 'soft', why: whyCat('soft', "The error lives in a software frame convention.") },
      { text: 'Rotation code passes its tests at 0 and 90 degrees only', bucket: 'soft', why: whyCat('soft', "The program was never checked against a known case at a general angle.") },
      { text: 'A new supplier defines the primary flat at a different crystal direction', bucket: 'third', why: whyCat('third', "A supplied convention entered without incoming verification.") },
      { text: 'Datasheet tensor signs follow a different axis convention', bucket: 'third', why: whyCat('third', "The convention came from outside, in a supplied document.") },
      { text: 'Unmarked dies are left in a shared gel-pak with no note', bucket: 'peer', why: whyCat('peer', "Orientation did not travel with the die inside the group.") },
      { text: 'A run-sheet template has no orientation column', bucket: 'peer', why: whyCat('peer', "A shared template inside the group lost the orientation.") }
    ]
  }]);
  chSrc.steps.splice(findStep(chSrc.steps, 'Personal inventory'), 0, {
    type: 'sort', title: 'Classify by source, part two',
    prompt: 'Drag each incident into its category of orientation misconduct.',
    narration: ["Part two. Drag each incident into its category of orientation misconduct."],
    buckets: [{ id: 'cross', label: CATS[4] }, { id: 'dom', label: CATS[5] }, { id: 'qpq', label: CATS[6] }, { id: 'hostile', label: CATS[7] }],
    items: [
      { text: 'A partner lab measures angles from the back side and calls it standard', bucket: 'cross', why: whyCat('cross', "Two institutions, no written frame.") },
      { text: 'A collaborator sends tensor constants in their own sign convention', bucket: 'cross', why: whyCat('cross', "The data crossed an organizational boundary without an agreed frame.") },
      { text: 'You mark your own die on the wrong corner', bucket: 'dom', why: whyCat('dom', "Your own bench, your own mark, no witnesses.") },
      { text: 'Your notebook records angles in a mirrored view', bucket: 'dom', why: whyCat('dom', "Your own records.") },
      { text: '"Take my stepper slot if you sign off the flat check"', bucket: 'qpq', why: whyCat('qpq', "A benefit offered for a skipped check.") },
      { text: '"Put me on the paper and I will not mention the frame"', bucket: 'qpq', why: whyCat('qpq', "Co-authorship traded for silence.") },
      { text: 'Every hotplate in the lab lacks ramp control, and sparks are called normal', bucket: 'hostile', why: whyCat('hostile', "The conditions are pervasive.") },
      { text: 'New hires are told "you get used to it"', bucket: 'hostile', why: whyCat('hostile', "Correct conduct is made impossible by the environment.") }
    ]
  });

  /* ======================= COURSE-WIDE PASSES ======================= */

  // Spoken strings spell acronyms and units the way the voice must read them (captions join spelled acronyms).
  function speakify(t) {
    return tts(t)
      .replace(/\bGHz\b/g, 'gigahertz').replace(/\bMHz\b/g, 'megahertz').replace(/\bkHz\b/g, 'kilohertz')
      .replace(/\bpm\/V\b/g, 'picometers per volt').replace(/\bkV\/mm\b/g, 'kilovolts per millimeter')
      .replace(/\br33\b/g, 'r three three').replace(/\br13\b/g, 'r one three').replace(/\br22\b/g, 'r two two')
      .replace(/\bdT\b/g, 'delta T').replace(/\bEO\b/g, 'E O').replace(/\bMZI\b/g, 'M Z I').replace(/\bPML\b/g, 'P M L');
  }
  function plain(t) { return String(t).replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim(); }
  function sayAll(a) { return Array.isArray(a) ? a.map(speakify) : a; }
  function cleanStep(st) {
    if (st.narration) st.narration = sayAll(st.narration);
    (st.tabs || []).forEach(function (t) { t.narration = sayAll(t.narration); });
    (st.spots || []).forEach(function (t) { t.narration = sayAll(t.narration); });
    (st.cards || []).forEach(function (c) { c.narration = sayAll(c.narration); if (c.why) c.why = speakify(c.why); });
    (st.targets || []).forEach(function (t) { if (t.why) t.why = speakify(t.why); });
    if (st.missWhy) st.missWhy = speakify(st.missWhy);
    (st.stages || []).forEach(function (sg) { sg.narration = sayAll(sg.narration); });
    if (st.revealNarration) st.revealNarration = sayAll(st.revealNarration);
    (st.items || []).forEach(function (it) { if (it.why) it.why = speakify(it.why); });
    if (st.correctSay) st.correctSay = speakify(st.correctSay);
    if (st.type === 'check') {
      if (!st.narration || !st.narration.length) st.narration = [speakify(plain(st.question))];
      (st.options || []).forEach(function (o) { if (o.why) o.why = speakify(o.why); });
    }
  }
  // The engine does not shuffle options: vary the position of the correct one (deterministic).
  var posSeq = [2, 0, 3, 1, 4, 2, 1, 3, 0, 2, 1];
  var posN = 0;
  function placeCorrect(st) {
    if (st.type !== 'check' || !st.options || st.options.length < 3) return;
    var ci = -1;
    st.options.forEach(function (o, i) { if (o.correct) ci = i; });
    var target = posSeq[posN % posSeq.length] % st.options.length;
    posN++;
    if (ci < 0 || ci === target) return;
    var c = st.options.splice(ci, 1)[0];
    st.options.splice(target, 0, c);
  }
  [ch0, ch1, chSrc, ch2, ch3, ch4, ch5, ch6].forEach(function (ch) { ch.steps.forEach(function (st) { cleanStep(st); placeCorrect(st); }); });
  pool.forEach(function (st) { cleanStep(st); placeCorrect(st); });

  /* ======================= COURSE OBJECT ======================= */

  var allRefs = [];
  Object.keys(R).forEach(function (k) { allRefs.push(R[k]); });

  window.LMS_COURSES = window.LMS_COURSES || {};
  window.LMS_COURSES.ferroelectric = {
    id: 'ferroelectric',
    code: 'FE-0042',
    title: 'Ferroelectric Workplace Conduct and Crystal Orientation Awareness',
    subtitle: 'Lithium niobate and related ferroelectrics: orientation, charge, and simulation integrity. Annual mandatory refresher.',
    duration: 'approximately 180 minutes',
    version: '2026.4',
    effective: '2026-12-01',
    policy: 'Policy PHC-17.3, Ferroelectric Conduct and Crystal Orientation',
    mantra: MANTRA,
    mantraShort: ['Know your axes.', 'Watch your charge.', 'Scan both ways.', 'Trust no symmetry plane.'],
    host: { name: 'Pat Reynolds', role: 'Director, Office of Photonic Conduct', img: 'media/render/host_pat_present.jpg',
      cam: { idle: 'media/render/host_pat_cam_idle.mp4', talk: ['media/render/host_pat_cam_talk.mp4', { src: 'media/render/host_pat_cam_gesture.mp4', weight: 0.5 }] } },
    employer: 'Five Sigma Photonics, Inc.',
    chapters: [ch0, ch1, chSrc, ch2, ch3, ch4, ch5, ch6],
    exam: { passPct: 80, count: 10, pool: pool },
    references: allRefs,
    widgets: widgets,
    speech: []
  };
})();
