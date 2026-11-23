/* Course: Feedback, Reflection and Zero Tolerance for Retaliation in the Optical Workplace (FB-1550)
   Engine contract: course/COURSE_SPEC.md. Classic script, no DOM access at load time. */
(function () {
  'use strict';

  var MANTRA = 'Every interface reflects. Every reflection is a cavity. Every cavity hops. Isolate, angle, and verify.';
  var MSHORT = ['Every interface reflects.', 'Every reflection is a cavity.', 'Every cavity hops.', 'Isolate. Angle. Verify.'];
  var GARY = 'Gary';
  var MGR = 'Morgan (Manager)';
  var BY = 'Dana Okafor (Labmate)';

  /* ---------- reference table (all from sourced research notes) ---------- */
  var R = {
    tkach: { cite: 'Tkach and Chraplyvy, "Regimes of feedback effects in 1.5-um distributed feedback lasers", J. Lightwave Technol. 4, 1655 (1986)', doi: '10.1109/JLT.1986.1074666' },
    marinins: { cite: 'Marinins et al., "Wafer-scale hybrid integration of InP DFB lasers on Si photonics by flip-chip bonding with sub-300nm alignment precision", IEEE J. Sel. Top. Quantum Electron., 2022', doi: '10.1109/jstqe.2022.3223641' },
    kurczveil: { cite: 'Kurczveil et al., "Characterization of insertion loss and back reflection in passive hybrid silicon tapers", IEEE Photonics Journal, 2013', doi: '10.1109/jphot.2013.2246559' },
    moscoso: { cite: 'Moscoso-Martir et al., "Hybrid silicon photonics flip-chip laser integration with vertical self-alignment", CLEO Pacific Rim, 2017', doi: '10.1109/cleopr.2017.8118971' },
    dekoninck: { cite: 'De Koninck et al., "GaAs nano-ridge laser diodes fully fabricated in a 300-mm CMOS pilot line", Nature, 2025 (statements quoted from the preprint, arXiv:2309.04473, 2023)', doi: '10.1038/s41586-024-08364-2' },
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
    lapointe: { cite: 'Lapointe et al., "Passive broadband Faraday isolator for hybrid integration to photonic circuits without lens and external magnet", Nature Photonics, 2025 (numbers quoted from the preprint, Research Square rs-3157765, 2023)', doi: '10.1038/s41566-024-01601-0' },
    kashyap: { cite: 'Kashyap and Blow, "Observation of catastrophic self-propelled self-focusing in optical fibres", Electronics Letters, 1988', doi: '10.1049/el:19880032' },
    ji2017: { cite: 'Ji et al., "Ultra-low-loss on-chip resonators with sub-milliwatt parametric oscillation threshold", Optica, 2017', doi: '10.1364/optica.4.000619' },
    hausmann: { cite: 'Hausmann et al., "Diamond nonlinear photonics", Nature Photonics, 2014', doi: '10.1038/nphoton.2014.72' },
    raja: { cite: 'Raja et al., "Electrically pumped photonic integrated soliton microcomb", Nature Communications, 2019', doi: '10.1038/s41467-019-08498-2' },
    white: { cite: 'White et al., "Unified laser stabilization and isolation on a silicon chip", Nature Photonics, 2024', doi: '10.1038/s41566-024-01539-3' },
    kramnik: { cite: 'Kramnik et al., "Scalable feedback stabilization of quantum light sources on a CMOS chip", Nature Electronics 8, 620, 2025 (preprint arXiv:2411.05921, 2024)', doi: '10.1038/s41928-025-01410-5' },
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
  /* judged chats: intro (read or watch, then judge) is spoken before the OK / Not OK popup; the verdict line after it */
  function isThisOk(line, intro) { return [].concat(intro, ['Pause and ask yourself. Is this okay?', line, 'Remember. ' + MANTRA]); }

  /* =================================================================== */
  /* INCIDENT REVIEWS                                                       */
  /* Numbers: Fresnel R = ((n-1)/(n+1))^2 with n = 1.444 fused silica      */
  /* (Malitson 1965) -> 3.3 %, -14.8 dB; n = 3.48 silicon (Li 1980) ->     */
  /* 31 %; air-gap etalon period lambda^2 / (2 L); Lapointe 2025 isolator  */
  /* > 25 dB isolation, < 1.5 dB insertion loss (stated as an assumption); */
  /* doublet numbers from the chapter-5 coupled-mode model.                */
  /* =================================================================== */
  var APPROVED = 'I noticed something on the optical path. Can we pause and check it together?';
  function OK(t) { return { text: t, correct: true, why: '' }; }
  function W(t, why) { return { text: t, correct: false, why: why + ' Isolate, angle, and verify. And log it.' }; }
  function WB(t, why) { return { text: t, correct: false, why: why + ' The approved phrase is: ' + APPROVED + '' }; }
  function tester(qn, opts, say) {
    return { type: 'check', kicker: 'What would you do?', question: qn, options: opts, correctSay: say,
      narration: ['What would you do?', 'Put yourself in Gary\'s position. There are no wrong answers. There are only answers that get struck out.'] };
  }
  var nDana = 0;
  function bystander(qn, opts, say) {
    var first = nDana++ === 0;
    return { type: 'check', kicker: first ? 'You are Dana, Gary\'s labmate' : 'You are Dana', question: qn, options: opts, correctSay: say,
      narration: [first ? 'You are Dana Okafor, Gary\'s labmate.' : 'You are Dana.', 'Dana saw it happen. Now Dana has a choice.', 'Remember the approved phrase. ' + APPROVED] };
  }
  function felt(situation, bullets, narr) {
    return { type: 'slide', layout: 'host', title: 'How Do You Think Gary Felt?', hostImg: 'media/render/host_pat_empathy.jpg',
      hostCam: { idle: 'media/render/host_pat_cam_empathy_idle.mp4', talk: ['media/render/host_pat_cam_empathy_talk.mp4'] },
      bullets: [situation, 'Gary felt ashamed', 'Shame is a normal response to a reflection, including your own', 'This is a judgment-free zone', 'The incident has still been logged'].concat(bullets || []),
      callout: 'Judgment-free. Still logged.',
      narration: ['Let us pause for a moment.', situation + '.', 'How do you think Gary felt?', 'Gary felt ashamed.', 'That is normal. Shame is a normal response to a reflection, including your own.', 'This is a judgment-free zone. Nobody in this course is judging Gary.', 'The incident has still been logged.'].concat(narr || []).concat(['Remember. ' + MANTRA]) };
  }
  function feltTabs(situation, tabs) {
    return { type: 'reveal', title: 'How Do You Think Gary Felt?', prompt: situation + '. How do you think Gary felt? Open each feeling. All feelings are valid. All feelings are logged.', tabs: tabs,
      narration: ['Let us pause for a moment.', situation + '.', 'How do you think Gary felt? Please open each feeling.', 'All feelings are valid. All feelings are logged.'] };
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
        narration: ['Welcome to Feedback, Reflection, and Zero Tolerance for Retaliation in the Optical Workplace.', 'This course is mandatory for all employees of Five Sigma Photonics, Incorporated.', 'It is also mandatory for contractors, interns, visitors, and any laser on the premises.', 'This course takes approximately one hundred and fifty-five minutes.', 'Thank you for taking the time to keep our photons safe, respected, and travelling forward.'] },
      { type: 'slide', layout: 'host', title: 'A Message From Pat Reynolds',
        bullets: ['Director, Office of Photonic Conduct', 'Every photon deserves to travel forward without fear', 'Feedback is welcome. Unsolicited optical feedback is not'],
        callout: MANTRA,
        narration: ['Hi. I am Pat Reynolds, Director of the Office of Photonic Conduct.', 'At Five Sigma Photonics, we believe in feedback.', 'We believe in open doors, open communication, and an open speak-up culture.', 'What we do not believe in is optical feedback that nobody asked for.', 'Light that comes back into a laser cavity is not a conversation. It is retaliation.', 'And we have zero tolerance for retaliation.', 'Over the next seven chapters, we will look at where retaliation comes from, how it forms cavities you never designed, and how those cavities make lasers hop.', 'Our mantra for this course is simple. ' + MANTRA, 'You will hear it often. That is intentional.'] },
      { type: 'slide', layout: 'policy', title: 'Course Enforcement Features',
        bullets: ['The Next button stays locked until narration ends', 'Playback speed cannot be changed. Light does not hurry; neither will you', 'Leaving this browser tab pauses the course', 'All scenarios are mandatory, including the extra ones', 'Knowledge checks allow unlimited retries', 'A certificate is issued after the final assessment (80 percent to pass)'],
        callout: 'These controls exist for your benefit and cannot be disabled.',
        narration: ['Before we begin, a few words about how this course works.', 'The Next button stays locked until the narration ends.', 'Playback speed cannot be changed.', 'If you leave this browser tab, the course will pause and wait for you, like a laser waiting for its cavity phase to be set.', 'All scenarios are mandatory. Including the extra ones. Especially the extra ones.', 'Knowledge checks allow unlimited retries. Wrong answers come with a detailed explanation, read aloud in full.', 'These controls exist for your benefit and cannot be disabled.'] },
      { type: 'reveal', title: 'Key Terms Used in This Course', prompt: 'Open each tab to review the policy definitions.',
        tabs: [
          { label: 'Feedback', body: '<p><b>Optical feedback</b>: any light that re-enters a laser cavity after leaving it. It adds a delayed field to the field inside the cavity. Its effect depends on how much returns, with what phase, and from how far away.</p>', narration: ['Optical feedback is any light that re-enters a laser cavity after leaving it.', 'It adds a delayed copy of the field to the field inside the cavity.', 'What happens next depends on how much light returns, with what phase, and from how far away.'] },
          { label: 'Retaliation', body: '<p><b>Retaliation</b> (policy term): a back-reflection. Every refractive index step, facet, coupler end, crossing, splice and connector can retaliate. Retaliation does not require intent.</p>', narration: ['In this course, retaliation is our policy term for a back-reflection.', 'Every refractive index step can retaliate. Facets, coupler ends, crossings, splices and connectors.', 'Retaliation does not require intent.'] },
          { label: 'Parasitic cavity', body: '<p><b>Parasitic cavity</b>: any two reflections on the same optical path. Together they form a Fabry-Perot etalon that nobody drew on the layout. Its free spectral range is $c/(2nL)$, with $n$ the group index and $L$ the spacing.</p>', narration: ['A parasitic cavity is any two reflections on the same optical path.', 'Together they form a Fabry-Perot etalon that nobody drew on the layout.', 'Its free spectral range is the speed of light divided by two times the group index times the spacing.'] },
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
          { t: 0, cast: ['Gary'], caption: 'Gary flip-chip bonds an InP laser onto a silicon photonics chip.', shot: 'Lab bench, medium shot from 35 degrees above bench height, 50 mm lens. Flip-chip bonder with a small InP laser die held by the collet, lowering onto a silicon photonics chip with silicon nitride waveguides and grating couplers, on the chuck. Gary leans in, safety glasses on forehead. Too-clean training-video lighting, soft key from left.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.flipChip(ctx, w, h, u, { mode: 'bond' }); } },
          { t: 7, cast: ['Gary'], caption: 'The light versus current curve has periodic ripples.', shot: 'Over-the-shoulder on Gary to a bench monitor. Light versus current curve plotting live, with a periodic ripple. Slow push-in on the monitor. Gary points at the gap between laser and chip on a second monitor showing a microscope view.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.liCurve(ctx, w, h, u); } },
          { t: 13, cast: ['Gary'], caption: 'Gary blames the sub-micron bond gap and reworks the bond. Three times.', shot: 'Montage, locked-off wide. Gary lifts the die, re-bonds, re-measures. Wall clock advances between cuts. The ripple on the monitor does not change. Freeze frame, red X stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.flipChip(ctx, w, h, u, { mode: 'gap', stamp: 'VIOLATION' }); } },
          { t: 19, cast: ['Pat Reynolds'], caption: 'Replay: measure the whole path. The far-end grating coupler is the second mirror.', shot: 'Top-down macro of the chip. An animated ruler extends from the laser facet across the entire chip to the grating coupler at the far end. The coupler glows red. Green check stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.flipChip(ctx, w, h, u, { mode: 'whole', stamp: 'COMPLIANT', ok: true }); } }
        ],
        narration: ['Meet Gary.', 'Gary has bonded a laser onto a silicon photonics chip.', 'The light versus current curve has ripples. Regular, periodic ripples.', 'Gary blames the tiny gap between the laser and the chip, because it is the closest interface.', 'He reworks the bond. Then he reworks it again.', 'Let us replay that. This time, Gary measures the whole optical path.', 'The second mirror was at the far end of the chip all along.'] },
      { type: 'slide', layout: 'host', title: 'What Is Unsolicited Feedback?',
        bullets: ['Light returning into the laser adds a delayed field to the cavity field', 'The effect depends on returned power, its phase, and the delay (distance)', 'Changing current or temperature changes the phase, so the effect is periodic', 'Feedback does not need your permission, or your layout'],
        callout: MANTRA,
        narration: ['Let us take a closer look.', 'When light returns into a laser, it adds a delayed copy of the field to the field already inside the cavity.', 'Whether that helps or hurts depends on three things. How much light returns. Its phase. And how long it took to come back.', 'Current and temperature both change the phase, so the symptoms repeat periodically as you tune.', 'That is why a ripple with a regular period is a confession, not a coincidence.', 'Remember. ' + MANTRA] },
      { type: 'reveal', title: 'The Five Regimes of Feedback', prompt: 'Tkach and Chraplyvy (1986) classified feedback effects on 1.5 micron distributed feedback lasers into five regimes, ordered by increasing feedback. Open each regime.',
        tabs: [
          { label: 'Regime I', media: { src: 'media/lit/fb_tkach1986_regime1.svg', caption: 'Tkach and Chraplyvy, J. Lightwave Technol. 1986. Redrawn, illustrative.', href: 'https://doi.org/10.1109/JLT.1986.1074666' }, body: '<p>Very weak feedback. The linewidth narrows or broadens depending on the phase of the returning light.</p><p class="muted">The thresholds between regimes depend on the laser and on the distance to the reflector. The Mode Hop Map activity later in this course shows the levels reported for the 1986 experiment.</p>', narration: ['Regime one. Very weak feedback.', 'The linewidth narrows or broadens, depending on the phase of the returning light.', 'Even a whisper of retaliation changes the laser.'] },
          { label: 'Regime II', media: { src: 'media/lit/fb_tkach1986_regime2.svg', caption: 'Tkach and Chraplyvy, J. Lightwave Technol. 1986. Redrawn, illustrative.', href: 'https://doi.org/10.1109/JLT.1986.1074666' }, body: '<p>Stronger feedback. The laser can split into or hop between external-cavity modes. This is where mode hopping from feedback lives.</p>', narration: ['Regime two. Stronger feedback.', 'The laser can hop between modes of the new, larger cavity that now includes the reflector.', 'Every reflection is a cavity. Every cavity hops.'] },
          { label: 'Regime III', media: { src: 'media/lit/fb_tkach1986_regime3.svg', caption: 'Tkach and Chraplyvy, J. Lightwave Technol. 1986. Redrawn, illustrative.', href: 'https://doi.org/10.1109/JLT.1986.1074666' }, body: '<p>A narrow range where the laser settles on a single external-cavity mode with a narrow linewidth, insensitive to the feedback phase.</p>', narration: ['Regime three. A narrow window where the laser settles on one external cavity mode, with a narrow line.', 'This regime is narrow. Do not plan your career around it.'] },
          { label: 'Regime IV', media: { src: 'media/lit/fb_tkach1986_regime4.svg', caption: 'Tkach and Chraplyvy, J. Lightwave Technol. 1986. Redrawn, illustrative.', href: 'https://doi.org/10.1109/JLT.1986.1074666' }, body: '<p><b>Coherence collapse.</b> The linewidth broadens dramatically and the laser becomes noisy. In this regime the behavior is largely insensitive to the distance and phase of the reflector.</p>', narration: ['Regime four. Coherence collapse.', 'The linewidth broadens dramatically and the laser becomes noisy.', 'Here, changing the distance to the reflector does not save you.'] },
          { label: 'Regime V', media: { src: 'media/lit/fb_tkach1986_regime5.svg', caption: 'Tkach and Chraplyvy, J. Lightwave Technol. 1986. Redrawn, illustrative.', href: 'https://doi.org/10.1109/JLT.1986.1074666' }, body: '<p>Strong feedback into a laser with an anti-reflection coated facet: stable operation as an external-cavity laser. This is feedback that was designed, with consent.</p>', narration: ['Regime five. Strong feedback into a laser whose facet is anti-reflection coated.', 'The laser now runs stably as an external cavity laser.', 'This is feedback with consent. It was designed, documented, and approved.'] }
        ], narration: ['Feedback is not one thing. It is five things.'] },
      { type: 'slide', layout: 'bullets', title: 'Retaliation From an Unexpected Direction', image: { src: 'media/lit/fb_marinins2022_far_end.svg', kb: [0.5, 0.5, 1, 0.5, 0.5, 1] },
        bullets: ['Flip-chip InP distributed feedback lasers on silicon photonics showed light versus current ripples with a 25 to 30 mA period (Marinins et al., 2022)', 'The cause: about 4 percent (-14 dB) feedback from a silicon nitride grating coupler at the far end of the waveguide', 'Not the sub-micron gap between the laser and the chip', 'Everyone suspects the closest interface. Measure the whole path'],
        callout: 'The far end of your chip is part of your laser until proven otherwise.',
        narration: ['This is not hypothetical.', 'Marinins and coworkers flip-chip bonded indium phosphide distributed feedback lasers onto silicon photonics.', 'The light versus current curves showed ripples with a period of twenty-five to thirty milliamps.', 'The cause was about four percent feedback, minus fourteen decibels, from a silicon nitride grating coupler at the far end of the waveguide.', 'Not the gap of less than one micron right next to the laser.', 'Everyone suspects the closest interface. Please measure the whole path.', 'Every interface reflects. Every reflection is a cavity.'] },
      { type: 'slide', layout: 'policy', title: 'Return Loss Budgets and Placement',
        bullets: ['Reflections from a III-V to silicon mode converter are cited as needing to be below -50 dB for feedback-sensitive circuits; measured passive tapers were below -41 dB (Kurczveil et al., 2013)', 'Tapers are usually optimized for insertion loss, not return loss. Budget both', 'Laser sensitivity to back-reflection depends strongly on the distance to the reflector; one argument is to place the isolator at the chip output rather than right after the laser (Moscoso-Martir et al., 2017)', 'An isolator anywhere is not the same as an isolator in the right place'],
        narration: ['Policy section four point two, Return Loss Budgets.', 'For feedback-sensitive circuits, reflections from a mode converter between the laser material and silicon are cited as needing to be below minus fifty decibels.', 'Kurczveil and coworkers measured passive hybrid tapers below minus forty-one decibels.', 'Read that again. The measured number and the cited requirement are not the same number.', 'Tapers are usually optimized for insertion loss. Please budget return loss too.', 'Placement also matters. Laser sensitivity depends strongly on the distance to the reflector.', 'That is one argument for placing the isolator at the chip output, after the reflectors, rather than directly after the laser.', 'Isolate, angle, and verify.'] },
      { type: 'chat', title: 'Scenario 1: The Rework', app: 'phone', skin: 'imessage', meta: { header: 'Messages', participants: [GARY, MGR] },
        messages: [
          { from: GARY, text: 'L-I ripple is back on the reworked part. Period about 25 mA, same as before', side: 'right' },
          { from: GARY, text: 'Most likely still the bond gap. I will redo the bond with more pressure, done by Thursday', side: 'right' },
          { from: MGR, text: 'OK. That is the third rework?', side: 'left' },
          { from: GARY, text: 'Fourth. The gap is under a micron now, so this should be the last one', side: 'right' },
          { from: MGR, text: 'Go ahead. If it is still there, call it a packaging artifact in the report and we move on', side: 'left' }
        ], narration: isThisOk('Gary has reworked the bond four times without asking why the ripple period never changed, and the fallback is to call it a packaging artifact.',
          ['Read this text exchange between Gary and his manager, Morgan, about a ripple on a bonded laser.', 'Then judge whether Gary\'s plan is proper.']) },
      { type: 'chat', title: 'Scenario 2: The Flat Cleave', app: 'email', skin: 'mail', meta: { header: 'Inbox', subject: 'Laser output unstable near facet, RMA request', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Hi Morgan,\n\nThe DFB on bench 2 jumps in output power whenever the fiber comes within a few tens of microns of the facet. The datasheet says it should be stable. I would like to open an RMA with the vendor while we are in warranty. Trace attached.\n\nThanks,\nGary', side: 'right', to: [MGR], cc: [BY], date: '2026-11-12 09:14', subject: 'Laser output unstable near facet, RMA request', attachments: [{ name: 'power_vs_gap_2026-11-12.csv', size: '48 KB' }] },
          { from: BY, text: 'Gary, quick question before the RMA: is the fiber flat cleaved? A bare glass end reflects a few percent (Fresnel), so the tip and the facet form a cavity, and moving the fiber sweeps its phase. Could we try an angle-cleaved fiber or a drop of gel first?\n\nDana', side: 'left', to: [GARY], cc: [MGR], date: '2026-11-12 09:41' },
          { from: GARY, text: 'It is a clean flat cleave, checked under the scope. I would rather not change the setup in the middle of the characterization. The vendor can test the laser on their side.', side: 'right', to: [BY], cc: [MGR], date: '2026-11-12 10:03' },
          { from: MGR, text: 'Agree, let us send it back while it is in warranty. We can revisit the fiber if they find nothing.\n\nMorgan', side: 'left', to: [GARY, BY], date: '2026-11-12 10:20' }
        ], narration: isThisOk('Gary wants to return the laser before testing the fiber tip he has just described, and Morgan agrees.',
          ['Read this email thread between Gary, Dana and Morgan, about an unstable laser on bench two.', 'Then judge whether Gary\'s request is proper.']) },
      { type: 'chat', title: 'Scenario 3: Isolator Placement', app: 'thread', skin: 'slack', meta: { header: '#laser-integration', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Isolator arrived, installed this morning', side: 'right' },
          { from: GARY, text: 'I put it at the chip output, on the first fiber after the chip. The spool, its connectors and the receiver are all behind it now', side: 'right' },
          { from: BY, text: 'The taper, the crossings and the grating coupler are still between the laser and the isolator though', side: 'left' },
          { from: GARY, text: 'Yes. Those are close to the laser and in the return-loss budget. I also took the L-I curve and the linewidth with the spool connected and disconnected: no ripple and no linewidth change either way. A flat L-I curve alone only rules out strong feedback. Both measurements are in the ticket', side: 'right' },
          { from: MGR, text: 'Thanks. Closing the ticket with the measurement attached', side: 'left' }
        ], narration: isThisOk('Gary placed the isolator at the chip output, in front of the distant reflectors, budgeted the near ones, and closed the ticket with a measurement attached.',
          ['Read this team chat about where Gary installed the new isolator.', 'Then judge whether his placement is proper.']) },
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
          { t: 0, cast: ['Gary'], caption: 'Gary measures a modulator transfer curve. It has wiggles.', shot: 'Office cubicle, medium close-up on Gary at a dual monitor. Left monitor: Mach-Zehnder modulator transfer curve with a fine periodic ripple riding on the cosine. Gary squints, then smiles.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.mzmTrace(ctx, w, h, u); } },
          { t: 7, cast: ['Gary'], caption: 'Gary adds a sinusoid to his fit model. The fit is excellent.', shot: 'Screen insert: fit dialog with a new sinusoid term being typed in. Fit residuals collapse to zero. Gary does a small fist pump. Red X stamp freezes the frame.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.mzmTrace(ctx, w, h, u, { fit: true, stamp: 'VIOLATION' }); } },
          { t: 13, cast: ['Pat Reynolds'], caption: 'Replay: the ripple period is a ruler. It points to two reflectors.', shot: 'Camera cranes up out of the monitor into a stylized top view of the chip, which becomes a hall of mirrors: each facet, crossing and coupler end flashes as a reflector, and pairs of them are joined by glowing arcs labelled with their implied spacing.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.mzmChip(ctx, w, h, u, { stamp: 'COMPLIANT', ok: true }); } }
        ],
        narration: ['Gary measures the transfer curve of his modulator.', 'It has wiggles. Small, regular, periodic wiggles.', 'Gary adds a sinusoid to his fit model. The fit is excellent.', 'Let us replay that.', 'The period of the wiggle is a ruler. It tells you the spacing between two reflectors.', 'Every interface reflects. Every reflection is a cavity.'] },
      { type: 'slide', layout: 'bullets', title: 'The Parasitic Etalon, in Two Formulas',
        bullets: ['Two reflectors with power reflectances $R_1$ and $R_2$, spacing $L$, group index $n$', 'Ripple period: $\\mathrm{FSR} = c/(2nL)$; in wavelength, $\\Delta\\lambda = \\lambda^2/(2nL)$', 'Lossless Airy transmission, with round-trip phase $\\phi$: $$T = \\frac{(1-R_1)(1-R_2)}{1 + R_1R_2 - 2\\sqrt{R_1R_2}\\,\\cos\\phi}$$', 'Peak-to-valley ripple: $20\\log_{10}\\left[(1+r)/(1-r)\\right]$ dB, with $r = \\sqrt{R_1R_2}$','Weak mirrors still make a cavity. The ripple is small, but the period is exact'],
        callout: 'Measure the period. Compute the length. Find the two reflectors.',
        narration: ['Every parasitic cavity is described by two formulas.', 'The ripple period, or free spectral range, is the speed of light divided by two times the group index times the spacing.', 'The ripple depth depends on the geometric mean of the two reflectances.', 'Weak mirrors still make a cavity. The ripple may be small, but its period is exact.', 'So measure the period, compute the length, and go find the two reflectors.', MANTRA] },
      { type: 'widget', title: 'Interactive: Parasitic Fabry-Perot Ripple', widget: 'fabryPerot', params: {},
        instructions: 'Run a laser sweep. Then change the two interfaces, the gap length $L$ or the group index $n_g$, and sweep again. Watch the ripple period $\\lambda^2 / (2 n_g L)$ and its depth, and try a long gap with a coarse step. Then use the ripple-to-length calculator.',
        narration: ['In this activity, you will build a cavity you did not design.', 'Press Run laser sweep. Then choose the two reflecting interfaces, change the gap, and sweep again.', 'Notice that the depth depends on the reflectances, and the period depends only on the spacing and the group index.', 'Now try a long gap with a coarse step. If the laser steps farther than half a ripple period, the sweep shows a slower, false ripple, or none at all. The cavity is still there.'] },
      { type: 'reveal', title: 'Cavities You Did Not Design', prompt: 'Open each tab to meet a reflector that was not on the schematic.',
        tabs: [
          { label: 'Amplifier facets', media: { src: 'media/lit/fb_bao2024_fig4ab.png', caption: 'Bao et al., arXiv 2024, Fig. 4(a,b).', href: 'https://arxiv.org/abs/2405.16583' }, body: '<p>In a high-gain erbium-doped thin film lithium niobate amplifier, facet back-reflections reinforced amplified spontaneous emission, and parasitic lasing set in at the 1531.5 nm fluorescence peak (Bao et al., 2024).</p><p>An amplifier has no mirrors only until it has gain.</p>', narration: ['An erbium-doped amplifier on thin film lithium niobate.', 'Facet back-reflections reinforced the amplified spontaneous emission, and parasitic lasing set in at fifteen thirty-one point five nanometers, at the fluorescence peak.', 'In the figure from the paper, the shaded band is the parasitic lasing regime, above twenty-eight decibels of gain.', 'An amplifier has no mirrors only until it has enough gain.', 'Every interface reflects.'] },
          { label: 'Crossings and splitters', media: { src: 'media/lit/fb_crossings_splitters.svg', caption: 'Celo et al., IEEE Group IV Photonics 2017; Tian et al., Opt. Quantum Electron. 2025. Illustrative.', href: 'https://doi.org/10.1109/group4.2017.8082188' }, body: '<p>A poorly designed waveguide crossing contributes crosstalk, parasitic cavities from back-reflection, and multipath (Celo et al., 2017). Splitter trees accumulate back-reflection with depth (Tian et al., 2025).</p>', narration: ['Waveguide crossings.', 'A poorly designed crossing contributes crosstalk, parasitic cavities, and multipath interference.', 'And splitter trees accumulate back-reflection with every level of depth.', 'Crossings and splitters are usually budgeted for loss only. Budget their reflections too.'] },
          { label: 'Multimode interferometer walls', media: { src: 'media/lit/fb_cherchi2014_mmi_walls.svg', caption: 'Cherchi et al., Opt. Express 2014. Redrawn, illustrative.', href: 'https://doi.org/10.1364/oe.22.009245' }, body: '<p>Back-reflection ripples from the end walls of multimode interferometers only appear in simulation when enough modes are included; one double-interferometer study needed more than 30 (Cherchi et al., 2014).</p><p>If your simulation has no ripple, check that it is allowed to have one.</p>', narration: ['Multimode interferometers have end walls, and end walls reflect.', 'In one study, the reflection ripples only appeared in simulation once more than thirty modes were included.', 'If your simulation shows no ripple, first check that it was allowed to show one.'] },
          { label: 'Vertical grating couplers', media: { src: 'media/lit/fb_vaerenbergh2021_vertical_gc.svg', caption: 'Van Vaerenbergh et al., Opt. Express 2021. Redrawn, illustrative.', href: 'https://doi.org/10.1364/oe.433744' }, body: '<p>A perfectly vertical grating coupler gives a large back-reflection into the waveguide, which is why grating couplers are usually tilted (Van Vaerenbergh et al., 2021).</p>', narration: ['Perfectly vertical grating couplers.', 'They give a large back-reflection into the waveguide, which is one reason grating couplers are usually tilted.', 'Angle is a policy, not a style choice.'] },
          { label: 'Offset splices', media: { src: 'media/lit/fb_contreras2023_offset_splice.svg', caption: 'Contreras-Teran et al., Meas. Sci. Technol. 2023. Redrawn, illustrative.', href: 'https://doi.org/10.1088/1361-6501/acb3ea' }, body: '<p>A single-mode fiber spliced with a deliberate lateral core offset excites cladding modes and becomes an in-fiber Mach-Zehnder interferometer, used as a strain sensor (Contreras-Teran et al., 2023).</p><p>An accidental offset can do the same: an unintended interferometer in your link.</p>', narration: ['An offset splice.', 'A single mode fiber spliced with a deliberate lateral offset excites cladding modes, and becomes an interferometer inside the fiber. It has been used as a strain sensor.', 'An accidental offset can do the same thing, uninvited.'] },
          { label: 'The dark port', body: '<p>In an on-chip interferometer that recombines through a 1x2 junction, there is no second guided output. Light that does not interfere constructively is not routed to another port: it is radiated or scattered, and part of it can reflect back toward the source.</p>', narration: ['The dark port of a one by two combiner.', 'There is no second guided output for the light to go to.', 'So the light that does not interfere constructively is radiated, or scattered, and part of it can come back.', 'Energy is conserved. Your budget is not.'] }
        ], narration: ['Parasitic cavities are everywhere. Let us meet some of them.'] },
      { type: 'chat', title: 'Scenario 1: The Excellent Fit', app: 'thread', skin: 'teams', meta: { header: '#modulator-team', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Transfer-curve fits are done. I added a small sinusoidal term for the residual wiggle, R squared 0.999 on all devices', side: 'right' },
          { from: BY, text: 'Nice. What is the period of that wiggle? Converted to a length it would tell us where it comes from', side: 'left' },
          { from: GARY, text: 'I treated it as a nuisance parameter. It is small and it does not change V pi', side: 'right' },
          { from: BY, text: 'It might not change V pi, but it is a cavity somewhere', side: 'left' },
          { from: MGR, text: 'Fits look clean. Let us use them in the review deck', side: 'left' }
        ], narration: isThisOk('Gary absorbed the ripple into a fit term without converting its period into a length.',
          ['Read this team chat about Gary\'s modulator transfer curve fits.', 'Then judge whether Gary\'s handling of the wiggle is proper.']) },
      { type: 'chat', title: 'Scenario 2: The Accidental Laser', app: 'phone', skin: 'whatsapp', meta: { header: 'Messages', participants: [GARY, MGR] },
        messages: [
          { from: GARY, text: 'Amplifier test: a narrow line shows up right at the fluorescence peak at high pump', side: 'right' },
          { from: GARY, text: 'Could be a new low-threshold regime. Worth a quick invention disclosure?', side: 'right' },
          { from: MGR, text: 'Is that expected for an amplifier?', side: 'left' },
          { from: GARY, text: 'Not that I know of. That is why it might be novel', side: 'right' }
        ], narration: isThisOk('Gary is considering an invention disclosure for a narrow line at the gain peak, without checking for facet feedback.',
          ['Read this message exchange. Gary is testing a high-gain erbium-doped waveguide amplifier.', 'Then judge whether his next step is proper.']) },
      { type: 'chat', title: 'Scenario 3: Certified Single-Mode', app: 'call', skin: 'meeting', meta: { header: 'Video call', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'There are fringes on the pigtail spectrum. It is single-mode fiber and the splicer reported a good splice, but the splicer loss figure is only an estimate', side: 'right' },
          { from: GARY, text: 'A small core offset would launch cladding modes, and core plus cladding is an interferometer. I will re-splice one unit, put a tight bend right after the splice on another to strip the cladding modes, and see if the fringes go', side: 'right' },
          { from: BY, text: 'Sounds right. Log the fringe period before you change anything', side: 'left' },
          { from: MGR, text: 'Fine by me. Post the before and after', side: 'left' }
        ], narration: isThisOk('Gary treated the splicer figure as an estimate, and planned a test that shows whether the fringes come from cladding modes.',
          ['Watch this video call about fringes on a fiber pigtail spectrum.', 'Then judge whether Gary\'s response is proper.']) },
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
      takeaways([['A regular ripple is a cavity: its period gives the spacing, $L = c/(2n\\,\\mathrm{FSR})$','A regular ripple is a cavity. Its period gives the spacing between the two reflectors.'], ['Ripple depth grows with $\\sqrt{R_1R_2}$; weak mirrors still set an exact period','Ripple depth grows with the square root of R one times R two. Weak mirrors still set an exact period.'], 'Facets, coupler ends, crossings, splitters and interferometer walls all reflect', 'Gain plus two facets is a laser, whether or not you ordered one', 'If the ripple period moves when the fiber moves, the ripple belongs to the gap, not the device']),
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
          { t: 0, cast: ['Gary'], caption: 'Gary sweeps the laser temperature up and logs the wavelength.', shot: 'Lab bench, wide. A butterfly-packaged laser on a temperature controller, fiber to a wavemeter. Gary turns the controller knob up in steps, writing values on a clipboard. The wavemeter readout climbs smoothly, then jumps.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.hopSweep(ctx, w, h, u, { dir: 'up' }); } },
          { t: 9, cast: ['Gary', 'Dana'], caption: 'Dana repeats the sweep downward. The jump happens at a different temperature.', shot: 'Same set, Dana now at the controller turning the knob down. Split screen: Gary clipboard and Dana clipboard, the jump temperatures circled in red, not matching.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.hopSweep(ctx, w, h, u, { dir: 'down', prior: true }); } },
          { t: 16, cast: ['Gary'], caption: 'Gary opens a calibration ticket for the temperature sensor.', shot: 'Close-up on a laptop screen: a ticket titled "Thermometer inconsistent" being submitted. Red X stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.doc(ctx, w, h, u, { app: 'Facilities ticketing', title: 'Ticket: thermometer is inconsistent', rows: [['Filed by', 'Gary'], ['Asset', 'Laser temperature sensor, bench 3'], ['Sweep up: jump at', '28.19 \u00b0C', 'bad'], ['Sweep down: jump at', '27.43 \u00b0C', 'bad'], ['Request', 'Recalibrate the temperature sensor', 'type']], button: 'Submit', stamp: 'VIOLATION' }); } },
          { t: 21, cast: ['Pat Reynolds'], caption: 'Replay: the hop points differ because the laser remembers. That is hysteresis.', shot: 'Host stands beside a large wall chart showing two staircases, up and down, offset from each other. Green check stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.hopLoop(ctx, w, h, u, { stamp: 'COMPLIANT', ok: true }); } }
        ],
        narration: ['Gary sweeps the laser temperature up, and logs the wavelength.', 'It climbs smoothly. Then it jumps.', 'Dana repeats the sweep, going down. The jump happens at a different temperature.', 'Gary opens a calibration ticket for the temperature sensor.', 'Let us replay that. The thermometer is fine. The laser remembers which mode it was in.', 'Every cavity hops.'] },
      { type: 'slide', layout: 'host', title: 'Why Lasers Hop',
        bullets: ['The laser cavity supports a comb of longitudinal modes, one free spectral range apart', 'A filter (grating, ring, gain peak, or a parasitic etalon) selects which mode wins', 'Temperature, current and phase move the comb and the filter at different rates', 'When another mode wins, the laser jumps. Once it has jumped, it tends to stay: hysteresis'],
        callout: MANTRA,
        narration: ['Let us take a closer look.', 'A laser cavity supports a comb of modes, one free spectral range apart.', 'A filter selects which mode wins. That filter might be a grating, a ring, the gain peak, or a parasitic etalon from Chapter Two.', 'Temperature and current move the comb and the filter, but at different rates.', 'When a different mode starts to win, the laser jumps.', 'And once it has jumped, it tends to stay there. That is hysteresis. Sweep direction matters.', MANTRA] },
      { type: 'widget', title: 'Interactive: Mode Hop Map', widget: 'modeHop', params: {},
        instructions: 'Run a sweep up, then a sweep down, and compare where the hops happen. Then change the feedback level and the distance to the reflector, and watch the feedback parameter $C$: below $C = 1$ the laser is pulled but cannot hop.',
        narration: ['In this activity, you will sweep a laser that can see its own reflection.', 'Press Sweep up, then press Sweep down.', 'Compare where the hops happen. Going up and going down, the laser hops at different points. That is hysteresis.', 'Then lower the feedback level, or move the reflector closer, until the feedback parameter C drops below one. The laser is still pulled. It can no longer hop.'] },
      { type: 'reveal', title: 'Mode Hop Fine Print', prompt: 'Open each tab to review the fine print.',
        tabs: [
          { label: 'Tuning range is not mode-hop-free range', media: { src: 'media/lit/fb_vanrees2020_mhf_tuning.svg', caption: 'van Rees et al., Opt. Express 2020. Redrawn.', href: 'https://doi.org/10.1364/oe.386356' }, body: '<p>Mode-hop-free tuning is bounded by the laser cavity free spectral range unless the cavity phase follows the filter. A ring-enhanced extended-cavity laser reached 0.22 nm (28 GHz) of mode-hop-free tuning, six times its cavity free spectral range (van Rees et al., 2020).</p><p>Read the abstract for the tuning range. Read the figures for the continuous range.</p>', narration: ['The total tuning range and the mode-hop-free range are different numbers.', 'Without help, continuous tuning is bounded by the cavity free spectral range.', 'One ring-enhanced extended cavity laser reached zero point two two nanometers, or twenty-eight gigahertz, of mode-hop-free tuning. That was six times its cavity free spectral range.', 'Read the abstract for the tuning range. Read the figures for the continuous range.'] },
          { label: 'Cavity phase', media: { src: 'media/lit/fb_xu2021_fig5a.png', caption: 'Xu et al., Sci. Rep. 2021, Fig. 5(a).', href: 'https://doi.org/10.1038/s41598-021-95981-w' }, body: '<p>A kink-free power versus current curve (kinks are the typical sign of mode hops) in a photonic-wire-bond external-cavity laser required a proper initial cavity-phase setting. Some integrated external-cavity lasers in the literature hop and others do not; the authors believe this depends strongly on the design of the reflective semiconductor optical amplifier (RSOA), in particular its length, and on the pump current (Xu et al., 2021).</p>', narration: ['Cavity phase.', 'A kink-free power versus current curve in a photonic wire bond external cavity laser required setting the cavity phase properly first.', 'The figure shows that measured curve. No kinks.', 'Single mode at one current does not mean single mode at every current.'] },
          { label: 'A regime, not a property', media: [{ src: 'media/lit/fb_franken2024_fig3a.png', caption: 'Franken et al., arXiv 2024, Fig. 3(a).', href: 'https://arxiv.org/abs/2407.00269' }, { src: 'media/lit/fb_franken2024_fig4d.png', caption: 'Franken et al., arXiv 2024, Fig. 4(d).', href: 'https://arxiv.org/abs/2407.00269' }], body: '<p>Residual wiggles on a thin film lithium niobate photonic-wire-bond laser power curve were longitudinal mode hops, even though the same laser also ran 58 hours mode-hop-free with 4.4 MHz per hour drift, with both amplifiers at 125 mA, just above threshold (Franken et al., 2025).</p><p>Mode-hop-free: conditions apply.</p>', narration: ['Mode-hop-free is a regime, not a property.', 'In one thin film lithium niobate laser, wiggles on the power curve were longitudinal mode hops.', 'In the first figure, the points wander around the straight line. Those are the hops.', 'The same laser also ran fifty-eight hours mode-hop-free, just above threshold, with four point four megahertz per hour of drift.', 'Both statements are true. Conditions apply.'] },
          { label: 'Feedback hops too', media: { src: 'media/lit/fb_tkach1986_regime2.svg', caption: 'Tkach and Chraplyvy, J. Lightwave Technol. 1986. Redrawn, illustrative.', href: 'https://doi.org/10.1109/JLT.1986.1074666' }, body: '<p>External feedback adds a second cavity. In the feedback regime classification of Tkach and Chraplyvy (1986), one regime is hopping between external-cavity modes. A parasitic etalon is a filter you did not choose.</p>', narration: ['Feedback hops too.', 'External feedback adds a second cavity, and with it a second comb.', 'A parasitic etalon is a filter you did not choose. Every reflection is a cavity. Every cavity hops.'] }
        ], narration: ['Mode hops come with fine print. Let us read it together.'] },
      { type: 'chat', title: 'Scenario 1: The Datasheet', app: 'phone', skin: 'imessage', meta: { header: 'Messages', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'The datasheet front page lists a wide tuning range. Before planning one continuous sweep for the spectroscopy run, I read the footnotes', side: 'right' },
          { from: GARY, text: 'The mode-hop-free range is a much smaller number. I will sweep in segments inside it, and log the wavemeter reading at every step', side: 'right' },
          { from: BY, text: 'Nice. Record the sweep direction too, the hop points have hysteresis', side: 'left' },
          { from: GARY, text: 'Added a column for it', side: 'right' }
        ], narration: isThisOk('Gary read past the headline tuning range, found the mode-hop-free range, and planned the sweep inside it.',
          ['Read this text exchange. Gary is planning a spectroscopy run with a tunable laser.', 'Then judge whether his preparation is proper.']) },
      { type: 'chat', title: 'Scenario 2: Up and Down', app: 'email', skin: 'mail', meta: { header: 'Inbox', subject: 'Wavelength calibration table (final_v3)', participants: [GARY, MGR, BY] },
        messages: [
          { from: GARY, text: 'Hi Morgan,\n\nAttached is the wavelength versus temperature table, measured going up. Dana measured going down and found different hop points. To keep a single table for the customer, I have averaged the two.\n\nGary', side: 'right', to: [MGR], cc: [BY], date: '2026-11-19 10:15', subject: 'Wavelength calibration table (final_v3)', attachments: [{ name: 'wavelength_vs_T_final_v3.xlsx', size: '64 KB' }] },
          { from: BY, text: 'Both sweeps are right. It is hysteresis. An averaged table puts the hop points where the laser never hops in either direction. Could we add a sweep-direction column instead?\n\nDana', side: 'left', to: [GARY, MGR], date: '2026-11-19 10:47' },
          { from: MGR, text: 'One table is easier for the customer. Averaging sounds like a fair compromise.\n\nMorgan', side: 'left', to: [GARY, BY], date: '2026-11-19 11:05' }
        ], narration: isThisOk('Gary averaged two hop maps that differ because of hysteresis, to keep one table.',
          ['Read this email thread about a wavelength calibration table for a customer.', 'Then judge whether Gary\'s table is proper.']) },
      { type: 'chat', title: 'Scenario 3: Qualified at One Current', app: 'thread', skin: 'teams', meta: { header: '#product-qual', participants: [GARY, MGR, BY] },
        messages: [
          { from: GARY, text: 'Laser qualified mode-hop-free: overnight run, zero hops', side: 'right' },
          { from: BY, text: 'At which current and temperature? The product spec runs hotter and at higher current', side: 'left' },
          { from: GARY, text: 'At the nominal point. The overnight run is our standard qualification', side: 'right' },
          { from: MGR, text: 'Standard qual passed. Mark it qualified', side: 'left' }
        ], narration: isThisOk('Gary qualified the laser at one operating point, and the result is being applied to the whole operating range.',
          ['Read this product qualification chat.', 'Then judge whether the qualification is proper.']) },
      { type: 'check', question: 'Gary sweeps the temperature up and records hop points. Dana sweeps down and records different hop points. What is the most likely explanation?',
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
          { t: 0, cast: ['Gary'], caption: 'Gary adds an asymmetric mode converter between his laser and his chip, and labels it isolator in the layout.', shot: 'Lab bench, macro on a chip under a probe station. On the layout screen, Gary types the cell name "ISOLATOR" onto a section of waveguide that tapers asymmetrically. Proud nod.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.modeConv(ctx, w, h, u, { variant: 'layout' }); } },
          { t: 7, cast: ['Gary'], caption: 'Forward and backward transmission differ on a single-mode detector. Gary declares victory.', shot: 'Monitor insert: two bars, forward tall, backward short, measured through a single-mode fiber detector. Gary prints the plot.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.modeConv(ctx, w, h, u, { variant: 'measure', stamp: 'VIOLATION' }); } },
          { t: 13, cast: ['Pat Reynolds'], caption: 'Replay: the reflection travels back in the mode it left in, and arrives home as the fundamental mode.', shot: 'Animated cutaway of the waveguide. Forward: fundamental mode enters, exits as a higher-order mode. A downstream reflection sends the higher-order mode back; it reconverts to the fundamental mode and enters the laser. Red glow on the laser facet. Then a Faraday rotator and polarizer slide into place. Green check.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.modeConv(ctx, w, h, u, { variant: 'replay', stamp: 'COMPLIANT', ok: true }); } }
        ],
        narration: ['Gary has built an isolator. It says so in the layout.', 'It is a mode converter that treats the two directions differently.', 'On a single-mode detector, forward transmission is high, and backward transmission is low.', 'Let us replay that.', 'Light reflected downstream travels back in the same mode it left in. The converter undoes its own work, and the reflection arrives home in the fundamental mode.', 'Reciprocity is not optional. Isolate, angle, and verify.'] },
      { type: 'slide', layout: 'policy', title: 'Policy 4.1: Definition of an Isolator',
        bullets: ['An optical isolator must break Lorentz reciprocity (Jalas et al., 2013)', 'Linear, time-invariant, non-magnetic structures are reciprocal: the scattering matrix is symmetric', 'Options that break reciprocity: magneto-optic materials with a bias field, or time modulation. Nonlinearity only partly: it works in a power window and is limited by dynamic reciprocity (Shi et al., 2015)', 'Asymmetric mode conversion, measured with a single-mode detector, can look like isolation. It is not', 'The test: launch backward into the forward output mode(s), count all ports and modes, and compare with the forward transmission. Equal means reciprocal'],
        callout: 'If the scattering matrix is symmetric, the device is not an isolator. Labels do not change matrices.',
        narration: ['Policy section four point one. Definition of an isolator.', 'An optical isolator must break Lorentz reciprocity. Jalas and coworkers wrote this down in two thousand thirteen, in a paper titled, What is, and what is not, an optical isolator.', 'A structure that is linear, time-invariant, and non-magnetic is reciprocal. Its scattering matrix is symmetric.', 'To break reciprocity you need a magneto-optic material with a bias field, or time modulation.', 'Nonlinearity counts only partly. A nonlinear isolator works only within a window of power, and it is limited by dynamic reciprocity, as Shi and coworkers showed in two thousand fifteen.', 'A device that converts modes differently in each direction can look like an isolator if you only measure one mode. It is not one.', 'The proper test is to launch light backward into the mode, or modes, that the forward light exits in. Count every port and every mode, and compare the transmission with the forward direction. If they match, the device is reciprocal.', 'Isolate, angle, and verify.'] },
      { type: 'reveal', title: 'Isolation Facts and Myths', prompt: 'Open each tab.',
        tabs: [
          { label: 'Why the reflection comes home', body: '<p>A reflector downstream returns light in the mode that arrived there. In a reciprocal device, light launched backward in the forward output mode reaches the forward input mode with the same transmission as the forward path ($S_{ij} = S_{ji}$): straight into the laser. Asymmetric mode conversion only changes what happens to <i>other</i> backward inputs.</p>', narration: ['Why the reflection comes home.', 'A downstream reflector sends light back in the mode that arrived there.', 'In a reciprocal device, that light reaches the mode that went in, with the same transmission as the forward path. Straight into the laser.', 'Mode conversion only changes what happens to other backward inputs. Not to your own reflection.'] },
          { label: 'Simulated isolation', body: '<p>A large isolation number from a simulation is not a working isolator. Check the boundary conditions, the materials (is anything actually nonreciprocal?), and which ports and modes were counted.</p>', narration: ['Simulated isolation.', 'A large number in a simulation is not a working isolator.', 'Check whether anything in the model is actually nonreciprocal, and which ports and modes were counted.'] },
          { label: 'Real isolators', media: [{ src: 'media/lit/fb_lapointe2023_fig3g.png', caption: 'Lapointe et al., Research Square 2023, Fig. 3(g).', href: 'https://doi.org/10.21203/rs.3.rs-3157765/v1' }, { src: 'media/lit/fb_lapointe2023_fig3k.png', caption: 'Lapointe et al., Research Square 2023, Fig. 3(k).', href: 'https://doi.org/10.21203/rs.3.rs-3157765/v1' }], body: '<p>The usual fix for feedback-sensitive semiconductor lasers is a polarizer plus a Faraday rotator in a magnetic field, the classic bulk isolator. A lens-free, magnet-free version for hybrid integration reached more than 25 dB isolation across the C-band, with a projected insertion loss below 1.5 dB: 0.8 dB measured, plus about 0.5 dB for the two polarizers (Lapointe et al., 2023).</p>', narration: ['Real isolators.', 'The classic fix is a polarizer plus a Faraday rotator in a magnetic field.', 'A compact version for hybrid integration, with no lens and no external magnet, reached more than twenty-five decibels of isolation across the C band, with an insertion loss projected below one and a half decibels.'] },
          { label: 'Angle, with refraction', body: '<p>Output waveguides are tilted at the facet so that the reflection does not couple back into the guided mode. By Snell\'s law, $n_\\text{wg}\\sin\\theta_\\text{wg} = n_\\text{out}\\sin\\theta_\\text{out}$, the beam leaves at a larger angle than the waveguide tilt. Rotate the chip or the waveguide so the <i>exiting</i> beam points where your fiber is.</p>', narration: ['Angle, with refraction.', 'Output waveguides are tilted at the facet so that the reflection does not couple back into the guided mode.', 'But by Snell\'s law, the beam leaves at a larger angle than the waveguide tilt.', 'So rotate things so the exiting beam points at your fiber. Not the waveguide.'] },
          { label: 'Retaliation at scale', media: { src: 'media/lit/fb_kashyap1988_fiber_fuse.svg', caption: 'Kashyap and Blow, Electron. Lett. 1988. Illustrative.', href: 'https://doi.org/10.1049/el:19880032' }, body: '<p>At high power, a fiber can carry a self-propelled damage front, the fiber fuse, that travels back along the fiber toward the source (Kashyap and Blow, 1988).</p>', narration: ['Retaliation at scale.', 'At high enough power, a fiber can carry a self-propelled damage front, called a fiber fuse.', 'It travels back along the fiber, toward the source.', 'Retaliation always knows where you live.'] }
        ], narration: ['Let us separate isolation facts from isolation myths.'] },
      { type: 'chat', title: 'Scenario 1: The Contrast Ratio', app: 'thread', skin: 'slack', meta: { header: '#integration-wins', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Asymmetric silicon device: strong forward to backward contrast on the single-mode detector. No magnets, no modulation, plain silicon', side: 'right' },
          { from: GARY, text: 'Before calling it anything, I launched backward into the mode the forward light exits in, and counted every output mode. Backward matches forward. It is reciprocal: a mode converter, not an isolator', side: 'right' },
          { from: BY, text: 'Agreed. Linear, static and non-magnetic means reciprocal', side: 'left' },
          { from: MGR, text: 'Thanks for checking before the press release. Mode converter it is', side: 'left' }
        ], narration: isThisOk('Gary ran the full reciprocity test before making a claim, and reported a mode converter as a mode converter.',
          ['Read this team chat about Gary\'s new silicon device.', 'Then judge whether Gary\'s handling of the result is proper.']) },
      { type: 'chat', title: 'Scenario 2: The Directional Modulator', app: 'call', skin: 'facetime', meta: { header: 'Video call', participants: [GARY, MGR, BY] },
        messages: [
          { from: GARY, text: 'Cost-down idea: our traveling-wave modulator is strongly directional, so it should suppress back-reflections. We could drop the isolator', side: 'right' },
          { from: BY, text: 'It modulates forward light efficiently. A continuous-wave reflection off the facet still passes back through it', side: 'left' },
          { from: MGR, text: 'Directional sounds close to one-way, no?', side: 'left' },
          { from: BY, text: 'Not for unmodulated light', side: 'left' }
        ], narration: isThisOk('Gary proposes removing the isolator because the modulator is directional.',
          ['Watch this video call about a cost-down idea.', 'Then judge whether Gary\'s proposal is proper.']) },
      { type: 'chat', title: 'Scenario 3: The Straight Arrow', app: 'email', skin: 'mail', meta: { header: 'Inbox', subject: 'Fiber attach drawing for review, rev B', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'Hi Dana,\n\nAttached is rev B of the fiber attach drawing. The output waveguide is tilted at the facet to reduce back-reflection, and the fiber is aligned along the waveguide axis. Could you review by Friday?\n\nThanks,\nGary', side: 'right', to: [BY], date: '2026-11-20 09:30', subject: 'Fiber attach drawing for review, rev B', attachments: [{ name: 'fiber_attach_revB.pdf', size: '860 KB' }] },
          { from: BY, text: 'Thanks Gary. One issue: the beam refracts at the facet and leaves at a larger angle than the waveguide tilt. The fiber should follow the refracted beam, not the waveguide.', side: 'left', to: [GARY], date: '2026-11-20 13:12' },
          { from: GARY, text: 'The assembly house aligns actively for maximum power anyway, so that should take care of it.', side: 'right', to: [BY], date: '2026-11-20 13:40' },
          { from: BY, text: 'Active alignment optimizes position. If the angle in the holder is wrong, it optimizes a lower coupling.\n\nDana', side: 'left', to: [GARY], date: '2026-11-20 14:02' }
        ], narration: isThisOk('Gary drew the fiber along the waveguide tilt, and is relying on active alignment to fix the refraction angle.',
          ['Read this email exchange about Gary\'s fiber attach drawing.', 'Then judge whether Gary\'s answer to Dana is proper.']) },
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
          { t: 0, cast: ['Gary'], caption: 'Gary measures a ring resonator with a very high Q. The dip has two bottoms.', shot: 'Dark lab, laser sweep across a ring on a chip under a microscope. Monitor shows a transmission dip with two closely spaced minima. Gary leans in, frowning.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.ringScatter(ctx, w, h, u); } },
          { t: 8, cast: ['Gary'], caption: 'Gary fits one Lorentzian through both. The reported Q drops.', shot: 'Screen insert: a single wide Lorentzian fit drawn over a doublet. The Q value field shows a smaller number. Gary emails "wafer is bad". Red X stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.doubletPlot(ctx, w, h, u, { stamp: 'VIOLATION' }); } },
          { t: 15, cast: ['Gary', 'Morgan'], caption: 'Months later: the laser team needs backscatter for self-injection locking.', shot: 'Conference room, wide. Morgan unveils a slide: "Backscatter is now mandatory." Gary looks at his wafer-scrap request on his laptop.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.silSlide(ctx, w, h, u); } },
          { t: 21, cast: ['Pat Reynolds'], caption: 'Replay: fit the doublet with a coupled-mode model. Know when backscatter helps and when it hurts.', shot: 'Host beside a ring model with standing-wave nodes glowing around the circumference. Green check stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.ringScatter(ctx, w, h, u, { variant: 'standing', stamp: 'COMPLIANT', ok: true }); } }
        ],
        narration: ['Gary measures a ring resonator with a very high quality factor.', 'The dip has two bottoms.', 'Gary fits a single Lorentzian through both. The reported Q drops, and Gary asks to scrap the wafer.', 'Months later, the laser team announces that backscatter is now mandatory.', 'Let us replay that. A tiny scatterer couples clockwise and counter-clockwise light. The two form standing waves, and the resonance splits.', 'Every interface reflects. Even a speck of dust is an interface.'] },
      { type: 'slide', layout: 'host', title: 'Why Resonances Split',
        bullets: ['A ring supports two degenerate traveling waves: clockwise and counter-clockwise', 'Any backscatter couples them; at high Q, a tiny coupling exceeds the linewidth', 'The new eigenmodes are standing waves, split in frequency by twice the coupling rate', 'Backscattered light also leaves the ring backward: toward your laser'],
        callout: MANTRA,
        narration: ['Let us take a closer look.', 'A ring supports two traveling waves with the same frequency, clockwise and counter-clockwise.', 'Any backscatter couples them. At high Q, even a tiny coupling can be larger than the linewidth.', 'The new eigenmodes are standing waves, and their frequencies split by twice the coupling rate.', 'And the backscattered light also leaves the ring backward. Toward your laser.', MANTRA] },
      { type: 'widget', title: 'Interactive: Backscatter Doublet', widget: 'doublet', params: {},
        instructions: 'Increase the backscatter rate $\\beta$ until the resonance splits into a doublet. Compare the single-Lorentzian fit Q with the true Q, switch the plot to dB to see the light reflected back toward the laser, then try the silicon ring.',
        narration: ['In this activity, you will add backscatter to a high Q ring.', 'Move the backscatter slider to the right, until the dip splits.', 'Compare the Q from a single Lorentzian fit with the true Q.', 'Then switch the plot to decibels, and watch the reflected light. That is the light going back toward your laser.'] },
      { type: 'reveal', title: 'Backscatter: Defect and Requirement', prompt: 'Open each tab.',
        tabs: [
          { label: 'The doublet misreports Q', media: { src: 'media/lit/fb_doublet_single_fit.svg', caption: 'Ji et al., Optica 2017. Computed, illustrative.', href: 'https://doi.org/10.1364/optica.4.000619' }, body: '<p>At high Q with tight confinement, very small defects or roughness split the clockwise and counter-clockwise degeneracy into a visible doublet (Ji et al., 2017).</p><p>The coupled-mode model (Kippenberg et al., 2002) shows the consequence: a single-Lorentzian fit to a doublet misreports Q.</p>', narration: ['At high Q, very small defects split the resonance into a doublet.', 'The coupled mode model shows the consequence. A single Lorentzian fit then misreports the Q.', 'Fit a doublet model. Report the intrinsic linewidth and the coupling separately.'] },
          { label: 'Your lithography is a scatterer', media: { src: 'media/lit/fb_hausmann2014_stitching.svg', caption: 'Hausmann et al., Nat. Photonics 2014. Illustrative.', href: 'https://doi.org/10.1038/nphoton.2014.72' }, body: '<p>Hausmann et al. (2014) report that their earlier electron-beam writing left periodic scattering centres around diamond rings, which split the clockwise and counter-clockwise resonances. Writing the pattern continuously, without beam jumps or small segments, potentially explains why their new rings showed no split resonances.</p>', narration: ['Your lithography is a scatterer.', 'In diamond rings, earlier electron beam writing left periodic scatterers around the circumference, and the resonances split.', 'The authors suggest that writing the pattern continuously is why the split went away. Each stitch was an interface, and every interface reflects.'] },
          { label: 'Now mandatory', media: { src: 'media/lit/fb_raja2019_fig2d_inset.png', caption: 'Raja et al., Nat. Commun. 2019, Fig. 2(d), inset.', href: 'https://doi.org/10.1038/s41467-019-08498-2' }, body: '<p>Self-injection locking relies on backscatter: the resonances that self-injection locked all showed backscatter splitting (Raja et al., 2019). The defect is the feature.</p>', narration: ['Policy reversal.', 'Self-injection locking relies on backscatter to send light back into the laser.', 'In one integrated soliton microcomb, the resonances that locked all showed backscatter splitting.', 'The figure shows one of them. Two dips, where one was expected.', 'This is feedback with consent.'] },
          { label: 'Until you turn it up', media: { src: 'media/lit/fb_white2024_backreflection.svg', caption: 'White et al., Nat. Photonics 2024. Computed, illustrative.', href: 'https://doi.org/10.1038/s41566-024-01539-3' }, body: '<p>Backscatter-based self-injection locking works at low power, but as power increases the ring\'s nonlinear response detunes it, and the backscattered power no longer reaches the laser (White et al., 2024).</p>', narration: ['Until you turn it up.', 'Backscatter-based locking works at low power.', 'At higher power, the nonlinear response of the ring detunes it, and the backscattered light no longer reaches the laser.', 'Characterize at the power you will operate at.'] },
          { label: 'The thermal cliff', media: { src: 'media/lit/fb_kramnik2024_fig2c.png', caption: 'Kramnik et al., arXiv 2024, Fig. 2(c).', href: 'https://arxiv.org/abs/2411.05921' }, body: '<p>Thermally locked microrings sit near the edge of instability: past a point, self-heating flips from negative to positive feedback and the resonance snaps to shorter wavelength. The balance point cannot be predicted, so a hot-to-cold calibration sweep is required (Kramnik et al., 2024).</p>', narration: ['The thermal cliff.', 'A thermally locked ring sits near the edge of instability.', 'Past a certain point, self-heating flips from negative to positive feedback, and the resonance snaps away to shorter wavelength.', 'In the figure, the forward and reverse sweeps snap at different wavelengths.', 'The balance point cannot be predicted from the design, so a hot to cold calibration sweep is required.'] },
          { label: 'Not every split is backscatter', media: { src: 'media/lit/fb_chen2026_fig2d.png', caption: 'Chen et al., Light Sci. Appl. 2026, Fig. 2(d).', href: 'https://doi.org/10.1038/s41377-026-02353-y' }, body: '<p>Fano lineshapes from multimode interference can coexist with backscatter doublets in the same resonator, doublets at one wavelength and Fano shapes at others (Chen et al., 2026). Not every distorted dip is backscatter.</p>', narration: ['Not every split is backscatter.', 'In one resonator, backscatter doublets appeared at one wavelength, and Fano lineshapes from multimode interference at others.', 'The figure shows both in one device. A doublet at ten sixty-four nanometers, and a Fano shape at seven eighty.', 'Label your diagnosis as an inference until you have checked it.'] }
        ], narration: ['Backscatter is a defect. Backscatter is a requirement. Both are policy.'] },
      { type: 'chat', title: 'Scenario 1: Scrap the Wafer', app: 'email', skin: 'mail', meta: { header: 'Inbox', subject: 'Ring Q below target, lot 7', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Hi Morgan,\n\nSingle-Lorentzian fits put the loaded Q well below target on most high-Q rings in lot 7. I suggest we scrap the lot and restart with the next one so the schedule holds. Fit summary attached.\n\nGary', side: 'right', to: [MGR], cc: [BY], date: '2026-11-18 08:50', subject: 'Ring Q below target, lot 7', attachments: [{ name: 'Q_fits_lot7.xlsx', size: '212 KB' }] },
          { from: BY, text: 'Before scrapping: are the dips split? If there are two minima, a single Lorentzian is measuring the splitting, not the loss.\n\nDana', side: 'left', to: [GARY, MGR], date: '2026-11-18 09:22' },
          { from: GARY, text: 'Most show two shallow minima. I treated the second one as sweep noise.', side: 'right', to: [BY, MGR], date: '2026-11-18 09:35' },
          { from: MGR, text: 'Let us scrap pending one more opinion. Dana, can you look at a few today?\n\nMorgan', side: 'left', to: [GARY, BY], date: '2026-11-18 10:02' }
        ], narration: isThisOk('Gary fit doublets with a single Lorentzian and treated the second minimum as noise, and a scrap decision is pending.',
          ['Read this email thread about ring resonators in lot seven.', 'Then judge whether Gary\'s recommendation is proper.']) },
      { type: 'chat', title: 'Scenario 2: Conflicting Policies', app: 'thread', skin: 'teams', meta: { header: '#all-hands', participants: [MGR, GARY, BY] },
        messages: [
          { from: MGR, text: 'Reminder: the resonator team target this quarter is minimum backscatter', side: 'left' },
          { from: MGR, text: 'Reminder: the laser team target is enough backscatter to self-injection lock', side: 'left' },
          { from: GARY, text: 'Same ring?', side: 'right' },
          { from: MGR, text: 'Same ring. Both targets are in the plan', side: 'left' },
          { from: BY, text: 'Could we specify the coupling rate relative to the linewidth, and the operating power? Then both teams can sign', side: 'left' }
        ], narration: isThisOk('The same ring has two opposite backscatter targets, and no operating power is specified.',
          ['Read this all-hands chat about this quarter\'s backscatter targets.', 'Then judge whether the plan is proper.']) },
      { type: 'chat', title: 'Scenario 3: The Snap', app: 'phone', skin: 'whatsapp', meta: { header: 'Messages', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'Ran a hot-to-cold calibration sweep on the ring before setting the thermal lock', side: 'right' },
          { from: GARY, text: 'Partway down, the resonance snapped to the blue. Not where the design thermal model put the balance point', side: 'right' },
          { from: GARY, text: 'I set the lock point on the stable side, with margin from the snap, and logged the sweep', side: 'right' },
          { from: BY, text: 'Good. Re-run the sweep if the operating power changes', side: 'left' }
        ], narration: isThisOk('Gary measured the balance point with a hot to cold sweep instead of trusting the design model, and set the lock with margin.',
          ['Read this message exchange about Gary\'s thermally locked ring.', 'Then judge whether Gary\'s handling is proper.']) },
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
          { t: 0, cast: ['Gary'], caption: 'The wavemeter shows a steady number. Gary concludes the laser is not hopping.', shot: 'Lab bench, close on a wavemeter display with a calm, steady reading. Behind it, out of focus, an oscilloscope trace of the laser intensity flickers between two levels. Gary gives a thumbs up to the camera.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.steadyNumber(ctx, w, h, u); } },
          { t: 8, cast: ['Gary'], caption: 'Gary measures the ring with an on-chip laser. The free spectral range looks larger than designed.', shot: 'Probe station, chip with an attached laser and a ring. A thermal camera inset shows the region around the laser and the ring warming as the laser sweeps. Gary writes "new physics?" on a sticky note. Red X stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.onChipHeat(ctx, w, h, u, { stamp: 'VIOLATION' }); } },
          { t: 15, cast: ['Pat Reynolds'], caption: 'Replay: compare against a separate benchtop laser, add an isolator, and look at fast signals.', shot: 'Same bench, now with a benchtop tunable laser through an isolator, a fast photodiode on the oscilloscope, and the thermal camera showing a cool chip. Green check stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.separateBench(ctx, w, h, u, { stamp: 'COMPLIANT', ok: true }); } }
        ],
        narration: ['The wavemeter shows a steady number. Gary concludes the laser is not hopping.', 'Meanwhile, the intensity flickers between two levels, faster than the wavemeter updates.', 'Next, Gary measures a ring with a laser on the same chip, and the free spectral range looks larger than designed.', 'Let us replay that. Use a separate source, add an isolator, and look at fast signals.', 'Isolate, angle, and verify. Verify is the part with the instruments.'] },
      { type: 'slide', layout: 'host', title: 'Every Instrument Integrates',
        bullets: ['A wavemeter reports one number per update; a hop faster than the update is averaged or missed', 'An optical spectrum analyzer reports power within its resolution bandwidth; ripples finer than that disappear', 'A power meter averages over its response time', 'The setup itself has interfaces: fiber tips, connectors, the detector face. They reflect too'],
        callout: MANTRA,
        narration: ['Let us take a closer look.', 'Every instrument integrates over something.', 'A wavemeter reports one number per update. A hop faster than the update can be averaged or missed.', 'An optical spectrum analyzer reports power within its resolution bandwidth. A ripple finer than that disappears.', 'A power meter averages over its response time.', 'And the setup has interfaces of its own. Fiber tips, connectors, the detector face. They reflect too.', MANTRA] },
      { type: 'reveal', title: 'When the Measurement Is the Problem', prompt: 'Open each tab.',
        tabs: [
          { label: 'The setup feeds back', media: { src: 'media/lit/fb_dekoninck2023_figS5a.png', caption: 'De Koninck et al., arXiv 2023, Fig. S5(a).', href: 'https://arxiv.org/abs/2309.04473' }, body: '<p>Without an isolator, parasitic reflections lead to an overestimate of laser linewidth, as stated for 300-mm GaAs nano-ridge lasers (De Koninck et al., 2023).</p>', narration: ['The setup feeds back.', 'Without an isolator in the measurement path, parasitic reflections lead to an overestimate of the laser linewidth.', 'The figure shows that setup. The lensed fiber goes straight into the interferometer, with no isolator.', 'Your measurement fiber is an interface. It reflects.'] },
          { label: 'The source heats the sample', media: { src: 'media/lit/fb_chowdhury2024_fsr.svg', caption: 'Chowdhury et al., Opt. Express 2024. Redrawn.', href: 'https://doi.org/10.1364/oe.510036' }, body: '<p>Sweeping an on-chip, photonic-wire-bonded distributed feedback laser heated the chip and red-shifted the rings under test: the free spectral range read 160 pm, against 120 pm with a benchtop laser and 110 pm simulated (Chowdhury et al., 2024).</p>', narration: ['The source heats the sample.', 'Sweeping a laser on the same chip heated the chip and red-shifted the rings under test.', 'The free spectral range read one hundred sixty picometers. With a separate benchtop laser, it read one hundred twenty. The simulation said one hundred ten.', 'The instrument was cooking the sample.'] },
          { label: 'Resolution hides ripple', body: '<p>Use the parasitic Fabry-Perot calculator from Chapter 2: a long fiber pigtail gives a ripple period far finer than a typical spectrum analyzer resolution. The ripple is still there; the instrument averages it.</p>', narration: ['Resolution hides ripple.', 'A long fiber pigtail makes a ripple period much finer than a typical analyzer resolution.', 'The ripple is still there. The instrument is averaging it for you, without asking.'] },
          { label: 'Simulations average too', body: '<p>An eigenmode solver can return a mode bound to the edge of the simulation domain. Before reporting a higher-order mode, check where its field lives and whether it moves when you move the boundary.</p>', narration: ['Simulations average too, in their own way.', 'An eigenmode solver can return a mode that lives at the edge of the simulation box.', 'Before reporting it, check whether it moves when you move the boundary.'] },
          { label: 'Label your inferences', body: '<p>When you attribute a ripple to a specific pair of reflectors, or a split to backscatter, write down whether you measured it or inferred it. This course labels its own inferences. So should you.</p>', narration: ['Label your inferences.', 'When you blame a ripple on two specific reflectors, write down whether you measured it, or inferred it.', 'This course labels its own inferences. So should you.'] }
        ], narration: ['Sometimes the measurement is the problem.'] },
      { type: 'chat', title: 'Scenario 1: The Calm Wavemeter', app: 'call', skin: 'meeting', meta: { header: 'Video call', participants: [GARY, MGR, BY] },
        messages: [
          { from: GARY, text: 'The wavemeter has been steady to the last digit all morning, but I am not calling it hop-free yet', side: 'right' },
          { from: GARY, text: 'It updates slowly, and the fast photodiode on the other bench shows the intensity switching between two levels. I will borrow the photodiode and check before I report', side: 'right' },
          { from: BY, text: 'It is free after lunch', side: 'left' },
          { from: MGR, text: 'OK. Report after the fast check', side: 'left' }
        ], narration: isThisOk('Gary did not take a steady reading from a slow instrument as proof, and checked with a fast detector before reporting.',
          ['Watch this video call about this morning\'s laser test.', 'Then judge whether Gary\'s conclusion is proper.']) },
      { type: 'chat', title: 'Scenario 2: New Physics', app: 'thread', skin: 'slack', meta: { header: '#ring-characterization', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Measured free spectral range with the on-chip laser is noticeably larger than the design. Could be a dispersion effect we have not modeled', side: 'right' },
          { from: BY, text: 'Did you compare with the benchtop laser? The on-chip laser heats the chip while it sweeps, and the rings shift with it', side: 'left' },
          { from: GARY, text: 'The on-chip laser is the configuration the product uses, so I think it is the more relevant number', side: 'right' },
          { from: MGR, text: 'Let us mention it as an unexplained dispersion effect in the abstract and check later', side: 'left' }
        ], narration: isThisOk('Gary is reporting a heating artifact from his own source as an unmodeled effect.',
          ['Read this team chat about a ring measured with a laser on the same chip.', 'Then judge whether Gary\'s interpretation is proper.']) },
      { type: 'chat', title: 'Scenario 3: The Record Linewidth', app: 'email', skin: 'mail', meta: { header: 'Inbox', subject: 'Linewidth result, DFB batch 12', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'Hi Dana,\n\nLinewidth result attached. I took the isolator out of the interferometer path to get more signal, since the beat note was close to the noise floor.\n\nGary', side: 'right', to: [BY], date: '2026-11-18 17:20', subject: 'Linewidth result, DFB batch 12', attachments: [{ name: 'beatnote_2026-11-18.csv', size: '1.2 MB' }] },
          { from: BY, text: 'Without an isolator, reflections from the setup feed back into the laser, so the number is the laser plus the setup.', side: 'left', to: [GARY], date: '2026-11-18 17:45' },
          { from: GARY, text: 'The setup reflections should be small compared with the chip. I will mention "no isolator" in the methods.', side: 'right', to: [BY], date: '2026-11-18 18:02' }
        ], narration: isThisOk('Gary removed the isolator from the linewidth setup to gain signal, and plans to footnote it.',
          ['Read this email exchange about a laser linewidth result.', 'Then judge whether Gary\'s setup choice is proper.']) },
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
        { t: 0, cast: ['Gary', 'Morgan'], caption: 'Friday afternoon. The customer demo is on Monday, and the link is short on power.', shot: 'Lab bench, wide, 35 mm lens, eye level. A fiber-pigtailed laser in a butterfly package on a temperature-controlled mount, a small cylindrical inline fiber isolator with a printed arrow, a fiber spool, a benchtop power meter. Gary seen from behind in a lab coat, gloved hands on the bench. Morgan in the doorway, out of focus, holding a laptop. Wall calendar with Monday circled in red.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.linkBench(ctx, w, h, u, { variant: 'budget' }); } },
        { t: 7, cast: ['Gary'], caption: 'Gary unplugs the isolator to win back its insertion loss. The power meter goes up.', shot: 'Macro, 100 mm lens. Gloved hands disconnect two fiber connectors, lift out the inline isolator and drop it into a small parts bag. The laser pigtail is mated directly to the link patch cord. Rack focus to the power meter as its reading rises. Slow push-in.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.linkBench(ctx, w, h, u, { variant: 'remove' }); } },
        { t: 14, cast: ['Gary'], caption: 'Monday, at the customer. An unused splitter port on the evaluation board ends in an open, flat-polished connector. The laser hears it.', shot: 'Customer lab, medium shot. Evaluation board on an anti-static mat; one connector with its dust cap off, flat ferrule end catching a highlight. Oscilloscope beside it: laser intensity jumping between two levels. Electrical spectrum analyzer: a broad self-heterodyne beat note. Customer engineers only as out-of-focus silhouettes. Red stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.linkBench(ctx, w, h, u, { variant: 'customer', stamp: 'VIOLATION' }); } },
        { t: 22, cast: ['Pat Reynolds'], caption: 'Replay: the isolator stays in. A link budget is fixed by fixing the link, not by removing the protection.', shot: 'Still (Blender render): butterfly laser bench with the inline isolator back in line, arrow pointing away from the laser. Planned live shot: the customer board with a dust cap and an angled connector on the open port; power meter lower but steady; beat note narrow.', image: { src: 'media/render/butterfly_isolator_fc_bench.jpg' }, draw: function (ctx, w, h, u) { LMS_DRAW.fb.linkBench(ctx, w, h, u, { variant: 'inline', stamp: 'COMPLIANT', ok: true }); } }
      ],
      narration: ['Friday afternoon. The customer demo is on Monday, and the link is short on power.', 'Gary looks at the link budget. The isolator costs its insertion loss.', 'Gary unplugs the isolator. The power meter goes up. Gary feels good about this.', 'Monday, at the customer. An unused splitter port on the evaluation board ends in an open, flat-polished connector.', 'A flat end of fused silica reflects about three point three percent of the light that reaches it.', 'Assume an even one by two splitter. The light passes it twice, so about zero point eight percent of the laser output comes home. That is about minus twenty-one decibels.', 'With no isolator, nothing stops it. The intensity flickers, and the line broadens.', 'Let us replay that. The isolator stays in. A link budget is fixed by fixing the link.', MANTRA] },
    { type: 'chat', title: 'Incident 1: How It Unfolded', app: 'thread', skin: 'slack', meta: { header: '#demo-prep', participants: [GARY, MGR, BY] },
      messages: [
          { from: MGR, text: 'Link budget for Monday is short. Any ideas?', side: 'left' },
          { from: GARY, text: 'The inline isolator is the biggest single loss in the path. We could leave it out for the demo and add it back for production', side: 'right' },
          { from: BY, text: 'Without it, reflections from the customer board go straight back into the laser. Their unused splitter port is an open connector', side: 'left' },
          { from: MGR, text: 'It is a DFB, and it is one afternoon. Can we live with it?', side: 'left' },
          { from: GARY, text: 'Isolator out. Power meter now reads above the requirement', side: 'right' },
          { from: BY, text: 'The meter averages. Can we look at the spectrum and the linewidth before Monday?', side: 'left' },
          { from: MGR, text: 'Let us do that after the demo. Dana, let us keep this thread to solutions for now, I will loop you back in next week', side: 'left' }
        ],
      narration: isThisOk('Gary removed the isolator to close the link budget for one demo, the spectrum check was postponed, and Morgan moved Dana off the thread.',
        ['Read this team chat from the Friday before the demo, one message at a time.', 'Then judge whether the decisions Gary and Morgan make are proper.']).concat(['Please note. Moving a coworker off a thread because they raised a concern is retaliation, even when it is polite.', 'Zero tolerance applies to people as well as to photons.']) },
    felt('At the customer, the laser line was broad, the intensity flickered, and the customer asked whether this was the product',
      ['The isolator was in Gary\'s bag the whole time'],
      ['The isolator was in Gary\'s bag the whole time. It was isolating the bag.', 'Gary will not be punished for reporting this incident. He will be enrolled in this course again.', 'In Chapter Three, you will learn the official name for what happened on Friday. Isolation, traded for insertion loss. It is called Quid Pro Quo Feedback.']),
    tester('You are Gary. It is Monday morning at the customer. The self-heterodyne beat note of your laser is far broader than the datasheet linewidth, and the intensity flickers. The isolator is in your bag. What would you do?', [
      OK('Stop, put the isolator back in line, re-measure the beat note, and tell the customer and Morgan that the earlier setup had no isolator. Then log it'),
      W('Open a ticket with the laser vendor: the beat note is far broader than their datasheet linewidth', 'The laser vendor measured the laser under their own conditions, with their own isolation. A laser\'s linewidth under feedback depends on the reflector outside the laser, and that reflector is on the customer board, which the vendor did not ship. Tkach and Chraplyvy classified how external feedback narrows or broadens a distributed feedback laser line, and in the regime called coherence collapse, described by Lenstra and coworkers in nineteen eighty-five, the line broadens dramatically. The open connector returns about zero point eight percent of the laser output through the splitter, about minus twenty-one decibels. If your isolator performs like the hybrid integration isolator reported by Lapointe and coworkers, with more than twenty-five decibels of isolation, removing it lets more than three hundred times more of that reflected light back in. The vendor did not remove your isolator. You did.'),
      W('Re-check the self-heterodyne setup first: the delay fiber was moved during shipping', 'The delayed self-heterodyne setup did exactly what it was built to do. It beat the laser against a delayed, frequency-shifted copy of itself, and showed you the laser\'s phase noise as the width of a beat note. If the beat note is broad, the laser\'s phase noise is high. The measurement path has interfaces of its own, which is why it should also be isolated, as De Koninck and coworkers note. But the measurement setup is not what changed since Friday. The isolator is what changed.'),
      W('Give the laser more time to warm up, then re-measure', 'Warming up changes the laser temperature. That moves the phase of the light returning from the open connector, so the symptoms may come and go as the laser warms. That is not the laser settling. That is the feedback phase sweeping through its cycle, which is exactly the periodic behavior described in this chapter. In coherence collapse, the broadening is largely insensitive to the phase anyway. Waiting does not put the isolator back.'),
      W('Raise the laser current a little: further above threshold, the line should be narrower', 'A higher current does narrow the intrinsic linewidth of a clean laser. But feedback effects are set by the fraction of light that returns, not by the absolute power. More current sends more light to the open connector, and more light comes back. The ratio is the same. You also move the laser to a different operating point that nobody characterized, in front of a customer.'),
      W('Average the beat note for longer, to get a cleaner width estimate', 'Averaging reduces the noise on the displayed trace. It does not reduce the width of the line, because the width is the physics, not the noise on the display. A longer average gives you a smoother picture of the same broad line. The instrument did what it was told. It averaged.')
    ], 'Restore the isolator, re-measure, disclose, and log it.'),
    bystander('You are Dana. On Friday you saw Gary unplug the isolator, and Morgan moved you off the thread until after the demo. It is now Friday evening. What do you do?', [
      OK('Speak up now, to Gary and Morgan, using the approved phrase, and log the setup change before the demo'),
      WB('Say nothing. Morgan approved it', 'Morgan approved a schedule. Nobody can approve a reflection. The open connector on the customer board reflects the same three point three percent regardless of who signed off, and after the splitter, about zero point eight percent of the laser output still comes home. And being moved off the thread does not change your responsibility. Excluding someone because they raised a quality concern is itself a reportable event under our non-retaliation policy.'),
      WB('Mention it at lunch on Tuesday, after the demo', 'Light reflected from a few meters of fiber returns to the laser within tens of nanoseconds. Lunch on Tuesday is several days later. The speak-up has to arrive before the demo, not after the light. Speak up while it can still change the outcome.'),
      WB('Quietly put the isolator back on Sunday without telling anyone', 'The intention is good. But now Gary\'s measurements from Friday describe a setup that no longer exists, and nobody knows it. Untracked changes to an optical path are how ripples become mysteries. Put it back, yes, but say so, and log it.'),
      WB('Note "possible vibration" in the lab log, since the bench was bumped on Friday', 'The bench may well have been bumped. Vibration can modulate the phase of light returning from a reflector, and it can make an existing feedback problem flicker. But vibration does not create the reflector, and it does not remove the isolator. Writing a false root cause into the log means the next engineer will chase vibration while the isolator sits in a bag.'),
      WB('Submit an anonymous hotline report about the isolator removal', 'The anonymous hotline is always available, and we thank everyone who uses it. But a hotline report is reviewed on its own schedule, and the demo is on Monday. When the people who can fix it are on the same thread, speak to them first, and use the hotline for what they will not fix.')
    ], 'Speak up early, use the approved phrase, and log it.')
  ];

  /* ---------- Incident 2 (Chapter 2): The Ripple in the Report ---------- */
  var INC2 = [
    { type: 'video', title: 'Incident Review 2: The Ripple in the Report', duration: 30,
      keyframes: [
        { t: 0, cast: ['Gary'], caption: 'Gary edge-couples a flat-cleaved fiber to a cleaved silicon chip facet and sweeps a tunable laser across the band.', shot: 'Edge-coupling stage, top-down through a stereo microscope, then cut to the monitor feed: a bare flat-cleaved fiber tip approaching a chip facet, the air gap visible as a dark band. Gloved hand on the piezo stage controller. Tunable laser and photodetector in the rack behind, out of focus.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.edgeCouple(ctx, w, h, u, { mode: 'approach' }); } },
        { t: 8, cast: ['Gary'], caption: 'The transmission has a ripple with a 24 nm period. Gary reports it as an intrinsic device ripple, and fits it out.', shot: 'Over-the-shoulder to a monitor showing a report document; a plot captioned "intrinsic device ripple" is highlighted in yellow. The send button is clicked. Red stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.gapTrace(ctx, w, h, u, { report: true, stamp: 'VIOLATION' }); } },
        { t: 16, cast: ['Dana'], caption: 'Dana backs the fiber off by 5 micrometers. The period changes. The chip did not move.', shot: 'Split screen. Left: Dana\'s gloved hand turning the stage micrometer a fraction of a turn. Right: the live trace, ripple period visibly shrinking. A ruler graphic marks the period before and after.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.edgeCouple(ctx, w, h, u, { mode: 'backoff', trace: true }); } },
        { t: 23, cast: ['Pat Reynolds'], caption: 'Replay: period to length. 24 nm at 1550 nm in air means a 50 micrometer gap: the fiber tip and the facet.', shot: 'Host beside a large graphic of the formula lambda squared over two L, with the fiber tip and the chip facet highlighted as the two mirrors of the cavity. A drop of index-matching gel fills the gap. Green check stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.edgeCouple(ctx, w, h, u, { mode: 'gel', stamp: 'COMPLIANT', ok: true }); } }
      ],
      narration: ['Gary edge-couples a flat-cleaved fiber to a silicon chip, and sweeps a tunable laser across the band.', 'The transmission has a ripple. Its period is twenty-four nanometers.', 'Gary reports it as an intrinsic device ripple. He fits it, and subtracts it.', 'Dana backs the fiber off by five micrometers. The period changes. The chip did not move.', 'Let us replay that. In air, the ripple period is lambda squared over two times the gap.', 'At fifteen fifty nanometers, a period of twenty-four nanometers means a gap of fifty micrometers. That is the gap between the fiber tip and the facet.', 'Back off to fifty-five micrometers, and the period becomes about twenty-two nanometers. The ripple belongs to the gap.', 'Every interface reflects. Every reflection is a cavity.'] },
    { type: 'chat', title: 'Incident 2: How It Unfolded', app: 'email', skin: 'mail', meta: { header: 'Inbox', subject: 'Final report: waveguide transmission, lot 3', participants: [GARY, BY, MGR] },
      messages: [
          { from: GARY, text: 'Hi Morgan,\n\nAttached is the final transmission report for the customer. Every die shows a ripple with a 24 nm period and about the same depth, so I have characterized it as an intrinsic waveguide property and removed it with a sinusoid fit for the loss numbers. Raw data attached.\n\nBest,\nGary', side: 'right', to: [MGR], cc: [BY], date: '2026-11-17 16:05', subject: 'Final report: waveguide transmission, lot 3', attachments: [{ name: 'transmission_report_lot3_v4.pdf', size: '2.1 MB' }, { name: 'sweeps_lot3_raw.zip', size: '38 MB' }] },
          { from: BY, text: 'Gary, 24 nm at 1550 nm in air corresponds to $\\lambda^2/(2L)$ with $L \\approx 50\\ \\mu\\text{m}$. What gap did you use for the edge coupling? If it was about 50 um on every die, the ripple would look the same on every die.\n\nDana', side: 'left', to: [GARY], cc: [MGR], date: '2026-11-17 16:38' },
          { from: GARY, text: 'About 50 um on all of them, that is the standard alignment recipe. But the waveguide simulation shows no ripple, and the depth is consistent die to die, so I am fairly confident it is the device.', side: 'right', to: [BY], cc: [MGR], date: '2026-11-17 16:52' },
          { from: BY, text: 'The simulation has absorbing boundaries at the facets, so it cannot show a facet cavity. A flat cleave is about 3.3 percent and a silicon facet about 31 percent. Could we re-measure one die with gel before it goes out?', side: 'left', to: [GARY], cc: [MGR], date: '2026-11-17 17:10' },
          { from: MGR, text: 'The customer asked for data this week. Let us send it as is, describe the ripple as intrinsic, and follow up if they ask.\n\nMorgan', side: 'left', to: [GARY, BY], date: '2026-11-17 17:31' }
        ], narration: isThisOk('Gary reported a ripple that matches his fiber gap as an intrinsic device property, and the report went out without a re-measurement.',
          ['Read this email thread about the final transmission report for the customer.', 'Then judge whether Gary\'s conclusion, and the decision to send it, are proper.']) },
    { type: 'slide', layout: 'bullets', title: 'Incident 2: The Numbers',
      bullets: ['Air-gap etalon period: $\\Delta\\lambda = \\lambda^2/(2L)$. At 1550 nm, $L = 50\\ \\mu\\text{m}$ gives 24 nm; $L = 55\\ \\mu\\text{m}$ gives 21.8 nm', 'Flat silica end, $n = 1.444$ (Malitson, 1965): $R = \\left(\\frac{n-1}{n+1}\\right)^2$ = 3.3 percent, $-14.8$ dB', 'Cleaved silicon facet, $n = 3.48$ (Li, 1980): $R$ = 31 percent (plane-wave Fresnel estimate)', 'Plane-wave peak-to-valley bound: $20\\log_{10}\\left[(1+r)/(1-r)\\right]$ with $r = \\sqrt{R_1R_2} = 0.10$, about 1.8 dB. Diffraction in the gap returns less light, so the real ripple is smaller', 'Diagnostic: change the gap. If the period follows the gap, the ripple belongs to the gap'],
      callout: 'If the period moves when the fiber moves, the ripple is not the device.',
      narration: ['Let us check the numbers together.', 'In air, the ripple period is lambda squared over two times the gap. Fifty micrometers gives twenty-four nanometers. Fifty-five micrometers gives about twenty-two.', 'A flat end of fused silica, with an index of about one point four four four, reflects about three point three percent, or minus fifteen decibels.', 'A cleaved silicon facet, with an index of about three point four eight, reflects about thirty-one percent, as a plane-wave estimate.', 'Together they can make a ripple of up to about one point eight decibels peak to valley. Diffraction in the gap makes the real ripple smaller.', 'The diagnostic is simple. Change the gap. If the period follows the gap, the ripple belongs to the gap.', 'Every interface reflects. Every reflection is a cavity.'] },
    feltTabs('The customer received a report describing the gap between Gary\'s fiber and the chip as a property of the chip', [
      { label: 'Ashamed', body: '<p>Gary felt ashamed. That is a normal response. Shame is a normal response to a reflection, including your own.</p><p>This is a judgment-free zone. The incident has still been logged.</p>', narration: ['Gary felt ashamed.', 'That is a normal response. Shame is a normal response to a reflection, including your own.', 'This is a judgment-free zone. The incident has still been logged.'] },
      { label: 'Reflective', body: '<p>Gary felt reflective. About 3.3 percent reflective, at the fiber end, and about 31 percent at the facet.</p><p>Reflection is encouraged in people. In optical paths, please angle it.</p>', narration: ['Gary felt reflective.', 'About three point three percent reflective at the fiber end, and about thirty-one percent at the facet.', 'Reflection is encouraged in people. In optical paths, please angle it.'] },
      { label: 'Isolated', body: '<p>Gary felt isolated. Unlike his laser, which was not.</p><p>If you feel isolated after an incident, the Employee Assistance Program is available. If your laser is not isolated, please refer to Chapter 5.</p>', narration: ['Gary felt isolated.', 'Unlike his laser, which was not.', 'If you feel isolated after an incident, the Employee Assistance Program is available. If your laser is not isolated, please refer to Chapter Five.'] },
      { label: 'Logged', body: '<p>The incident has been logged. Being logged is not a punishment. It is a record. Records are how the next engineer finds the second mirror faster.</p>', narration: ['The incident has been logged.', 'Being logged is not a punishment. It is a record.', 'Records are how the next engineer finds the second mirror faster.', 'In Chapter Three, you will classify this incident. A flat fiber end, outside the chip, on the path Gary connected. That is External Feedback.', MANTRA] }
    ]),
    tester('You are Gary. The report with the intrinsic ripple went to the customer yesterday. Dana has just shown that the ripple period changes when the fiber gap changes. What would you do?', [
      OK('Log it, tell the report owner and the customer, re-measure with index-matching gel or an angled fiber, and issue a corrected report'),
      W('Re-align for maximum transmission and re-measure, so the report uses the best-coupled data', 'Re-aligning changes the gap, which moves the ripple, which is the proof that the ripple belongs to the gap. Maximizing transmission at one wavelength picks a gap that puts a ripple peak there. It does not remove the cavity. It hides it at one wavelength and keeps it at every other wavelength. A measurement that depends on which run you kept is not a measurement of the device.'),
      W('Ask the simulation team to explain why their model missed the ripple', 'The simulation did exactly what it was told. A perfectly matched layer absorbs all outgoing light by construction, and a matched port reflects nothing by construction. Your model had no fiber, no air gap, and no facet reflection, so it could not show a ripple from them. If you want a simulation to predict a parasitic cavity, you must include both mirrors. The reflection in the lab came from the flat fiber end and the cleaved facet, fifty micrometers apart.'),
      W('Re-normalize to the tunable laser reference sweep, in case the ripple is laser power variation', 'A tunable laser\'s output power can vary with wavelength, which is why transmission is normalized to a reference measurement of the laser alone. But the tunable laser does not know where your fiber is. The period changed when only the gap changed, from twenty-four nanometers at fifty micrometers to about twenty-two at fifty-five. That is lambda squared over two L, in air.'),
      W('Re-cleave the fiber more carefully and re-measure', 'A better cleave is still a flat cleave. The cleaver did exactly what a standard cleaver is designed to do: a flat end, perpendicular to the fiber axis. A flat end of fused silica reflects about three point three percent straight back. If you need that reflection steered away from the guided mode, use an angle cleave, an angle-polished fiber, or index-matching gel. The cleaver followed the procedure. The procedure needed an angle.'),
      W('Add a note to the report that no gel was used, and leave the data as is', 'A note describes the setup. It does not correct the conclusion: the report still calls the gap ripple intrinsic. Applied properly, a gel with an index close to that of glass would have largely removed the glass-to-air reflection at the fiber end, and with it most of this cavity. That is a good idea for the re-measurement.'),
      W('Report the loss from a moving average over one ripple period, which is common practice', 'A moving average one period wide removes the ripple from the plot. It also removes any real feature narrower than twenty-four nanometers, and it leaves the cavity in the setup. Smoothing is a choice about the picture. The cavity is a fact about the bench.')
    ], 'Log it, correct the report, and remove the second mirror.'),
    bystander('You are Dana. You found the cause. Gary seems embarrassed and has asked you not to mention it to anyone. What do you do?', [
      OK('Use the approved phrase with Gary, offer to help re-measure, and make sure the correction is logged and sent to the customer'),
      WB('Say nothing. It is his report', 'It is his report, and it is now the customer\'s data. A wrong root cause in a customer report will be designed around by people who never met Gary. Respecting a colleague means helping him correct it, not helping him keep it.'),
      WB('Mention it at lunch, in a general way, so he gets the hint', 'A hint at lunch is not a correction. The report stays wrong, and nothing is logged. Speak to the owner directly, with the approved phrase.'),
      WB('Email the customer the correction yourself, since you found it', 'The physics is correct, and we appreciate your initiative. But the customer relationship has an owner. Raise it with the report owner first, so the correction goes out once, from the owner, with the re-measured data.'),
      WB('Let Gary handle it in his own time; he knows now', 'Knowing is not correcting. Until a correction is logged and sent, the customer has the wrong report, and they may not notice. They may design a compensation filter for a ripple that exists only on Gary\'s bench. Waiting moves the reflection downstream, to a place where it costs more.')
    ], 'Speak up kindly, help fix it, log it.')
  ];

  /* ---------- Incident 3 (Chapter 4): The Demo Restart ---------- */
  var INC3 = [
    { type: 'video', title: 'Incident Review 3: The Demo Restart', duration: 34,
      keyframes: [
        { t: 0, cast: ['Gary'], caption: 'The live customer demo. The laser is tuned to the customer\'s channel. The isolator is still in Gary\'s bag from Friday.', shot: 'Customer conference room, wide. Demo rack with the laser, a wavemeter and a spectrum display on a large monitor. Gary from behind, presenting with a clicker. Customer engineers as out-of-focus silhouettes. A parts bag with a small cylindrical isolator visible in an open laptop bag on the floor.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.wmTimeline(ctx, w, h, u, { phase: 'steady' }); } },
        { t: 8, cast: ['Gary'], caption: 'Midway through, the wavemeter jumps. The customer\'s receiver loses lock.', shot: 'Close on the wavemeter display: the reading jumps to a new value. Cut to the customer receiver status panel turning from green to red. Gary freezes mid-sentence.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.wmTimeline(ctx, w, h, u, { phase: 'hop' }); } },
        { t: 15, cast: ['Gary'], caption: 'Gary power-cycles the laser. It comes back on the right channel. He tells the room it re-locked, and carries on.', shot: 'Close on a gloved hand flipping the laser driver output off and on. The wavemeter returns to the original value. Gary turns back to the room and continues the presentation. Red stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.wmTimeline(ctx, w, h, u, { phase: 'restart', stamp: 'VIOLATION' }); } },
        { t: 22, cast: ['Gary', 'Morgan'], caption: 'Afterwards, Morgan: "Good save. Let us not log a one-off. And let us keep Dana off the follow-up call."', shot: 'Corridor outside the conference room, medium two-shot from behind both figures. Morgan pats Gary on the shoulder. Through the glass, Dana is packing up alone.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.doc(ctx, w, h, u, { app: 'Calendar', title: 'Follow-up call: customer demo', rows: [['Organizer', 'Morgan'], ['Attendees', 'Gary, Morgan'], ['Dana Okafor', 'removed from the invitation', 'bad'], ['Message', 'Good save. Let us not log a one-off.', 'type']], stamp: 'VIOLATION' }); } },
        { t: 28, cast: ['Pat Reynolds'], caption: 'Replay: log the hop, the temperature, the current and the sweep direction. Put the isolator back. Retaliation is prohibited, by lasers and by managers.', shot: 'Host in the empty conference room beside the demo rack, isolator back in line, a lab log open on the table with a new entry. Green check stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.doc(ctx, w, h, u, { app: 'Lab log', title: 'Entry: mode hop during the customer demo', rows: [['Laser temperature', '29.60 \u00b0C'], ['Laser current', '120.0 mA'], ['Sweep direction', 'up, warming'], ['Hop', '1550.392 nm to 1550.542 nm, one cavity mode', 'bad'], ['Isolator', 'back in line, arrow away from the laser', 'ok'], ['Follow-up call', 'Dana invited', 'ok']], stamp: 'COMPLIANT', ok: true }); } }
      ],
      narration: ['The live customer demo. The laser is tuned to the customer\'s channel.', 'The isolator is still in Gary\'s bag from Friday.', 'Midway through, the wavemeter jumps. The laser has hopped. The customer\'s receiver loses lock.', 'Gary power-cycles the laser. It comes back on the right channel.', 'Gary tells the room it re-locked, and carries on.', 'Afterwards, Morgan says, good save. Let us not log a one-off. And let us keep Dana off the follow-up call.', 'Let us replay that.', 'Restarting a laser picks a mode again from a fresh start. It does not remove the reflector, and it does not remove the hop point. The next temperature drift brings the laser back to it.', 'Log the hop, the temperature, the current, and the sweep direction. Put the isolator back.', 'And retaliation is prohibited. By lasers, and by managers.', 'Every cavity hops.'] },
    { type: 'chat', title: 'Incident 3: How It Unfolded', app: 'phone', skin: 'signal', meta: { header: 'Messages', participants: [MGR, GARY] },
      messages: [
          { from: MGR, text: 'How did it go after I left?', side: 'left' },
          { from: GARY, text: 'Fine. The wavelength jumped once mid-demo, I restarted the laser and it came back on channel', side: 'right' },
          { from: MGR, text: 'Good save. Let us not put a one-off in the customer log until we know what it was', side: 'left' },
          { from: GARY, text: 'Makes sense. I will log it if it happens again. The isolator is still out from Friday, by the way', side: 'right' },
          { from: MGR, text: 'Leave it for now. And let us keep Dana off the follow-up call, we need a confident tone', side: 'left' },
          { from: GARY, text: 'ok', side: 'right' }
        ], narration: isThisOk('The mode hop is not in the log, the isolator is still out, and Morgan has taken Dana off the follow-up call.',
          ['Read these messages between Morgan and Gary, sent after the customer demo.', 'Then judge whether what they agree on is proper.']).concat(['Removing a coworker from a call because they raised concerns is retaliation, even when it is phrased as tone.', 'Retaliation against people is prohibited. So is keeping a hop out of the log.']) },
    felt('The laser hopped in front of the customer, Gary restarted it without logging it, and Dana was taken off the follow-up call',
      ['Gary also felt afraid to report. That is normal too', 'Retaliation against Gary for reporting is prohibited'],
      ['Gary also felt afraid to report. That is normal too.', 'Fear of retaliation is the most common reason people do not speak up. That is why retaliation is prohibited.', 'The laser is allowed to hop. It is a laser. Morgan is not allowed to retaliate. Morgan is a manager.', 'For the record, this incident is classified as Quid Pro Quo Feedback, followed by Retaliation by the laser. Taking Dana off the call is handled under a separate policy, with the same zero tolerance.']),
    tester('You are Gary. The hop is not in the log, Morgan suggested not logging a one-off, and the customer wants a second demo next week. What would you do?', [
      OK('Log the hop with temperature, current and sweep direction, put the isolator back, map the hop points in both sweep directions, and raise the logging decision and Dana\'s removal through the speak-up channel'),
      W('Restart the laser before every demo, so it always starts in a good mode', 'A restart picks a mode again, starting from wherever the laser comes up. Because hop points show hysteresis, the restart decides which side of the hysteresis loop you start on. It does not remove the loop. The reflector on the customer board is still there, the isolator is still in your bag, and the next temperature drift will carry the laser back to the hop point, possibly in the middle of the next demo.'),
      W('Change the temperature setpoint slightly, so the laser sits further from where it hopped', 'Moving the setpoint moves you away from one hop point, but nobody has mapped the others. The temperature controller holds the temperature at its sensor, and that is all it can do. It does not regulate the length of fiber between the laser and the open connector on the customer board, whose optical length changes with the room temperature and moves the feedback phase. Without the isolator, that external cavity adds its own modes to the competition, and one of them won.'),
      W('Use the previous shift\'s operating point, since it never hopped for them', 'An operating point is good or bad only relative to a hop map, and nobody has measured the hop map of this laser with this setup. The previous shift set a current and a temperature. You removed the isolator and connected a reflector. A hop map, measured in both directions, is how you choose an operating point with margin.'),
      W('Tell the customer the laser re-locked after a brief interruption, which is accurate', 'It is accurate, and it is incomplete. The wavemeter showed the optical frequency jump. A jump in optical frequency is a mode hop, and the cause, an unisolated reflection, is still in place. Re-locked withholds the one fact the customer needs to keep their receiver locked next time.'),
      W('Add vibration damping to the demo rack, since the hop happened while people were moving around', 'Vibration can shake the fibers and modulate the phase of the light coming back from a reflector. With feedback present, that can trigger hops sooner. But vibration needs a reflector to work with, and an isolator would have kept that light out of the laser. Vibration is an accomplice. The open connector, with no isolator, is the root cause.'),
      W('Warm the laser up for longer before the next demo, since the hop came early', 'Warming up moves the temperature of the laser, which moves the cavity modes and the filter at different rates. That is exactly how a laser walks into a hop point. A laser that hops while warming up will hop again whenever the customer\'s system warms up. Warming up is the trigger, not the excuse.')
    ], 'Log it, fix it, and report the retaliation.'),
    bystander('You are Dana. You were in the room and saw the wavemeter jump. Afterwards, Morgan took you off the follow-up call, for tone. What do you do?', [
      OK('Speak up with the approved phrase, log what you saw, and report your removal from the call through the non-retaliation channel'),
      WB('Accept it. Managers decide who attends customer calls', 'Managers do decide who attends. Removing the one person who raised a quality concern, right after they raised it, is still retaliation, because it teaches the whole team to stop raising concerns. And the mode hop is the customer\'s business. You saw both. Witnesses are part of the speak-up process.'),
      WB('Mention the hop to Gary informally, and let him decide whether to log it', 'Gary has already decided not to log it, on his manager\'s suggestion. An informal mention leaves both the hop and your removal unrecorded. Report first. Tell the story after the incident is closed, with names removed.'),
      WB('Email the customer about the hop yourself, since you are off the call anyway', 'The customer does need to know. But going around your own team to the customer is not a speak-up channel, and it turns a quality issue into a relationship issue. Speak to the owner and log it.'),
      WB('Raise it in the annual engagement survey, which is anonymous', 'The annual survey closes in eleven months. The second demo is next week. Speak up on the time scale of the problem, not the time scale of the survey.')
    ], 'Speak up, log it, and report retaliation.')
  ];

  /* ---------- Incident 4 (Chapter 5): Installed Backwards ---------- */
  var INC4 = [
    { type: 'video', title: 'Incident Review 4: Installed Backwards', duration: 34,
      keyframes: [
        { t: 0, cast: ['Gary'], caption: 'Gary installs a new inline fiber isolator. The arrow on the housing points at the laser.', shot: 'Still (Blender render): butterfly laser on a finned mount, yellow pigtail through an inline isolator spliced in backwards (splice sleeves on both sides), its printed arrow pointing toward the laser; capped FC/APC and FC/PC connectors in the foreground. Planned live shot: gloved hands mating the isolator, slow push-in on the arrow.', image: { src: 'media/render/butterfly_isolator_reversed_bench.jpg' }, draw: function (ctx, w, h, u) { LMS_DRAW.fb.linkBench(ctx, w, h, u, { variant: 'reversed' }); } },
        { t: 8, cast: ['Gary'], caption: 'The output power drops by about the isolation ratio. Gary files a return request against the laser.', shot: 'Over-the-shoulder to a laptop: a vendor return form, reason field "laser output dead". Beside it, the power meter reading near the noise floor. Red stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.doc(ctx, w, h, u, { app: 'Vendor returns', title: 'Return request', rows: [['Product', 'DFB laser, butterfly package'], ['Reason', 'Laser output dead', 'type'], ['Measured output', '-34.6 dBm', 'bad'], ['Datasheet output', '+2.1 dBm']], readouts: [{ label: 'POWER METER', value: '-34.6', unit: 'dBm', col: '#ff9d96' }], button: 'Submit', stamp: 'VIOLATION' }); } },
        { t: 15, cast: ['Gary'], caption: 'Downstream, an optical amplifier emits spontaneous emission backward. The reversed isolator passes it straight into the laser.', shot: 'Animated cutaway of the link: laser, reversed isolator, fiber, an amplifier module. A faint red glow leaves the amplifier input traveling backward, passes the isolator in its low-loss direction and lands on the laser facet.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.linkBench(ctx, w, h, u, { variant: 'reversed' }); } },
        { t: 22, cast: ['Gary'], caption: 'On the free-space bench, Gary rotates a polarization-dependent isolator by 90 degrees to fit the mount. Nothing comes out.', shot: 'Optical table, medium close. A free-space isolator in a rotation mount, engraved input polarizer axis now horizontal while the laser polarization is vertical. Gary\'s gloved hand tightens the setscrew. The detector card behind it is dark.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.fsIso(ctx, w, h, u); } },
        { t: 28, cast: ['Pat Reynolds'], caption: 'Replay: the arrow points away from the laser, and the input polarizer is aligned to the laser polarization. Then measure forward power before you close the setup.', shot: 'Still (Blender render): the same bench with the isolator arrow pointing away from the laser. Planned live shot: host beside both setups, a printed checklist with two ticked lines, direction and rotation.', image: { src: 'media/render/butterfly_isolator_fc_bench.jpg' }, draw: function (ctx, w, h, u) { LMS_DRAW.fb.fsIso(ctx, w, h, u, { fixed: 0, stamp: 'COMPLIANT', ok: true }); } }
      ],
      narration: ['Gary installs a new fiber isolator. The arrow on the housing points at the laser.', 'The output power drops by about the isolation ratio. Gary files a return request against the laser.', 'Here is the subtle part. For the laser\'s own light reflected downstream, a reversed isolator attenuates the round trip just as much as a correct one. Forward loss times backward loss is the same product either way.', 'What the reversed isolator does is block almost all of the laser\'s output, and pass any light that starts downstream, such as amplified spontaneous emission traveling backward out of an amplifier, straight into the laser.', 'Meanwhile, on the free-space bench, Gary rotates a polarization-dependent isolator by ninety degrees to fit the mount.', 'Its input polarizer is now crossed with the laser polarization. By the law of Malus, the transmission goes as cosine squared of the angle. At ninety degrees, nothing comes out.', 'Let us replay that. The arrow points away from the laser. The input polarizer is aligned with the laser polarization.', 'Reciprocity is not optional. Isolate, angle, and verify.'] },
    { type: 'chat', title: 'Incident 4: How It Unfolded', app: 'thread', skin: 'slack', meta: { header: '#lab-help', participants: [GARY, BY, MGR] },
      messages: [
          { from: GARY, text: 'Output power dropped to almost nothing after the isolator swap. Driver current is nominal, so it looks like the laser. I opened a vendor RMA', side: 'right' },
          { from: BY, text: 'Which way does the arrow on the new isolator point?', side: 'left' },
          { from: GARY, text: 'Toward the laser, but I assumed that does not matter for loss. Reciprocity says transmission is the same both ways, right?', side: 'right' },
          { from: BY, text: 'Reciprocity is exactly what an isolator breaks. That is its job', side: 'left' },
          { from: GARY, text: 'My link model shows no difference between orientations though. I modeled it as a two-port with the datasheet loss', side: 'right' },
          { from: BY, text: 'A symmetric two-port is reciprocal by construction. The model cannot show it', side: 'left' },
          { from: MGR, text: 'Can we hold off on a vendor ticket? Renewal negotiations are next week', side: 'left' }
        ], narration: isThisOk('Gary installed the isolator backwards, trusted a symmetric model, and opened a vendor return, and Morgan is worried about the renewal.',
          ['Read this lab help chat about a laser that went dark after an isolator swap.', 'Then judge whether Gary\'s diagnosis, and the response to it, are proper.']) },
    { type: 'slide', layout: 'policy', title: 'Policy 4.3: Isolator Orientation',
      bullets: ['A Faraday isolator is non-reciprocal: forward and backward transmission differ by design (Jalas et al., 2013)', 'Polarization-dependent isolator: input polarizer, 45 degree Faraday rotator, output polarizer at 45 degrees. Backward light is rotated a further 45 degrees in the same sense, reaches 90 degrees, and is blocked (Saleh and Teich, 1991)', 'Reversed (180 degrees): the laser output is attenuated by about the isolation ratio; light that starts downstream passes into the laser with only the insertion loss', 'For the laser\'s own reflection, the round trip is forward transmission times backward transmission: the same product in either orientation', 'Rotated about its axis: a polarization-dependent isolator transmits as $\\cos^2\\theta$, with $\\theta$ the angle between laser polarization and input polarizer (Malus): 0 at 90 degrees, 0.5 at 45 degrees. A polarization-independent fiber isolator does not care about rotation. It cares about direction'],
      callout: 'Direction is not angle. Check both.',
      narration: ['Policy section four point three. Isolator orientation.', 'A Faraday isolator is non-reciprocal by design.', 'In a polarization-dependent isolator, light passes an input polarizer, is rotated by forty-five degrees in a Faraday rotator, and leaves through an output polarizer set at forty-five degrees.', 'Light coming backward is rotated another forty-five degrees in the same sense. It arrives at ninety degrees to the input polarizer, and is blocked. That is non-reciprocity.', 'If you install the isolator backwards, the laser output is attenuated by about the isolation ratio, and light that starts downstream gets into the laser with only the insertion loss.', 'For the laser\'s own reflection, the round trip is the forward transmission times the backward transmission. That product is the same in either orientation. So a reversed isolator does not protect you more, or less, from your own echo. It just takes your power, and lets strangers in.', 'If you rotate a polarization-dependent isolator about its axis, the transmission follows the law of Malus. Cosine squared. Zero at ninety degrees, one half at forty-five.', 'A polarization-independent fiber isolator does not care about rotation. It cares about direction.', 'Direction is not angle. Check both.'] },
    felt('The laser vendor returned the laser marked "no fault found", with a photo of the isolator arrow',
      ['The vendor\'s photo was circled in red'],
      ['The vendor\'s photo was circled in red. Red circles are a normal part of the return process.', 'Gary will be asked to re-read Policy section four point three. Twice. Once in each direction.', 'For the record, the light from the amplifier is External Feedback, admitted by an isolator installed backwards. The lost power is classified as embarrassing.']),
    tester('You are Gary. The vendor has returned the laser marked "no fault found". You have just noticed that the arrow on the isolator points at the laser. What would you do?', [
      OK('Turn the isolator around, re-measure output power, log the orientation error, tell the laser owner, and close the vendor return with the true cause'),
      W('Turn the isolator around and close the vendor return as resolved, without going into detail', 'The vendor found no fault because there was no fault in the laser. Closing the return as resolved, with no cause, leaves an open question in two companies\' records. The next engineer who sees low power will also send a good laser back. The true cause is one line in the log: isolator installed backwards.'),
      W('Note that the arrow on this housing is easy to misread, and ask purchasing for a different model', 'The arrow on an inline isolator marks the direction of forward propagation, from the input port to the output port. It was printed for exactly this moment. The procedure is to check the arrow against the drawing, and then to measure forward power through the isolator before closing the setup. Both steps would have caught it.'),
      W('Check the patch cords first; the drop could be a bad connector from the shared drawer', 'Patch cords from the shared drawer deserve suspicion. Mating an angle-polished connector to a flat one, for example, leaves a gap and a tilted interface, and adds loss. Swap the cord and check. But a backwards isolator attenuates by roughly its isolation ratio, which is far more than a mismatched connector is designed to explain, and the isolator is the part you changed.'),
      W('Trust the link model, which shows no difference between the two orientations', 'The simulation did exactly what it was told. You modeled the isolator as a symmetric two-port, and a symmetric scattering matrix describes a reciprocal device. A reciprocal device has no preferred direction, so the model could not tell the orientations apart. The entire function of an isolator is an asymmetric scattering matrix, as Jalas and coworkers put it. Model the device you installed.'),
      W('Rotate the isolator by 90 degrees, in case it is polarization sensitive', 'Direction is not angle. A polarization-independent fiber isolator transmits the same for any rotation about its axis, so rotating it does nothing. A polarization-dependent isolator rotated by ninety degrees crosses its input polarizer with the laser polarization, and by the law of Malus, cosine squared of ninety degrees is zero. Turning it around end to end is the fix.'),
      W('Turn the laser current up to make up for the lost power', 'The light is not lost in the laser. It is being blocked by the isolator, which is doing its job in the wrong direction. Raising the current pushes the laser to an operating point nobody characterized, and almost all of the extra light is blocked as well.')
    ], 'Turn it around, re-measure, and log the true cause.'),
    bystander('You are Dana. Gary has posted a photo of his setup in the lab channel. The isolator arrow points at the laser. The vendor return has already shipped. What do you do?', [
      OK('Speak up now with the approved phrase, point to the arrow, promptly and privately, so the return and the log can be corrected'),
      WB('React to the photo, and wait for Gary to notice', 'A reaction is not a correction. It tells Gary that something is funny, and it does not tell him what. It also tells the rest of the channel. Speak to the owner directly, and point to the arrow.'),
      WB('Say nothing. The vendor will find no fault and he will work it out', 'He may work it out after a second return, or a third. Each return costs time, shipping, and the vendor\'s goodwill. You can save all of it with one sentence today.'),
      WB('Mention it at lunch', 'The return has already shipped. Lunch is later. Speak up on the time scale of the problem.'),
      WB('Turn the isolator around yourself after hours', 'The physics is right. The process is not. Gary\'s records and the vendor return now describe a setup that no longer exists, and nobody knows it changed. Untracked changes to an optical path are how future incidents begin.')
    ], 'One sentence, today, to the owner.')
  ];

  /* ---------- Incident 5 (Chapter 6): Two Modes and a Record ---------- */
  var INC5 = [
    { type: 'video', title: 'Incident Review 5: Two Modes and a Record', duration: 30,
      keyframes: [
        { t: 0, cast: ['Gary'], caption: 'Gary sweeps a high-Q ring. Every resonance has two dips.', shot: 'Dark lab, probe station, top-down through the microscope onto a ring resonator chip with lensed fibers on both sides. Monitor in the foreground: a transmission sweep with a doublet at each resonance. Gary from behind, leaning in.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.ringScatter(ctx, w, h, u, { b: 90 }); } },
        { t: 7, cast: ['Gary'], caption: 'Gary reports the pair as two transverse mode families: "a dual-mode ring".', shot: 'Slide on a monitor titled "Dual-mode ring", two arrows labelling the two dips "mode 1" and "mode 2". Red stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.doubletPlot(ctx, w, h, u, { variant: 'twomodes', stamp: 'VIOLATION' }); } },
        { t: 14, cast: ['Gary'], caption: 'For the intrinsic Q, Gary fits one shallow dip with the over-coupled branch, the fitting tool default for this gap. It is a record.', shot: 'Screen insert: a fitting tool dialog with a toggle "under-coupled / over-coupled". The cursor clicks over-coupled; the intrinsic Q field triples. Gary types "record" into the slide title. Red stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.fitTool(ctx, w, h, u, { stamp: 'VIOLATION' }); } },
        { t: 21, cast: ['Dana', 'Pat Reynolds'], caption: 'Replay: light comes back out of the input port exactly at the doublet. Two independent mode families would not do that, and would walk apart across the spectrum.', shot: 'Same bench; Dana adds a circulator at the input and a second detector on the reflection port. The monitor shows a reflection peak aligned with each doublet. Then a wide sweep: the splitting stays pinned to every resonance. Green check stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.ringReflect(ctx, w, h, u, { stamp: 'COMPLIANT', ok: true }); } }
      ],
      narration: ['Gary sweeps a high Q ring. Every resonance has two dips.', 'Gary reports the pair as two mode families, and calls it a dual-mode ring.', 'For the intrinsic Q, Gary fits one of the shallow dips with the over-coupled formula. It is the fitting tool default for his design gap, and the result is a record.', 'Let us replay that.', 'Dana looks at the light coming back out of the input port. There is a reflection peak, exactly at each doublet.', 'Two independent mode families would not send light back. And because they have different group indices, they would have different free spectral ranges, and walk apart across the spectrum.', 'A backscatter doublet stays pinned to every resonance, and it reflects. Every interface reflects. Even a speck is an interface.'] },
    { type: 'chat', title: 'Incident 5: How It Unfolded', app: 'email', skin: 'mail', meta: { header: 'Inbox', subject: 'Abstract draft: record intrinsic Q, lot 7', participants: [GARY, BY, MGR] },
      messages: [
          { from: GARY, text: 'Hi all,\n\nDraft abstract attached. Every resonance shows two mode families, consistent with the wide waveguide. Fitting one dip with the over-coupled branch, which is the fitting tool default for our design gap, gives an intrinsic Q above our previous best.\n\nGary', side: 'right', to: [MGR], cc: [BY], date: '2026-11-21 15:40', subject: 'Abstract draft: record intrinsic Q, lot 7', attachments: [{ name: 'abstract_draft_v2.docx', size: '41 KB' }, { name: 'Q_fits_lot7_v2.xlsx', size: '230 KB' }] },
          { from: BY, text: 'Is the over-coupled branch confirmed independently? Transmission alone cannot tell the branches apart. Also, I see light coming back out of the input port at every doublet, which looks like backscatter splitting rather than two families.\n\nDana', side: 'left', to: [GARY, MGR], date: '2026-11-21 16:25' },
          { from: GARY, text: 'The branch matches our design gap and the trend from the last lot, so I think it is reasonable.', side: 'right', to: [BY, MGR], date: '2026-11-21 16:48' },
          { from: MGR, text: 'Deadline is tonight. Let us submit with the record and refine in the full paper.\n\nMorgan', side: 'left', to: [GARY, BY], date: '2026-11-21 17:10' }
        ], narration: isThisOk('Gary chose the coupling branch that matches expectation, called a backscatter doublet two mode families, and the abstract is going out tonight.',
          ['Read this email thread about an abstract due tonight.', 'Then judge whether Gary\'s analysis, and the decision to submit, are proper.']) },
    { type: 'slide', layout: 'bullets', title: 'Incident 5: The Numbers',
      bullets: ['Backscatter splitting of high-Q modes: Weiss et al., 1995; modal coupling model: Kippenberg, Spillane and Vahala, 2002', 'Single-mode extraction: $Q_i = 2Q_L/(1+\\sqrt{T_0})$ if under-coupled, $Q_i = 2Q_L/(1-\\sqrt{T_0})$ if over-coupled. The all-pass transmission is symmetric in loss and coupling (Yariv, 2000), so $|T|$ alone cannot choose the branch', 'Coupled-mode model of this chapter, critically coupled before splitting, doublet well resolved: each dip bottoms at $T_0 = 0.25$, and about 25 percent of the light comes back out of the input port at each dip', 'With $T_0 = 0.25$ and each dip about one loaded linewidth wide: under-coupled branch gives $Q_i = 1.33\\,Q_L$, over-coupled branch gives $Q_i = 4\\,Q_L$. The true value in this model is $2\\,Q_L$', 'Two transverse mode families have different group indices, hence different $\\mathrm{FSR} = c/(2\\pi R\\,n_g)$: their spacing changes from resonance to resonance'],
      callout: 'A default is not a coupling regime.',
      narration: ['Let us check the numbers together.', 'Backscatter splitting of high Q modes was reported by Weiss and coworkers in nineteen ninety-five, and modeled as coupling between counter-propagating waves by Kippenberg, Spillane, and Vahala in two thousand two.', 'For a single resonance, you can get the intrinsic Q from the loaded Q and the dip depth. But there are two formulas, one for under-coupling and one for over-coupling.', 'The transmission of a ring on a bus is symmetric in the loss and the coupling, as Yariv wrote down in two thousand. So the transmission magnitude alone cannot tell you which formula applies.', 'In the coupled mode model of this chapter, take a ring that was critically coupled before splitting. Once the doublet is well resolved, each dip bottoms out at one quarter transmission, and about one quarter of the light comes back out of the input port.', 'Plug that into the two formulas. The under-coupled branch gives one point three three times the loaded Q. The over-coupled branch gives four times. The true value in this model is two times.', 'Gary chose four. It was the default.', 'And two real mode families have different group indices, so their free spectral ranges differ, and their spacing changes from resonance to resonance. A doublet does not walk. It reflects.'] },
    feltTabs('Gary\'s record Q and his dual-mode ring were in the draft abstract, and both were one ring with a speck on it', [
      { label: 'Ashamed', body: '<p>Gary felt ashamed. That is a normal response. Shame is a normal response to a reflection, including your own.</p><p>This is a judgment-free zone. The incident has still been logged.</p>', narration: ['Gary felt ashamed.', 'That is a normal response. Shame is a normal response to a reflection, including your own.', 'This is a judgment-free zone. The incident has still been logged.'] },
      { label: 'Split', body: '<p>Gary felt split: one part wanted to report the record, one part knew the branch was chosen by default, not by evidence. Like his resonance, the two parts are coupled.</p><p>Fit both. Report both. Choose with evidence.</p>', narration: ['Gary felt split.', 'One part of Gary wanted to report the record. The other part knew the branch was chosen by default, not by evidence.', 'Like his resonance, the two parts are coupled. Fit both. Report both. Choose with evidence.'] },
      { label: 'Over-coupled', body: '<p>Gary felt over-coupled to the result. Strong coupling to a result broadens judgment the way strong coupling to a bus broadens a resonance.</p>', narration: ['Gary felt over-coupled to the result.', 'Strong coupling to a result broadens judgment, the way strong coupling to a bus broadens a resonance.'] },
      { label: 'Logged', body: '<p>The incident has been logged. Being logged is not a punishment. It is a record. Records are how the next engineer fits a doublet the first time.</p>', narration: ['The incident has been logged.', 'Being logged is not a punishment. It is a record.', 'Records are how the next engineer fits a doublet the first time.', 'For the record, a speck on your own ring is Intracavity, or Domestic, Feedback. It lives in the ring.', MANTRA] }
    ]),
    tester('You are Gary. The record Q is in the draft abstract, due tonight. The reflection port shows light at every doublet. What would you do?', [
      OK('Fit a coupled-mode doublet model, settle the coupling regime with independent evidence, report the intrinsic linewidth and the backscatter rate separately, and correct the abstract'),
      W('Keep the over-coupled branch: it is the tool default for this gap, and it matches the last lot', 'The fitting tool did exactly what it was told. It offered both branches because, as Yariv showed in two thousand, the transmission of a ring on a bus is symmetric in the loss and the coupling, so the transmission magnitude cannot choose between them. The choice has to come from independent evidence, for example a series of rings with different bus gaps, or a phase-sensitive measurement. In the model of this chapter, the over-coupled branch reports twice the true intrinsic Q.'),
      W('Re-sweep more slowly, in case the two dips are a sweep artifact', 'A tunable laser sweep can distort a trace if it hops or sweeps non-linearly, which is why sweeps are calibrated against a wavelength reference. But a sweep artifact does not send light back out of the input port exactly at the resonance, and it does not repeat the same splitting at every resonance. The reflection port says backscatter. Repeat the sweep in both directions, and with a reference, and see if the doublet stays.'),
      W('Cite the mode solver, which shows a second mode family for this waveguide width', 'The mode solver did exactly what it was told. It found the transverse modes that the waveguide cross-section supports. Whether the second dip belongs to that family is a separate question, and the answer is in the free spectral range. A second family has a different group index, so its resonances drift relative to the first family across the spectrum. Your doublets stay pinned, with the same small splitting, and they reflect. That is backscatter.'),
      W('Submit as is, and add the doublet analysis to the full paper', 'Reviewers can only judge what you show them. If the abstract says two modes and does not show the reflection port, the reviewers will agree with you, and the error moves into the literature. The data to decide is already on your bench.'),
      W('Average more sweeps to reduce noise before fitting', 'Averaging a stable doublet does not merge it. If the dips do merge in the average, it is because the resonance drifted between sweeps, and the averaged dip is then broader than either real dip, so the Q you extract is lower and still wrong. The instrument averaged. The physics did not.')
    ], 'Fit the doublet, report both numbers, choose the regime with evidence.'),
    bystander('You are Dana, a co-author. The abstract deadline is tonight. Gary says the record Q is "basically right". What do you do?', [
      OK('Use the approved phrase with Gary and the corresponding author today, and offer to run the doublet fit before the deadline'),
      WB('Let it go; abstracts are preliminary and get corrected in the full paper', 'Abstracts get cited. A record that is a factor of two too high is good for nobody, including the next group that tries to reproduce it. You are a co-author. Your name will be on the number.'),
      WB('Ask to be removed from the author list, quietly', 'Removing yourself protects you. It does not correct the abstract. The approved process is to raise the concern with the authors, and help fix it.'),
      WB('Send Gary a paper on doublet fitting, without comment', 'The paper is relevant, and we admire your restraint. But a link without comment cannot answer questions, cannot run the fit, and will not be read before tonight. Speak up in person.'),
      WB('Mention it at lunch next week', 'The deadline is tonight. Lunch next week is after the abstract is submitted. Speak up on the time scale of the problem.')
    ], 'Speak up before the deadline, and help fix it.')
  ];

  /* ---------- Incident 6 (Chapter 7): Passed by Averaging ---------- */
  var INC6 = [
    { type: 'video', title: 'Incident Review 6: Passed by Averaging', duration: 34,
      keyframes: [
        { t: 0, cast: ['Gary'], caption: 'Outgoing inspection. Gary screens lasers with a swept optical spectrum analyzer and a power meter with a long averaging time.', shot: 'Inspection station, wide, cool fluorescent light. A rack with a spectrum analyzer, a power meter and a wavemeter; a tray of butterfly lasers in anti-static foam; inspection sheets on a clipboard. Gary from behind in a lab coat, gloved hand on the analyzer start-sweep key.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.inspect(ctx, w, h, u, { variant: 'station' }); } },
        { t: 8, cast: ['Gary'], caption: 'One laser hops between two adjacent modes. Most sweeps show only one of them. The power meter shows a steady average.', shot: 'Split screen. Left: fast photodiode trace on an oscilloscope switching between two levels. Right: the analyzer trace showing a single peak and the power meter showing a calm number. The oscilloscope is not connected in the inspection procedure; it is shown only to the viewer.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.inspect(ctx, w, h, u, { variant: 'split' }); } },
        { t: 15, cast: ['Gary'], caption: 'Some runs show a second peak. Gary treats them as analyzer artifacts, attaches the representative run, and the laser passes.', shot: 'Close on the inspection sheet: "Run 9 of 9: PASS" written in, runs 1 to 8 not recorded. Runs 1 to 8 go into a folder labeled analyzer artifacts. Red stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.inspect(ctx, w, h, u, { variant: 'runs', stamp: 'VIOLATION' }); } },
        { t: 22, cast: ['Gary', 'Morgan'], caption: 'Two weeks later, the customer\'s coherent receiver finds the hops in the first hour.', shot: 'Conference room, a customer complaint email on the wall screen with a fast intensity trace attached. Morgan and Gary seen from behind.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.inspect(ctx, w, h, u, { variant: 'customer', stamp: 'VIOLATION' }); } },
        { t: 28, cast: ['Pat Reynolds'], caption: 'Replay: a fast photodiode, a resolution bandwidth narrower than the mode spacing, sweeps in both directions, and every run kept.', shot: 'Same station, now with a fast photodiode and oscilloscope wired into the procedure, the analyzer set to a narrow resolution, and nine printouts stapled to the sheet. Green check stamp.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.inspect(ctx, w, h, u, { variant: 'replay', stamp: 'COMPLIANT', ok: true }); } }
      ],
      narration: ['Outgoing inspection. Gary screens lasers with a swept optical spectrum analyzer, and a power meter with a long averaging time.', 'One laser hops back and forth between two adjacent modes.', 'The analyzer can resolve the two modes. But it sweeps. It looks at each wavelength at a different moment, so a mode only appears if the laser is in it when the sweep passes that wavelength.', 'The power meter averages the two intensity levels into one calm number.', 'So most sweeps show one peak, some show two, and some show one peak that is lower than it should be.', 'Gary treats the second peaks as analyzer artifacts, and attaches the representative run. The laser passes.', 'Two weeks later, the customer finds the hops in the first hour.', 'Let us replay that. A fast photodiode. A resolution bandwidth narrower than the mode spacing. Sweeps in both directions. Every run kept.', 'Isolate, angle, and verify. Verify is the part with the instruments.'] },
    { type: 'chat', title: 'Incident 6: How It Unfolded', app: 'thread', skin: 'teams', meta: { header: '#outgoing-inspection', participants: [GARY, BY, MGR] },
      messages: [
          { from: GARY, text: 'Last laser in the tray passed on run 9', side: 'right' },
          { from: BY, text: 'What did runs 1 to 8 show?', side: 'left' },
          { from: GARY, text: 'A few had a second peak. They looked like analyzer artifacts, so I attached the representative run', side: 'right' },
          { from: BY, text: 'Runs 1 to 8 are data too. If the laser hops, some sweeps catch it and some do not', side: 'left' },
          { from: MGR, text: 'Let us keep the summary to representative runs so the review is not confusing. Yield is tight this quarter', side: 'left' }
        ], narration: isThisOk('Gary attached only the run that looked representative, and the laser passed.',
          ['Read this outgoing inspection chat.', 'Then judge whether the way the laser passed is proper.']) },
    felt('The customer found the hops in the first hour, and attached a fast photodiode trace to the complaint',
      ['The customer\'s trace was very clear'],
      ['The customer\'s trace was very clear. Clarity is a gift, even when it arrives as a complaint.', 'Gary\'s inspection procedure will now include a fast photodiode. Gary will now include all runs.', 'For the record, measurement time traded for averaging is Quid Pro Quo Feedback. And the hops the customer found are Retaliation, delivered late.']),
    tester('You are Gary. The laser passed on run 9 and is packed for shipping. Several of runs 1 to 8 show a second peak. What would you do?', [
      OK('Hold the shipment, attach every run, re-test with a fast photodiode and a resolution bandwidth narrower than the mode spacing, log it, and tell the laser owner'),
      W('Treat the second peak as an analyzer artifact, since it does not appear on every sweep', 'The analyzer did exactly what it was set to do. At each point of its sweep, it reports the power within its resolution bandwidth at that moment. If the laser hops during the sweep, some sweeps catch both modes and some catch one. The second peak on some runs is the analyzer working. A clean run is the run where the hop happened while it was looking elsewhere.'),
      W('Trust the power meter, which was steady on every run', 'The power meter did exactly what it was set to do. It averaged the power over its averaging time. A laser switching between two intensity levels faster than that gives a steady average. A steady average proves the average is steady. Use a fast photodiode to see what the laser is doing.'),
      W('Exclude runs 1 to 8 as instrument warm-up', 'An instrument warming up drifts slowly in calibration. It does not create a second laser line one mode spacing away and then remove it on the next sweep. And if anything was warming up, it was the laser, which moves its modes and its filter at different rates, and that is exactly how a laser walks into a hop point.'),
      W('Widen the resolution bandwidth so the two peaks merge, then run it once', 'With a resolution bandwidth wider than the mode spacing, two modes appear as one peak. That is not a laser with one mode. That is an instrument told not to tell you. Set the resolution narrower than the mode spacing, so the analyzer can show what is there.'),
      W('Recalibrate the analyzer and re-run, in case the second peak is a calibration issue', 'A calibration error shifts or scales a reading. It does not make a second peak appear on some sweeps and not on others. The previous shift calibrated the instrument. The laser hopped.'),
      W('Keep run 9 as the representative run; it shows the laser in its normal state', 'Run 9 shows the laser during one sweep. A clean run proves the laser was single-mode during that sweep. The other runs prove it was not single-mode during theirs. Keeping only the clean run is selecting the answer, not measuring it. The customer\'s receiver will see every run.')
    ], 'Hold it, keep every run, and measure fast.'),
    bystander('You are Dana, the second signature on the inspection sheet. Only run 9 is attached. What do you do?', [
      OK('Do not sign yet. Use the approved phrase, ask for all runs, and log the request'),
      WB('Sign. Gary is the one who ran it', 'The second signature exists so that someone other than the tester looks at the data. If you sign without looking, the sheet has two signatures and one check. Your signature is a measurement too.'),
      WB('Sign, and mention it at lunch', 'Once you sign, the laser ships. Lunch does not recall shipments. Ask before you sign.'),
      WB('Sign, and add a note asking for the other runs', 'A note on a signed sheet is read after the laser ships. A question that does not hold the shipment is below the resolution bandwidth of everyone who reads the sheet. Ask before you sign.'),
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
          { t: 0, cast: ['Gary'], caption: 'Gary\'s bench. Every reflector on it belongs to a category. Gary does not know any of them.', shot: 'Optical bench, slow lateral dolly from left to right at bench height, 35 mm lens. Laser in a butterfly package, a chip on a stage, patch cords from an open drawer, a spectrum analyzer, a box with another institution\'s shipping label. Gary from behind, hands on hips.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.benchTop(ctx, w, h, u, { stage: 'pan' }); } },
          { t: 8, cast: [], caption: 'Category labels appear one by one: Intracavity. External. Instrument-Originated. Third-Party. Intra-Group. Cross-Institutional.', shot: 'Same dolly, repeated. As the camera passes each item, a printed policy tag on a lanyard drops into frame beside it: on the chip, on the patch cord, on the analyzer input, on an isolator box, on the drawer, on the shipping box. Tags sway gently.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.benchTop(ctx, w, h, u, { stage: 'tags' }); } },
          { t: 16, cast: ['Gary', 'Morgan'], caption: 'Morgan offers a trade: take out the isolator, get the insertion loss back. Quid Pro Quo.', shot: 'Two-shot from behind. Morgan holds out a small cylindrical isolator in one hand and a printed link budget in the other, as if weighing them. A tag reading Quid Pro Quo drops between them.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.benchTop(ctx, w, h, u, { stage: 'trade', stamp: 'VIOLATION' }); } },
          { t: 23, cast: ['Pat Reynolds'], caption: 'When nothing is angled and everything reflects: a Hostile Optical Environment. What the laser does next: Retaliation.', shot: 'Crane up to a top-down view of the whole bench. Every flat surface flashes in turn; arcs connect every pair. The laser at the left edge glows red and its spectrum on the monitor broadens. Host steps into frame. Green check stamp once the arcs fade.', draw: function (ctx, w, h, u) { LMS_DRAW.fb.benchTop(ctx, w, h, u, { stage: 'arcs', stamp: 'COMPLIANT', ok: true }); } }
        ],
        narration: ['This is Gary\'s bench.', 'Every reflector on it belongs to a category. Gary does not know any of them.', 'Intracavity. External. Instrument-Originated. Third-Party. Intra-Group. Cross-Institutional.', 'Morgan offers a trade. Take out the isolator, and get the insertion loss back. That is Quid Pro Quo.', 'When nothing is angled and everything reflects, that is a Hostile Optical Environment.', 'And what the laser does next is called Retaliation.', 'By the end of this chapter, you will be able to name all nine. You will hear them more than once. That is intentional.'] },
      { type: 'slide', layout: 'host', title: 'Why We Classify',
        bullets: ['Every reflection is a cavity, but not every cavity has the same owner', 'Classification decides who fixes it and who must be told', 'Two questions: where is the reflector, and who put it there?', 'An incident may belong to more than one category. For training purposes, select the primary category', 'Rule: classify the source, unless the question asks about the laser\'s response. The response is always Retaliation, classified separately', 'Nine categories. All mandatory'],
        callout: MANTRA,
        narration: ['Why do we classify unsolicited feedback?', 'Every reflection is a cavity. But not every cavity has the same owner.', 'Classification decides who fixes it, and who must be told.', 'To classify an incident, ask two questions. Where is the reflector? And who put it there?', 'An incident may belong to more than one category. For training purposes, please select the primary category.', 'Classify the source, unless the question asks about the laser\'s response. The response is always Retaliation, and it is classified separately.', 'There are nine categories. All nine are mandatory.', MANTRA] },

      /* 1 Intracavity */
      catSlide(1, 'domestic',
        ['A far-end silicon nitride grating coupler on the same chip returned about 4 percent (-14 dB) into flip-chip lasers (Marinins et al., 2022); facet reflections made an amplifier lase (Bao et al., 2024)', 'A far-end grating coupler on the same chip returned about four percent of the light into flip-chip bonded lasers, as Marinins and coworkers found. And facet reflections made an erbium-doped amplifier lase, as Bao and coworkers reported.'],
        [['The symptom survives every change outside the package', 'The symptom survives every change you make outside the package.'], ['The ripple period implies a length that fits on your chip, $L = c/(2n\\,\\mathrm{FSR})$','The ripple period implies a length that fits on your own chip.'], ['Every unit of the same design shows it', 'Every unit of the same design shows it. It is not personal. It is in the layout.']],
        'the chip or package designer'),
      { type: 'chat', title: 'Category 1 in the Workplace', app: 'phone', skin: 'signal', meta: { header: 'Messages', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'Swapped all the patch cords on the bench. L-I ripple is still there', side: 'right' },
          { from: BY, text: 'Did the period change at all?', side: 'left' },
          { from: GARY, text: 'No, identical on all six units. I will try the cords from the other bench tomorrow', side: 'right' },
          { from: BY, text: 'If it survives every swap and shows on every unit, it is probably on the chip. Measure the distances on the layout', side: 'left' }
        ], narration: isThisOk('Gary is about to swap a second set of patch cords to fix a ripple that has the same period on every unit.',
          ['Read this message exchange between Gary and Dana.', 'Then judge whether Gary\'s next step is proper.']) },
      classify('A laser flip-chip bonded on a silicon photonics chip shows a light versus current ripple. It does not change when any fiber or instrument is swapped, and it appears on every unit of the design. The ripple period matches the distance from the laser to an on-chip grating coupler.', 'domestic', [
        ['external', 'Swapping every fiber and instrument did not change the ripple, so the reflector is not on the path you connected.'],
        ['intragroup', 'Nothing shared by the group was involved. Every unit of the design shows it, including units nobody in the group has touched.'],
        ['hostile', 'One reflector, at a length that fits on the chip, explains the ripple. A Hostile Optical Environment has reflectors everywhere and many ripple periods at once.']
      ]),

      /* 2 External */
      catSlide(2, 'external',
        ['A flat end of fused silica, $n = 1.444$ (Malitson, 1965), reflects about 3.3 percent (-14.8 dB) back along the fiber', 'A flat end of fused silica, with an index of about one point four four four, reflects about three point three percent of the light back along the fiber. A lensed fiber, a lens, a detector, and the device under test all have surfaces.'],
        [['The symptom changes when you move, bend or disconnect a fiber', 'The symptom changes when you move, bend, or disconnect a fiber.'], ['The implied length matches a fiber or free-space distance on your bench', 'The implied cavity length matches a fiber or free-space distance on your bench.'], ['It improves with index-matching gel, an angled connector, or an isolator', 'It improves with index-matching gel, an angled connector, or an isolator in the right direction.']],
        'the owner of the setup'),
      { type: 'chat', title: 'Category 2 in the Workplace', app: 'thread', skin: 'slack', meta: { header: '#test-bench-2', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'Laser output jumps a little whenever I move the lensed fiber in front of the device under test. Probably mechanical', side: 'right' },
          { from: BY, text: 'Could be the device facet. It is a mirror too, so you are tuning a cavity with the stage', side: 'left' },
          { from: GARY, text: 'Possible, but it settles once aligned, so I will just measure at the best spot', side: 'right' },
          { from: BY, text: 'The best spot is a point on the cavity fringe', side: 'left' }
        ], narration: isThisOk('Gary plans to measure at the alignment position where the cavity happens to look best.',
          ['Read this team chat from test bench two.', 'Then judge whether Gary\'s plan is proper.']) },
      classify('The output of a fiber-coupled laser jumps whenever Gary moves a lensed fiber in front of the facet of the device under test. The laser has no isolator.', 'external', [
        ['domestic', 'The reflector here is the facet of the device under test and the lensed fiber, both outside the laser package. Intracavity Feedback lives inside your own chip or package.'],
        ['instrument', 'The lensed fiber and the device under test are not measurement instruments. They are the optical path you connected.'],
        ['retaliation', 'The power jumps are the laser\'s response, and yes, that response is Retaliation. But this question asks for the source. The rule: classify the source, unless the question asks about the laser\'s response. Retaliation is classified separately, and both are reported.']
      ]),
      tester('You are the tester. A ripple on your laser does not change when you swap fibers, connectors or instruments, and it appears on every unit. What would you do?', [
        OK('Classify it as Intracavity Feedback, compute the implied length from the period, find the matching on-chip reflector, and report it to the chip design owner'),
        W('Try the patch cords from the other bench as well, to be thorough', 'You have already replaced the patch cords, and the period did not change. Replacing them again tests the same hypothesis a second time. The period is a ruler: length equals the speed of light over two times the group index times the free spectral range. Use it to find the reflector on the chip.'),
        W('Order new, labeled patch cords for the bench, since the shared ones are unlabeled', 'The shared drawer deserves suspicion in general, and it has its own category. But in this incident the symptom survived every swap. Suspicion is not evidence. The period is evidence.'),
        W('Classify it as External Feedback for now, since that is the most common source', 'Common is not the same as this one. Classification is not a way to assign a problem to whoever is usual. It is a way to assign a problem to the person who can fix it. A reflector on the chip can only be fixed in the layout.'),
        W('Re-run the layout simulation, which showed no reflections', 'The layout simulation did exactly what it was told. If each component was simulated with matched ports or absorbing boundaries, no component could reflect into another, so no parasitic cavity could appear. The chip, unfortunately, was not built with matched ports. Simulate the reflection of each component, and the cavity between them.')
      ], 'Intracavity Feedback goes to the chip design owner.'),

      /* 3 Instrument-Originated */
      catSlide(3, 'instrument',
        ['Without an isolator in the measurement path, parasitic reflections lead to an overestimate of laser linewidth (De Koninck et al., 2025). A flat detector window or instrument input reflects a few percent per surface (Fresnel)', 'Without an isolator in the measurement path, parasitic reflections lead to an overestimate of the laser linewidth, as De Koninck and coworkers state. A flat, uncoated detector window, or an instrument input, reflects a few percent at each surface.'],
        [['The result depends on which instrument is connected', 'The result depends on which instrument is connected.'], ['The symptom appears only during measurement', 'The symptom appears only while you are measuring. Which is, unfortunately, the only time you look.'], ['A tunable source misbehaves only with one device under test connected. Your source is a laser too', 'A tunable source misbehaves only when one particular device under test is connected. Your source is a laser too, and the device under test reflects into it.']],
        'the owner of the measurement setup, and the instrument custodian'),
      { type: 'chat', title: 'Category 3 in the Workplace', app: 'email', skin: 'mail', meta: { header: 'Inbox', subject: 'Linewidth differs between setup A and setup B', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'Hi Dana,\n\nSame laser, two linewidth setups, two answers. Setup B reads noticeably broader. A has an input isolator and B does not, so B may be measuring the laser plus its own reflections.\n\nBefore I report either number, I would like to add an isolator to B and compare again. Could I borrow yours?\n\nGary', side: 'right', to: [BY], date: '2026-11-14 11:02', subject: 'Linewidth differs between setup A and setup B', attachments: [{ name: 'linewidth_A_vs_B_2026-11-14.xlsx', size: '96 KB' }] },
          { from: BY, text: 'Of course, it is in the second drawer, labelled. If the two agree once B is isolated, the difference was B feeding back into the laser.\n\nDana', side: 'left', to: [GARY], date: '2026-11-14 11:20' },
          { from: GARY, text: 'Thanks. I will log the comparison either way.', side: 'right', to: [BY], date: '2026-11-14 11:26' }
        ], narration: isThisOk('Gary treated the disagreement between the two setups as a finding, and planned the isolator comparison before reporting any number.',
          ['Read this email exchange about two linewidth setups that disagree.', 'Then judge whether Gary\'s plan is proper.']) },
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
      { type: 'chat', title: 'Category 4 in the Workplace', app: 'thread', skin: 'teams', meta: { header: '#purchasing', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Isolators arrived, one is installed on bench 4. Feedback symptoms are reduced but not gone', side: 'right' },
          { from: BY, text: 'Which band are they specified for?', side: 'left' },
          { from: GARY, text: 'The label says 1310. We run at 1550, but I assumed it would still be mostly fine', side: 'right' },
          { from: BY, text: 'Faraday rotation depends on wavelength. At 1550 it will not be 45 degrees, so the isolation drops', side: 'left' },
          { from: MGR, text: 'The 1550 version has a long lead time. Can we live with these for now?', side: 'left' }
        ], narration: isThisOk('An isolator specified for another band was installed to avoid a long lead time, without measuring its isolation at the operating wavelength.',
          ['Read this purchasing chat about the new isolators.', 'Then judge whether the way they are being used is proper.']) },
      classify('An isolator labelled for the 1310 nm band is installed in a 1550 nm link. Backward light now reaches the laser.', 'thirdparty', [
        ['qpq', 'Nobody traded isolation for anything on purpose. The part was assumed to work at a wavelength it was not built for.'],
        ['instrument', 'The isolator is not measurement equipment. It is a purchased component on the optical path.'],
        ['domestic', 'The isolator is outside the laser package, and it is a supplier part, not your own chip.']
      ]),
      bystander('You are Dana. While helping on Gary\'s bench, you notice the isolator in his 1550 nm link is labelled for 1310 nm. He is about to start a week-long measurement. What do you do?', [
        OK('Use the approved phrase now, show him the label, and help him log it and find an isolator for the right band'),
        WB('Say nothing; the label may just be a generic part number', 'Isolators are not all the same. The Faraday rotation angle depends on wavelength, so an isolator built for one band does not rotate by forty-five degrees in another, and its isolation falls. A week of measurement will record the laser plus its feedback.'),
        WB('Mention it at the end of the week, when the data is in', 'At the end of the week, the data is in, and it is a week of data with feedback. Speak up before the measurement, on the time scale of the problem.'),
        WB('Post in the group channel that purchasing ordered the wrong isolators', 'Purchasing bought what was ordered, and the price is not the root cause. The part was received and installed without checking the band. Third-Party Feedback is enabled by the supplier and completed by us. Speak to the bench owner first.'),
        WB('Swap in a 1550 nm isolator from your own bench while Gary is at lunch', 'The physics is right. The process is not. Gary\'s log now describes a different setup than the one on the bench. Untracked changes to an optical path are how future incidents begin.')
      ], 'Speak up before the measurement starts.'),

      /* 5 Intra-Group */
      catSlide(5, 'intragroup',
        ['An open, flat-polished port on a shared splitter reflects about 3.3 percent of what reaches it (Fresnel, $n = 1.444$) back through the splitter toward every laser on that bench', 'An open, flat-polished port on a shared splitter reflects about three point three percent of the light that reaches it, back through the splitter, toward every laser connected to it.'],
        [['Nobody knows who connected it', 'Nobody knows who connected it.'], ['The setup "has always been like that"', 'The setup has always been like that.'], ['Patch cords in the shared drawer have no labels, or labels in a previous student\'s handwriting', 'The patch cords in the shared drawer have no labels, or labels in the handwriting of a student who graduated.']],
        'the group lab manager, and every member of the group'),
      { type: 'chat', title: 'Category 5 in the Workplace', app: 'thread', skin: 'slack', meta: { header: '#group-lab', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'Who owns the 1x4 splitter on bench 3? Two ports are open, with flat connectors', side: 'right' },
          { from: BY, text: 'It was set up before I joined. I think it has always been like that', side: 'left' },
          { from: GARY, text: 'Every laser on bench 3 sees those two reflections. I would like to put angled terminators on both ports at 3 pm today and note it in the bench log. Any objections?', side: 'right' },
          { from: MGR, text: 'No objection. Please also add the splitter to the bench checklist, with an owner', side: 'left' }
        ], narration: isThisOk('Gary found two open ports on a shared splitter, and announced the fix and the log entry before touching the shared bench.',
          ['Read this group lab chat about a shared splitter on bench three.', 'Then judge whether Gary\'s response is proper.']) },
      classify('A shared splitter on the group\'s test bench has two ports left open with flat connectors. It was set up by a student who has since graduated. Every laser on that bench is a little noisier than elsewhere.', 'intragroup', [
        ['external', 'The reflectors are outside each laser package, yes. But the defining feature is that the resource is shared inside the group and nobody owns it. Primary category, please.'],
        ['hostile', 'Two identifiable open ports explain it. A Hostile Optical Environment has reflectors everywhere, with no single one to blame.'],
        ['crossinst', 'The student was in your own group. Graduation does not make someone another institution, although it sometimes feels that way.']
      ]),

      /* 6 Cross-Institutional */
      catSlide(6, 'crossinst',
        ['A partner\'s chip arrives with flat, uncoated facets although the email said they were angled. A cleaved silicon facet, $n = 3.48$ (Li, 1980), reflects about 31 percent as a plane-wave estimate', 'A partner\'s chip arrives with flat, uncoated facets, although the email said they would be angled. A cleaved silicon facet reflects about thirty-one percent, as a plane-wave estimate.'],
        [['The specification is an email that says "should be fine"', 'The specification is an email that says, should be fine.'], ['Nobody wrote down the return loss', 'Nobody wrote down the return loss.'], ['"The other group already measured it"', 'The other group already measured it. With their setup. Which you have not seen.']],
        'both principal investigators, in writing'),
      { type: 'chat', title: 'Category 6 in the Workplace', app: 'email', skin: 'mail', meta: { header: 'Inbox', subject: 'RE: Chips shipped (angled facets as discussed)', participants: [GARY, BY] },
        messages: [
          { from: GARY, text: 'Hi Dana,\n\nThe chips from the partner group arrived. Under the scope the facets look flat to me, and I do not see a coating. Their email said angled, and the agreement only says "standard facets".\n\nBefore any device measurements, I will measure the return loss at our wavelength, log it, and send it to both principal investigators, with a proposal to put return loss into the specification.\n\nGary', side: 'right', to: [BY], date: '2026-11-15 14:12', subject: 'RE: Chips shipped (angled facets as discussed)', attachments: [{ name: 'facet_inspection_2026-11-15.jpg', size: '1.4 MB' }] },
          { from: BY, text: 'Good. Until then, keep an isolator between the laser and the chip.\n\nDana', side: 'left', to: [GARY], date: '2026-11-15 14:30' },
          { from: GARY, text: 'It is already in. I will copy you on the email to both sides.', side: 'right', to: [BY], date: '2026-11-15 14:41' }
        ], narration: isThisOk('Gary found a gap in the collaboration specification, and will measure the reflection and tell both sides in writing before building on it.',
          ['Read this email exchange about chips that arrived from a partner group.', 'Then judge whether Gary\'s plan is proper.']) },
      classify('Chips from a partner university arrive with flat, uncoated facets, although an email promised angled facets. The collaboration agreement never mentions return loss.', 'crossinst', [
        ['thirdparty', 'A partner university is not a supplier with a datasheet. The defining feature is a collaboration in which the reflections were never specified.'],
        ['domestic', 'The facets are on a chip from another institution. It will become part of your package, but the specification gap is between institutions.'],
        ['qpq', 'Nobody offered a trade. The angled facets were promised and simply did not arrive.']
      ]),
      tester('You are the tester. The partner\'s chips have flat, uncoated facets, and your laser is edge-coupled to them with no isolator. Your partner\'s student says "should be fine". What would you do?', [
        OK('Measure the return loss, log it, tell both principal investigators in writing, add an isolator or angled interface, and put return loss into the collaboration specification'),
        W('Accept the partner\'s "should be fine"; they know their process', 'They know their process. They do not know your laser. A cleaved silicon facet reflects about thirty-one percent as a plane-wave estimate, and that light enters your laser if nothing stops it. Whose chip it is does not change where the light goes.'),
        W('Ask the partner to check their dicing recipe, and keep measuring in the meantime', 'Asking about the recipe is reasonable. Measuring in the meantime, with flat facets, no isolator and no return loss on record, is not. A dicing saw cuts where and how it was told. If the facets were supposed to be angled, the angle had to be specified in the process, by someone. The tool did what it was told. Nobody told it about the angle.'),
        W('Proceed, and write "facets as received" in the methods', '"As received" describes the facets. It does not describe their reflection, or what it did to your laser. Readers cannot correct for a number you did not measure.'),
        W('Angle-polish the facets in your own lab to keep the schedule', 'The physics improves. The collaboration does not, unless you tell them. The partner\'s records now describe a different chip than the one you measured, and the next shipment will arrive flat again.'),
        W('Use the partner group\'s own return loss measurement', 'The other group measured with their setup, at their wavelength, with their isolation. Return loss is a property of the interface in your setup. Measure it where it matters.')
      ], 'Measure it, write it down, and tell both sides.'),

      /* 7 Quid Pro Quo */
      catSlide(7, 'qpq',
        ['Figures reported for one integrated isolator: more than 25 dB isolation, less than 1.5 dB insertion loss projected (Lapointe et al., 2023). Removing such an isolator wins back at most 1.5 dB and gives up at least 25 dB of protection', 'Here are example figures for one integrated isolator: more than twenty-five decibels of isolation, and less than one and a half decibels of insertion loss, as projected by Lapointe and coworkers. Removing such an isolator wins back at most one and a half decibels, and gives up at least twenty-five.'],
        [['Any sentence that begins "If you just remove the isolator"', 'Any sentence that begins, if you just remove the isolator.'], ['"We will put it back after the demo"', 'We will put it back after the demo.'], ['Averaging time increased to make a measurement faster to pass', 'Averaging time increased to make a measurement easier to pass. Averaging is time traded for truth.']],
        'the person who proposed the trade, and their manager',
        ['Please note. A Quid Pro Quo offer does not need to be explicit. A link budget that only closes without the isolator is an offer.']),
      { type: 'chat', title: 'Category 7 in the Workplace', app: 'phone', skin: 'signal', meta: { header: 'Messages', participants: [MGR, GARY] },
        messages: [
          { from: MGR, text: 'The isolator costs us the margin we need. Take it out for the demo, we can add it back for production', side: 'left' },
          { from: GARY, text: 'And the reflections from their board?', side: 'right' },
          { from: MGR, text: 'It is one afternoon. Also, can you increase the averaging on the noise test? It passes marginally and the review is tomorrow', side: 'left' }
        ], narration: isThisOk('Morgan suggested trading the isolator for insertion loss, and more averaging for a passing noise test.',
          ['Read this message exchange between Morgan and Gary, the day before a demo.', 'This time, judge whether Morgan\'s suggestions are proper.']) },
      classify('Morgan tells Gary that if he removes the isolator, the link budget closes, and that the isolator can go back in after the demo.', 'qpq', [
        ['retaliation', 'Retaliation is the laser\'s response, and it may well follow. But the defining feature here is the offer: isolation in exchange for insertion loss.'],
        ['intragroup', 'Morgan is in Gary\'s group, yes. But the defining feature is not a shared resource. It is a trade.'],
        ['thirdparty', 'The isolator works as specified. Nobody is proposing to replace it with a worse part. They are proposing to remove it.']
      ]),

      /* 8 Hostile Optical Environment */
      catSlide(8, 'hostile',
        ['Every pair of reflectors on a path is a cavity: $N$ reflectors make $N(N-1)/2$ of them. Six flat surfaces make fifteen cavities, each with its own period $c/(2nL)$','Every pair of reflectors on a path is a cavity. N reflectors make N times N minus one over two of them. Six flat surfaces make fifteen cavities, each with its own ripple period.'],
        [['Ripples at several periods at once', 'Ripples at several periods at once.'], ['The symptom moves when anything on the bench is touched', 'The symptom moves when anything on the bench is touched. Including the bench.'], ['Nothing on the bench is angled, coated, gelled or isolated', 'Nothing on the bench is angled, coated, gelled, or isolated.']],
        'everyone who uses the bench, coordinated by the lab manager',
        ['A Hostile Optical Environment is not one bad reflector. It is a culture.']),
      { type: 'chat', title: 'Category 8 in the Workplace', app: 'call', skin: 'meeting', meta: { header: 'Video call', participants: [GARY, BY, MGR] },
        messages: [
          { from: GARY, text: 'I see ripples at several periods. Before I pick one: flat cleaves, flat connectors, an uncoated lens, a flat detector window, an open splitter port', side: 'right' },
          { from: GARY, text: 'Any pair of those is a cavity, so the ripple closest to the expected device length is not evidence. I will gel or angle them one at a time and see which ripple goes away', side: 'right' },
          { from: MGR, text: 'Every bench in the building looks like that', side: 'left' },
          { from: GARY, text: 'Then I will send the list to the lab manager as well', side: 'right' },
          { from: BY, text: 'I can help with the connectors', side: 'left' }
        ], narration: isThisOk('Gary listed every reflector on the bench, did not pick the ripple that matched his expectation, and raised the bench with the lab manager.',
          ['Watch this video call about the ripples on Gary\'s spectrum.', 'Then judge whether Gary\'s approach is proper.']) },
      classify('On Gary\'s bench: flat-cleaved fibers, flat connectors, an uncoated lens, a flat detector window and an open splitter port. The spectrum shows ripples at several periods at once, and they all move when someone leans on the table.', 'hostile', [
        ['external', 'Each reflector is External, yes. But with reflectors everywhere and ripples at several periods, there is no single reflector to blame. That is the defining feature of another category.'],
        ['intragroup', 'Some of these may be shared resources. But the defining feature is that everything reflects, and no single reflector explains it.'],
        ['instrument', 'One of the reflectors is a detector window. One. There are at least five.']
      ]),
      bystander('You are Dana. You have just joined a bench where every surface is flat and nothing is angled. Everyone has adapted. What do you do?', [
        OK('Use the approved phrase with the lab manager, list the reflectors, and propose fixing them one at a time with the bench owners'),
        WB('Work with the bench as it is; others have published from it', 'Adapting to a Hostile Optical Environment means every measurement on the bench carries a mix of cavities, and nobody can tell which ripple belongs to which device. Adaptation is not mitigation.'),
        WB('Schedule sensitive measurements at night, when the building is quieter', 'Reflections do not keep office hours. A flat connector reflects the same three point three percent at midnight. You may see less vibration at night, but vibration only moves the phase of reflections that are already there.'),
        WB('Increase averaging to suppress the ripples', 'Averaging can hide a ripple finer than your resolution, or a fluctuation faster than your averaging time. It does not remove a single reflector. It removes your ability to see them.'),
        WB('Move your measurements to a cleaner bench', 'Moving protects your measurement and leaves the bench as it is for the next person. Speak up, so that the bench gets fixed, not just vacated.')
      ], 'Name it, list it, and fix it together.'),

      /* 9 Retaliation */
      catSlide(9, 'retaliation',
        ['Depending on strength and distance, feedback narrows or broadens the line, causes hops between external-cavity modes, or drives coherence collapse (Tkach and Chraplyvy, 1986; Lenstra et al., 1985)', 'Depending on the strength of the feedback and the distance to the reflector, the laser narrows or broadens its line, hops between external cavity modes, or falls into coherence collapse, as classified by Tkach and Chraplyvy, and described by Lenstra and coworkers.'],
        [['Mode hops after a change in the optical path', 'Mode hops after a change in the optical path.'], ['A periodic ripple in the light versus current curve', 'A periodic ripple in the light versus current curve.'], ['A dramatically broadened line that does not improve when you move the reflector', 'A dramatically broadened line that does not improve when you move the reflector.']],
        'the owner of the reflector that caused it. Never the laser',
        ['Retaliation is classified separately from the feedback that caused it. Both are reported.', 'Zero tolerance for Retaliation means zero tolerance for the conditions that cause it. We do not punish the laser.']),
      { type: 'chat', title: 'Category 9 in the Workplace', app: 'thread', skin: 'teams', meta: { header: '#all-hands', participants: [MGR, GARY, BY] },
        messages: [
          { from: MGR, text: 'The laser on bench 2 is unstable again. Let us order a replacement, this unit has been trouble from the start', side: 'left' },
          { from: BY, text: 'It is responding to feedback. A new laser on the same path will behave the same way', side: 'left' },
          { from: GARY, text: 'Should I log it as a laser failure?', side: 'right' },
          { from: BY, text: 'Log the reflector as the cause, and the laser response as the symptom', side: 'left' }
        ], narration: isThisOk('Morgan wants to replace a laser that is responding to feedback, instead of removing the feedback.',
          ['Read this all-hands chat about an unstable laser on bench two.', 'Then judge whether Morgan\'s proposal is proper.']) },
      classify('After a flat connector is added close to the laser, the line broadens dramatically and the laser becomes noisy. Moving the connector a little does not help.', 'retaliation', [
        ['external', 'The connector is indeed External Feedback, and it is reported as the source. But this question asks about the laser\'s response, not the source: a dramatic broadening that does not improve when the reflector moves. Responses have their own category.'],
        ['hostile', 'One connector was added. A Hostile Optical Environment has reflectors everywhere. And in any case, this question asks about the laser\'s response, not the source.'],
        ['qpq', 'Nobody traded anything. And this question asks about the laser\'s response, not about how the reflector got there.']
      ], 'Which category describes the laser\'s response?'),
      tester('You are the tester. Your laser\'s line has broadened dramatically since a connector was added nearby, and Morgan suggests replacing the laser, since this unit has been trouble from the start. What would you do?', [
        OK('Treat the laser\'s response as a symptom: find and remove the feedback, log both the reflector and the Retaliation, and keep the laser'),
        W('Replace the laser as Morgan suggests; this unit has been trouble from the start', 'The new laser will sit on the same optical path, in front of the same connector, and it will respond in the same way. Retaliation is the response, not the cause. Zero tolerance for Retaliation means zero tolerance for the conditions that cause it. We do not punish the laser.'),
        W('Raise the current; a laser further above threshold is usually more stable', 'Feedback effects depend on the fraction of light that returns, not on the absolute power. More current sends more light to the connector, and more comes back. The ratio is the same.'),
        W('Move the connector a little further away', 'In the regime called coherence collapse, the broadening is largely insensitive to the distance to the reflector, as Tkach and Chraplyvy found. A small move will not change the regime. Remove, angle, or isolate the reflector.'),
        W('Log it as a laser defect, so the replacement is covered by warranty', 'A warranty claim records the wrong root cause, and the next laser on this path will do the same thing. Report the reflector to its owner, and the Retaliation to the setup log.'),
        W('Swap the laser driver first, since a noisy driver can broaden the line', 'A noisy current source can add noise to a laser, and it is worth checking. But the line broadened when the connector was added, and nothing changed in the driver. The driver did exactly what it did yesterday. The optical path is what changed.')
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
      [['The color of the resist', 'Hausmann and coworkers report that earlier electron beam writing left periodic scatterers along the ring circumference that split the resonances, and suggest that continuous writing removed them.']]),
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

  /* fabryPerot, modeHop and doublet live in widgets/*.js (sources in tools/widgets_src/fb/); they register on window.LMS_WIDGETS. */

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
    photo('Angled and Flat', RD + 'fc_apc_vs_pc_ferrules.jpg', 'Two FC connectors on an anti-static mat, dust caps off. The green-boot connector (upper) has an angle-polished ferrule tip; the blue-boot connector (lower) has a flat, slightly domed tip.',
      ['Here, the green connector is angle polished. The blue one is flat.', 'Angled sends the reflection into the cladding. Flat sends it home. Please do not mate green with blue.'], [0.5, 0.5, 1.0, 0.8, 0.5, 1.3]),
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
        { x: 0.445, y: 0.695, label: 'Rotation mount', body: '<p>Rotation mounts usually carry a wave plate or polarizer: a thin plate with two parallel faces. A plate of thickness $L$ and group index $n$ is a low-finesse Fabry-Perot with free spectral range $c/(2nL)$.</p>', narration: ['A thin plate with two parallel faces is a Fabry-Perot, whether or not it was ordered as one.'] },
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
  patch(ch5, 'Why Resonances Split', { layout: 'split', image: { src: FB + 'microdisk_sem.jpg' }, heading: 'Why Resonances Split', kicker: 'Clockwise and counter-clockwise' });
  after(ch5, 'Why Resonances Split', [
    photo('A Resonator Awaiting Light', RD + 'sin_microring_lensed_fiber.jpg', 'A lensed fiber at the edge of a silicon nitride chip, a bus waveguide, a ring resonator of 100 micrometer radius, and a drop waveguide.',
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

  /* =================================================================== */
  /* INTERACTIVITY: stages, tap-to-advance chats, sort, reflect.             */
  /* Mostly re-cuts existing narration into stages (engine contract in       */
  /* COURSE_SPEC.md, "Interactivity: stages, reflect, sort").                */
  /* =================================================================== */
  function allSteps() { var a = []; [ch0, ch1, ch2, chCat, ch3, ch4, ch5, ch6].forEach(function (ch) { ch.steps.forEach(function (st) { a.push(st); }); }); return a; }
  function byTitle(ch, t) { return ch.steps[findStep(ch, t)]; }
  /* split a slide into a base and stages: cuts = [[bulletStart, narrationStart, prompt], ...] */
  function stageify(st, cuts) {
    var B = st.bullets, N = st.narration, stages = [];
    cuts.forEach(function (c, i) {
      var b1 = i + 1 < cuts.length ? cuts[i + 1][0] : B.length, n1 = i + 1 < cuts.length ? cuts[i + 1][1] : N.length;
      stages.push({ prompt: c[2], bullets: B.slice(c[0], b1), narration: N.slice(c[1], n1) });
    });
    st.bullets = B.slice(0, cuts[0][0]); st.narration = N.slice(0, cuts[0][1]); st.stages = stages;
  }

  var MISS = {
    'Scenario 1: The Rework': 'Nobody asked why the ripple period stayed the same through four reworks.',
    'Scenario 2: The Flat Cleave': 'The fiber tip was never tested, and the laser is going back.',
    'Incident 1: How It Unfolded': 'The isolator came out with no spectrum or linewidth check, and the person who asked for one was moved off the thread.',
    'Scenario 1: The Excellent Fit': 'The ripple period was never converted into a length, so nobody knows which two surfaces made it.',
    'Scenario 2: The Accidental Laser': 'Nobody checked the facets before calling it novel.',
    'Incident 2: How It Unfolded': 'The ripple matches the fiber gap, and nobody changed the gap before calling it intrinsic.',
    'Category 1 in the Workplace': 'A ripple that survives every swap and appears on every unit points at the chip, not the cords.',
    'Category 2 in the Workplace': 'The best-looking alignment is just one point on a cavity fringe.',
    'Category 4 in the Workplace': 'Nobody measured the isolation at the wavelength actually used.',
    'Category 7 in the Workplace': 'Protection and measurement quality are being traded for a schedule.',
    'Category 9 in the Workplace': 'The laser is responding to a reflector that nobody has found.',
    'Scenario 2: Up and Down': 'The two maps differ for a physical reason, and averaging hides it.',
    'Scenario 3: Qualified at One Current': 'One operating point was tested, and the whole range is being signed off.',
    'Incident 3: How It Unfolded': 'The hop is unlogged, the cause is still in place, and the person who raised concerns was removed from the call.',
    'Scenario 2: The Directional Modulator': 'Directional modulation is not isolation.',
    'Scenario 3: The Straight Arrow': 'The drawing ignores refraction, and active alignment cannot fix an angle.',
    'Incident 4: How It Unfolded': 'The orientation was never checked, and the vendor is being blamed for it.',
    'Scenario 1: Scrap the Wafer': 'The second minimum was dismissed as noise, and the fit model was never questioned.',
    'Scenario 2: Conflicting Policies': 'Two teams have opposite targets for the same ring, and neither target is a number.',
    'Incident 5: How It Unfolded': 'The coupling branch was assumed, not measured, and the reflection port was ignored.',
    'Scenario 2: New Physics': 'The source was never cross-checked against a separate laser.',
    'Scenario 3: The Record Linewidth': 'A footnote does not remove the feedback from the measurement.',
    'Incident 6: How It Unfolded': 'The runs that showed the hop were left out.'
  };
  /* judged chats where the judged person acts properly: the correct answer is OK */
  var PROPER = {
    'Scenario 3: Isolator Placement': 'The isolator sits in front of the distant reflectors, and the near ones are budgeted and measured.',
    'Scenario 3: Certified Single-Mode': 'The fringes are being tested, not explained away by a label and an estimate.',
    'Category 3 in the Workplace': 'The disagreement between the setups is treated as the finding, and no number is reported until it is explained.',
    'Category 5 in the Workplace': 'The shared reflectors were found, the fix was announced before anyone touched the shared bench, and the splitter gets an owner and a log entry.',
    'Category 6 in the Workplace': 'The return loss will be measured, logged, and sent to both sides before any work builds on it.',
    'Category 8 in the Workplace': 'The reflectors are listed and will be removed one at a time, and the bench is reported, not adapted to.',
    'Scenario 1: The Datasheet': 'The mode-hop-free range was checked before the run, and the sweep was planned inside it.',
    'Scenario 1: The Contrast Ratio': 'Every mode was counted before any claim, and the device is described as what it is.',
    'Scenario 3: The Snap': 'The balance point was measured, not predicted, and the lock has margin.',
    'Scenario 1: The Calm Wavemeter': 'A slow instrument was cross-checked with a fast one before anything was reported.'
  };
  /* 1. every "Is this okay?" chat: an OK / Not OK popup, then the answer as a stage */
  allSteps().forEach(function (st) {
    if (st.type !== 'chat' || !st.narration) return;
    var k = st.narration.indexOf('Pause and ask yourself. Is this okay?');
    if (k < 0) return;
    var rest = st.narration.slice(k + 1), good = PROPER[st.title], why = good || MISS[st.title];
    if (!why) throw new Error('no MISS or PROPER for chat ' + st.title);
    if (k < 1 || st.narration.slice(0, k).some(function (l) { return typeof l !== 'string' || !l; })) throw new Error('no judge intro for chat ' + st.title);
    st.narration = st.narration.slice(0, k);
    st.ask = { question: 'Is this OK?', options: [{ text: 'OK', correct: !!good, say: good ? why : 'Not quite. ' + why }, { text: 'Not OK', correct: !good, say: good ? 'Not quite. ' + why : why }] };
    st.stages = [{ prompt: 'Show why', title: good ? 'It is OK.' : 'It is not OK.', body: '<p>' + why + '</p>', narration: rest }];
  });
  /* incident chats unfold one message at a time */
  allSteps().forEach(function (st) { if (st.type === 'chat' && /^Incident \d: How It Unfolded$/.test(st.title)) st.tapToAdvance = true; });

  /* 2. "How Do You Think Gary Felt?" slides: the feelings are revealed */
  allSteps().forEach(function (st) {
    if (st.type !== 'slide' || st.title !== 'How Do You Think Gary Felt?') return;
    stageify(st, [[1, 3, 'See how Gary felt']]);
    st.narration = st.narration.filter(function (l) { return l !== 'How do you think Gary felt?'; });
    var fs = 'All feelings are valid. Let us see how Gary felt.';
    st.ask = { question: 'How do you think Gary felt?', options: [{ text: 'Ashamed', say: fs }, { text: 'Relieved it was caught', say: fs }, { text: 'Unbothered', say: fs }, { text: 'I prefer not to say', say: 'That is also a feeling, and it has been logged. Let us see how Gary felt.' }] };
  });

  /* 3. category policy slides: warning signs and owner are revealed */
  allSteps().forEach(function (st) {
    if (st.type !== 'slide' || !/^Category \d of 9: /.test(st.title)) return;
    var nb = 2, nn = st.narration.indexOf('Warning signs include the following.');
    stageify(st, [[nb, nn, 'Show the warning signs']]);
  });

  /* 4. slides whose narration already evolves: symptom then cause, setup then numbers */
  stageify(byTitle(ch1, 'Retaliation From an Unexpected Direction'), [[1, 3, 'Reveal the cause']]);
  stageify(byTitle(ch2, 'Incident 2: The Numbers'), [[1, 2, 'Show the two mirrors'], [3, 4, 'Show the ripple depth'], [4, 5, 'Show the diagnostic']]);
  stageify(byTitle(ch5, 'Incident 5: The Numbers'), [[2, 4, 'Plug in the doublet'], [4, 7, 'Two modes, or one doublet?']]);
  stageify(byTitle(ch4, 'Policy 4.3: Isolator Orientation'), [[2, 4, 'What if it is installed backwards?'], [4, 6, 'What if it is rotated 90 degrees?']]);
  byTitle(ch4, 'Policy 4.3: Isolator Orientation').ask = { question: 'If the isolator is installed backwards, what happens to the laser output?', options: [
    { text: 'Nothing changes; by reciprocity it works both ways', say: 'Not quite. An isolator is the one component that breaks reciprocity.' },
    { text: 'It is attenuated by about the isolation ratio', correct: true, say: 'And light from downstream gets in.' },
    { text: 'It is protected twice as well against reflections', say: 'Not quite. The round trip is the same product either way. Only the output suffers.' }] };
  byTitle(ch1, 'Retaliation From an Unexpected Direction').ask = { question: 'Which reflector caused the ripple?', options: [
    { text: 'The sub-micron gap right next to the laser', say: 'Not quite. That was everyone\'s first suspect. It was the far end.' },
    { text: 'A grating coupler at the far end of the waveguide', correct: true, say: 'The far end.' },
    { text: 'The laser\'s own grating', say: 'Not quite. The laser\'s own grating is designed feedback. It was the far end.' }] };
  (function () {
    var st = byTitle(chCat, 'Why We Classify');
    st.stages = [{ prompt: 'Show the policy', bullets: st.bullets, narration: st.narration.slice(1) }];
    st.bullets = []; st.narration = ['Before we meet the categories, a quick check.'];
    st.ask = { question: 'Why do we classify unsolicited feedback?', options: [
      { text: 'To decide who fixes it, and who must be told', correct: true, say: 'Classification is about ownership, not blame.' },
      { text: 'To decide who is to blame', say: 'Not quite. Classification is about ownership, not blame.' },
      { text: 'For the quarterly metrics', say: 'Not quite. The metrics are a side effect. Classification is about ownership.' }] };
  })();

  /* 5. photo pairs become one step with an image stage */
  (function () {
    var i = findStep(ch1, 'Angled and Flat'), fer = ch1.steps[i];
    var cord = byTitle(ch1, 'The Angle Is the Point');
    cord.stages = [{ prompt: 'Look closer at the ferrules', image: { src: fer.image.src, kb: fer.kb }, body: '<p>' + fer.caption + '</p>', narration: fer.narration }];
    ch1.steps.splice(i, 1);
    var j = findStep(ch5, 'A Resonator Awaiting Light'), ring = ch5.steps[j];
    var split = byTitle(ch5, 'Why Resonances Split');
    split.stages = [{ prompt: 'Show a ring on a chip', image: { src: ring.image.src, kb: ring.kb }, body: '<p>' + ring.caption + '</p>', narration: ring.narration }];
    ch5.steps.splice(j, 1);
  })();

  /* 6. sort exercises in the classification module */
  after(chCat, 'You are Dana. While helping on Gary\'s bench', [
    { type: 'sort', title: 'Sort the Reflectors',
      prompt: 'Drag each reflector into its category of unsolicited feedback. Ask where the reflector is, and who put it there.',
      buckets: [{ id: 'domestic', label: CAT.domestic.name }, { id: 'external', label: CAT.external.name }, { id: 'instrument', label: CAT.instrument.name }, { id: 'thirdparty', label: CAT.thirdparty.name }],
      items: [
        { text: 'A grating coupler at the far end of your own chip', bucket: 'domestic', why: 'It is inside your own chip. Intracavity, or Domestic, Feedback. Owner: the chip designer.' },
        { text: 'The facets of the erbium amplifier on your chip', bucket: 'domestic', why: 'Facets on your own chip. With enough gain, they make a laser nobody ordered.' },
        { text: 'A flat-cleaved fiber tip held in front of the laser', bucket: 'external', why: 'It is on the optical path you connected, outside the package. External Feedback.' },
        { text: 'The facet of the device under test', bucket: 'external', why: 'The device under test is on the path you connected, and its facet is a mirror. External Feedback.' },
        { text: 'The input of a linewidth setup with no isolator', bucket: 'instrument', why: 'The measurement equipment itself reflects into the laser. Instrument-Originated Feedback.' },
        { text: 'A flat, uncoated detector window', bucket: 'instrument', why: 'Each flat surface of the window reflects a few percent. The instrument is part of the cavity.' },
        { text: 'An isolator built for 1310 nm, installed in a 1550 nm link', bucket: 'thirdparty', why: 'The Faraday rotation depends on wavelength, so the purchased part does not do what was assumed. Third-Party Feedback, completed by us.' },
        { text: 'A pigtailed laser that shipped with no internal isolator', bucket: 'thirdparty', why: 'The supplier part lacks a protection you assumed. Read the datasheet. And the footnote.' }
      ],
      narration: ['Sort the reflectors.', 'Drag each reflector into its category. Where is the reflector, and who put it there?'] }
  ]);
  after(chCat, 'You are Dana. You have just joined a bench where', [
    { type: 'sort', title: 'Reflects or Does Not Reflect',
      prompt: 'Drag each item into the correct column. Policy PHC-1550.4 anticipates that one column will be nearly empty.',
      buckets: [{ id: 'yes', label: 'Reflects' }, { id: 'no', label: 'Does not reflect' }],
      items: [
        { text: 'A cleaved chip facet', bucket: 'yes', why: 'An index step. A cleaved silicon facet reflects about 31 percent as a plane-wave estimate.' },
        { text: 'An angle-polished connector', bucket: 'yes', why: 'It still reflects. The angle sends the reflection out of the core instead of straight back. Angled is not silent.' },
        { text: 'A lens with an anti-reflection coating', bucket: 'yes', why: 'A coating reduces a reflection. It does not remove it.' },
        { text: 'A fiber joint filled with index-matching gel', bucket: 'yes', why: 'The gel makes the index step small. Small is not zero.' },
        { text: 'A high-Q ring resonator', bucket: 'yes', why: 'Backscatter sends light back toward the laser. See Chapter 6.' },
        { text: 'The isolator itself', bucket: 'yes', why: 'Its polarizers and faces are interfaces. Light reflected before the Faraday rotator returns to the laser unprotected.' },
        { text: 'Your measurement fiber', bucket: 'yes', why: 'Its ends, its connectors, and the fiber itself. Every interface reflects.' },
        { text: 'A uniform medium with no index step anywhere', bucket: 'no', why: 'No index step, no Fresnel reflection. This is the only item in this column. It is also the only item not found in a laboratory.' }
      ],
      narration: ['Reflects, or does not reflect.', 'Drag each item into the correct column. One column will be nearly empty. That is intentional.'] }
  ]);
  after(chCat, 'You are the tester. Your laser\'s line has broadened dramatically', [
    { type: 'sort', title: 'Source or Response',
      prompt: 'The rule: classify the source, unless the question asks about the laser\'s response. Drag each item to the correct side.',
      buckets: [{ id: 'src', label: 'Source of the feedback' }, { id: 'resp', label: 'The laser\'s response (Retaliation)' }],
      items: [
        { text: 'An open, flat-polished port on a shared splitter', bucket: 'src', why: 'A reflector, shared by the group. A source: Intra-Group Feedback.' },
        { text: 'Mode hops after a connector was changed', bucket: 'resp', why: 'Hops are what the laser does. Retaliation.' },
        { text: 'Removing the isolator to close the link budget', bucket: 'src', why: 'A trade that lets the feedback in. A source: Quid Pro Quo Feedback.' },
        { text: 'A periodic ripple in the light versus current curve', bucket: 'resp', why: 'The ripple is the laser responding to a cavity. Retaliation. Then go and find the source.' },
        { text: 'Coherence collapse', bucket: 'resp', why: 'A dramatic broadening of the line. The most dramatic form of Retaliation.' },
        { text: 'A flat detector window', bucket: 'src', why: 'A reflector in the instrument. A source: Instrument-Originated Feedback.' }
      ],
      narration: ['Source, or response.', 'Classify the source, unless the question asks about the laser\'s response. The response is always Retaliation.'] }
  ]);

  /* 7. reflect moments */
  after(ch0, 'Speak-Up Procedure for Optical Incidents', [
    { type: 'reflect', title: 'A Moment of Reflection', mode: 'text',
      prompt: 'In a few words, describe a time when something you sent out came back to you.',
      reveal: '<p>Thank you for sharing. Whatever came back, it came back because something along the way reflected it.</p><p>Every interface reflects. Including yours.</p>',
      narration: ['A moment of reflection.', 'In a few words, describe a time when something you sent out came back to you.'],
      revealNarration: ['Thank you for sharing.', 'Whatever came back, it came back because something along the way reflected it. Every interface reflects. Including yours.'] }
  ]);
  after(ch1, 'Incident 1: How It Unfolded', [
    { type: 'reflect', title: 'Honest Self-Assessment', mode: 'scale',
      prompt: 'If my demo were on Monday and the link were short on power, I would also consider removing the isolator.',
      reveal: '<p>Many engineers agree with this statement. That is why this course exists.</p><p>Removing an isolator wins back at most its insertion loss, and gives up all of its isolation. The link budget is fixed by fixing the link.</p>',
      narration: ['Honest self-assessment.', 'There are no wrong answers on this scale. Only logged ones.'],
      revealNarration: ['Many engineers agree with this statement. That is why this course exists.', 'Removing an isolator wins back at most its insertion loss, and gives up all of its isolation.'] }
  ]);
  after(ch4, 'Inspect This Bench', [
    { type: 'reflect', title: 'Check Your Own Bench', mode: 'choice',
      prompt: 'Which way does the isolator on your own bench point right now?',
      choices: ['Away from the laser', 'Toward the laser', 'I do not know', 'I do not have an isolator'],
      reveal: '<p>If you chose anything other than the first option, please check before your next measurement.</p><p>If you chose the last option, please repeat Chapter 1.</p>',
      narration: ['Check your own bench.', 'Which way does the isolator on your bench point right now?'],
      revealNarration: ['If you chose anything other than the first option, please check before your next measurement.', 'If you chose the last option, please repeat Chapter One.'] }
  ]);
  after(ch6, 'Incident 6: How It Unfolded', [
    { type: 'reflect', title: 'Your Last Measurement', mode: 'choice',
      prompt: 'In your last measurement, which runs did you keep?',
      choices: ['All of them', 'The clean ones', 'The last one', 'I prefer not to say'],
      reveal: '<p>Every run is data. A clean run is the run where the hop happened while the instrument was looking elsewhere.</p><p>"I prefer not to say" has been logged.</p>',
      narration: ['Your last measurement.', 'Which runs did you keep?'],
      revealNarration: ['Every run is data. A clean run is the run where the hop happened while the instrument was looking elsewhere.', 'I prefer not to say, has been logged.'] }
  ]);

  /* =================================================================== */
  /* MODULE OPENERS: a hero title card as the first step of every module.  */
  /* Drawn cards (opener_*.jpg) come from tools/draw_fb_openers.py.          */
  /* =================================================================== */
  (function () {
    var OPEN = [
      [ch1, 1, RD + 'butterfly_isolator_fc_bench.jpg', [0.55, 0.45, 1.0, 0.6, 0.35, 1.12], 'Somewhere between your laser and your customer, something is reflecting. This module will help you recognize it.'],
      [ch2, 2, FB + 'opener_airy.jpg', null, 'Two reflections are all it takes. Most benches have more.'],
      [chCat, 3, SH + 'cube_farm.jpg', [0.5, 0.5, 1.0, 0.5, 0.45, 1.1], 'Every reflection has an owner. Today, we find out who.'],
      [ch3, 4, FB + 'opener_modehop.jpg', null, 'Lasers do not drift away. They hop. And they remember where they came from.'],
      [ch4, 5, FB + 'opener_faraday.jpg', null, 'Light goes both ways. Isolators are the exception, and only when installed the right way round.'],
      [ch5, 6, FB + 'opener_doublet.jpg', null, 'The better your resonator, the more it notices a single speck.'],
      [ch6, 7, FB + 'fp_scanning_interferometer.jpg', [0.5, 0.5, 1.0, 0.45, 0.5, 1.12], 'Every instrument has an opinion about time. This module asks what it is not telling you.']
    ];
    var WORD = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven'];
    OPEN.forEach(function (o) {
      var ch = o[0], img = { src: o[2] };
      if (o[3]) img.kb = o[3];
      ch.steps.unshift({ type: 'slide', layout: 'hero', title: ch.title, heading: ch.title, kicker: 'Module ' + o[1], subtitle: ch.subtitle, image: img,
        narration: ['Module ' + WORD[o[1]] + '. ' + ch.title + '.', o[4]] });
    });
    var t = ch0.steps[0];
    t.image = { src: FB + 'tisa_laser_table.jpg', kb: [0.55, 0.5, 1.0, 0.6, 0.45, 1.1] };
  })();

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
    host: { name: 'Pat Reynolds', role: 'Director, Office of Photonic Conduct', img: 'media/render/host_pat_present.jpg',
      cam: { idle: 'media/render/host_pat_cam_idle.mp4', talk: ['media/render/host_pat_cam_talk.mp4', { src: 'media/render/host_pat_cam_gesture.mp4', weight: 0.5 }] } },
    employer: 'Five Sigma Photonics, Inc.',
    chapters: [ch0, ch1, ch2, chCat, ch3, ch4, ch5, ch6],
    exam: { passPct: 80, count: 10, pool: pool },
    references: Object.keys(R).map(function (k) { return R[k]; }),
    speech: [],
    widgets: widgets
  };
})();
