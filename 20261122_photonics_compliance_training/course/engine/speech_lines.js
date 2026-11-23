/* Every line the engine can speak, built in one place so the player and the voice generator agree on the exact strings.
   Classic script (browser) and CommonJS (node). Contract: course/COURSE_SPEC.md (Tranche 2, Recorded voice).
   Hashing (LMS_normText, LMS_hash) lives in engine/speech_hash.js. */
(function (G) {
  'use strict';

  /* Fixed engine lines. Static strings only (no numbers or names spliced in) so each can be pre-recorded once. */
  var LINES = {
    right: "That's right!",
    notQuite: "Not quite. Let's take a moment to understand why.",
    tryAgain: 'Please try again.',
    ackDefault: 'Please read the acknowledgment carefully, enter your full legal name, and tick the box to continue.',
    spotFound: 'Hazard identified.',
    spotMiss: 'There is no hazard at that location. Please look again carefully.',
    spotAll: 'You have identified every hazard. Thank you for speaking up.',
    examPass: 'Congratulations. You have passed the final assessment.',
    examPassNext: 'Your score is shown on screen. Select Next to receive your certificate.',
    examFail: 'You did not meet the passing score.',
    examFailScore: 'Your score is shown on screen.',
    examRetake: 'Good news. You may retake the assessment as many times as needed.',
    certValid: 'Your certificate is valid for twelve months.',
    certReenroll: 'You will be automatically re-enrolled in eleven months.',
    recorded: 'Your response has been recorded.',
    sortRight: 'Correct. That item is in the right place.',
    sortWrong: 'Not quite. That item belongs in a different category.',
    sortAll: 'Every item is now in the right place.',
    askRight: "That's right.",
    askWrong: "Let's look at this more closely."
  };

  var ENT = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', ndash: '–', mdash: '—', hellip: '…', rsquo: '’', lsquo: '‘', rdquo: '”', ldquo: '“', times: '×', deg: '°', micro: 'µ', mu: 'μ', pi: 'π', lambda: 'λ', Delta: 'Δ', minus: '−', plusmn: '±' };
  /* Plain text of an HTML snippet. Deterministic and DOM-free so node and browser produce identical strings. */
  function stripHtml(s) {
    return String(s == null ? '' : s)
      .replace(/<br\s*\/?>/gi, ' ')
      .replace(/<[^>]*>/g, '')
      .replace(/&(#x[0-9a-f]+|#[0-9]+|[a-z]+);/gi, function (m, e) {
        if (e[0] === '#') { var n = e[1] === 'x' || e[1] === 'X' ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10); return isFinite(n) ? String.fromCodePoint(n) : m; }
        return Object.prototype.hasOwnProperty.call(ENT, e) ? ENT[e] : m;
      })
      .replace(/\s+/g, ' ').trim();
  }
  /* Sentence split for long explanations: break after . ! ? when followed by whitespace and a capital, digit, or quote. Decimals stay intact. */
  function splitSentences(s) {
    s = String(s || '').replace(/\s+/g, ' ').trim();
    if (!s) return [];
    return s.split(/(?<=[.!?]["'”)]?)\s+(?=[A-Z0-9"'“(])/).map(function (x) { return x.trim(); }).filter(Boolean);
  }
  var arr = function (a) { return Array.isArray(a) ? a.filter(function (x) { return typeof x === 'string' && x.trim(); }) : []; };

  /* Line sequences per engine event. player.js calls these; LMS_collectSpeech enumerates them. */
  var S = {
    narration: function (st) { return arr(st && st.narration); },
    checkIntro: function (st) { var n = arr(st.narration); return n.length ? n : (st.question ? [stripHtml(st.question)] : []); },
    checkCorrect: function (st, opt) {
      var tail = st.correctSay ? splitSentences(stripHtml(st.correctSay)) : opt && opt.why ? splitSentences(stripHtml(opt.why)) : [];
      return [LINES.right].concat(tail);
    },
    checkWrong: function (opt) { return [LINES.notQuite].concat(splitSentences(stripHtml((opt && opt.why) || ''))).concat([LINES.tryAgain]); },
    ack: function (st) { var n = arr(st.narration); return n.length ? n : [LINES.ackDefault]; },
    interstitial: function (course) { return ['Reminder. ' + (course.mantraShort || [course.mantra]).join(' ')]; },
    examResult: function (course, pass) {
      var m = 'Remember: ' + course.mantra;
      return pass ? [LINES.examPass, LINES.examPassNext, m] : [LINES.examFail, LINES.examFailScore, LINES.examRetake, m];
    },
    certificate: function (course) { return ['Congratulations on completing ' + course.title + '.', LINES.certValid, LINES.certReenroll, 'Remember: ' + course.mantra]; },
    hotspot: function (spot) { return arr(spot && spot.narration); },
    cardOpen: function (card) { return arr(card && card.narration); },
    cardCorrect: function (st, card) { return arr(card.narration).concat(S.checkCorrect(st, card)); },
    cardWrong: function (card) { return arr(card.narration).concat(S.checkWrong(card)); },
    spotHit: function (t) { return [LINES.spotFound].concat(splitSentences(stripHtml((t && t.why) || ''))); },
    spotMiss: function (st) { var w = splitSentences(stripHtml(st.missWhy || '')); return w.length ? w : [LINES.spotMiss]; },
    spotAll: function () { return [LINES.spotAll]; },
    stage: function (sg) { return arr(sg && sg.narration); },
    reflectSubmit: function (st) { return [LINES.recorded].concat(arr(st.revealNarration)); },
    sortRight: function () { return [LINES.sortRight]; },
    sortWrong: function (item) { return [LINES.sortWrong].concat(splitSentences(stripHtml((item && item.why) || ''))); },
    sortAll: function () { return [LINES.sortAll]; },
    askQuestion: function (a) { return a && a.question ? [stripHtml(a.question)] : []; },
    askAnswer: function (a, o) {
      var anyCorrect = (a.options || []).some(function (x) { return x.correct === true; });
      var say = splitSentences(stripHtml((o && o.say) || ''));
      if (o && (o.correct === true || !anyCorrect)) return [LINES.askRight].concat(say);
      return say.length ? say : [LINES.askWrong];
    }
  };

  function stepLines(c, st, out) {
    var push = function (a) { for (var i = 0; i < a.length; i++) out.push(a[i]); };
    (st.stages || []).forEach(function (sg) { push(S.stage(sg)); });
    if (st.ask && st.ask.question) { push(S.askQuestion(st.ask)); (st.ask.options || []).forEach(function (o) { push(S.askAnswer(st.ask, o)); }); }
    switch (st.type) {
      case 'reflect': push(S.narration(st)); push(S.reflectSubmit(st)); return;
      case 'sort':
        push(S.narration(st));
        (st.items || []).forEach(function (it) { push(S.sortWrong(it)); });
        push(S.sortRight()); push(S.sortAll());
        return;
      case 'check':
        push(S.checkIntro(st));
        (st.options || []).forEach(function (o) { push(o.correct ? S.checkCorrect(st, o) : S.checkWrong(o)); });
        return;
      case 'ack': push(S.ack(st)); return;
      case 'reveal': push(S.narration(st)); (st.tabs || []).forEach(function (t) { push(arr(t.narration)); }); return;
      case 'hotspots': push(S.narration(st)); (st.spots || []).forEach(function (sp) { push(S.hotspot(sp)); }); return;
      case 'cards':
        push(S.narration(st));
        (st.cards || []).forEach(function (cd) {
          if (st.mode === 'choose') push(cd.correct ? S.cardCorrect(st, cd) : S.cardWrong(cd));
          else push(S.cardOpen(cd));
        });
        return;
      case 'spot':
        push(S.narration(st));
        (st.targets || []).forEach(function (t) { push(S.spotHit(t)); });
        push(S.spotMiss(st)); push(S.spotAll());
        return;
      default: push(S.narration(st));
    }
  }

  /* Every string the engine can pass to the speech layer for this course, de-duplicated after whitespace normalization, in first-use order. */
  function collectSpeech(c) {
    var out = [];
    (c.chapters || []).forEach(function (ch) { (ch.steps || []).forEach(function (st) { stepLines(c, st, out); }); });
    out = out.concat(S.interstitial(c));
    if (c.exam) {
      (c.exam.pool || []).forEach(function (q) { stepLines(c, Object.assign({ type: 'check' }, q), out); });
      out = out.concat(S.examResult(c, true), S.examResult(c, false), S.certificate(c));
    }
    arr(c.speech).forEach(function (l) { out.push(l); });
    /* Widget files under widgets/ register spoken strings per course id: window.LMS_WIDGET_SPEECH[courseId] = [...] */
    arr((G.LMS_WIDGET_SPEECH || {})[c.id]).forEach(function (l) { out.push(l); });
    var seen = {}, res = [];
    out.forEach(function (l) { var k = String(l).replace(/\s+/g, ' ').trim(); if (k && !seen[k]) { seen[k] = 1; res.push(l); } });
    return res;
  }

  /* Cast profile picture for a chat name: exact key in LMS_CAST, then case-insensitive key or alias. */
  function castFor(name) {
    var C = G.LMS_CAST || {};
    if (name == null) return null;
    var n = String(name).trim();
    if (Object.prototype.hasOwnProperty.call(C, n)) return C[n];
    var l = n.toLowerCase();
    for (var k in C) {
      if (!Object.prototype.hasOwnProperty.call(C, k)) continue;
      if (k.toLowerCase() === l) return C[k];
      var al = C[k] && C[k].aliases || [];
      for (var j = 0; j < al.length; j++) if (String(al[j]).toLowerCase() === l) return C[k];
    }
    return null;
  }
  /* Every name a chat step can show (senders, participants, header). */
  function chatNames(st) {
    var out = [];
    if (st.type !== 'chat') return out;
    (st.messages || []).forEach(function (m) { if (m.from) out.push(m.from); });
    ((st.meta || {}).participants || []).forEach(function (p) { out.push(p); });
    if (st.meta && st.meta.header) out.push(st.meta.header);
    return out;
  }
  /* reveal tab media: one { src, alt, caption, href } or an array of up to three */
  function tabMedia(tb) { var m = tb && tb.media; return (Array.isArray(m) ? m : m ? [m] : []).filter(function (x) { return x && x.src; }).slice(0, 3); }
  /* presenter still for host slides: path string or { src } */
  function hostImage(v) { return typeof v === 'string' && v ? { src: v } : v && v.src ? v : null; }
  /* Every image src a course uses (also reveal tab media and presenter stills), (slide, photo, hotspots, cards, spot, check incl. exam pool, video keyframes, stages, cast avatars shown in chats). */
  function collectImages(c) {
    var out = [];
    var add = function (im) { if (im && im.src && out.indexOf(im.src) < 0) out.push(im.src); };
    var step = function (st) {
      chatNames(st).forEach(function (n) { var cc = castFor(n); if (cc && cc.avatar) add({ src: cc.avatar }); });
      (st.stages || []).forEach(function (sg) { add(sg.image); });
      add(st.image);
      (st.cards || []).forEach(function (cd) { add(cd.image); });
      (st.keyframes || []).forEach(function (k) { add(k.image); });
      (st.tabs || []).forEach(function (tb) { tabMedia(tb).forEach(add); });
      if (st.type === 'slide' && st.layout === 'host') add(hostImage(st.hostImg || (c.host && c.host.img)));
    };
    (c.chapters || []).forEach(function (ch) { (ch.steps || []).forEach(step); });
    if (c.exam) (c.exam.pool || []).forEach(step);
    return out;
  }

  /* Presenter cam clip srcs a course plays (course.host.cam on host slides without their own hostImg, step hostCam). */
  function collectHostCams(c) {
    var out = [];
    var add = function (cam) {
      if (!cam || typeof cam !== 'object') return;
      ['idle', 'talk'].forEach(function (k) { [].concat(cam[k] || []).forEach(function (x) { var s = typeof x === 'string' ? x : x && x.src; if (s && out.indexOf(s) < 0) out.push(s); }); });
    };
    var step = function (st) {
      if (st.type !== 'slide' || st.layout !== 'host') return;
      add(st.hostCam !== undefined ? st.hostCam : st.hostImg ? null : c.host && c.host.cam);
    };
    (c.chapters || []).forEach(function (ch) { (ch.steps || []).forEach(step); });
    return out;
  }

  G.LMS_LINES = LINES;
  G.LMS_SPEECH = S;
  G.LMS_stripHtml = stripHtml;
  G.LMS_splitSentences = splitSentences;
  G.LMS_collectSpeech = collectSpeech;
  G.LMS_collectImages = collectImages;
  G.LMS_collectHostCams = collectHostCams;
  G.LMS_castFor = castFor;
  G.LMS_tabMedia = tabMedia;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LMS_LINES: LINES, LMS_SPEECH: S, LMS_stripHtml: stripHtml, LMS_splitSentences: splitSentences, LMS_collectSpeech: collectSpeech, LMS_collectImages: collectImages, LMS_collectHostCams: collectHostCams, LMS_castFor: castFor, LMS_tabMedia: tabMedia };
  }
})(typeof window !== 'undefined' ? window : globalThis);
