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
    certReenroll: 'You will be automatically re-enrolled in eleven months.'
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
    spotAll: function () { return [LINES.spotAll]; }
  };

  function stepLines(c, st, out) {
    var push = function (a) { for (var i = 0; i < a.length; i++) out.push(a[i]); };
    switch (st.type) {
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
    var seen = {}, res = [];
    out.forEach(function (l) { var k = String(l).replace(/\s+/g, ' ').trim(); if (k && !seen[k]) { seen[k] = 1; res.push(l); } });
    return res;
  }

  /* Every image src a course uses (slide, photo, hotspots, cards, spot, check incl. exam pool, video keyframes). */
  function collectImages(c) {
    var out = [];
    var add = function (im) { if (im && im.src && out.indexOf(im.src) < 0) out.push(im.src); };
    var step = function (st) {
      add(st.image);
      (st.cards || []).forEach(function (cd) { add(cd.image); });
      (st.keyframes || []).forEach(function (k) { add(k.image); });
    };
    (c.chapters || []).forEach(function (ch) { (ch.steps || []).forEach(step); });
    if (c.exam) (c.exam.pool || []).forEach(step);
    return out;
  }

  G.LMS_LINES = LINES;
  G.LMS_SPEECH = S;
  G.LMS_stripHtml = stripHtml;
  G.LMS_splitSentences = splitSentences;
  G.LMS_collectSpeech = collectSpeech;
  G.LMS_collectImages = collectImages;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LMS_LINES: LINES, LMS_SPEECH: S, LMS_stripHtml: stripHtml, LMS_splitSentences: splitSentences, LMS_collectSpeech: collectSpeech, LMS_collectImages: collectImages };
  }
})(typeof window !== 'undefined' ? window : globalThis);
