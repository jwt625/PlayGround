/* LMS parody course player. Classic script, no build. Contract: course/COURSE_SPEC.md */
(function () {
  'use strict';

  const qs = new URLSearchParams(location.search);
  /* Dev mode: ?dev=1 on a dev-allowed host only (engine/config.js). ?review=1 is no longer honored. */
  const DEV = typeof window.LMS_devMode === 'function' ? window.LMS_devMode() : false;
  const CID = qs.get('c');
  const SP = window.LMS_SPEECH;
  const LINES = window.LMS_LINES;
  const stripHtml = window.LMS_stripHtml;
  const splitSentences = window.LMS_splitSentences;
  const AUDIO = window.LMS_AUDIO || { files: {} };
  const clipFor = (text) => {
    if (typeof window.LMS_hash !== 'function' || !AUDIO.files) return null;
    const k = window.LMS_hash(text);
    return Object.prototype.hasOwnProperty.call(AUDIO.files, k) ? { hash: k, sec: +AUDIO.files[k] || 0, url: 'audio/' + k + '.' + (AUDIO.ext || 'mp3') } : null;
  };
  const course = (window.LMS_COURSES || {})[CID];
  const $ = (sel, el) => (el || document).querySelector(sel);
  const h = (tag, attrs, ...kids) => {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      const v = attrs[k];
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v;
      else if (k === 'html') e.innerHTML = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return e;
  };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  /* append, skipping null / false children (Node.append would print them as text) */
  const put = (el, ...kids) => { el.append(...kids.flat().filter((k) => k != null && k !== false)); return el; };
  const fmt = (s) => { s = Math.max(0, Math.floor(s)); const hh = Math.floor(s / 3600), mm = Math.floor(s / 60) % 60, ss = s % 60; return (hh ? String(hh).padStart(2, '0') + ':' : '') + String(mm).padStart(2, '0') + ':' + String(ss).padStart(2, '0'); };
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  /* ---------- storage (per-viewer convenience only) ---------- */
  const KEY = 'lms_parody_' + CID;
  const store = {
    get() { try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { return {}; } },
    set(o) { try { localStorage.setItem(KEY, JSON.stringify(o)); } catch (e) { /* ignore */ } },
  };
  const theme = {
    init() { let t = null; try { t = localStorage.getItem('lms_parody_theme'); } catch (e) {} if (t) document.documentElement.dataset.theme = t; },
    toggle() { const cur = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'; document.documentElement.dataset.theme = cur; try { localStorage.setItem('lms_parody_theme', cur); } catch (e) {} },
  };
  theme.init();

  if (!course) {
    document.body.innerHTML = '<div style="padding:40px">Course not found. <a href="index.html">Return to My Learning</a></div>';
    return;
  }

  /* ---------- state ---------- */
  const saved = store.get();
  const state = {
    completed: new Set(saved.completed || []),
    name: saved.name || '',
    seat: saved.seat || 0,
    pos: 0,
    sess: 0,
    pauseReasons: new Set(),
    lastInput: Date.now(),
    cc: true,
    volume: 1,
  };
  const persist = () => store.set({ completed: [...state.completed], name: state.name, seat: Math.floor(state.seat), pos: state.pos });

  /* ---------- flatten steps ---------- */
  const flat = [];
  course.chapters.forEach((ch, ci) => ch.steps.forEach((st, si) => flat.push({ ci, si, ch, st, id: ch.id + ':' + si })));
  const examChapter = { id: '__exam', title: 'Final Assessment', subtitle: 'Demonstrate your understanding', objectives: ['Answer at least ' + (course.exam ? course.exam.passPct : 80) + ' percent of questions correctly on the first attempt.', 'Receive your certificate of completion.', 'Remember: ' + course.mantra] };
  const chapters = course.chapters.concat(course.exam ? [examChapter] : []);
  const examIdx = course.chapters.length;
  let examSteps = [];
  function buildExam() {
    const pool = (course.exam.pool || []).slice();
    for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
    const n = Math.min(course.exam.count || 10, pool.length);
    examSteps = pool.slice(0, n).map((q, i) => ({ ci: examIdx, si: i, ch: examChapter, st: Object.assign({ type: 'check', exam: true, examNo: i + 1, examOf: n }, q), id: '__exam:' + i }));
    examSteps.push({ ci: examIdx, si: n, ch: examChapter, st: { type: 'examResult', title: 'Assessment Results' }, id: '__exam:result' });
    examSteps.push({ ci: examIdx, si: n + 1, ch: examChapter, st: { type: 'certificate', title: 'Certificate of Completion' }, id: '__exam:cert' });
    exam.answers = {};
  }
  const exam = { answers: {} };
  const allSteps = () => flat.concat(examSteps);
  if (course.exam) buildExam();

  /* ---------- speech: recorded clip (audio/<hash>.mp3) if present, else browser speech, else timed captions ---------- */
  const Speech = (function () {
    const synth = window.speechSynthesis || null;
    const missing = new Map(); /* text -> times spoken without a recorded clip (dev panel) */
    const spoken = new Set(); /* every text passed to the speech layer this session (dev and test checks) */
    let voice = null;
    let curAudio = null, curFinish = null;
    function pickVoice() {
      if (!synth) return;
      const vs = synth.getVoices().filter((v) => /^en/i.test(v.lang));
      const pref = ['Samantha', 'Google US English', 'Microsoft Aria', 'Microsoft Jenny', 'Karen', 'Daniel', 'Moira'];
      voice = null;
      for (const p of pref) { voice = vs.find((v) => v.name.indexOf(p) >= 0); if (voice) break; }
      if (!voice) voice = vs.find((v) => /en-US/i.test(v.lang)) || vs[0] || null;
    }
    if (synth) { pickVoice(); synth.onvoiceschanged = pickVoice; }
    let interrupted = false;
    function paused() { return state.pauseReasons.size > 0; }
    async function waitUnpaused(sess) { while (paused() && sess === state.sess) await sleep(150); }
    function playClip(clip, est) {
      return new Promise((resolve) => {
        const a = new Audio(clip.url);
        a.preload = 'auto';
        a.volume = state.volume;
        let fin = false;
        const start = Date.now();
        const limit = (clip.sec ? clip.sec * 1000 : est) * 2 + 5000;
        const finish = (why) => {
          if (fin) return; fin = true; clearInterval(wd);
          try { a.pause(); a.removeAttribute('src'); a.load(); } catch (e) { /* ignore */ }
          if (curAudio === a) { curAudio = null; curFinish = null; }
          resolve(why);
        };
        curAudio = a; curFinish = finish;
        a.onended = () => finish('end');
        a.onerror = () => finish('err');
        const wd = setInterval(() => {
          if (interrupted) finish('int');
          else if (Date.now() - start > limit) finish('end');
        }, 100);
        let p = null;
        try { p = a.play(); } catch (e) { finish('err'); return; }
        if (p && p.catch) p.catch(() => finish(interrupted ? 'int' : 'err'));
      });
    }
    function once(text) {
      return new Promise((resolve) => {
        const words = text.split(/\s+/).length;
        const est = Math.max(1.2, words / 2.6) * 1000;
        const clip = DEV_MUTE ? null : clipFor(text);
        if (!clip && !DEV_MUTE) { missing.set(text, (missing.get(text) || 0) + 1); devPanelMissing(); }
        if (clip) {
          playClip(clip, est).then((r) => {
            if (r !== 'err') { resolve(r); return; }
            missing.set(text, (missing.get(text) || 0) + 1); devPanelMissing();
            fallback(text, est, resolve);
          });
          return;
        }
        fallback(text, est, resolve);
      });
    }
    function fallback(text, est, resolve) {
      if (!synth || DEV_MUTE) {
        const t0 = Date.now(); let acc = 0; let last = t0;
        const iv = setInterval(() => {
          const now = Date.now();
          if (!paused()) acc += now - last; last = now;
          if (interrupted) { clearInterval(iv); resolve('int'); }
          else if (acc >= est * (DEV_FAST ? 0.25 : 1)) { clearInterval(iv); resolve('end'); }
        }, 100);
        return;
      }
      const u = new SpeechSynthesisUtterance(text);
      if (voice) u.voice = voice;
      u.rate = 1.0; u.pitch = 1.0; u.volume = state.volume;
      let fin = false;
      const finish = (why) => { if (!fin) { fin = true; clearInterval(wd); resolve(why); } };
      u.onend = () => finish(interrupted ? 'int' : 'end');
      u.onerror = () => finish(interrupted ? 'int' : 'end');
      const start = Date.now();
      const wd = setInterval(() => {
        if (interrupted) { finish('int'); return; }
        if (Date.now() - start > est * 3 + 4000) finish('end');
      }, 200);
      synth.speak(u);
    }
    async function speakNow(text, sess) {
      for (;;) {
        await waitUnpaused(sess);
        if (sess !== state.sess) return;
        spoken.add(text);
        setCaption(text);
        interrupted = false;
        const r = await once(text);
        if (sess !== state.sess) return;
        if (r === 'int' && paused()) { continue; } /* blur or idle pause: repeat the sentence from its start */
        return;
      }
    }
    /* One sentence at a time: calls made while a sentence plays wait their turn (browser speech queued the same way). */
    let chain = Promise.resolve();
    function speak(text, sess) {
      if (!text) return Promise.resolve();
      const p = chain.then(() => speakNow(String(text), sess));
      chain = p.catch(() => {});
      return p;
    }
    function interrupt() { interrupted = true; if (curFinish) curFinish('int'); if (synth) synth.cancel(); }
    function setVolume(v) { if (curAudio) curAudio.volume = v; }
    return { speak, interrupt, setVolume, missing, spoken, available: !!synth, recorded: () => !!(AUDIO.files && Object.keys(AUDIO.files).length && typeof window.LMS_hash === 'function') };
  })();
  let DEV_MUTE = false, DEV_FAST = false;

  /* Narration spells acronyms for the speech engine ("D C", "T E"); captions show them joined ("DC", "TE"). */
  const capText = (t) => String(t || '').replace(/\b(?:[A-Z] ){1,5}[A-Z]\b/g, (m) => m.replace(/ /g, ''));
  function setCaption(t) { const c = $('.cc-text'); if (c) { c.textContent = capText(t); c.dataset.raw = t || ''; } }
  /* Natural 300 ms gap between sentences (pause-aware). */
  async function gap(sess) { let acc = 0; const ms = DEV_FAST ? 75 : 300; while (acc < ms) { await sleep(50); if (sess !== state.sess) return; if (state.pauseReasons.size === 0) acc += 50; } }
  async function sayLines(lines, sess) {
    const ls = (lines || []).filter(Boolean);
    for (let i = 0; i < ls.length; i++) {
      if (sess !== state.sess) return false;
      await Speech.speak(ls[i], sess);
      if (i < ls.length - 1) await gap(sess);
    }
    return sess === state.sess;
  }

  /* ---------- pause / enforcement ---------- */
  function addPause(reason) {
    if (state.pauseReasons.has(reason)) return;
    state.pauseReasons.add(reason);
    Speech.interrupt();
  }
  function removePause(reason) { state.pauseReasons.delete(reason); }

  let modalEl = null;
  function modal({ title, body, kind, buttons, onShow }) {
    closeModal();
    const foot = h('div', { class: 'mf' });
    const box = h('div', { class: 'modal', role: 'dialog' }, h('div', { class: 'mh ' + (kind || '') }, title), h('div', { class: 'mb', html: body }), foot);
    (buttons || []).forEach((b) => foot.append(h('button', { class: 'btn ' + (b.primary ? 'primary' : ''), onclick: b.onclick }, b.label)));
    modalEl = h('div', { class: 'modal-back' }, box);
    document.body.append(modalEl);
    if (onShow) onShow(box);
    return box;
  }
  function closeModal() { if (modalEl) { modalEl.remove(); modalEl = null; } }
  let toastT = null;
  function toast(msg) {
    let t = $('.toast'); if (!t) { t = h('div', { class: 'toast' }); document.body.append(t); }
    t.textContent = msg; t.classList.remove('hidden'); clearTimeout(toastT); toastT = setTimeout(() => t.classList.add('hidden'), 3200);
  }

  let started = false;
  function onBlur() {
    if (!started || state.pauseReasons.has('blur')) return;
    addPause('blur');
    modal({
      title: 'Your training has been paused', kind: 'warn',
      body: '<p>This window lost focus. Training content pauses automatically when the course window is not active.</p><p>Time spent outside this window does not count toward your required seat time. For compliance, the current statement will be repeated in full when you resume.</p><p style="color:var(--ink-3);font-size:12px">Reference: ' + esc(course.policy || 'Training Integrity Policy') + ', Section 3.1 (Attentive Presence).</p>',
      buttons: [{ label: 'Resume training', primary: true, onclick: () => { closeModal(); removePause('blur'); } }],
    });
  }
  window.addEventListener('blur', onBlur);
  document.addEventListener('visibilitychange', () => { if (document.hidden) onBlur(); });
  ['mousemove', 'keydown', 'click', 'touchstart'].forEach((ev) => window.addEventListener(ev, () => { state.lastInput = Date.now(); }, { passive: true }));

  setInterval(() => {
    if (!started) return;
    const active = state.pauseReasons.size === 0 && document.hasFocus();
    if (active) { state.seat += 1; const s = $('.seat-v'); if (s) s.textContent = fmt(state.seat); }
    if (Math.floor(state.seat) % 10 === 0) persist();
    if (!DEV && active && Date.now() - state.lastInput > 180000) idleCheck();
  }, 1000);

  function idleCheck() {
    addPause('idle');
    let left = 60;
    const box = modal({
      title: 'Are you still there?', kind: 'warn',
      body: '<p>We have not detected any activity for 3 minutes.</p><p>To confirm your continued attentive presence, click <b>I am still here</b> within <b class="idle-left">60</b> seconds. Otherwise, this slide will restart from the beginning.</p>',
      buttons: [{ label: 'I am still here', primary: true, onclick: () => { clearInterval(iv); closeModal(); state.lastInput = Date.now(); removePause('idle'); } }],
    });
    const iv = setInterval(() => {
      left -= 1; const e = $('.idle-left', box); if (e) e.textContent = left;
      if (left <= 0) {
        clearInterval(iv); closeModal(); state.lastInput = Date.now(); removePause('idle');
        modal({ title: 'Session timed out', kind: 'bad', body: '<p>Your session timed out due to inactivity. For compliance, the current slide will restart from the beginning.</p>', buttons: [{ label: 'Restart slide', primary: true, onclick: () => { closeModal(); go(state.pos, true); } }] });
      }
    }, 1000);
  }

  /* ---------- layout ---------- */
  const mantraBeats = course.mantraShort || [course.mantra];
  document.title = course.title;
  document.body.append(
    h('div', { class: 'topbar' },
      h('div', { class: 'logo' }, 'Intracavity Photonics', h('small', null, 'Learning Center')),
      h('div', { class: 'course-name' }, (course.code ? course.code + '  |  ' : '') + course.title),
      DEV ? h('span', { class: 'review-badge', title: 'Dev mode: gating disabled. Remove ?dev=1 for the learner experience.' }, 'DEV MODE') : null,
      DEV ? h('button', { onclick: () => devPanelToggle(), title: 'Show or hide the developer panel' }, 'Dev panel') : null,
      h('div', { class: 'seat', title: 'Only time with this window focused and playing counts toward the requirement.' }, 'Seat time ', h('span', { class: 'seat-v' }, fmt(state.seat)), ' / required ' + esc(course.requiredSeat || '01:34:00')),
      h('button', { onclick: theme.toggle, title: 'Toggle light / dark' }, 'Theme'),
      h('button', { onclick: () => { if (confirm('Exit the course? Progress on the current slide will not be saved. Partial completion does not satisfy your training requirement.')) location.href = 'index.html'; } }, 'Exit')
    ),
    h('div', { class: 'shell' },
      h('aside', { class: 'sidebar' },
        h('div', { class: 'progress-block' },
          h('h3', { style: 'padding:0 0 6px' }, 'Course progress'),
          h('div', { class: 'progress-bar' }, h('div', { class: 'pbar' })),
          h('div', { class: 'progress-label' }, h('span', { class: 'pct' }, '0% complete'), h('span', null, 'Due 2026-12-31'))
        ),
        h('h3', null, 'Course outline'),
        h('ul', { class: 'toc' }),
        h('div', { class: 'policy-note' }, 'This course is required under ' + (course.policy || 'company policy') + '. Modules unlock in sequence. Content may not be skipped, sped up, or viewed in a background tab.')
      ),
      h('main', { class: 'main' },
        h('div', { class: 'stage-wrap' },
          h('section', { class: 'stage' }, h('div', { class: 'stage-head' }, h('span', { class: 'chap' }), h('h2', { class: 'stitle' })), h('div', { class: 'stage-body' })),
          h('div', { class: 'captions' }, h('span', { class: 'cc-tag' }, 'CC'), h('span', { class: 'cc-text' }))
        ),
        h('div', { class: 'controls' },
          h('button', { class: 'btn back', onclick: () => go(state.pos - 1) }, 'Back'),
          h('span', { class: 'count' }),
          h('button', { class: 'btn small', onclick: replay, title: 'Replay narration for this slide' }, 'Replay audio'),
          h('button', { class: 'btn small', onclick: () => { state.cc = !state.cc; $('.captions').classList.toggle('off', !state.cc); } }, 'CC'),
          h('span', { class: 'speed' }, h('button', { class: 'btn small', onclick: speedMenu }, 'Speed 1.0x')),
          h('label', { class: 'lock-hint', title: 'Minimum volume is enforced. Audio is required for this course.' }, 'Vol ', h('input', { class: 'vol', type: 'range', min: '0.35', max: '1', step: '0.05', value: '1', oninput: (e) => { state.volume = +e.target.value; Speech.setVolume(state.volume); if (+e.target.value <= 0.35) toast('Audio is required for this course. Minimum volume is enforced by your administrator.'); } })),
          DEV ? h('button', { class: 'btn small', onclick: () => { DEV_MUTE = !DEV_MUTE; DEV_FAST = DEV_MUTE; Speech.interrupt(); toast(DEV_MUTE ? 'Dev: voice off, captions at 4x' : 'Dev: voice on'); } }, 'Voice on/off') : null,
          h('span', { class: 'spacer' }),
          h('span', { class: 'lock-hint gate-hint' }),
          h('button', { class: 'btn primary next', onclick: next }, 'Next')
        )
      )
    ),
    h('div', { class: 'ticker' }, h('span', { class: 'lbl' }, 'REMEMBER'), h('div', { class: 'run' }, ...Array.from({ length: 4 }, () => [h('span', null, course.mantra), ...(course.ticker || []).map((t) => h('span', null, t))]).flat()))
  );

  function speedMenu(e) {
    const host = e.target.parentElement;
    const old = $('.speed-menu', host); if (old) { old.remove(); return; }
    const m = h('div', { class: 'speed-menu' },
      ...['0.75x', '1.0x', '1.25x', '1.5x', '2.0x'].map((s) => h('div', { class: s === '1.0x' ? 'on' : '' }, s + (s === '1.0x' ? '  (selected)' : '  (disabled)'))),
      h('div', { class: 'note' }, 'Playback speed is managed by your administrator to ensure full comprehension.'));
    host.append(m);
    setTimeout(() => document.addEventListener('click', function off(ev) { if (!host.contains(ev.target)) { m.remove(); document.removeEventListener('click', off); } }), 0);
  }

  /* ---------- toc ---------- */
  function chapterComplete(ci) {
    if (ci === examIdx) return state.completed.has('__exam:result');
    return flat.filter((f) => f.ci === ci).every((f) => state.completed.has(f.id));
  }
  function chapterUnlocked(ci) { return DEV || ci === 0 || chapterComplete(ci - 1); }
  const ICON = {
    lock: '<svg class="ico" viewBox="0 0 16 16"><rect x="3" y="7" width="10" height="8" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M5 7V5a3 3 0 0 1 6 0v2" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>',
    done: '<svg class="ico" viewBox="0 0 16 16"><rect x="1" y="1" width="14" height="14" fill="var(--ok)"/><path d="M4 8.5l2.5 2.5L12 5.5" fill="none" stroke="#fff" stroke-width="2"/></svg>',
    open: '<svg class="ico" viewBox="0 0 16 16"><rect x="1.5" y="1.5" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>',
  };
  function renderToc() {
    const ul = $('.toc'); ul.innerHTML = '';
    const cur = allSteps()[state.pos];
    chapters.forEach((ch, ci) => {
      const unlocked = chapterUnlocked(ci);
      const ico = chapterComplete(ci) ? ICON.done : unlocked ? ICON.open : ICON.lock;
      const b = h('button', { disabled: !unlocked, title: unlocked ? '' : 'Complete the previous module to unlock', onclick: () => jumpChapter(ci) });
      b.innerHTML = ico + '<span><span class="num">Module ' + ci + '</span>' + esc(ch.title) + '</span>';
      ul.append(h('li', { class: cur && cur.ci === ci ? 'active' : '' }, b));
    });
    const total = flat.length + 1; const done = flat.filter((f) => state.completed.has(f.id)).length + (state.completed.has('__exam:result') ? 1 : 0);
    const pct = Math.round((100 * done) / total);
    $('.pbar').style.width = pct + '%'; $('.pct').textContent = pct + '% complete';
  }
  function jumpChapter(ci) {
    const idx = allSteps().findIndex((f) => f.ci === ci);
    if (idx >= 0) go(idx);
  }

  /* ---------- gating ---------- */
  let gate = { narr: true, req: true };
  let cleanups = [];
  function setGate(g) { gate = Object.assign({ narr: false, req: true }, g); updateNext(); }
  function done(which) { gate[which] = true; updateNext(); }
  function updateNext() {
    const f = allSteps()[state.pos];
    const ok = gate.narr && gate.req;
    if (ok && f) { state.completed.add(f.id); persist(); renderToc(); }
    const already = f && state.completed.has(f.id);
    const btn = $('.next');
    const isLast = state.pos >= allSteps().length - 1;
    btn.disabled = isLast || !(ok || already || DEV);
    btn.classList.toggle('ready', ok && !isLast);
    $('.gate-hint').textContent = ok || already ? '' : !gate.req ? 'Complete the activity to continue' : 'Please listen to the narration in full';
    $('.back').disabled = state.pos === 0;
    btn.textContent = f && allSteps()[state.pos + 1] && allSteps()[state.pos + 1].ci !== f.ci ? 'Next module' : 'Next';
  }
  function next() {
    const steps = allSteps();
    const f = steps[state.pos];
    if (!f) return;
    const n = steps[state.pos + 1];
    if (!n) return;
    if (n.ci !== f.ci) return interstitial(n.ci, () => go(state.pos + 1));
    go(state.pos + 1);
  }
  function interstitial(ci, cont) {
    const ch = chapters[ci];
    Speech.interrupt(); state.sess++;
    let left = DEV ? 0 : 6;
    const box = modal({
      title: 'Module ' + ci + ': ' + ch.title,
      body: '<p style="margin-top:0"><b>Before you continue, a reminder:</b></p><div class="mantra-chips">' + mantraBeats.map((m) => '<span>' + esc(m) + '</span>').join('') + '</div>' +
        '<p><b>In this module you will learn to:</b></p><ul>' + (ch.objectives || []).map((o) => '<li>' + esc(o) + '</li>').join('') + '</ul>',
      buttons: [{ label: 'Continue', primary: true, onclick: () => { if (left > 0) { toast('Please review this reminder in full. ' + left + ' seconds remaining.'); return; } closeModal(); cont(); } }],
    });
    const sess = state.sess;
    sayLines(SP.interstitial(course), sess);
    const btn = $('.mf .btn', box);
    const tick = () => { btn.textContent = left > 0 ? 'Continue (' + left + ')' : 'Continue'; btn.disabled = left > 0; };
    tick();
    const iv = setInterval(() => { left -= 1; tick(); if (left <= 0) clearInterval(iv); }, 1000);
  }
  function replay() { go(state.pos, true); }

  /* ---------- navigation ---------- */
  function go(i, force) {
    const steps = allSteps();
    if (i < 0 || i >= steps.length) return;
    const target = steps[i];
    if (!DEV && !chapterUnlocked(target.ci)) { toast('This module is locked. Complete the previous module first.'); return; }
    cleanups.forEach((fn) => { try { fn(); } catch (e) {} }); cleanups = [];
    Speech.interrupt();
    state.sess++;
    state.pos = i; persist();
    setCaption('');
    const body = $('.stage-body'); body.innerHTML = '';
    $('.chap').textContent = target.ci === examIdx ? 'Final Assessment' : 'Module ' + target.ci + '  |  ' + target.ch.title;
    const st = target.st;
    $('.stitle').textContent = st.title || (st.type === 'check' ? (st.exam ? 'Question ' + st.examNo + ' of ' + st.examOf : 'Knowledge Check') : st.type === 'takeaways' ? 'Key Takeaways' : st.type === 'ack' ? 'Acknowledgment' : st.type === 'refs' ? 'References' : '');
    const n = steps.filter((f) => f.ci === target.ci).length;
    $('.count').textContent = 'Slide ' + (target.si + 1) + ' of ' + n;
    $('.stage').classList.toggle('is-hero', st.type === 'slide' && st.layout === 'hero');
    renderToc();
    const fn = R[st.type] || R.unknown;
    fn(body, st, target, state.sess);
    updateNext();
    $('.stage-wrap').scrollTop = 0;
    preloadStep(steps[i + 1]);
    devPanelUpdate();
  }
  /* Warm the browser cache with the next step's images so full-bleed photos appear without a pop-in. */
  const preloaded = new Set();
  function preloadStep(f) {
    if (!f || !window.LMS_collectImages) return;
    window.LMS_collectImages({ chapters: [{ steps: [f.st] }] }).forEach((src) => { if (preloaded.has(src)) return; preloaded.add(src); const im = new Image(); im.decoding = 'async'; im.src = src; });
  }

  /* ---------- renderers ---------- */
  const R = {};
  const narr = (st, sess) => { setGate({ narr: false, req: true }); sayLines(st.narration, sess).then((ok) => { if (ok) { document.querySelectorAll('.stage-body .bullets li').forEach((li) => li.classList.add('in')); done('narr'); } }); };

  function hostSvg() {
    return '<svg viewBox="0 0 300 400"><defs><linearGradient id="hg" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#2a3a4e"/><stop offset="1" stop-color="#121a24"/></linearGradient></defs>' +
      '<rect width="300" height="400" fill="url(#hg)"/><rect x="0" y="300" width="300" height="100" fill="#3b2f26"/>' +
      '<rect x="190" y="40" width="80" height="110" fill="none" stroke="#4a6380" stroke-width="3"/><path d="M200 140 L220 95 L235 120 L250 80 L262 140Z" fill="#4a6380"/>' +
      '<circle cx="150" cy="140" r="44" fill="#c9a98c"/><path d="M106 132 Q150 70 194 132 Q190 100 150 92 Q110 100 106 132Z" fill="#3a2a20"/>' +
      '<rect x="128" y="176" width="44" height="22" fill="#c9a98c"/><path d="M70 300 Q76 200 150 194 Q224 200 230 300Z" fill="#26476e"/>' +
      '<path d="M138 196 L150 260 L162 196" fill="none" stroke="#d8e0ea" stroke-width="3"/><rect x="141" y="258" width="18" height="24" fill="#f1f1f1" stroke="#999"/>' +
      '<rect x="120" y="128" width="26" height="16" fill="none" stroke="#222" stroke-width="2"/><rect x="154" y="128" width="26" height="16" fill="none" stroke="#222" stroke-width="2"/><line x1="146" y1="136" x2="154" y2="136" stroke="#222" stroke-width="2"/>' +
      '<path d="M135 166 Q150 174 165 166" fill="none" stroke="#7a4a3a" stroke-width="3"/></svg>';
  }
  function bulletsList(items, sess, lines) {
    const ul = h('ul', { class: 'bullets' }, ...(items || []).map((b, i) => h('li', null, h('span', { class: 'b' }, String(i + 1)), h('span', { html: b }))));
    const lis = [...ul.children];
    (async () => {
      if (lines && lines.length >= lis.length && lis.length) {
        /* reveal bullet k when narration line (offset) k starts */
        const off = lines.length - lis.length;
        let k = 0;
        const obs = setInterval(() => {
          if (sess !== state.sess) return clearInterval(obs);
          const cap = $('.cc-text').dataset.raw || '';
          const idx = lines.indexOf(cap);
          if (idx >= 0) { const show = Math.min(lis.length, Math.max(0, idx - off + 1)); for (; k < show; k++) lis[k].classList.add('in'); }
          if (k >= lis.length) clearInterval(obs);
        }, 150);
        cleanups.push(() => clearInterval(obs));
        setTimeout(() => { if (sess === state.sess && k === 0 && (lines.length - lis.length) === 0) { /* first line starts immediately */ } }, 0);
      } else {
        for (const li of lis) { if (sess !== state.sess) return; li.classList.add('in'); await sleep(DEV ? 60 : 900); }
      }
    })();
    if (DEV) lis.forEach((li) => li.classList.add('in'));
    return ul;
  }

  /* ---------- media (registry: window.LMS_MEDIA[src] = { alt, credit, license, source, kind, w, h }) ---------- */
  const mediaInfo = (src) => (window.LMS_MEDIA || {})[src] || null;
  /* <figure> with the image, its credit in small type in a corner, and (dev mode) a path + credit overlay label. */
  function figure(image, opts) {
    opts = opts || {};
    const src = image && image.src;
    const m = mediaInfo(src) || {};
    const im = h('img', { src: src, alt: image.alt || m.alt || '', draggable: 'false', decoding: 'async' });
    const fig = h('figure', { class: 'media ' + (opts.cls || '') });
    fig.append(im);
    if (opts.exact) {
      /* exact aspect so 0..1 coordinates map onto the picture; cap height to the viewport */
      const setAR = (w, hh) => { if (w && hh) { fig.style.aspectRatio = w + ' / ' + hh; fig.style.maxWidth = 'calc(' + (opts.maxVh || 68) + 'vh * ' + (w / hh).toFixed(4) + ')'; } };
      setAR(m.w, m.h);
      if (!m.w || !m.h) im.addEventListener('load', () => setAR(im.naturalWidth, im.naturalHeight));
    }
    if (m.credit) fig.append(h('span', { class: 'credit' + (opts.creditTop ? ' top' : ''), title: [m.credit, m.license].filter(Boolean).join('  |  ') }, m.credit));
    if (DEV) fig.append(h('span', { class: 'devpath' }, src + '  |  ' + (m.credit || 'NO REGISTRY ENTRY')));
    im.addEventListener('error', () => { fig.classList.add('broken'); fig.append(h('span', { class: 'broken-msg' }, 'Image unavailable')); });
    return { fig, img: im };
  }
  /* Ken Burns: kb = [x0, y0, s0, x1, y1, s1] (center 0..1 of the image, scale >= 1); clamped so no edge ever shows. */
  function kbTransform(cx, cy, sc) {
    sc = Math.max(1, +sc || 1);
    const lim = (1 - 1 / sc) / 2;
    const tx = Math.max(-lim, Math.min(lim, 0.5 - (+cx || 0.5))), ty = Math.max(-lim, Math.min(lim, 0.5 - (+cy || 0.5)));
    return 'scale(' + sc.toFixed(4) + ') translate(' + (tx * 100).toFixed(3) + '%, ' + (ty * 100).toFixed(3) + '%)';
  }
  function kenBurns(img, kb, lines) {
    const k = Array.isArray(kb) && kb.length === 6 ? kb : [0.5, 0.5, 1.0, 0.5, 0.5, 1.07];
    const words = (lines || []).join(' ').split(/\s+/).length;
    const ms = Math.min(45000, Math.max(14000, (words / 2.6) * 1000));
    if (!img.animate) return;
    const an = img.animate([{ transform: kbTransform(k[0], k[1], k[2]) }, { transform: kbTransform(k[3], k[4], k[5]) }], { duration: ms, easing: 'ease-in-out', fill: 'forwards' });
    const iv = setInterval(() => { if (state.pauseReasons.size > 0) { if (an.playState === 'running') an.pause(); } else if (an.playState === 'paused') an.play(); }, 250);
    cleanups.push(() => { clearInterval(iv); try { an.cancel(); } catch (e) { /* ignore */ } });
  }
  const ltag = (text, kind) => h('span', { class: 'ltag' + (kind ? ' ' + kind : '') }, text);

  R.slide = (body, st, f, sess) => {
    let layout = st.layout || 'bullets';
    if (st.image && st.image.src && (layout === 'bullets' || layout === 'title') && layout !== 'hero') layout = layout === 'title' ? 'hero' : 'split';
    const callout = st.callout ? h('div', { class: 'callout', html: st.callout }) : null;
    if (layout === 'hero' && st.image) {
      const { fig, img } = figure(st.image, { cls: 'hero-fig', creditTop: true });
      fig.append(h('div', { class: 'hero-band' },
        st.kicker ? ltag(st.kicker, 'on-dark') : ltag(f.ci === examIdx ? 'Final Assessment' : 'Module ' + f.ci, 'on-dark'),
        h('h1', { html: st.heading || st.title || course.title }),
        st.subtitle ? h('p', { html: st.subtitle }) : null));
      put(body, fig, st.bullets || callout ? h('div', { class: 'hero-after' }, st.bullets ? bulletsList(st.bullets, sess, st.narration) : null, callout) : null);
      kenBurns(img, st.image.kb || st.kb, st.narration);
    } else if (layout === 'split' && st.image) {
      const { fig, img } = figure(st.image, { cls: 'split-fig' });
      put(body, h('div', { class: 'split-layout' }, fig, h('div', { class: 'split-text' }, st.kicker ? ltag(st.kicker) : null, st.heading ? h('h3', { class: 'split-h', html: st.heading }) : null, st.bullets ? bulletsList(st.bullets, sess, st.narration) : null, callout)));
      kenBurns(img, st.image.kb || st.kb || [0.5, 0.5, 1.0, 0.5, 0.5, 1.05], st.narration);
    } else if (layout === 'title') {
      put(body, h('div', { class: 'slide-title' }, h('div', { class: 'code' }, (course.code || '') + '  |  Version ' + (course.version || '1.0') + '  |  Effective ' + (course.effective || '')), h('h1', { html: st.heading || course.title }), h('p', { html: st.subtitle || course.subtitle || '' }), st.bullets ? bulletsList(st.bullets, sess, st.narration) : null, callout));
    } else if (layout === 'host') {
      const frame = h('div', { class: 'host-frame', html: hostSvg() }, h('span', { class: 'placeholder-tag' }, 'PRESENTER FOOTAGE PENDING'), h('div', { class: 'lower-third' }, h('b', null, (course.host && course.host.name) || 'Host'), (course.host && course.host.role) || ''));
      put(body, h('div', { class: 'host-layout' }, frame, h('div', null, st.bullets ? bulletsList(st.bullets, sess, st.narration) : null, callout)));
    } else if (layout === 'policy') {
      put(body, h('div', { class: 'policy-doc' }, h('div', { class: 'ph' }, (course.policy || 'POLICY') + '  |  CONTROLLED DOCUMENT  |  DO NOT PARAPHRASE'), st.bullets ? bulletsList(st.bullets, sess, st.narration) : null), callout);
    } else {
      put(body, h('div', { class: 'objectives' }, st.kicker ? h('h3', null, st.kicker) : null, st.bullets ? bulletsList(st.bullets, sess, st.narration) : null, callout));
    }
    narr(st, sess);
  };

  /* photo: full-bleed still with a slow Ken Burns move and a caption bar */
  R.photo = (body, st, f, sess) => {
    const { fig, img } = figure(st.image || {}, { cls: 'photo-fig' });
    put(body, fig, st.caption ? h('div', { class: 'photo-cap' }, h('span', { html: st.caption })) : null);
    kenBurns(img, st.kb || (st.image && st.image.kb), st.narration);
    narr(st, sess);
  };

  /* video placeholder with keyframe animatic, locked scrubber */
  R.video = (body, st, f, sess) => {
    const D = st.duration || 20;
    const kfs = (st.keyframes || []).slice().sort((a, b) => a.t - b.t);
    const canvas = h('canvas');
    const shot = h('div', { class: 'shot' + (DEV ? '' : ' dev-only-hidden') }); /* production shot notes: dev mode only */
    const vcap = h('div', { class: 'vcap' });
    const big = h('button', { class: 'bigplay' }, h('span', null, 'PLAY'));
    const fill = h('div');
    const track = h('div', { class: 'vtrack', title: 'Seeking is disabled for this content', onclick: () => toast('Seeking is disabled for this content. Please watch the video in full.') }, fill, ...kfs.map((k) => h('div', { class: 'kf', style: 'left:' + (100 * k.t / D) + '%' })));
    const tlabel = h('span', null, '00:00 / ' + fmt(D));
    const pp = h('button', null, 'Pause');
    const screen = h('div', { class: 'vscreen' }, canvas, vcap, big);
    put(body, h('div', { class: 'vplayer' }, screen, h('div', { class: 'vbar' }, pp, track, tlabel, h('button', { onclick: (e) => { e.stopPropagation(); toast('Playback speed is locked at 1.0x by your administrator.'); } }, '1.0x'), h('button', { onclick: () => toast('Full screen is unavailable during compliance content.') }, 'Full screen')), shot));
    setGate({ narr: false, req: false });
    big.classList.add('hint');
    let t = 0, playing = false, last = 0, raf = 0, narrStarted = false;
    const ctx = canvas.getContext('2d');
    /* keyframe stills: { image: { src } } drawn cover-fit with a slow push-in */
    const stills = kfs.map((k) => { if (!k.image || !k.image.src) return null; const im = new Image(); im.decoding = 'async'; im.src = k.image.src; return im; });
    function size() { const r = screen.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1; canvas.width = Math.max(320, r.width * dpr); canvas.height = Math.max(180, r.height * dpr); }
    size();
    function draw() {
      const W = canvas.width, H = canvas.height;
      let i = 0; for (let k = 0; k < kfs.length; k++) if (kfs[k].t <= t) i = k;
      const kf = kfs[i] || { shot: 'Scene', caption: '' };
      const t0 = kf.t || 0, t1 = kfs[i + 1] ? kfs[i + 1].t : D;
      const u = Math.min(1, Math.max(0, (t - t0) / Math.max(0.001, t1 - t0)));
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = '#0e1217'; ctx.fillRect(0, 0, W, H);
      const still = stills[i];
      if (still) {
        if (still.complete && still.naturalWidth) {
          const k = 1 + 0.08 * u, iw = still.naturalWidth, ih = still.naturalHeight;
          const sc = Math.max(W / iw, H / ih) * k;
          ctx.drawImage(still, (W - iw * sc) / 2, (H - ih * sc) / 2, iw * sc, ih * sc);
        }
        const m = mediaInfo(kf.image.src) || {};
        const sx = W / 900;
        const lines = [];
        if (m.credit) lines.push(m.credit);
        if (DEV && document.body.classList.contains('dev-overlay')) lines.push(kf.image.src + '  |  ' + (m.credit || 'NO REGISTRY ENTRY'));
        ctx.font = (10 * sx) + 'px ' + css('--font');
        lines.forEach((ln, j) => {
          const tw = ctx.measureText(ln).width;
          const y = 6 * sx + j * 18 * sx;
          ctx.fillStyle = j ? 'rgba(160,20,20,0.85)' : 'rgba(0,0,0,0.55)'; ctx.fillRect(W - tw - 12 * sx, y, tw + 12 * sx, 16 * sx);
          ctx.fillStyle = '#fff'; ctx.fillText(ln, W - tw - 6 * sx, y + 12 * sx);
        });
      } else {
        if (typeof kf.draw === 'function') { try { ctx.save(); kf.draw(ctx, W, H, u); ctx.restore(); } catch (e) { ctx.restore(); } }
        else defaultCard(ctx, W, H, kf, u);
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        const lbl = 'ANIMATIC  |  BLENDER RENDER PENDING  |  KEYFRAME ' + (i + 1) + '/' + kfs.length;
        ctx.font = (11 * W / 900) + 'px ui-monospace, Menlo, monospace';
        ctx.fillStyle = 'rgba(0,0,0,0.7)'; ctx.fillRect(8 * (W / 900), 8 * (W / 900), ctx.measureText(lbl).width + 12 * (W / 900), 22 * (W / 900));
        ctx.fillStyle = '#e3b04b'; ctx.fillText(lbl, 14 * (W / 900), 23 * (W / 900));
      }
      shot.innerHTML = '<b>SHOT ' + (i + 1) + (kf.cast && kf.cast.length ? '  |  CAST: ' + esc(kf.cast.join(', ')) : '') + '</b>' + esc(kf.shot || '');
      vcap.textContent = kf.caption || '';
      fill.style.width = (100 * t / D) + '%';
      tlabel.textContent = fmt(t) + ' / ' + fmt(D);
    }
    function loop(now) {
      const paused = state.pauseReasons.size > 0;
      if (playing && !paused) { t = Math.min(D, t + (now - last) / 1000 * (DEV_FAST ? 4 : 1)); }
      last = now;
      draw();
      if (t >= D) { playing = false; pp.textContent = 'Replay'; done('req'); }
      raf = requestAnimationFrame(loop);
    }
    function play() {
      big.classList.add('hidden'); big.classList.remove('hint');
      if (t >= D) { if (!gate.req) return; t = 0; }
      playing = true; pp.textContent = 'Pause'; removePause('video');
      if (!narrStarted) { narrStarted = true; sayLines(st.narration, sess).then((ok) => { if (ok) done('narr'); }); }
    }
    big.onclick = play;
    pp.onclick = () => { if (playing) { playing = false; pp.textContent = 'Play'; addPause('video'); } else play(); };
    raf = requestAnimationFrame((n) => { last = n; loop(n); });
    const onR = () => size(); window.addEventListener('resize', onR);
    cleanups.push(() => { cancelAnimationFrame(raf); window.removeEventListener('resize', onR); removePause('video'); });
    if (!st.narration || !st.narration.length) done('narr');
  };
  function defaultCard(ctx, W, H, kf, u) {
    const s = W / 900;
    ctx.fillStyle = '#1a222c'; ctx.fillRect(0, H * 0.72, W, H * 0.28);
    ctx.strokeStyle = '#2c3846'; ctx.lineWidth = 1 * s;
    for (let x = 0; x < W; x += 40 * s) { ctx.beginPath(); ctx.moveTo(x, H * 0.72); ctx.lineTo(x - 80 * s, H); ctx.stroke(); }
    const cast = kf.cast && kf.cast.length ? kf.cast : ['Gary'];
    cast.forEach((name, i) => {
      const x = W * (0.25 + (0.5 * (i + 0.5)) / cast.length) + Math.sin((u + i) * 6.28) * 6 * s, y = H * 0.74;
      ctx.fillStyle = name === 'Gary' ? '#c98a4a' : '#6a87a8';
      ctx.fillRect(x - 22 * s, y - 120 * s, 44 * s, 80 * s);
      ctx.beginPath(); ctx.arc(x, y - 140 * s, 20 * s, 0, 6.283); ctx.fill();
      ctx.fillRect(x - 18 * s, y - 40 * s, 14 * s, 40 * s); ctx.fillRect(x + 4 * s, y - 40 * s, 14 * s, 40 * s);
      ctx.fillStyle = '#cfd8e3'; ctx.font = 'bold ' + (13 * s) + 'px sans-serif'; ctx.textAlign = 'center'; ctx.fillText(name, x, y + 18 * s); ctx.textAlign = 'left';
    });
  }

  R.reveal = (body, st, f, sess) => {
    const tabs = st.tabs || [];
    const viewed = new Set();
    let busy = true;
    const panel = h('div', { class: 'reveal-panel' }, h('div', { class: 'empty' }, 'Select each topic above to learn more. You must view all ' + tabs.length + ' topics to continue.'));
    const btns = tabs.map((tb, i) => h('button', { onclick: () => open(i) }, tb.label));
    const must = h('div', { class: 'must' }, 'Viewed 0 of ' + tabs.length);
    put(body, st.prompt ? h('p', { style: 'font-size:15px;margin-top:0', html: st.prompt }) : null, h('div', { class: 'reveal-tabs' }, ...btns), panel, must);
    setGate({ narr: false, req: tabs.length === 0 });
    function pulse() { btns.forEach((b, i) => b.classList.toggle('hint', !busy && !viewed.has(i) && i === [...Array(tabs.length).keys()].find((k) => !viewed.has(k)))); }
    async function open(i) {
      if (busy && !DEV) { toast('Please wait for the current narration to finish.'); return; }
      busy = true; pulse();
      btns.forEach((b, k) => b.classList.toggle('open', k === i));
      panel.innerHTML = tabs[i].body || '';
      const ok = await sayLines(tabs[i].narration || [], sess);
      if (!ok) return;
      viewed.add(i); btns[i].classList.add('done'); busy = false;
      must.textContent = 'Viewed ' + viewed.size + ' of ' + tabs.length + (viewed.size === tabs.length ? '. Thank you for reviewing all topics.' : '');
      pulse();
      if (viewed.size === tabs.length) done('req');
    }
    sayLines(st.narration, sess).then((ok) => { if (ok) { done('narr'); busy = false; pulse(); } });
    if (!st.narration || !st.narration.length) { busy = false; pulse(); }
  };

  const AVC = ['#c0582f', '#2f6fc0', '#3f8f4f', '#8a4fb0', '#b08a2f', '#2f9aa0'];
  const avColor = (name) => { let s = 0; for (const c of String(name)) s += c.charCodeAt(0); return name === 'Gary' ? '#c0582f' : AVC[s % AVC.length]; };
  const initials = (n) => String(n).split(/\s+/).map((w) => w[0]).join('').slice(0, 2).toUpperCase();

  R.chat = (body, st, f, sess) => {
    const app = st.app || 'phone';
    const meta = st.meta || {};
    const msgs = st.messages || [];
    let list, add;
    const tag = h('div', { class: 'scenario-tag' }, 'SCENARIO' + (st.scenarioNo ? ' ' + st.scenarioNo : '') + '  |  HYPOTHETICAL. ANY RESEMBLANCE TO YOUR LAB IS INTENTIONAL.');
    if (app === 'call') {
      const people = meta.participants || [...new Set(msgs.map((m) => m.from))];
      const tiles = {};
      const grid = h('div', { class: 'grid' }, ...people.map((p) => (tiles[p] = h('div', { class: 'tile' }, h('div', { class: 'av', style: 'background:' + avColor(p) }, initials(p)), h('span', { class: 'nm' }, p)))));
      list = h('div', { class: 'chatlog' });
      put(body, tag, h('div', { class: 'call' }, h('div', { class: 'call-top' }, h('span', null, meta.header || 'Weekly sync'), h('span', null, 'Recording  |  ' + people.length + ' participants')), grid, list));
      add = (m) => { Object.values(tiles).forEach((t) => t.classList.remove('speaking')); if (tiles[m.from]) tiles[m.from].classList.add('speaking'); list.append(h('div', null, h('b', null, m.from + ': '), m.text)); list.scrollTop = 1e6; };
    } else if (app === 'email') {
      list = h('div', { class: 'eb' });
      put(body, tag, h('div', { class: 'email' }, h('div', { class: 'eh', html: '<b>Subject:</b> ' + esc(meta.subject || meta.header || '(no subject)') + '<br><b>To:</b> ' + esc((meta.participants || []).join(', ') || 'team') }), list));
      add = (m) => { list.append(h('div', { class: 'em' }, h('div', { class: 'who' }, m.from), h('div', { html: esc(m.text).replace(/\n/g, '<br>') }))); };
    } else if (app === 'thread') {
      list = h('div');
      put(body, tag, h('div', { class: 'thread' }, h('div', { class: 'th-top' }, (/^#/.test(meta.header || '') ? meta.header : '# ' + (meta.header || 'lab-general')), h('small', null, (meta.participants || []).length ? (meta.participants.length + ' members') : '')), list));
      add = (m) => { list.append(h('div', { class: 'tm' }, h('div', { class: 'av', style: 'background:' + avColor(m.from) }, initials(m.from)), h('div', null, h('div', { class: 'who' }, m.from, h('small', null, m.time || '')), h('div', null, m.text)))); };
    } else {
      list = h('div', { class: 'msgs' });
      put(body, tag, h('div', { class: 'phone' }, h('div', { class: 'ph-top' }, meta.header || 'Messages', h('small', null, (meta.participants || []).join(', '))), list));
      add = (m) => { list.append(h('div', { class: 'msg ' + (m.side || 'left') }, m.side === 'right' ? null : h('span', { class: 'who' }, m.from), m.text)); list.scrollTop = 1e6; };
    }
    setGate({ narr: false, req: false });
    (async () => {
      for (const m of msgs) {
        if (sess !== state.sess) return;
        let typing = null;
        if (app === 'phone' || app === 'thread') { typing = h('div', { class: 'typing' }, m.from + ' is typing...'); list.append(typing); list.scrollTop = 1e6; }
        const wait = DEV ? 120 : Math.min(2600, 700 + m.text.length * 22);
        let acc = 0; while (acc < wait) { await sleep(100); if (state.pauseReasons.size === 0) acc += 100; if (sess !== state.sess) return; }
        if (typing) typing.remove();
        add(m);
      }
      if (sess !== state.sess) return;
      put(body, h('div', { class: 'isok' }, st.isok || 'Pause and reflect: Is this OK?'));
      done('req');
      const ok = await sayLines(st.narration, sess);
      if (ok) done('narr');
    })();
  };

  R.check = (body, st, f, sess) => {
    const L = 'ABCDEFGH';
    const opts = st.options || [];
    const fb = h('div', { class: 'kc-fb' });
    const att = h('div', { class: 'kc-attempts' });
    let attempts = 0, locked = false;
    const btns = opts.map((o, i) => h('button', { onclick: () => choose(i) }, h('span', { class: 'l' }, L[i]), h('span', { html: o.text })));
    const optsBox = h('div', { class: 'kc-opts' }, ...btns);
    const parts = [
      h('div', { class: 'kc-label' }, st.exam ? 'FINAL ASSESSMENT  |  QUESTION ' + st.examNo + ' OF ' + st.examOf : 'KNOWLEDGE CHECK  |  THIS QUESTION IS NOT SCORED. ALL ANSWERS ARE VALUED.'),
      st.kicker ? h('div', { class: 'kc-kicker' }, ltag(st.kicker, 'strong')) : null,
      h('div', { class: 'kc-q', html: st.question }), optsBox, fb, att,
    ];
    if (st.image && st.image.src) {
      const { fig, img } = figure(st.image, { cls: 'kc-fig' });
      put(body, h('div', { class: 'kc-split' }, fig, h('div', { class: 'kc-main' }, ...parts)));
      kenBurns(img, st.image.kb || [0.5, 0.5, 1.0, 0.5, 0.5, 1.05], SP.checkIntro(st));
    } else put(body, ...parts);
    setGate({ narr: false, req: false });
    sayLines(SP.checkIntro(st), sess).then((ok) => { if (ok) { done('narr'); optsBox.classList.add('hint'); } });
    async function choose(i) {
      if (locked) { toast('Please listen to the feedback in full before selecting another answer.'); return; }
      const o = opts[i];
      attempts++;
      optsBox.classList.remove('hint');
      if (st.exam && attempts === 1) exam.answers[st.examNo] = !!o.correct;
      Speech.interrupt(); state.sess++; const my = state.sess;
      if (o.correct) {
        locked = true;
        btns.forEach((b, k) => { b.disabled = true; if (k === i) b.classList.add('right'); });
        fb.className = 'kc-fb good'; fb.innerHTML = capText('<h4>' + esc(LINES.right) + '</h4>' + (o.why || st.correctSay || ''));
        att.textContent = attempts === 1 ? 'Correct on the first attempt. Excellent photonic conduct.' : 'Correct after ' + attempts + ' attempts. Every attempt is a learning opportunity.';
        await sayLines(SP.checkCorrect(st, o), my);
        if (my === state.sess) { done('narr'); done('req'); }
      } else {
        locked = true;
        btns[i].classList.add('wrong'); btns[i].disabled = true;
        fb.className = 'kc-fb bad';
        fb.innerHTML = capText('<h4>' + esc(LINES.notQuite) + '</h4>' + (o.why || 'This answer does not reflect the expected photonic conduct.') + '<p style="margin-bottom:0"><i>Please review the explanation, then select another answer.</i></p>');
        att.textContent = 'Attempt ' + attempts + '. You may try again as many times as needed. Remember: ' + course.mantra;
        await sayLines(SP.checkWrong(o), my);
        if (my === state.sess) { locked = false; done('narr'); optsBox.classList.add('hint'); }
      }
    }
  };

  /* hotspots: points to open on a real scene. Three looks: 'pins' (plus markers, popover), 'numbered' (numbered squares + side list), 'callout' (labelled leader lines, panel below) */
  R.hotspots = (body, st, f, sess) => {
    const style = ['pins', 'numbered', 'callout'].indexOf(st.style) >= 0 ? st.style : 'pins';
    const spots = st.spots || [];
    const viewed = new Set();
    let busy = true;
    const { fig } = figure(st.image || {}, { cls: 'hs-fig hs-' + style, exact: true, maxVh: style === 'numbered' ? 64 : 60 });
    const marks = spots.map((sp, i) => {
      const lbl = stripHtml(sp.label || '');
      let inner;
      if (style === 'pins') inner = [h('span', { class: 'pin-head' }, h('span', { class: 'pin-plus' }))];
      else if (style === 'numbered') inner = [h('span', { class: 'num-badge' }, String(i + 1))];
      else inner = [h('span', { class: 'cx-dot' }), h('span', { class: 'cx-line' }), h('span', { class: 'cx-lbl', html: sp.label || '' })];
      const b = h('button', { class: 'hs-mk', style: 'left:' + (100 * (+sp.x || 0)).toFixed(2) + '%;top:' + (100 * (+sp.y || 0)).toFixed(2) + '%', title: lbl, 'aria-label': lbl, onclick: (e) => { e.stopPropagation(); open(i); } }, ...inner);
      if (style === 'callout') { if (sp.x > 0.6) b.classList.add('flipx'); if (sp.y < 0.22) b.classList.add('flipy'); }
      fig.append(b);
      return b;
    });
    const must = h('div', { class: 'must' }, 'Viewed 0 of ' + spots.length);
    const prompt = st.prompt ? h('p', { class: 'prompt', html: st.prompt }) : null;
    let pop = null, rows = null, panel = null;
    if (style === 'pins') {
      pop = h('div', { class: 'hs-pop hidden' });
      fig.append(pop);
      put(body, prompt, fig, must);
    } else if (style === 'numbered') {
      rows = spots.map((sp, i) => h('li', null, h('button', { class: 'hs-row', onclick: () => open(i) }, h('span', { class: 'num-badge' }, String(i + 1)), h('span', { html: sp.label || '' })), h('div', { class: 'hs-body hidden', html: sp.body || '' })));
      put(body, prompt, h('div', { class: 'hs-numbered-layout' }, h('div', { class: 'hs-figwrap' }, fig), h('ol', { class: 'hs-list' }, ...rows)), must);
    } else {
      panel = h('div', { class: 'reveal-panel hs-panel' }, h('div', { class: 'empty' }, 'Select each label on the image to learn more. You must view all ' + spots.length + ' items to continue.'));
      put(body, prompt, fig, panel, must);
    }
    setGate({ narr: false, req: spots.length === 0 });
    function hint() {
      const nextI = spots.findIndex((s, k) => !viewed.has(k));
      marks.forEach((m, k) => m.classList.toggle('hint', !busy && k === nextI));
      if (rows) rows.forEach((r, k) => r.firstChild.classList.toggle('hint', !busy && k === nextI));
    }
    async function open(i) {
      if (busy && !DEV) { toast('Please wait for the current narration to finish.'); return; }
      Speech.interrupt(); state.sess++; const my = state.sess;
      busy = true; hint();
      marks.forEach((m, k) => m.classList.toggle('open', k === i));
      const sp = spots[i];
      if (pop) {
        const narrow = fig.clientWidth < 620;
        pop.innerHTML = '';
        pop.append(h('div', { class: 'hs-pop-h' }, h('span', { html: sp.label || '' }), h('button', { class: 'hs-x', title: 'Close', onclick: (e) => { e.stopPropagation(); pop.classList.add('hidden'); marks[i].classList.remove('open'); } }, 'Close')), h('div', { class: 'hs-pop-b', html: sp.body || '' }));
        pop.classList.remove('hidden');
        pop.classList.toggle('below', narrow);
        if (narrow) { fig.after(pop); pop.removeAttribute('style'); }
        else {
          if (pop.parentNode !== fig) fig.append(pop);
          const left = sp.x < 0.55;
          const top = sp.y < 0.5;
          pop.style.cssText = (left ? 'left:calc(' + (100 * sp.x).toFixed(2) + '% + 26px);' : 'right:calc(' + (100 * (1 - sp.x)).toFixed(2) + '% + 26px);') +
            (top ? 'top:max(8px, calc(' + (100 * sp.y).toFixed(2) + '% - 22px));' : 'bottom:max(8px, calc(' + (100 * (1 - sp.y)).toFixed(2) + '% - 22px));');
        }
      } else if (rows) {
        rows.forEach((r, k) => { r.classList.toggle('open', k === i); r.lastChild.classList.toggle('hidden', k !== i); });
      } else {
        panel.innerHTML = '';
        panel.append(h('h4', { class: 'hs-panel-h', html: sp.label || '' }), h('div', { html: sp.body || '' }));
      }
      const ok = await sayLines(SP.hotspot(sp), my);
      if (!ok) return;
      viewed.add(i); marks[i].classList.add('done'); if (rows) rows[i].classList.add('done');
      busy = false;
      must.textContent = 'Viewed ' + viewed.size + ' of ' + spots.length + (viewed.size === spots.length ? '. Thank you for reviewing every item.' : '');
      hint();
      done('narr');
      if (viewed.size === spots.length) done('req');
    }
    sayLines(SP.narration(st), sess).then((ok) => { if (ok) { done('narr'); busy = false; hint(); } });
  };

  /* cards: 'grid' (photo cards, panel below) or 'flip' (cards turn over). mode 'choose' (one correct, wrong ones struck out after their why is read) or 'explore' (open every card) */
  R.cards = (body, st, f, sess) => {
    const style = st.style === 'flip' ? 'flip' : 'grid';
    const mode = st.mode === 'choose' ? 'choose' : 'explore';
    const cards = st.cards || [];
    const opened = new Set(), struck = new Set();
    let busy = true, finished = false;
    const els = cards.map((cd, i) => {
      const { fig } = figure(cd.image || {}, { cls: 'card-fig' });
      const state1 = h('span', { class: 'card-state' });
      if (style === 'grid') {
        const b = h('button', { class: 'card', onclick: () => pick(i) }, fig, h('div', { class: 'card-lbl', html: cd.label || '' }), state1);
        return { b, state: state1 };
      }
      const back = h('div', { class: 'flip-back' });
      const b = h('button', { class: 'flip', onclick: () => pick(i) }, h('div', { class: 'flip-inner' }, h('div', { class: 'flip-front' }, fig, h('div', { class: 'card-lbl', html: cd.label || '' }), h('span', { class: 'flip-cue' }, 'Select to turn over')), back));
      return { b, back, state: state1 };
    });
    const wrap = h('div', { class: style === 'grid' ? 'cards-grid' : 'cards-flip' }, ...els.map((e) => e.b));
    const fb = h('div', { class: 'kc-fb' });
    const panel = style === 'grid' && mode === 'explore' ? h('div', { class: 'reveal-panel card-panel' }, h('div', { class: 'empty' }, 'Select each card to learn more. You must view all ' + cards.length + ' cards to continue.')) : null;
    const must = h('div', { class: 'must' }, mode === 'choose' ? 'Select one card. You may try again as many times as needed.' : 'Viewed 0 of ' + cards.length);
    put(body, st.prompt ? h('p', { class: 'prompt', html: st.prompt }) : null,
      h('div', { class: 'kc-label' }, mode === 'choose' ? 'SELECT THE BEST RESPONSE' : 'SELECT EACH CARD'), wrap, style === 'grid' && mode === 'choose' ? fb : panel, must);
    setGate({ narr: false, req: cards.length === 0 });
    function hint() {
      if (mode === 'choose') { wrap.classList.toggle('hint', !busy && !finished); return; }
      const nextI = cards.findIndex((c, k) => !opened.has(k));
      els.forEach((e, k) => e.b.classList.toggle('hint', !busy && k === nextI));
    }
    async function pick(i) {
      const cd = cards[i], el = els[i];
      if (mode === 'explore' && style === 'flip' && opened.has(i)) { el.b.classList.toggle('flipped'); return; }
      if (busy && !DEV) { toast('Please wait for the current narration to finish.'); return; }
      if (mode === 'choose' && (finished || struck.has(i))) return;
      Speech.interrupt(); state.sess++; const my = state.sess;
      busy = true; hint();
      if (mode === 'choose') {
        if (cd.correct) {
          finished = true;
          el.b.classList.add('right'); els.forEach((e, k) => { if (k !== i) e.b.disabled = true; });
          if (style === 'grid') { el.state.textContent = 'Correct'; el.b.append(el.state); fb.className = 'kc-fb good'; fb.innerHTML = capText('<h4>' + esc(LINES.right) + '</h4>' + (cd.why || cd.body || st.correctSay || '')); }
          else { el.back.innerHTML = ''; el.back.append(ltag('Correct', 'ok'), h('div', { class: 'flip-lbl', html: cd.label || '' }), h('div', { class: 'flip-txt', html: capText(cd.why || cd.body || st.correctSay || '') })); el.b.classList.add('flipped'); }
          const ok = await sayLines(SP.cardCorrect(st, cd), my);
          if (ok) { done('narr'); done('req'); }
        } else {
          struck.add(i);
          el.b.classList.add('wrong');
          if (style === 'grid') { el.state.textContent = 'Not this one'; el.b.append(el.state); fb.className = 'kc-fb bad'; fb.innerHTML = capText('<h4>' + esc(LINES.notQuite) + '</h4>' + (cd.why || '') + '<p style="margin-bottom:0"><i>Please review the explanation, then select another card.</i></p>'); }
          else { el.back.innerHTML = ''; el.back.append(ltag('Not this one', 'bad'), h('div', { class: 'flip-lbl', html: cd.label || '' }), h('div', { class: 'flip-txt', html: capText(cd.why || '') })); el.b.classList.add('flipped'); }
          const ok = await sayLines(SP.cardWrong(cd), my);
          if (ok) { busy = false; done('narr'); hint(); }
        }
        return;
      }
      els.forEach((e, k) => e.b.classList.toggle('open', k === i));
      if (style === 'grid') { panel.innerHTML = ''; panel.append(h('h4', { class: 'hs-panel-h', html: cd.label || '' }), h('div', { html: cd.body || '' })); }
      else { el.back.innerHTML = ''; el.back.append(h('div', { class: 'flip-lbl', html: cd.label || '' }), h('div', { class: 'flip-txt', html: cd.body || '' })); el.b.classList.add('flipped'); }
      const ok = await sayLines(SP.cardOpen(cd), my);
      if (!ok) return;
      opened.add(i); el.b.classList.add('done');
      if (style === 'grid') { el.state.textContent = 'Viewed'; el.b.append(el.state); }
      busy = false;
      must.textContent = 'Viewed ' + opened.size + ' of ' + cards.length + (opened.size === cards.length ? '. Thank you for reviewing every card.' : '');
      hint(); done('narr');
      if (opened.size === cards.length) done('req');
    }
    sayLines(SP.narration(st), sess).then((ok) => { if (ok) { done('narr'); busy = false; hint(); } });
  };

  /* spot: click the hazards in a real scene. Misses read missWhy; found hazards get a square outline and a label */
  R.spot = (body, st, f, sess) => {
    const targets = st.targets || [];
    const found = new Set();
    let busy = true;
    const { fig } = figure(st.image || {}, { cls: 'spot-fig', exact: true, maxVh: 62 });
    const boxes = targets.map((t, ti) => {
      const pos = 'left:' + (100 * t.x).toFixed(2) + '%;top:' + (100 * t.y).toFixed(2) + '%;width:' + (100 * t.w).toFixed(2) + '%;height:' + (100 * t.h).toFixed(2) + '%';
      if (DEV) fig.append(h('div', { class: 'spot-dev', style: pos }));
      const b = h('div', { class: 'spot-box hidden', style: pos }, h('span', { class: 'spot-tag', title: t.label || '' }, String(ti + 1)));
      if (t.y + t.h > 0.88) b.classList.add('tag-top');
      if (t.x + t.w > 0.72) b.classList.add('tag-right');
      fig.append(b);
      return b;
    });
    const count = h('div', { class: 'spot-count' }, 'Hazards found: 0 of ' + targets.length);
    const items = targets.map((t, i) => h('li', null, h('span', { class: 'spot-n' }, String(i + 1)), h('span', { class: 'spot-l' }, 'Not yet identified')));
    const fb = h('div', { class: 'kc-fb' });
    put(body, st.prompt ? h('p', { class: 'prompt', html: st.prompt }) : null,
      h('div', { class: 'spot-layout' }, h('div', { class: 'spot-figwrap' }, fig), h('div', { class: 'spot-side' }, ltag('Find the hazards', 'strong'), count, h('ol', { class: 'spot-list' }, ...items))), fb);
    setGate({ narr: false, req: targets.length === 0 });
    const hint = () => fig.classList.toggle('hint', !busy && found.size < targets.length);
    fig.addEventListener('click', async (ev) => {
      if (ev.target.closest('button')) return;
      if (busy && !DEV) { toast('Please wait for the current narration to finish.'); return; }
      if (found.size === targets.length) return;
      const r = fig.getBoundingClientRect();
      const x = (ev.clientX - r.left) / r.width, y = (ev.clientY - r.top) / r.height;
      const hit = targets.findIndex((t) => x >= t.x && x <= t.x + t.w && y >= t.y && y <= t.y + t.h);
      if (hit >= 0 && found.has(hit)) { toast('You have already identified this hazard. Please keep looking.'); return; }
      Speech.interrupt(); state.sess++; const my = state.sess;
      busy = true; hint();
      let lines;
      if (hit >= 0) {
        const t = targets[hit];
        found.add(hit);
        boxes[hit].classList.remove('hidden');
        items[hit].classList.add('found'); items[hit].lastChild.innerHTML = t.label || '';
        count.textContent = 'Hazards found: ' + found.size + ' of ' + targets.length;
        fb.className = 'kc-fb good'; fb.innerHTML = capText('<h4>' + esc(LINES.spotFound) + ' ' + (t.label || '') + '</h4>' + (t.why || ''));
        lines = SP.spotHit(t);
        if (found.size === targets.length) lines = lines.concat(SP.spotAll());
      } else {
        const mk = h('span', { class: 'spot-miss', style: 'left:' + (100 * x).toFixed(2) + '%;top:' + (100 * y).toFixed(2) + '%' });
        fig.append(mk); setTimeout(() => mk.remove(), 2600);
        fb.className = 'kc-fb bad'; fb.innerHTML = capText('<h4>Not a hazard</h4>' + (st.missWhy || LINES.spotMiss));
        lines = SP.spotMiss(st);
      }
      const ok = await sayLines(lines, my);
      if (!ok) return;
      busy = false; hint(); done('narr');
      if (found.size === targets.length) done('req');
    });
    sayLines(SP.narration(st), sess).then((ok) => { if (ok) { done('narr'); busy = false; hint(); } });
  };

  R.takeaways = (body, st, f, sess) => {
    put(body, h('div', { class: 'mantra-chips' }, ...mantraBeats.map((m) => h('span', null, m))), bulletsList(st.items || [], sess, st.narration), h('div', { class: 'callout', html: '<b>Remember:</b> ' + esc(course.mantra) }));
    narr(st, sess);
  };

  R.ack = (body, st, f, sess) => {
    const cb = h('input', { type: 'checkbox' });
    const nm = h('input', { type: 'text', placeholder: 'Full legal name', value: state.name });
    const lab = h('label', null, cb, h('span', null, 'I acknowledge that I have read, understood, and will comply with the above. I understand that photons do not accept ignorance as an excuse.'));
    put(body, h('div', { class: 'ack-box' }, h('div', { html: st.text || '' }), h('div', { style: 'margin-top:14px' }, 'Name: ', nm), lab));
    setGate({ narr: false, req: false });
    sayLines(SP.ack(st), sess).then((ok) => { if (ok) { done('narr'); if (!gate.req) lab.classList.add('hint'); } });
    const check = () => { state.name = nm.value.trim(); persist(); if (cb.checked && state.name.length > 1) { lab.classList.remove('hint'); done('req'); } else if (cb.checked) { toast('Please enter your full legal name to acknowledge.'); cb.checked = false; } };
    cb.addEventListener('change', check);
  };

  R.refs = (body, st, f, sess) => {
    put(body, h('p', { style: 'margin-top:0;color:var(--ink-2)' }, 'The following sources informed this module. Reading them is strongly encouraged and, per policy, assumed.'),
      h('ol', { class: 'refs' }, ...(st.items || []).map((r) => h('li', { html: esc(r.cite) + (r.doi ? ' <a target="_blank" rel="noopener" href="https://doi.org/' + esc(r.doi) + '">doi:' + esc(r.doi) + '</a>' : r.url ? ' <a target="_blank" rel="noopener" href="' + esc(r.url) + '">link</a>' : '') }))));
    const imgs = window.LMS_collectImages ? window.LMS_collectImages(course) : [];
    if (imgs.length) {
      put(body, h('h3', { class: 'refs-h' }, 'Image credits'),
        h('ol', { class: 'refs credits' }, ...imgs.map((src) => {
          const m = mediaInfo(src) || {};
          const lic = m.license && String(m.credit || '').toLowerCase().indexOf(String(m.license).toLowerCase()) < 0 ? '. ' + esc(m.license) : '';
          return h('li', { html: (m.alt ? '<span class="cr-alt">' + esc(m.alt) + '.</span> ' : '') + esc(m.credit || 'Credit pending') + lic + (/^https?:\/\//.test(m.source || '') ? ' <a target="_blank" rel="noopener" href="' + esc(m.source) + '">source</a>' : '') });
        })));
    }
    setGate({ narr: true, req: true });
    if (st.narration) narr(st, sess);
  };

  R.widget = (body, st, f, sess) => {
    const root = h('div', { class: 'widget-root' });
    const flag = h('div', { class: 'widget-done hidden' }, 'Activity complete. You may continue.');
    put(body, st.instructions ? h('div', { class: 'widget-instr', html: st.instructions }) : null, root, flag);
    setGate({ narr: false, req: false });
    sayLines(st.narration, sess).then((ok) => { if (ok) done('narr'); });
    const fn = course.widgets && course.widgets[st.widget];
    const api = {
      complete: () => { if (sess === state.sess) { flag.classList.remove('hidden'); done('req'); } },
      /* widgets speak static strings only; list each one in course.speech so it can be pre-recorded */
      say: (t) => Speech.speak(t, sess),
      onCleanup: (fn2) => cleanups.push(fn2),
      css,
      review: DEV,
      dev: DEV,
    };
    if (!fn) { root.append(h('div', { style: 'padding:20px' }, 'Interactive activity "' + st.widget + '" is not available.')); done('req'); return; }
    requestAnimationFrame(() => { try { fn(root, st.params || {}, api); } catch (e) { root.append(h('pre', { style: 'padding:12px;color:var(--bad)' }, 'Widget error: ' + e.message)); console.error(e); done('req'); } });
  };

  R.examResult = (body, st, f, sess) => {
    const n = examSteps.length - 2;
    const right = Object.values(exam.answers).filter(Boolean).length;
    const pct = n ? Math.round((100 * right) / n) : 0;
    const pass = pct >= (course.exam.passPct || 80);
    setGate({ narr: false, req: pass });
    put(body, h('div', { class: 'start' },
      h('h1', null, pass ? 'Congratulations. You passed.' : 'You did not meet the passing score.'),
      h('div', { style: 'font-size:48px;font-weight:700;color:' + (pass ? 'var(--ok)' : 'var(--bad)') }, pct + '%'),
      h('p', null, right + ' of ' + n + ' answered correctly on the first attempt. Passing score: ' + (course.exam.passPct || 80) + '%.'),
      pass ? h('p', null, 'Select Next to receive your certificate.') : h('div', null, h('p', null, 'Good news: you may retake the assessment as many times as needed. Questions are drawn from a randomized pool.'), h('button', { class: 'btn primary', onclick: () => { buildExam(); go(flat.length); } }, 'Retake assessment'))
    ));
    /* spoken lines are fixed sentences; the score itself is shown on screen only */
    sayLines(SP.examResult(course, pass), sess).then((ok) => { if (ok) done('narr'); });
  };

  R.certificate = (body, st, f, sess) => {
    const no = (course.code || 'CERT') + '-' + Math.random().toString(16).slice(2, 10).toUpperCase();
    const today = new Date().toISOString().slice(0, 10);
    put(body, h('div', { class: 'cert' },
      h('div', { style: 'font:12px var(--mono);letter-spacing:2px;color:var(--ink-3)' }, 'INTRACAVITY PHOTONICS, INC.  |  OFFICE OF PHOTONIC CONDUCT'),
      h('h1', null, 'Certificate of Completion'),
      h('div', null, 'This certifies that'),
      h('div', { class: 'nm' }, state.name || 'Valued Team Member'),
      h('div', null, 'has completed'),
      h('h2', { style: 'margin:8px 0;color:var(--ink)' }, course.title),
      h('div', null, (course.code || '') + '  |  Completed ' + today + '  |  Valid for 12 months'),
      h('div', { class: 'mantra-chips', style: 'justify-content:center;margin-top:16px' }, ...mantraBeats.map((m) => h('span', null, m))),
      h('div', { class: 'sig' }, h('div', null, (course.host && course.host.name) || 'Host', h('br'), (course.host && course.host.role) || ''), h('div', null, 'Chief Photonic Conduct Officer')),
      h('div', { class: 'fine' }, 'Certificate No. ' + no + '. This certificate does not certify that any device will work. Photons are not bound by this certificate.')
    ), h('div', { style: 'text-align:center;margin-top:14px' }, h('button', { class: 'btn primary', onclick: () => window.print() }, 'Print / Save as PDF'), ' ', h('a', { class: 'btn', href: 'index.html', style: 'text-decoration:none;display:inline-block' }, 'Return to My Learning')));
    setGate({ narr: false, req: true });
    sayLines(SP.certificate(course), sess).then((ok) => { if (ok) done('narr'); });
  };

  R.unknown = (body, st) => { put(body, h('div', null, 'Unsupported content type: ' + st.type)); setGate({ narr: true, req: true }); };

  /* ---------- start screen ---------- */
  function startScreen() {
    const body = $('.stage-body');
    $('.chap').textContent = course.code || ''; $('.stitle').textContent = 'Welcome';
    $('.count').textContent = '';
    const resumeAt = saved.pos || 0;
    put(body, h('div', { class: 'start' },
      h('div', { style: 'font:600 12px var(--mono);letter-spacing:1px;color:var(--ink-3)' }, 'REQUIRED TRAINING  |  ASSIGNED BY: OFFICE OF PHOTONIC CONDUCT'),
      h('h1', null, course.title),
      h('p', { style: 'margin:0;color:var(--ink-2);font-size:16px' }, course.subtitle || ''),
      h('div', { class: 'meta' },
        h('span', null, 'Course code'), h('b', null, course.code || ''),
        h('span', null, 'Estimated duration'), h('b', null, course.duration || ''),
        h('span', null, 'Version'), h('b', null, (course.version || '') + ' (effective ' + (course.effective || '') + ')'),
        h('span', null, 'Governing policy'), h('b', null, course.policy || ''),
        h('span', null, 'Audio'), h('b', null, 'Required. Please ensure your speakers are on.'),
        h('span', null, 'Narration'), h('b', null, Speech.recorded() ? 'Recorded narration' : Speech.available ? 'Browser voice' : 'Captions only (no speech engine found)')
      ),
      h('div', { class: 'mantra-chips', style: 'justify-content:center' }, ...mantraBeats.map((m) => h('span', null, m))),
      h('div', null,
        h('button', { class: 'btn primary', style: 'font-size:16px;padding:10px 26px', onclick: () => begin(0) }, 'Begin course'),
        resumeAt > 0 ? h('button', { class: 'btn', style: 'margin-left:8px;font-size:16px;padding:10px 20px', onclick: () => begin(resumeAt) }, 'Resume where I left off') : null),
      DEV ? h('p', { style: 'color:var(--warn);font-size:12px' }, 'Dev mode: Next is always enabled, modules are unlocked, idle timeout is off, arrow keys step. "Voice on/off" switches to captions at 4x speed. The dev panel lists every step, missing voice clips, and an image path overlay.') : null
    ));
    $('.next').disabled = true; $('.back').disabled = true;
    renderToc();
  }
  function begin(i) {
    started = true;
    if (window.speechSynthesis) { try { window.speechSynthesis.cancel(); } catch (e) {} }
    go(Math.min(i, allSteps().length - 1));
  }

  document.addEventListener('keydown', (e) => {
    if (!DEV || e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowRight') next();
    if (e.key === 'ArrowLeft') go(state.pos - 1);
  });

  /* ---------- dev panel (dev mode only) ---------- */
  var devEl = null; /* var: hoisted, read by speech callbacks */
  const DEVKEY = 'lms_parody_devpanel';
  function devPanelBuild() {
    if (!DEV) return;
    let collapsed = true; try { collapsed = localStorage.getItem(DEVKEY) !== 'open'; } catch (e) { /* ignore */ } /* collapsed by default; the topbar 'Dev panel' button toggles it */
    const lines = window.LMS_collectSpeech ? window.LMS_collectSpeech(course) : [];
    const rec = lines.filter((l) => clipFor(l)).length;
    const overlay = h('input', { type: 'checkbox', onchange: (e) => document.body.classList.toggle('dev-overlay', e.target.checked) });
    devEl = h('div', { class: 'devpanel' + (collapsed ? ' collapsed' : '') },
      h('div', { class: 'dp-head' }, h('b', null, 'Dev panel'), h('span', { class: 'dp-pos' }), h('span', { style: 'flex:1' }), h('button', { class: 'btn small', onclick: () => devPanelToggle() }, 'Hide')),
      h('div', { class: 'dp-body' },
        h('label', { class: 'dp-row' }, overlay, ' Image path and credit overlay'),
        h('div', { class: 'dp-row dp-dim' }, 'Recorded clips: ' + rec + ' of ' + lines.length + ' course lines' + (typeof window.LMS_hash === 'function' ? '' : ' (hash function not loaded)')),
        h('div', { class: 'dp-h' }, 'Steps'),
        h('div', { class: 'dp-steps' }),
        h('div', { class: 'dp-h' }, 'Spoken without a recorded clip (this session)'),
        h('ol', { class: 'dp-missing' })));
    document.body.append(devEl);
    devPanelSteps();
  }
  function devPanelToggle() {
    if (!devEl) return;
    const c = devEl.classList.toggle('collapsed');
    try { localStorage.setItem(DEVKEY, c ? 'closed' : 'open'); } catch (e) { /* ignore */ }
    devPanelUpdate();
  }
  function stepLabel(st) {
    const t = st.type + (st.layout ? ' ' + st.layout : '') + (st.style ? ' ' + st.style : '') + (st.mode ? ' ' + st.mode : '');
    const title = st.title || (st.question ? stripHtml(st.question) : st.type === 'widget' ? st.widget : '') || '';
    return { t, title: title.length > 70 ? title.slice(0, 68) + '...' : title };
  }
  function devPanelSteps() {
    if (!devEl) return;
    const box = $('.dp-steps', devEl); box.innerHTML = '';
    allSteps().forEach((f, i) => {
      const L = stepLabel(f.st);
      box.append(h('button', { class: 'dp-step', 'data-i': String(i), onclick: () => go(i) },
        h('span', { class: 'dp-i' }, String(i)), h('span', { class: 'dp-m' }, f.ci === examIdx ? 'Exam' : 'M' + f.ci), h('span', { class: 'dp-t' }, L.t), h('span', { class: 'dp-ti' }, L.title)));
    });
  }
  function devPanelUpdate() {
    if (!devEl) return;
    const steps = allSteps();
    if ($('.dp-steps', devEl).children.length !== steps.length) devPanelSteps();
    $('.dp-pos', devEl).textContent = 'Step ' + state.pos + ' of ' + (steps.length - 1);
    devEl.querySelectorAll('.dp-step').forEach((b) => b.classList.toggle('cur', +b.dataset.i === state.pos));
    const cur = $('.dp-step.cur', devEl);
    if (cur && !devEl.classList.contains('collapsed')) { const box = $('.dp-steps', devEl); box.scrollTop = Math.max(0, cur.offsetTop - box.clientHeight / 2); }
  }
  function devPanelMissing() {
    if (!devEl) return;
    const ol = $('.dp-missing', devEl); ol.innerHTML = '';
    [...Speech.missing.entries()].slice(-200).forEach(([t, n]) => ol.append(h('li', { title: window.LMS_hash ? window.LMS_hash(t) : '' }, t + (n > 1 ? '  (x' + n + ')' : ''))));
  }

  startScreen();
  devPanelBuild();
  window.__lms = { go, state, course, allSteps, dev: DEV, speech: Speech };
})();
