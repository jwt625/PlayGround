/* Sound layer: intro / outro music bed and a few subtle sound effects. Contract: course/COURSE_SPEC.md (Sound).
   Files come from window.LMS_SOUND (media/sound/manifest.js); everything is a no-op when a file is missing.
   Music: Web Audio (gapless buffer loop) over http(s), HTMLAudioElement on file://. Effects: HTMLAudioElement. */
(function () {
  'use strict';
  const src = (v) => (v && typeof v === 'object' ? v.src : v) || null;
  const ENGINE_SFX = ['correct', 'wrong', 'reveal', 'message', 'popup', 'sortPlace', 'moduleDone', 'certificate', 'perfectScore'];

  /* opts: { volume(): 0..1, enabled(): bool, paused(): bool } */
  function create(opts) {
    const S = () => window.LMS_SOUND || {};
    const gain = () => Object.assign({ music: 0.35, sfx: 0.6 }, S().gain || {});
    const log = [];
    const note = (kind, name, action, played) => { log.push({ t: Math.round(performance.now()), kind, name, action, played: !!played }); if (log.length > 500) log.shift(); };

    /* ---------- music ----------
       Over http(s) a track is decoded with Web Audio and the AudioBuffer loops gaplessly. On file:// (where fetch of a
       local file is refused) or when decoding fails, an HTMLAudioElement plays it instead. LMS_SOUND.loop[name]
       (default true) decides whether a track loops. */
    let want = null;          /* 'intro' | 'outro' | null */
    let cur = null;           /* track: { name, fade, mode, setVol(v), play(), pause(), isPaused(), stop() } */
    const fading = [];
    let duck = 1, speaking = false, unlockArmed = false, ctx = null;
    const buffers = {};       /* name -> Promise<AudioBuffer | null> */
    const canWebAudio = () => /^https?:$/.test(location.protocol) && !!(window.AudioContext || window.webkitAudioContext) && typeof fetch === 'function';
    const loops = (name) => { const L = S().loop || {}; return L[name] !== false; };
    function getCtx() { if (!ctx) { try { ctx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { ctx = null; } } return ctx; }
    function decode(name, f) {
      if (!buffers[name]) {
        buffers[name] = fetch(f).then((r) => { if (!r.ok) throw new Error('http ' + r.status); return r.arrayBuffer(); })
          .then((ab) => new Promise((res, rej) => getCtx().decodeAudioData(ab, res, rej)))
          .catch(() => { note('music', name, 'decode-failed', false); return null; });
      }
      return buffers[name];
    }
    function elementTrack(name, f) {
      const el = new Audio(f); el.loop = loops(name); el.preload = 'auto'; el.volume = 0;
      el.addEventListener('error', () => note('music', name, 'error', false));
      el.addEventListener('ended', () => note('music', name, 'ended', true));
      return {
        name, fade: 0, mode: 'element', el,
        setVol(v) { el.volume = v; },
        play() { const p = el.play(); if (p && p.catch) p.catch(armUnlock); },
        pause() { el.pause(); },
        isPaused() { return el.paused && !el.ended; },
        stop() { try { el.pause(); el.removeAttribute('src'); el.load(); } catch (e) { /* ignore */ } },
        level() { return el.volume; },
      };
    }
    function bufferTrack(name, f) {
      const c = getCtx();
      const g = c.createGain(); g.gain.value = 0; g.connect(c.destination);
      const t = { name, fade: 0, mode: 'webaudio', ready: false, stopped: false, src: null, started: false, wantPlay: false };
      decode(name, f).then((buf) => {
        if (t.stopped) return;
        if (!buf) { /* fall back to an element */ const e = elementTrack(name, f); Object.assign(t, { mode: 'element', el: e.el, setVol: e.setVol, play: e.play, pause: e.pause, isPaused: e.isPaused, stop: e.stop, level: e.level }); if (t.wantPlay) t.play(); return; }
        const sn = c.createBufferSource(); sn.buffer = buf; sn.loop = loops(name); sn.connect(g);
        sn.onended = () => { if (!t.stopped) note('music', name, 'ended', true); };
        t.src = sn; t.ready = true;
        note('music', name, 'decoded', true);
        if (t.wantPlay) t.play();
      });
      Object.assign(t, {
        setVol(v) { try { g.gain.setTargetAtTime(v, c.currentTime, 0.04); } catch (e) { g.gain.value = v; } },
        play() {
          t.wantPlay = true;
          if (c.state !== 'running') { armUnlock(); c.resume().catch(() => {}); }
          if (t.ready && !t.started) { t.started = true; t.src.start(); }
        },
        pause() { t.wantPlay = false; if (c.state === 'running' && !(cur && cur !== t && cur.mode === 'webaudio' && cur.wantPlay)) c.suspend().catch(() => {}); },
        isPaused() { return !t.wantPlay || c.state !== 'running'; },
        stop() { t.stopped = true; try { if (t.started) t.src.stop(); } catch (e) { /* ignore */ } try { g.disconnect(); } catch (e) { /* ignore */ } },
        level() { return g.gain.value; },
      });
      return t;
    }
    function armUnlock() {
      if (unlockArmed) return;
      unlockArmed = true;
      const u = () => { unlockArmed = false; window.removeEventListener('pointerdown', u, true); window.removeEventListener('keydown', u, true); if (ctx && ctx.state !== 'running' && cur && cur.mode === 'webaudio' && active()) ctx.resume().catch(() => {}); if (cur && active()) cur.play(); };
      window.addEventListener('pointerdown', u, true); window.addEventListener('keydown', u, true);
    }
    const active = () => opts.enabled() && !opts.paused();
    function music(name) {
      if (name === want) return;
      want = name;
      if (cur) { fading.push(cur); note('music', cur.name, 'fadeout', true); cur = null; }
      if (!name) return;
      const f = src((S().music || {})[name]);
      note('music', name, 'start', !!f);
      if (!f) return;
      cur = canWebAudio() && getCtx() ? bufferTrack(name, f) : elementTrack(name, f);
      if (active()) cur.play();
    }
    let last = performance.now();
    function tick() {
      const now = performance.now(), dt = Math.min(0.25, (now - last) / 1000); last = now;
      const target = speaking ? 0.3 : 1;
      duck += (target - duck) * Math.min(1, dt * (speaking ? 9 : 4)); /* duck fast, restore smoothly */
      const on = active();
      if (cur) {
        cur.fade = Math.min(1, cur.fade + dt / 1.0);
        if (!on && !cur.isPaused()) cur.pause();
        if (on && cur.isPaused()) cur.play();
        cur.setVol(Math.max(0, Math.min(1, gain().music * opts.volume() * duck * cur.fade)));
      }
      for (let i = fading.length - 1; i >= 0; i--) {
        const t = fading[i];
        t.fade = Math.max(0, t.fade - dt / 1.5);
        t.setVol(Math.max(0, Math.min(1, gain().music * opts.volume() * duck * t.fade)));
        if (t.fade <= 0 || !on) { t.stop(); fading.splice(i, 1); }
      }
    }
    setInterval(tick, 50);

    /* ---------- sound effects: subtle, at most one per 400 ms ---------- */
    let lastSfx = -1e9;
    function sfx(name) {
      const now = performance.now();
      if (now - lastSfx < 400) { note('sfx', name, 'ratelimited', false); return false; }
      if (!opts.enabled()) { note('sfx', name, 'off', false); return false; }
      lastSfx = now;
      const f = src((S().sfx || {})[name]);
      if (!f) { note('sfx', name, 'nofile', false); return false; }
      const el = new Audio(f);
      el.volume = Math.max(0, Math.min(1, gain().sfx * opts.volume()));
      const p = el.play(); if (p && p.catch) p.catch(() => {});
      note('sfx', name, 'play', true);
      return true;
    }
    return {
      music, sfx, log,
      speaking(on) { speaking = !!on; },
      refresh() { tick(); },
      current: () => want,
      debug: () => ({ want, mode: cur ? cur.mode : null, loop: cur ? (cur.src ? cur.src.loop : cur.el ? cur.el.loop : null) : null, vol: cur ? +cur.level().toFixed(3) : null, paused: cur ? cur.isPaused() : null, fading: fading.length, ctx: ctx ? ctx.state : null }),
    };
  }

  /* Sound files a course can play (engine effects, intro / outro, step `sfx` opt-ins), for the media credits list. */
  function collectSounds(course) {
    const L = window.LMS_SOUND || {};
    const names = new Set(ENGINE_SFX);
    const step = (st) => { if (st && typeof st.sfx === 'string') names.add(st.sfx); };
    (course.chapters || []).forEach((ch) => (ch.steps || []).forEach(step));
    const out = [];
    const add = (f) => { if (f && out.indexOf(f) < 0) out.push(f); };
    add(src((L.music || {}).intro)); add(src((L.music || {}).outro));
    names.forEach((n) => add(src((L.sfx || {})[n])));
    return out;
  }

  window.LMS_SoundLayer = create;
  window.LMS_collectSounds = collectSounds;
  window.LMS_ENGINE_SFX = ENGINE_SFX;
})();
