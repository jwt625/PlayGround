// node tools/validate.js courses/<id>.js  -- schema check against COURSE_SPEC.md (incl. Tranche 2 additions)
// Errors exit 1. Warnings (prefixed WARN) do not fail the run.
const path = require('path');
const fs = require('fs');
global.window = {};
const file = path.resolve(process.argv[2]);
const courseDir = path.resolve(path.dirname(file), '..'); /* course/ : image src and audio paths are relative to it */
const engineDir = path.resolve(__dirname, '..', 'engine'); /* engine next to this tool; media and audio next to the course file */
const tryLoad = (p) => { if (fs.existsSync(p)) { require(p); return true; } return false; };
tryLoad(path.join(engineDir, 'speech_hash.js'));
const SL = require(path.join(engineDir, 'speech_lines.js'));
const mediaDir = path.join(courseDir, 'media');
const creditFiles = fs.existsSync(mediaDir) ? fs.readdirSync(mediaDir).filter((f) => /^credits_.*\.js$/.test(f)) : [];
creditFiles.forEach((f) => require(path.join(mediaDir, f)));
tryLoad(path.join(courseDir, 'audio', 'manifest.js'));
tryLoad(path.join(mediaDir, 'cast.js'));
tryLoad(path.join(mediaDir, 'sound', 'manifest.js'));
tryLoad(path.join(engineDir, 'sound.js'));
/* widgets/*.js override in-course widgets (window.LMS_WIDGETS[name]) and register their spoken lines (window.LMS_WIDGET_SPEECH[courseId]) */
const vm = require('vm');
const widgetDir = path.join(courseDir, 'widgets');
const widgetFiles = fs.existsSync(widgetDir) ? fs.readdirSync(widgetDir).filter((f) => f.endsWith('.js')).sort() : [];
{
  const noop = () => null;
  const fakeEl = () => ({ style: {}, classList: { add: noop, remove: noop, toggle: noop }, setAttribute: noop, appendChild: noop, append: noop, addEventListener: noop, getContext: () => null });
  const sb = { console, window: global.window, document: { createElement: fakeEl, createElementNS: fakeEl, getElementById: noop, querySelector: noop, querySelectorAll: () => [], head: fakeEl(), body: fakeEl(), documentElement: fakeEl(), addEventListener: noop },
    getComputedStyle: () => ({ getPropertyValue: () => '' }), navigator: { userAgent: 'node' }, location: { search: '', hostname: 'localhost', protocol: 'file:' },
    setTimeout: noop, clearTimeout: noop, setInterval: noop, clearInterval: noop, requestAnimationFrame: noop, cancelAnimationFrame: noop, addEventListener: noop, removeEventListener: noop, Math, JSON };
  vm.createContext(sb);
  /* widget globals land on window: run each file with window as the global object's window property and copy top-level assignments through it */
  widgetFiles.forEach((f) => {
    try { vm.runInContext(fs.readFileSync(path.join(widgetDir, f), 'utf8'), sb, { filename: 'widgets/' + f }); }
    catch (e) { console.log('WARN  widgets/' + f + ' failed to load in node: ' + e.message); }
  });
}
/* KaTeX for math markup checks ($...$ inline, $$...$$ display) */
let katex = null;
try { katex = require(path.join(engineDir, 'vendor', 'katex', 'katex.min.js')); } catch (e) { katex = null; }
const MEDIA = window.LMS_MEDIA || {};
const hashFn = window.LMS_hash || globalThis.LMS_hash;

/* anims/*.js define code-drawn video keyframes (window.LMS_DRAW) used by the course files */
const animDir = path.join(courseDir, 'anims');
if (fs.existsSync(animDir)) fs.readdirSync(animDir).filter((f) => f.endsWith('.js')).sort().forEach((f) => require(path.join(animDir, f)));
require(file);
const ids = Object.keys(window.LMS_COURSES || {});
let errs = 0, warns = 0;
const err = (m) => { errs++; console.log('ERROR', m); };
const warn = (m) => { warns++; console.log('WARN ', m); };
const TYPES = ['slide', 'video', 'reveal', 'chat', 'widget', 'check', 'takeaways', 'ack', 'refs', 'photo', 'hotspots', 'cards', 'spot', 'reflect', 'sort'];
const LAYOUTS = ['title', 'bullets', 'host', 'policy', 'hero', 'split'];
const EMOJI = /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u;
const in01 = (v) => typeof v === 'number' && v >= 0 && v <= 1;
const src = fs.readFileSync(file, 'utf8');

for (const id of ids) {
  const c = window.LMS_COURSES[id];
  ['id', 'code', 'title', 'mantra', 'chapters', 'exam'].forEach((k) => { if (!c[k]) err(id + ' missing ' + k); });
  if (JSON.stringify(c).match(EMOJI)) err(id + ' contains emoji');
  if (c.speech !== undefined && !(Array.isArray(c.speech) && c.speech.every((l) => typeof l === 'string'))) err(id + ' course.speech must be an array of strings');
  let n = 0, words = 0;
  const imgs = new Map(); /* src -> first location */
  const image = (im, where, required) => {
    if (!im) { if (required) err(where + ' needs image: { src }'); return; }
    if (typeof im.src !== 'string' || !im.src) { err(where + ' image.src must be a non-empty string'); return; }
    if (!imgs.has(im.src)) imgs.set(im.src, where);
  };
  /* presenter cam clips (course.host.cam, step hostCam): { idle: src | [src], talk: [src | { src, weight }] } or false */
  const cams = new Map(); /* src -> first location */
  const camAdd = (cam, where) => {
    if (cam === undefined || cam === null || cam === false) return;
    if (typeof cam !== 'object' || Array.isArray(cam)) { err(where + ' must be { idle, talk } or false'); return; }
    Object.keys(cam).forEach((k) => { if (k !== 'idle' && k !== 'talk') err(where + ' unknown key ' + k + ' (idle, talk)'); });
    let count = 0;
    ['idle', 'talk'].forEach((k) => [].concat(cam[k] || []).forEach((x, j) => {
      const s = typeof x === 'string' ? x : x && x.src;
      if (typeof s !== 'string' || !s) { err(where + ' ' + k + '[' + j + '] needs a src'); return; }
      if (x && typeof x === 'object' && x.weight !== undefined && !(typeof x.weight === 'number' && x.weight > 0)) err(where + ' ' + k + '[' + j + '] weight must be a number > 0');
      count++;
      if (!cams.has(s)) cams.set(s, where + ' ' + k);
    }));
    if (!count) err(where + ' has no idle or talk clips');
  };
  camAdd(c.host && c.host.cam, id + ' host.cam');
  const check = (st, where) => {
    if (!TYPES.includes(st.type)) err(where + ' bad type ' + st.type);
    if (st.type === 'check') {
      if (!st.question || !Array.isArray(st.options)) err(where + ' check missing question/options');
      else if (st.options.filter((o) => o.correct).length !== 1) err(where + ' check must have exactly one correct option');
      if (st.image) image(st.image, where, false);
      if (st.kicker !== undefined && typeof st.kicker !== 'string') err(where + ' kicker must be a string');
    }
    if (st.type === 'slide') {
      if (st.layout && !LAYOUTS.includes(st.layout)) err(where + ' bad slide layout ' + st.layout);
      if (st.layout === 'hero' || st.layout === 'split') image(st.image, where, true);
      else if (st.image) image(st.image, where, false);
    }
    if (st.type === 'photo') { image(st.image, where, true); if (st.kb && !(Array.isArray(st.kb) && st.kb.length === 6 && st.kb.every((v) => typeof v === 'number'))) err(where + ' kb must be [x0, y0, s0, x1, y1, s1]'); }
    if (st.type === 'hotspots') {
      image(st.image, where, true);
      if (!['pins', 'numbered', 'callout'].includes(st.style)) err(where + ' hotspots style must be pins / numbered / callout');
      if (!(st.spots && st.spots.length)) err(where + ' hotspots needs spots');
      (st.spots || []).forEach((sp, k) => { if (!in01(sp.x) || !in01(sp.y)) err(where + ' spot ' + k + ' x, y must be 0..1'); if (!sp.label) err(where + ' spot ' + k + ' needs label'); });
    }
    if (st.type === 'cards') {
      if (!['grid', 'flip'].includes(st.style)) err(where + ' cards style must be grid / flip');
      if (!['choose', 'explore'].includes(st.mode)) err(where + ' cards mode must be choose / explore');
      if (!(st.cards && st.cards.length)) err(where + ' cards needs cards');
      (st.cards || []).forEach((cd, k) => { image(cd.image, where + ' card ' + k, true); if (!cd.label) err(where + ' card ' + k + ' needs label'); if (st.mode === 'choose' && !cd.correct && !cd.why) err(where + ' card ' + k + ' wrong card needs why'); });
      if (st.mode === 'choose' && (st.cards || []).filter((cd) => cd.correct).length !== 1) err(where + ' choose cards must have exactly one correct card');
    }
    if (st.type === 'spot') {
      image(st.image, where, true);
      if (!(st.targets && st.targets.length)) err(where + ' spot needs targets');
      (st.targets || []).forEach((t, k) => {
        if (![t.x, t.y, t.w, t.h].every(in01) || t.x + t.w > 1.0001 || t.y + t.h > 1.0001) err(where + ' target ' + k + ' x, y, w, h must be 0..1 and inside the image');
        if (!t.label || !t.why) err(where + ' target ' + k + ' needs label and why');
      });
      if (!st.missWhy) warn(where + ' spot has no missWhy (engine falls back to a generic line)');
    }
    if (st.sfx !== undefined) {
      if (typeof st.sfx !== 'string' || !st.sfx) err(where + ' sfx must be a sound effect name');
      else if (window.LMS_SOUND && window.LMS_SOUND.sfx && Object.keys(window.LMS_SOUND.sfx).length && !(st.sfx in window.LMS_SOUND.sfx)) warn(where + ' sfx "' + st.sfx + '" is not in media/sound/manifest.js');
    }
    if (st.stages !== undefined) {
      if (!Array.isArray(st.stages) || !st.stages.length) err(where + ' stages must be a non-empty array');
      else st.stages.forEach((sg, k) => {
        if (!sg || typeof sg.prompt !== 'string' || !sg.prompt) err(where + ' stage ' + k + ' needs a prompt (button label)');
        if (sg && sg.image) image(sg.image, where + ' stage ' + k, false);
        if (sg && !(sg.body || sg.bullets || sg.image || (sg.narration || []).length)) warn(where + ' stage ' + k + ' reveals nothing');
      });
    }
    if (st.ask !== undefined) {
      const a = st.ask;
      if (!a || typeof a.question !== 'string' || !a.question) err(where + ' ask needs a question');
      if (!a || !Array.isArray(a.options) || a.options.length < 2) err(where + ' ask needs at least two options');
      else a.options.forEach((o, k) => { if (!o.text) err(where + ' ask option ' + k + ' needs text'); if (o.correct !== undefined && typeof o.correct !== 'boolean') err(where + ' ask option ' + k + ' correct must be true or false'); if (o.say !== undefined && typeof o.say !== 'string') err(where + ' ask option ' + k + ' say must be a string'); });
      if (a && a.after !== undefined && !['narration', 'messages'].includes(a.after)) err(where + " ask.after must be 'narration' or 'messages'");
      if (a && a.after === 'messages' && st.type !== 'chat') err(where + " ask.after 'messages' only applies to chat steps");
      /* the narrator introduces every judged scenario before the popup (chat steps: before the messages play) */
      if (!(Array.isArray(st.narration) && st.narration.some((l) => typeof l === 'string' && l.trim()))) err(where + ' ask needs at least one narration line that introduces the scenario before the question');
    }
    if (st.type === 'chat' && st.app === 'email') {
      (st.messages || []).forEach((m, k) => {
        ['to', 'cc'].forEach((f) => { if (m[f] !== undefined && !(Array.isArray(m[f]) && m[f].every((x) => typeof x === 'string'))) err(where + ' message ' + k + ' ' + f + ' must be an array of names'); });
        if (m.date !== undefined && !/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/.test(m.date)) err(where + " message " + k + " date must be 'YYYY-MM-DD HH:MM'");
        if (m.subject !== undefined && typeof m.subject !== 'string') err(where + ' message ' + k + ' subject must be a string');
        if (m.attachments !== undefined && !(Array.isArray(m.attachments) && m.attachments.every((a) => a && typeof a.name === 'string'))) err(where + ' message ' + k + ' attachments must be [{ name, size }]');
      });
    }
    if (st.type === 'reflect') {
      if (!st.prompt) err(where + ' reflect needs prompt');
      if (!['text', 'choice', 'scale'].includes(st.mode)) err(where + ' reflect mode must be text / choice / scale');
      if (st.mode === 'choice' && !(Array.isArray(st.choices) && st.choices.length >= 2)) err(where + ' reflect choice needs at least two choices');
      if (st.scale !== undefined && !(Array.isArray(st.scale) && st.scale.length >= 2)) err(where + ' reflect scale must be an array of at least two labels');
      if (!st.reveal) warn(where + ' reflect has no reveal');
    }
    if (st.type === 'sort') {
      const ids = (st.buckets || []).map((b) => b.id);
      if (ids.length < 2 || new Set(ids).size !== ids.length || (st.buckets || []).some((b) => !b.label)) err(where + ' sort needs at least two buckets with unique id and a label');
      if (!(st.items && st.items.length)) err(where + ' sort needs items');
      (st.items || []).forEach((it, k) => { if (!it.text) err(where + ' sort item ' + k + ' needs text'); if (ids.indexOf(it.bucket) < 0) err(where + ' sort item ' + k + ' bucket not found: ' + it.bucket); if (!it.why) err(where + ' sort item ' + k + ' needs why (read aloud on a wrong drop)'); });
    }
    if (st.type === 'widget' && !(c.widgets && typeof c.widgets[st.widget] === 'function') && !(window.LMS_WIDGETS && typeof window.LMS_WIDGETS[st.widget] === 'function')) err(where + ' widget not found: ' + st.widget);
    if (st.type === 'video') {
      if (!(st.keyframes && st.keyframes.length && st.duration)) err(where + ' video needs keyframes + duration');
      (st.keyframes || []).forEach((k, j) => { if (k.image) image(k.image, where + ' keyframe ' + j, false); });
    }
    if (st.type === 'reveal' && !(st.tabs && st.tabs.length)) err(where + ' reveal needs tabs');
    (st.tabs || []).forEach((tb, k) => {
      if (tb.media === undefined) return;
      const list = Array.isArray(tb.media) ? tb.media : [tb.media];
      if (!list.length || list.length > 3) err(where + ' tab ' + k + ' media must be one object or an array of one to three');
      list.forEach((md, j) => {
        const w = where + ' tab ' + k + ' media ' + j;
        if (!md || typeof md.src !== 'string' || !md.src) { err(w + ' needs src'); return; }
        image(md, w, true);
        if (!md.caption || typeof md.caption !== 'string') err(w + ' needs a caption with the short citation (shown on screen): ' + md.src);
        if (md.href !== undefined && !/^https?:\/\//.test(md.href)) err(w + ' href must be an http(s) link');
        const m = MEDIA[md.src];
        if (m && !m.license) err(w + ' registry entry has no license: ' + md.src);
        if (m && m.licenseUrl !== undefined && !/^https?:\/\//.test(m.licenseUrl)) err(w + ' registry licenseUrl must be an http(s) link: ' + md.src);
        if (m && /^CC BY/.test(m.license || '') && !m.licenseUrl) warn(w + ' CC licensed but the registry entry has no licenseUrl: ' + md.src);
        if (!md.alt && !(m && m.alt)) warn(w + ' has no alt text (media field or registry)');
      });
    });
    if (st.hostImg !== undefined) { const v = typeof st.hostImg === 'string' ? { src: st.hostImg } : st.hostImg; image(v, where + ' hostImg', true); }
    if (st.hostCam !== undefined) { if (st.layout !== 'host') warn(where + ' hostCam only plays on layout host slides'); camAdd(st.hostCam, where + ' hostCam'); }
    if (st.type === 'chat') {
      if (!(st.messages && st.messages.length)) err(where + ' chat needs messages');
      const SK = { phone: ['imessage', 'signal', 'whatsapp'], thread: ['slack', 'teams'], call: ['facetime', 'meeting'], email: ['mail'] };
      const app = st.app || 'phone';
      if (!SK[app]) err(where + ' chat app must be phone / thread / call / email');
      else if (st.skin !== undefined && SK[app].indexOf(st.skin) < 0) err(where + ' chat skin for app ' + app + ' must be one of ' + SK[app].join(' / '));
      if (st.tapToAdvance !== undefined && typeof st.tapToAdvance !== 'boolean') err(where + ' tapToAdvance must be true or false');
    }
    const narrs = [st.narration].concat((st.tabs || []).map((t) => t.narration), (st.spots || []).map((t) => t.narration), (st.cards || []).map((t) => t.narration), (st.stages || []).map((t) => t.narration), [st.revealNarration]);
    narrs.forEach((arr) => (arr || []).forEach((l) => { if (typeof l !== 'string') err(where + ' narration must be strings'); else words += l.split(/\s+/).length; }));
  };
  c.chapters.forEach((ch, ci) => {
    if (!ch.id || !ch.title || !Array.isArray(ch.steps)) err('chapter ' + ci + ' missing id/title/steps');
    (ch.steps || []).forEach((st, si) => { n++; check(st, id + ' ch' + ci + ' step' + si); });
  });
  (c.exam.pool || []).forEach((q, i) => check(Object.assign({ type: 'check' }, q), id + ' exam' + i));

  /* images the engine collects but the checks above do not visit (cast avatars shown in chats, stage images) */
  SL.LMS_collectImages(c).forEach((s) => { if (!imgs.has(s)) imgs.set(s, id + ' (cast avatar or stage)'); });
  /* images: file on disk + registry entry with credit */
  for (const [s, where] of imgs) {
    if (/^(https?:)?\/\//.test(s) || path.isAbsolute(s)) { err(where + ' image must be a relative path under course/: ' + s); continue; }
    if (!fs.existsSync(path.join(courseDir, s))) err(where + ' image file missing: ' + s);
    const m = MEDIA[s];
    if (!m) err(where + ' image has no registry entry (media/credits_*.js): ' + s);
    else if (!m.credit) err(where + ' image registry entry has no credit: ' + s);
  }

  /* presenter cam clips: relative .mp4 under course/, on disk, registry entry with credit and kind 'video' */
  for (const [s, where] of cams) {
    if (/^(https?:)?\/\//.test(s) || path.isAbsolute(s)) { err(where + ' clip must be a relative path under course/: ' + s); continue; }
    if (!/\.mp4$/i.test(s)) err(where + ' clip must be an .mp4 file: ' + s);
    if (!fs.existsSync(path.join(courseDir, s))) err(where + ' clip file missing: ' + s);
    const m = MEDIA[s];
    if (!m) err(where + ' clip has no registry entry (media/credits_*.js): ' + s);
    else { if (!m.credit) err(where + ' clip registry entry has no credit: ' + s); if (m.kind !== 'video') err(where + " clip registry entry needs kind: 'video': " + s); }
  }

  /* sound files the course can play: on disk and credited (warnings: the sound lane may still be delivering) */
  (window.LMS_collectSounds ? window.LMS_collectSounds(c) : []).forEach((f) => {
    if (!fs.existsSync(path.join(courseDir, f))) warn(id + ' sound file missing: ' + f);
    else if (!(MEDIA[f] && MEDIA[f].credit)) warn(id + ' sound file has no credit in the media registry: ' + f);
  });
  /* interactive families: each used at least once per course (spec style rule) */
  const used = new Set();
  c.chapters.forEach((ch) => ch.steps.forEach((st) => used.add(st.type)));
  ['hotspots', 'cards', 'spot'].forEach((t) => { if (!used.has(t)) warn(id + ' does not use step type ' + t + ' yet (spec: each family at least once per course)'); });

  /* widget speech: say(...) / api.say(...) must get a string literal listed in course.speech */
  const speechSet = new Set((Array.isArray(c.speech) ? c.speech : []).map((l) => String(l).replace(/\s+/g, ' ').trim()));
  const re = /(?:\bapi\.say|(?<![\w.])say)\s*\(/g;
  let m2;
  while ((m2 = re.exec(src))) {
    const line = src.slice(0, m2.index).split('\n').length;
    const lineText = src.split('\n')[line - 1];
    if (/function\s+say\s*\(/.test(lineText)) continue; /* the helper that forwards to api.say */
    let rest = src.slice(m2.index + m2[0].length).replace(/^\s*api\s*,\s*/, '');
    const lit = rest.match(/^\s*(['"])((?:\\.|(?!\1).)*)\1\s*\)/);
    if (!lit) { warn(id + ' line ' + line + ': say() with a non-literal argument cannot be pre-recorded: ' + lineText.trim().slice(0, 110)); continue; }
    const text = lit[2].replace(/\\(['"\\])/g, '$1').replace(/\s+/g, ' ').trim();
    if (!speechSet.has(text)) warn(id + ' line ' + line + ': widget line not listed in course.speech: "' + text.slice(0, 80) + '"');
  }

  const lines = SL.LMS_collectSpeech(c);
  /* nothing with math markup may reach the speech layer */
  lines.forEach((l) => { if (String(l).indexOf('$') >= 0) err(id + ' spoken line contains "$" (math markup is for displayed text only; write narration in words): "' + String(l).slice(0, 100) + '"'); });
  /* math markup in displayed text must parse */
  const seenMath = new Set();
  const walk = (v, where) => {
    if (typeof v === 'string') {
      if (v.indexOf('$') < 0 || seenMath.has(v)) return;
      seenMath.add(v);
      const t = v.replace(/<span class="nomath">[^<]*<\/span>/g, '');
      const left = t.replace(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g, (m, d, i) => {
        const tex = d != null ? d : i;
        if (katex) { try { katex.renderToString(tex, { throwOnError: true, displayMode: d != null, strict: 'ignore' }); } catch (e) { err(where + ' math does not parse: $' + tex.slice(0, 60) + '$ (' + String(e.message).slice(0, 90) + ')'); } }
        return '';
      });
      if (left.indexOf('$') >= 0) err(where + ' unbalanced "$" in displayed text (use <span class="nomath">$</span> for a literal dollar sign): "' + v.slice(0, 90) + '"');
      return;
    }
    if (Array.isArray(v)) { v.forEach((x, k) => walk(x, where)); return; }
    if (v && typeof v === 'object') for (const k of Object.keys(v)) { if (typeof v[k] !== 'function') walk(v[k], where); }
  };
  Object.keys(c).forEach((k) => { if (k !== 'chapters' && k !== 'exam' && typeof c[k] !== 'function') walk(c[k], id + '.' + k); });
  c.chapters.forEach((ch, ci) => { Object.keys(ch).forEach((k) => { if (k !== 'steps') walk(ch[k], id + ' ch' + ci + ' ' + k); }); (ch.steps || []).forEach((st, si) => walk(st, id + ' ch' + ci + ' step' + si)); });
  (c.exam.pool || []).forEach((q, i) => walk(q, id + ' exam' + i));
  let rec = 'N/A (no hash function)';
  if (typeof hashFn === 'function') {
    const files = (window.LMS_AUDIO && window.LMS_AUDIO.files) || {};
    rec = lines.filter((l) => Object.prototype.hasOwnProperty.call(files, hashFn(l))).length + ' recorded';
  }
  console.log(id, 'chapters', c.chapters.length, 'steps', n, 'exam pool', (c.exam.pool || []).length, 'narration words', words, '(~' + Math.round(words / 150) + ' min spoken)',
    'speech lines', lines.length, '(' + rec + ')', 'images', imgs.size, 'widgets', Object.keys(c.widgets || {}).join(','));
}
/* widget files: say literals must be registered in window.LMS_WIDGET_SPEECH (or a course.speech list) so they can be pre-recorded */
{
  const reg = new Set();
  Object.values(window.LMS_WIDGET_SPEECH || {}).forEach((a) => (Array.isArray(a) ? a : []).forEach((l) => reg.add(String(l).replace(/\s+/g, ' ').trim())));
  Object.values(window.LMS_COURSES || {}).forEach((c) => (Array.isArray(c.speech) ? c.speech : []).forEach((l) => reg.add(String(l).replace(/\s+/g, ' ').trim())));
  reg.forEach((l) => { if (l.indexOf('$') >= 0) err('widget speech line contains "$": "' + l.slice(0, 100) + '"'); });
  widgetFiles.forEach((f) => {
    const wsrc = fs.readFileSync(path.join(widgetDir, f), 'utf8');
    const re2 = /\bapi\.say\s*\(\s*(['"])((?:\\.|(?!\1).)*)\1\s*\)/g;
    let mm;
    while ((mm = re2.exec(wsrc))) {
      const text = mm[2].replace(/\\(['"\\])/g, '$1').replace(/\s+/g, ' ').trim();
      if (!reg.has(text)) warn('widgets/' + f + ': spoken line not registered in LMS_WIDGET_SPEECH: "' + text.slice(0, 80) + '"');
    }
  });
}
console.log(errs ? errs + ' errors' : 'OK', warns ? '(' + warns + ' warnings)' : '');
process.exit(errs ? 1 : 0);
