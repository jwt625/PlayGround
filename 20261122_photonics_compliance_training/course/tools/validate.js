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
const MEDIA = window.LMS_MEDIA || {};
const hashFn = window.LMS_hash || globalThis.LMS_hash;

require(file);
const ids = Object.keys(window.LMS_COURSES || {});
let errs = 0, warns = 0;
const err = (m) => { errs++; console.log('ERROR', m); };
const warn = (m) => { warns++; console.log('WARN ', m); };
const TYPES = ['slide', 'video', 'reveal', 'chat', 'widget', 'check', 'takeaways', 'ack', 'refs', 'photo', 'hotspots', 'cards', 'spot'];
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
    if (st.type === 'widget' && !(c.widgets && typeof c.widgets[st.widget] === 'function')) err(where + ' widget not found: ' + st.widget);
    if (st.type === 'video') {
      if (!(st.keyframes && st.keyframes.length && st.duration)) err(where + ' video needs keyframes + duration');
      (st.keyframes || []).forEach((k, j) => { if (k.image) image(k.image, where + ' keyframe ' + j, false); });
    }
    if (st.type === 'reveal' && !(st.tabs && st.tabs.length)) err(where + ' reveal needs tabs');
    if (st.type === 'chat' && !(st.messages && st.messages.length)) err(where + ' chat needs messages');
    const narrs = [st.narration].concat((st.tabs || []).map((t) => t.narration), (st.spots || []).map((t) => t.narration), (st.cards || []).map((t) => t.narration));
    narrs.forEach((arr) => (arr || []).forEach((l) => { if (typeof l !== 'string') err(where + ' narration must be strings'); else words += l.split(/\s+/).length; }));
  };
  c.chapters.forEach((ch, ci) => {
    if (!ch.id || !ch.title || !Array.isArray(ch.steps)) err('chapter ' + ci + ' missing id/title/steps');
    (ch.steps || []).forEach((st, si) => { n++; check(st, id + ' ch' + ci + ' step' + si); });
  });
  (c.exam.pool || []).forEach((q, i) => check(Object.assign({ type: 'check' }, q), id + ' exam' + i));

  /* images: file on disk + registry entry with credit */
  for (const [s, where] of imgs) {
    if (/^(https?:)?\/\//.test(s) || path.isAbsolute(s)) { err(where + ' image must be a relative path under course/: ' + s); continue; }
    if (!fs.existsSync(path.join(courseDir, s))) err(where + ' image file missing: ' + s);
    const m = MEDIA[s];
    if (!m) err(where + ' image has no registry entry (media/credits_*.js): ' + s);
    else if (!m.credit) err(where + ' image registry entry has no credit: ' + s);
  }

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
  let rec = 'N/A (no hash function)';
  if (typeof hashFn === 'function') {
    const files = (window.LMS_AUDIO && window.LMS_AUDIO.files) || {};
    rec = lines.filter((l) => Object.prototype.hasOwnProperty.call(files, hashFn(l))).length + ' recorded';
  }
  console.log(id, 'chapters', c.chapters.length, 'steps', n, 'exam pool', (c.exam.pool || []).length, 'narration words', words, '(~' + Math.round(words / 150) + ' min spoken)',
    'speech lines', lines.length, '(' + rec + ')', 'images', imgs.size, 'widgets', Object.keys(c.widgets || {}).join(','));
}
console.log(errs ? errs + ' errors' : 'OK', warns ? '(' + warns + ' warnings)' : '');
process.exit(errs ? 1 : 0);
