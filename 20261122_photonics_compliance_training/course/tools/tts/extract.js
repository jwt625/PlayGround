// node tools/tts/extract.js [--course id] [--chapter chapterId] [--pretty]
// --chapter keeps only that chapter (and an empty exam pool) in each course before collecting, for pilots.
// Prints JSON { source, voice_lines: [{ hash, text, courses: [...] }] } : every string the engine can speak.
// Loads engine/speech_hash.js, engine/speech_lines.js (engine lane; defines LMS_collectSpeech) when present, and
// courses/*.js in a vm with a fake window. If LMS_collectSpeech is not defined yet, falls back to collecting
// narration arrays (steps, tabs, spots, cards), check question / option why texts (as player.js speaks them),
// correctSay, missWhy, exam pool, course.speech, widget say(api, '...') literals, and the fixed engine lines.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..', '..');
const args = process.argv.slice(2);
const only = args.includes('--course') ? args[args.indexOf('--course') + 1] : null;
const chap = args.includes('--chapter') ? args[args.indexOf('--chapter') + 1] : null;

function noop() { return null; }
const fakeEl = () => ({ style: {}, classList: { add: noop, remove: noop }, setAttribute: noop, appendChild: noop, addEventListener: noop, getContext: () => null });
const sandbox = {
  console,
  document: { createElement: fakeEl, getElementById: noop, querySelector: noop, querySelectorAll: () => [], head: fakeEl(), body: fakeEl(), documentElement: fakeEl(), addEventListener: noop },
  getComputedStyle: () => ({ getPropertyValue: () => '' }),
  navigator: { userAgent: 'node' }, location: { search: '', hostname: 'localhost', protocol: 'file:' },
  setTimeout: noop, clearTimeout: noop, setInterval: noop, clearInterval: noop, requestAnimationFrame: noop,
  addEventListener: noop, removeEventListener: noop,
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
const load = (rel) => vm.runInContext(fs.readFileSync(path.join(ROOT, rel), 'utf8'), sandbox, { filename: rel });

load('engine/speech_hash.js');
const hasLines = fs.existsSync(path.join(ROOT, 'engine/speech_lines.js'));
if (hasLines) load('engine/speech_lines.js');
const courseFiles = fs.readdirSync(path.join(ROOT, 'courses')).filter((f) => f.endsWith('.js')).sort();
if (fs.existsSync(path.join(ROOT, 'anims'))) fs.readdirSync(path.join(ROOT, 'anims')).filter((f) => f.endsWith('.js')).sort().forEach((f) => load('anims/' + f));
courseFiles.forEach((f) => load('courses/' + f));
if (fs.existsSync(path.join(ROOT, 'widgets'))) fs.readdirSync(path.join(ROOT, 'widgets')).filter((f) => f.endsWith('.js')).sort().forEach((f) => load('widgets/' + f));
const COURSES = sandbox.LMS_COURSES || {};
const useEngine = typeof sandbox.LMS_collectSpeech === 'function';

/* ---- fallback (mirrors player.js as of 2026-11-22) ---- */
const ENT = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', ndash: '–', mdash: '—', times: '×', deg: '°', micro: 'µ' };
const stripHtml = (s) => String(s || '').replace(/<[^>]*>/g, '').replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (m, e) => {
  if (e[0] === '#') return String.fromCodePoint(e[1] === 'x' || e[1] === 'X' ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10));
  return ENT[e.toLowerCase()] !== undefined ? ENT[e.toLowerCase()] : m;
});
const splitSentences = (s) => (s.match(/[^.!?]+[.!?]+(\s|$)|[^.!?]+$/g) || []).map((x) => x.trim()).filter(Boolean);

function fallbackCollect(course, src) {
  const out = [];
  const add = (t) => { if (typeof t === 'string' && t.trim()) out.push(t); };
  const addN = (n) => { if (Array.isArray(n)) n.forEach(add); else add(n); };
  const mantraBeats = course.mantraShort || [course.mantra];
  add('Reminder. ' + mantraBeats.join(' '));
  function check(st) {
    if (st.narration && st.narration.length) addN(st.narration); else add(stripHtml(st.question));
    (st.options || []).forEach((o) => {
      if (o.correct) { if (st.correctSay) add(st.correctSay); else if (o.why) add(stripHtml(o.why)); }
      else splitSentences(stripHtml(o.why || '')).forEach(add);
    });
  }
  function walk(o) {
    if (Array.isArray(o)) return o.forEach(walk);
    if (!o || typeof o !== 'object') return;
    if (Array.isArray(o.options)) { check(o); }
    else if (o.narration) addN(o.narration);
    if (o.correctSay && !Array.isArray(o.options)) add(o.correctSay);
    if (o.missWhy) splitSentences(stripHtml(o.missWhy)).forEach(add);
    ['tabs', 'spots', 'cards', 'targets', 'keyframes', 'steps'].forEach((k) => {
      if (Array.isArray(o[k])) o[k].forEach((x) => {
        if (x && typeof x === 'object' && x.why && (k === 'cards' || k === 'targets')) splitSentences(stripHtml(x.why)).forEach(add);
        walk(x);
      });
    });
  }
  (course.chapters || []).forEach((ch) => walk(ch.steps));
  ((course.exam && course.exam.pool) || []).forEach(check);
  (course.speech || []).forEach(add);
  // widget literals: say(api, '...') / api.say('...')
  const re = /\bsay\(\s*(?:api\s*,\s*)?('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")\s*\)/g;
  let m;
  while ((m = re.exec(src))) { try { add(vm.runInNewContext(m[1])); } catch (e) { /* ignore */ } }
  return out;
}
const ENGINE_FIXED = [
  "That's right!",
  "Not quite. Let's take a moment to understand why.",
  'Please try again.',
  'Please read the acknowledgment carefully, enter your full legal name, and tick the box to continue.',
];

const byHash = new Map();
const put = (t, cid) => {
  const n = sandbox.LMS_normText(t);
  if (!n) return;
  const h = sandbox.LMS_hash(n);
  if (!byHash.has(h)) byHash.set(h, { hash: h, text: n, courses: [] });
  const e = byHash.get(h);
  if (!e.courses.includes(cid)) e.courses.push(cid);
};
const srcOf = {};
courseFiles.forEach((f) => { const s = fs.readFileSync(path.join(ROOT, 'courses', f), 'utf8'); const id = (s.match(/LMS_COURSES\.(\w+)\s*=/) || [])[1]; if (id) srcOf[id] = s; });
for (const id of Object.keys(COURSES)) {
  if (only && id !== only) continue;
  let c = COURSES[id];
  if (chap) c = Object.assign({}, c, { chapters: c.chapters.filter((ch) => ch.id === chap), exam: Object.assign({}, c.exam, { pool: [] }) });
  const lines = useEngine ? sandbox.LMS_collectSpeech(c) : fallbackCollect(c, chap ? '' : (srcOf[id] || '')).concat(ENGINE_FIXED);
  lines.forEach((t) => put(t, id));
}
// every fixed engine line (LMS_LINES), used or not, so the engine never falls back for its own phrases
if (useEngine && sandbox.LMS_LINES) Object.values(sandbox.LMS_LINES).forEach((t) => put(t, 'engine'));
else ENGINE_FIXED.forEach((t) => put(t, 'engine'));
const voice_lines = [...byHash.values()];
// widget literals passed to say(...) that the collector does not list (they would fall back to browser speech)
const warnings = [];
if (useEngine && !chap) {
  const re = /\bsay\(\s*(?:api\s*,\s*)?('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")\s*\)/g;
  for (const id of Object.keys(srcOf)) {
    if (only && id !== only) continue;
    let m;
    while ((m = re.exec(srcOf[id]))) {
      let t; try { t = vm.runInNewContext(m[1]); } catch (e) { continue; }
      if (!byHash.has(sandbox.LMS_hash(t))) warnings.push(id + ': widget line not in course.speech: ' + t);
    }
  }
  warnings.forEach((w) => process.stderr.write('WARN ' + w + '\n'));
}
process.stdout.write(JSON.stringify({ source: useEngine ? 'LMS_collectSpeech' : 'fallback', speech_lines_js: hasLines, courses: Object.keys(COURSES), warnings, voice_lines }, null, args.includes('--pretty') ? 1 : 0) + '\n');
