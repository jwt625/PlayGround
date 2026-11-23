# Course content spec (engine contract)

Static, no-build, classic `<script>` files (no ES modules) so `player.html` also works from `file://`.

```
course/
  index.html               LMS dashboard (catalog)
  player.html?c=<id>       course player; ?dev=1 = dev mode on dev hosts only (see Tranche 2)
  engine/lms.css           theme tokens, layout
  engine/player.js         engine (steps, speech, enforcement, quiz, certificate)
  courses/<id>.js          one file per course: data + widgets
  tools/validate.js        node tools/validate.js courses/<id>.js
```

## Course object

```js
window.LMS_COURSES = window.LMS_COURSES || {};
window.LMS_COURSES['<id>'] = {
  id: '<id>', code: 'FE-0042', title: '...', subtitle: '...',
  duration: 'approximately 94 minutes',      // parody: always overstated
  version: '2026.4', effective: '2026-12-01',
  policy: 'Policy PHC-17.3 ...',             // fake internal policy name
  mantra: 'Long mantra sentence.',           // repeated everywhere (ticker, takeaways, interstitials)
  mantraShort: ['Word.', 'Word.', 'Word.'],  // 3-4 beats, shown as chips
  host: { name: 'Pat Reynolds', role: 'Director, Office of Photonic Conduct' },
  chapters: [ { id, title, subtitle, objectives: ['...'], steps: [ Step, ... ] } ],
  exam: { passPct: 80, count: 10, pool: [ CheckStep, ... ] },
  references: [ { cite: 'Author et al., Journal Vol, page (Year)', doi: '10.xxxx/...' } ],
  widgets: { name(root, params, api) { ... } }
};
```

## Step types (every step may have `narration: ['sentence', ...]`, spoken in order; Next stays locked until narration ends and the step's own requirement is met)

| type | fields | requirement to unlock Next |
|---|---|---|
| `slide` | `title`, `layout` ('title' / 'bullets' / 'host' / 'policy'), `bullets[]`, `callout` (optional string) | narration done |
| `video` | `title`, `duration` (s), `keyframes: [{ t, shot, caption, cast: ['Gary'], draw(ctx, w, h, u) }]` (`u` = 0..1 progress inside the keyframe; `draw` optional animatic), `narration` | video timer reaches duration AND narration done |
| `reveal` | `title`, `prompt`, `tabs: [{ label, body (HTML string), narration[] }]` | every tab opened and its narration done |
| `chat` | `title`, `app` ('phone' / 'call' / 'email' / 'thread'), `meta: { header, subject, participants[] }`, `messages: [{ from, text, side: 'left'/'right' }]` | all messages shown + narration done |
| `widget` | `title`, `widget` (key in `widgets`), `params`, `instructions` | widget calls `api.complete()` |
| `check` | `question`, `options: [{ text, correct: true/false, why: 'long explanation' }]`, `correctSay` (optional) | correct option chosen (unlimited retries; wrong options get struck out) |
| `takeaways` | `title`, `items[]` | narration done |
| `ack` | `text` (attestation paragraph) | box ticked |
| `refs` | `items: [{ cite, doi or url }]` | none |

Quiz behavior (engine): correct -> voice "That's right!" + `correctSay`. Wrong -> the long `why` is read aloud in full and shown, the option is struck out and disabled, learner retries. Write `why` texts long, patient, repetitive, and technically correct.

## Widget API

`widgets.name = function (root, params, api) { ... }`
- `root`: empty `<div>` to draw into (use SVG or `<canvas>`; size to `root.clientWidth`).
- `api.complete()`: unlocks Next. Call when the learner has done the required interaction.
- `api.say(text)`: speak and caption a line (returns a Promise).
- `api.onCleanup(fn)`: called when the step is left (stop timers / animation frames).
- `api.css(name)`: read a theme token, e.g. `api.css('--accent')`, so drawings work in light and dark.
- Show an animated hint hand / pulsing outline on the control the learner must use (he wants affordances obvious).

## Content rules

- No emoji. No internal shorthands in anything shown (no cluster ids, no DevLog refs, no file paths). Real citations as "Author et al., Journal, Year" + DOI.
- Physics must be correct. Models in widgets either use real formulas with sourced constants, or are labeled on screen "Illustrative model" with normalized axes. Never invent numbers; a number shown must come from a cited source or a stated formula.
- Parody at maximum: overstated duration, repeated objectives, mantra in every chapter (at least 4 times per chapter), 2-3 more hypothetical scenarios than needed, earnest host voice, corporate euphemisms, fake policy numbers, "Speak up" and "zero tolerance" framing applied to physics.
- Narration in plain spoken English; spell acronyms the way they should be pronounced (TTS reads it), e.g. "V pi", "x-cut".
- Cast: Gary (engineer who does everything wrong), his manager, a bystander coworker, the host. Generic chat apps only, no real product branding.
- Blender scenes are placeholders now: each `video` keyframe has a precise `shot` description (camera, set, props, action) for the later Blender build, and an optional `draw` animatic.

---

# Tranche 2 additions (2026-11-22): dev mode, recorded voice, real media

## Dev mode (replaces `?review=1`)

- `engine/config.js` (loaded first): `window.LMS_CONFIG = { devHosts: ['localhost', '127.0.0.1'], fileDev: true }`.
- Dev mode is ON only if the URL has `?dev=1` AND (`location.hostname` is in `devHosts` OR protocol is `file:` and `fileDev`). On any other host the flag is ignored, so a deployed page never has dev mode. `?review=1` is no longer honored.
- Dev mode = old reviewer mode (Next always enabled, modules unlocked, idle off, voice on/off with 4x captions, arrow keys) plus a dev panel: step jump list (index, chapter, type, title), current step index, overlay toggle that prints each image's path and credit, list of lines with no recorded audio clip. The panel starts collapsed; the topbar "Dev panel" button toggles it (remembered per viewer). Gate functions live in `engine/config.js`: `LMS_isDevHost(loc)`, `LMS_devMode(loc)`.
- Dashboard shows the dev links only when the host is dev-allowed.

## Recorded voice (Kokoro)

- `engine/speech_hash.js` (voice lane; browser globals and node `module.exports`): defines
  - `LMS_normText(t)`: `String(t).replace(/\s+/g, ' ').trim()`
  - `LMS_hash(t)`: on the UTF-8 bytes of `LMS_normText(t)`, FNV-1a 32-bit (prime `0x01000193`) twice, offsets `0x811c9dc5` and `0x050c5d1f`; result = 8 hex chars of the first + 8 hex chars of the second (lowercase, zero padded) = 16 chars.
- `engine/speech_lines.js` (engine lane; loaded after `speech_hash.js`, also `require`-able by node): defines
  - `LMS_LINES` (fixed engine sentences), `LMS_SPEECH` (the line sequence for each engine event; `player.js` speaks exactly these), `LMS_stripHtml`, `LMS_splitSentences` (long `why` / `correctSay` / `missWhy` texts are spoken one sentence per clip), `LMS_collectImages(course)`.
  - `LMS_collectSpeech(course)`: every string the engine can speak for that course (narration of every step and tab and hotspot and card, check `why` texts, `correctSay`, fixed engine lines such as "That's right!", interstitial and blur lines, exam lines, widget lines registered in `course.speech` [array of static strings]). Exactly the strings later passed to the speech layer.
- `audio/manifest.js`: `window.LMS_AUDIO = { voice: 'af_heart', ext: 'mp3', files: { '<hash>': seconds } }`; clips at `audio/<hash>.mp3`.
- Engine: if a clip exists for the text, play it (HTMLAudioElement; pause/resume on blur repeats the sentence from its start); else fall back to speechSynthesis. Widgets must speak only static strings listed in `course.speech` (no numbers spliced in at run time).

## Media registry

- Images live in `media/fe/`, `media/fb/`, `media/shared/`, `media/render/`. JPEG, long side at most 2400 px, under about 700 KB.
- Each lane writes its own `media/credits_<lane>.js`:
  `window.LMS_MEDIA = Object.assign(window.LMS_MEDIA || {}, { 'media/fe/x.jpg': { alt, credit: 'Author, via Wikimedia Commons, CC BY-SA 4.0', license, source: 'https://...', kind: 'photo' | 'render', w, h } })`.
- The engine prints the credit in small type in the image corner from this registry; the references step lists all image credits used by the course.
- People: no identifiable real person may be shown as Gary, his manager, or anyone doing something wrong. Cast members appear only as anonymous figures (bunny suit and mask, from behind, gloved hands, out of focus). Generic identifiable people only in neutral background roles, preferably none.
- No image-generation models. Real photos (public domain or CC licensed, with attribution) and Blender renders only.

## New and extended step types

| type | fields | unlock |
|---|---|---|
| `slide` (extended) | optional `image: { src }`; new layouts `'hero'` (full-bleed photo, title on a dark band) and `'split'` (photo left, bullets right) | narration done |
| `photo` | `title`, `image: { src }`, `caption`, optional `kb: [x0, y0, s0, x1, y1, s1]` (slow Ken Burns from/to: center 0..1, scale), `narration` | narration done |
| `hotspots` | `title`, `prompt`, `image: { src }`, `style: 'pins' / 'numbered' / 'callout'`, `spots: [{ x, y (0..1 of image), label, body (HTML), narration[] }]` | every spot opened and its narration done |
| `cards` | `title`, `prompt`, `style: 'grid' / 'flip'`, `mode: 'choose' / 'explore'`, `cards: [{ image: { src }, label, body, correct, why, narration[] }]`, `correctSay` | choose: correct card picked (wrong cards struck out after their `why` is read, like `check`); explore: every card opened |
| `spot` | `title`, `prompt`, `image: { src }`, `targets: [{ x, y, w, h (0..1), label, why }]`, `missWhy` (read when the click hits no target) | all targets found |
| `check` (extended) | optional `image: { src }` shown beside the question; optional `kicker` (e.g. 'What would you do?', 'You are the bystander') | as before |
| `video` (extended) | a keyframe may have `image: { src }` (still with slow push-in) instead of `draw` | as before |

Engine details (as implemented 2026-11-22):
- Optional extras: `slide` hero shows `kicker` (light-fill tag, default "Module N"), `heading || title`, `subtitle` on the band; split shows `kicker`, `heading`, `bullets`, `callout`. A `bullets` slide with `image` renders as split. Any `image` may carry `kb` (Ken Burns, same format as `photo.kb`); hero / split / photo default to a slow push-in.
- Spoken sequences: check / cards choose wrong = "Not quite. ..." + `why` split into sentences + "Please try again."; correct = "That's right!" + `correctSay` (or the correct item's `why`) split into sentences; a `cards` item's `narration` is spoken before its verdict. `spot` hit = "Hazard identified." + `why` sentences (+ "You have identified every hazard. Thank you for speaking up." after the last); miss = `missWhy` sentences (generic fixed line if absent). Exam result and certificate lines are fixed sentences; the score is shown on screen only.
- `hotspots` / `spot` images keep their true aspect (registry `w`, `h`) so 0..1 coordinates map onto the picture. Keep `callout` spots apart (labels sit up-right of the point, or up-left when `x > 0.6`, below when `y < 0.22`).
- Registry `credit` is printed verbatim in the final disclaimer step's media credits list (and the dev overlay), so put the license in `credit`.

Style rule: the three interactive families (hotspots, cards, spot) must look clearly different from each other and from plain `check`, and each must be used at least once per course.

## Chat skins and cast avatars (engine, 2026-11-22)

- `chat` steps keep `app` and gain optional `skin` (default = first in each list):
  - `app: 'phone'`: `skin: 'imessage' | 'signal' | 'whatsapp'`
  - `app: 'thread'`: `skin: 'slack' | 'teams'` (`slack` also honors `meta.thread: true`: first message in the channel, the rest as replies in a thread pane)
  - `app: 'call'`: `skin: 'facetime' | 'meeting'` (`facetime` = phone video call in the device frame; `meeting` = desktop grid with transcript)
  - `app: 'email'`: `skin: 'mail'`
- Engine file `engine/chat_skins.js` (loaded after `speech_lines.js`). Phone skins render in a CSS device frame (status bar 9:41, dynamic island, home indicator); on phone-width screens only the app screen shows. Desktop skins render in an app window. No real logos or app names on screen.
- Who is "me": senders whose messages have `side: 'right'` (right bubbles on phone skins, right cards in teams). `meta.header` names a group (a generic header such as "Messages" falls back to the participant names); `meta.subject` is the mail subject. Message times are a fixed fictional clock (9:12 AM, two minutes apart).
- `tapToAdvance: true` (optional): after the first message, each message waits for a "Tap to continue" button inside the app frame (or a tap on the message area), then shows its typing indicator. Next unlocks after the last message and the step narration.
- The in-content scenario banner is gone; the final disclaimer step covers dramatization.
- Cast avatars: `media/cast.js` (loaded before the course files) defines `window.LMS_CAST = Object.assign(window.LMS_CAST || {}, { 'Gary': { avatar: 'media/avatars/gary.jpg', aliases: ['Gary M.'] } })`. Lookup by exact `from` / participant / `meta.header` name, then case-insensitive key or alias; initials otherwise. Avatar images must be registered in the media registry (`media/credits_avatars.js` is loaded for that); `LMS_collectImages` includes every avatar a course's chats actually show, so they appear in the final media credits.

## Interactivity: stages, reflect, sort (engine, 2026-11-22)

- `stages` (optional, any step): `stages: [{ prompt: 'See what happened', title?, body? (HTML), bullets?, image?: { src, kb? }, narration: [] }]`.
  Flow: base content and base narration (plus the step's own requirement) first; then a primary button labeled `prompt` appears with the pulsing hint; a click reveals that stage below the earlier ones (stacked, all stay visible) and plays its narration; repeat. An image stage swaps the step's photo when the step has one (`photo`, hero / split `slide`, `check` with image), otherwise it shows its own image in the stage. Next unlocks after the last stage's narration. "Replay audio" replays the current stage. Stage images count toward media credits.
- `chat` `tapToAdvance: true`: see Chat skins above.
- `reflect` (ungraded): `prompt`, `mode: 'text' | 'choice' | 'scale'`, `choices: []` (choice mode), optional `scale: []` labels (default: Strongly disagree, Disagree, Neutral, Agree, Strongly agree), `reveal` (HTML, shown under "What the Office of Photonic Conduct would like you to consider"), `narration[]`, `revealNarration[]`. The learner must respond (text: at least 4 words; or a choice; or a scale point), then Submit shows "Your response has been recorded." (spoken) plus the reveal and its narration. Responses are kept only in this browser's course state. Next after the reveal narration.
- `sort`: `prompt`, `buckets: [{ id, label }]`, `items: [{ text, bucket, why }]`, `narration[]`. Drag an item onto a category, or select an item then a category (click, touch, or Enter / Space). Wrong: the item bounces back to the tray and "Not quite. That item belongs in a different category." + its `why` (split into sentences) is read and shown. Correct: the item locks into the category with "Correct. That item is in the right place." (its `why` is shown). After the last item: "Every item is now in the right place." Next after all items are placed.
- `video`: a REPLAY overlay appears after the end (the bar's Replay button stays).

## Mail layout, ask popup, speech pacing (engine, 2026-11-22)

- `skin: 'mail'` reads like a desktop mail client: folder pane, message list (sender, subject, preview, time; newest on top, active item highlighted, messages arrive one by one as new mail), reading pane with the subject as title, Reply / Reply all / Forward (non-functional; a toast answers), and the conversation oldest first with older messages collapsed (sender, address, one-line preview, date) and the newest expanded: avatar, `Name <address>`, To and Cc lines with addresses, date, attachment chips, body (`\n\n` = new paragraph, `\n` = line break; write greetings and sign-offs in the text), then a signature block (name, title, organization).
- Optional fields on email messages: `to: ['Name', ...]` (default: every other participant), `cc: ['Name', ...]`, `date: 'YYYY-MM-DD HH:MM'` (default: fixed fictional clock 2026-11-16 09:12, two minutes apart), `subject` (default: `meta.subject` for the first message, `'RE: ' + meta.subject` for replies unless it already starts with RE / FW), `attachments: [{ name: 'lot7_vpi_sweep.csv', size: '48 KB' }]`. `meta.subject` stays the thread subject.
- Addresses: `media/cast.js` entries carry `email`, `title`, `org` (company "Five Sigma Photonics, Inc." at the fictional domain `fivesigma.photonics`; vendors and partner labs use their own fictional domains, e.g. `support@lumenfold.lasers`, `sales@northcut.crystal`, `collaborations@partnerlab.photonics`). Display name = cast key without the parenthetical role ("Dana Okafor (Labmate)" shows as "Dana Okafor"). A name without a cast entry gets `firstname.lastname@fivesigma.photonics` derived from the name; people known by one name get `firstname@`.
- `ask` (optional, any step): `ask: { question, options: [{ text, correct?: true|false, say?: 'short spoken response' }], after?: 'narration' | 'messages' }`.
  - After the base narration and requirement (`after: 'narration'`, default), or for chat steps right after the last message with the chat narration played after the answer (`after: 'messages'`), a popup card over the content asks `question` (spoken first) with one button per option.
  - Choice: if it is correct, or no option is marked correct, the engine says "That's right." plus the option's `say`; if wrong, it says the option's `say` (or "Let's look at this more closely." when absent), marks the choice and the correct option, and continues (no retry).
  - Then the popup closes and the first stage reveals automatically if the step has `stages` (stage buttons stay hidden until the ask is answered); otherwise Next unlocks.
  - Under chats the old "Pause and reflect: Is this OK?" box shows only when the step has neither `ask` nor `stages`.
- Pacing: 300 ms between sentences; 1.2 s after a sentence ending in "?" when more narration follows in the same sequence; 600 ms before the first stage's narration.

## Sound: music bed and sound effects (engine, 2026-11-22)

- Files: `media/sound/manifest.js` defines `window.LMS_SOUND = { music: { intro, outro }, sfx: { correct, wrong, reveal, message, popup, sortPlace, moduleDone, certificate, perfectScore, memeBoom, sadTrombone, ... }, gain: { music, sfx } }` (paths relative to `course/`, or `{ src }`). A missing file or a `null` entry is silent. Credits: register each sound file in the media registry (`media/credits_sound.js`, same shape as image entries, keyed by path); the disclaimer step lists them under "Music and sound effects".
- Engine file `engine/sound.js`. Music over http(s) is fetched and decoded with Web Audio and the AudioBuffer loops gaplessly; on file:// (local fetch is refused there) or if decoding fails it falls back to an HTMLAudioElement. `LMS_SOUND.loop: { intro: true, outro: false }` decides looping per track (default true). Effects use HTMLAudioElement.
- Music: the intro bed loops on the start screen and Module 0's opening slides (every step before the first non-slide step of Module 0), fading in over about 1 s and out over about 1.5 s; the outro loops from the certificate through the disclaimer. Music ducks to about 30 percent while narration speaks and recovers smoothly between sentences; it pauses with the blur pause and follows the volume slider.
- Engine effects (subtle, at most one effect per 400 ms): `correct` (check, cards, ask, spot hit), `wrong` (check, cards, ask, spot miss, sort), `reveal` (stage reveal), `message` (chat message from someone other than the learner), `popup` (ask popup), `sortPlace` (sort item placed), `moduleDone` (module interstitial), `certificate` (certificate step), `perfectScore` (exam passed at 100 percent).
- Opt-in effect on any step: `sfx: '<key>'`, any key present in `LMS_SOUND.sfx` (e.g. 'memeBoom', 'sadTrombone'), played once when the step opens. Use sparingly. New meme sounds need only a file and a manifest entry.
- Controls bar: "Sound effects: on / off" toggle (on by default, remembered per viewer) switches music and effects together. Dev mode with voice off keeps music and effects off.
