/* Chat step skins: phone apps in a CSS device frame, desktop apps in a window. Generic mock-ups: no real logos or app names.
   Contract: course/COURSE_SPEC.md (Tranche 2, Chat skins). Loaded before player.js. */
(function () {
  'use strict';
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
  const svg = (s) => { const d = document.createElement('span'); d.className = 'cs-ico'; d.innerHTML = s; return d; };
  const I = {
    signal: '<svg viewBox="0 0 18 12"><rect x="0" y="8" width="3" height="4" rx="1"/><rect x="5" y="5.5" width="3" height="6.5" rx="1"/><rect x="10" y="3" width="3" height="9" rx="1"/><rect x="15" y="0" width="3" height="12" rx="1"/></svg>',
    wifi: '<svg viewBox="0 0 16 12"><path d="M8 2.2c2.3 0 4.4.9 6 2.4l1.2-1.3C13.3 1.5 10.8.4 8 .4S2.7 1.5.8 3.3L2 4.6c1.6-1.5 3.7-2.4 6-2.4zm0 3.6c1.3 0 2.5.5 3.4 1.3l1.2-1.3C11.4 4.7 9.8 4 8 4s-3.4.7-4.6 1.8l1.2 1.3c.9-.8 2.1-1.3 3.4-1.3zm0 3.6c-.6 0-1.1.2-1.5.6L8 11.6l1.5-1.6c-.4-.4-.9-.6-1.5-.6z"/></svg>',
    battery: '<svg viewBox="0 0 27 13"><rect x="0.5" y="0.5" width="23" height="12" rx="3.5" fill="none" stroke="currentColor" opacity=".45"/><rect x="2" y="2" width="20" height="9" rx="2"/><path d="M25 4.5v4c.8-.3 1.5-1.1 1.5-2s-.7-1.7-1.5-2z" opacity=".45"/></svg>',
    back: '<svg viewBox="0 0 12 20"><path d="M10 2L2 10l8 8" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    video: '<svg viewBox="0 0 24 16"><rect x="1" y="2" width="15" height="12" rx="3" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M17 6.5l5-3v9l-5-3z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>',
    phone: '<svg viewBox="0 0 20 20"><path d="M5 2l3 .5 1.5 4-2 1.5a10 10 0 0 0 4.5 4.5l1.5-2 4 1.5.5 3c0 1-1 2-2 2C9 17 3 11 3 4c0-1 1-2 2-2z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>',
    plus: '<svg viewBox="0 0 20 20"><path d="M10 4v12M4 10h12" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>',
    mic: '<svg viewBox="0 0 20 20"><rect x="7" y="2" width="6" height="11" rx="3" fill="currentColor"/><path d="M4 10a6 6 0 0 0 12 0M10 16v3" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>',
    micOff: '<svg viewBox="0 0 20 20"><rect x="7" y="2" width="6" height="11" rx="3" fill="currentColor"/><path d="M4 10a6 6 0 0 0 12 0M10 16v3" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/><path d="M3 3l14 14" stroke="#e5484d" stroke-width="2.2" stroke-linecap="round"/></svg>',
    cam: '<svg viewBox="0 0 24 16"><rect x="1" y="2" width="15" height="12" rx="3" fill="currentColor"/><path d="M17 6.5l5-3v9l-5-3z" fill="currentColor"/></svg>',
    share: '<svg viewBox="0 0 20 20"><rect x="2" y="3" width="16" height="11" rx="2" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M10 12V6M7 8.5L10 5.5l3 3M6 17h8" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>',
    end: '<svg viewBox="0 0 24 24"><path d="M3 14c5-5 13-5 18 0l-2 3-4-1v-3a10 10 0 0 0-6 0v3l-4 1z" fill="currentColor"/></svg>',
    chat: '<svg viewBox="0 0 20 20"><path d="M3 4h14v9H8l-4 3v-3H3z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>',
    tick2: '<svg viewBox="0 0 18 11"><path d="M1 5.8l3 3 6.4-7.4M7.2 8.4l.6.5 6.4-7.4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    read: '<svg viewBox="0 0 22 12"><circle cx="6" cy="6" r="5.2" fill="currentColor"/><circle cx="15" cy="6" r="5.2" fill="currentColor"/><path d="M3.6 6.2l1.7 1.6 3-3.4M12.6 6.2l1.7 1.6 3-3.4" stroke="var(--cs-ink-on)" stroke-width="1.4" fill="none" stroke-linecap="round"/></svg>',
    search: '<svg viewBox="0 0 20 20"><circle cx="8.5" cy="8.5" r="5.5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M13 13l4.5 4.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
    send: '<svg viewBox="0 0 20 20"><path d="M3 10l14-7-5 15-2.5-6z" fill="currentColor"/></svg>',
    up: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="9.5" fill="currentColor"/><path d="M10 15V6M6 9.5l4-4 4 4" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  };
  const initials = (n) => String(n).split(/\s+/).filter(Boolean).map((w) => w[0]).join('').slice(0, 2).toUpperCase();
  const PAL = ['#d9534f', '#3b7dd8', '#2e9e5b', '#8e5bd1', '#c98a1b', '#1f9aa8', '#c2457d', '#5c6bc0'];
  const color = (name) => { let s = 7; for (const c of String(name)) s = (s * 31 + c.charCodeAt(0)) >>> 0; return PAL[s % PAL.length]; };
  /* profile pictures from window.LMS_CAST (media/cast.js): exact key, then case-insensitive key or alias; initials otherwise */
  const castFor = (name) => (window.LMS_castFor ? window.LMS_castFor(name) : null);
  const av = (name, cls) => {
    const c = castFor(name);
    const e = h('span', { class: 'cs-av ' + (cls || '') + (c && c.avatar ? ' pic' : ''), style: '--av:' + color(name), title: String(name) }, initials(name));
    if (c && c.avatar) { const im = h('img', { src: c.avatar, alt: '', draggable: 'false' }); im.addEventListener('error', () => { im.remove(); e.classList.remove('pic'); }); e.append(im); }
    return e;
  };
  /* fixed fictional clock: first message 9:12 AM, two minutes apart; status bar reads 9:41 */
  const clock = (i) => { const m = 9 * 60 + 12 + 2 * i; const hh = Math.floor(m / 60), mm = m % 60; return (hh > 12 ? hh - 12 : hh) + ':' + String(mm).padStart(2, '0') + ' AM'; };
  const short = (i) => clock(i).replace(' AM', '');
  const text = (t) => h('span', { class: 'cs-txt' }, String(t == null ? '' : t));

  const SKINS = { phone: ['imessage', 'signal', 'whatsapp'], thread: ['slack', 'teams'], call: ['facetime', 'meeting'], email: ['mail'] };

  function people(st) {
    const msgs = st.messages || [];
    const meta = st.meta || {};
    const all = [...new Set([].concat(meta.participants || [], msgs.map((m) => m.from)))].filter(Boolean);
    const me = [...new Set(msgs.filter((m) => m.side === 'right').map((m) => m.from))];
    const others = all.filter((p) => me.indexOf(p) < 0);
    return { all, me, others, meta, msgs };
  }
  const generic = (s) => !s || /^(messages|inbox|chat|text messages?)$/i.test(String(s).trim());

  /* ---------- phone device ---------- */
  function device(skin, inner, dark) {
    return h('div', { class: 'cs-device cs-' + skin + (dark ? ' cs-darkbar' : '') },
      h('div', { class: 'cs-screen' },
        h('div', { class: 'cs-island' }),
        h('div', { class: 'cs-sbar' }, h('span', { class: 'cs-clock' }, '9:41'), h('span', { class: 'cs-glyphs' }, svg(I.signal), svg(I.wifi), svg(I.battery))),
        inner,
        h('div', { class: 'cs-home' })));
  }

  function phoneSkin(st, skin) {
    const P = people(st);
    const group = P.others.length > 1;
    const title = P.others.length === 1 ? P.others[0] : !generic(P.meta.header) ? P.meta.header : P.others.join(', ') || 'Conversation';
    const list = h('div', { class: 'cs-list' });
    const input = h('span', { class: 'cs-input-t' }, 'Message');
    let head, foot, sub = null;
    if (skin === 'whatsapp') {
      sub = h('small', null, group ? P.all.join(', ') : 'online');
      head = h('div', { class: 'cs-head' }, svg(I.back), av(title, 'grey'), h('div', { class: 'cs-hname' }, h('b', null, title), sub), h('span', { class: 'cs-hicons' }, svg(I.video), svg(I.phone)));
      foot = h('div', { class: 'cs-foot' }, svg(I.plus), h('div', { class: 'cs-input' }, input), h('span', { class: 'cs-mic' }, svg(I.mic)));
      list.append(h('div', { class: 'cs-chip' }, 'TODAY'));
    } else if (skin === 'signal') {
      head = h('div', { class: 'cs-head' }, svg(I.back), av(title), h('div', { class: 'cs-hname' }, h('b', null, title), group ? h('small', null, P.all.length + ' members') : null), h('span', { class: 'cs-hicons' }, svg(I.video), svg(I.phone)));
      foot = h('div', { class: 'cs-foot' }, svg(I.plus), h('div', { class: 'cs-input' }, input), h('span', { class: 'cs-mic' }, svg(I.mic)));
      list.append(h('div', { class: 'cs-chip' }, 'Today'));
    } else {
      head = h('div', { class: 'cs-head' }, h('span', { class: 'cs-hback' }, svg(I.back)),
        h('div', { class: 'cs-hmid' }, group ? h('span', { class: 'cs-avs' }, ...P.others.slice(0, 3).map((p) => av(p, 'grey'))) : av(title, 'grey'), h('span', { class: 'cs-hn' }, title)),
        h('span', { class: 'cs-hicons' }, svg(I.video)));
      foot = h('div', { class: 'cs-foot' }, h('span', { class: 'cs-plus' }, svg(I.plus)), h('div', { class: 'cs-input' }, input, h('span', { class: 'cs-send' }, svg(I.up))));
      list.append(h('div', { class: 'cs-stamp' }, h('b', null, 'Today'), ' ' + clock(0)));
    }
    const el = device(skin, h('div', { class: 'cs-app' }, head, list, foot), skin === 'whatsapp');
    let prev = null; /* { from, bubble, row } */
    let delivered = null;
    function add(m, i) {
      const me = m.side === 'right';
      const cont = prev && prev.from === m.from;
      const bubble = h('div', { class: 'cs-b' }, (group && !me && !cont && skin === 'whatsapp') ? h('b', { class: 'cs-bname', style: 'color:' + color(m.from) }, m.from) : null, text(m.text));
      if (skin === 'whatsapp') bubble.append(h('span', { class: 'cs-meta' }, short(i), me ? svg(I.tick2) : null));
      if (skin === 'signal') bubble.append(h('span', { class: 'cs-meta' }, short(i), me ? svg(I.read) : null));
      const row = h('div', { class: 'cs-row ' + (me ? 'me' : 'them') + (cont ? ' cont' : '') },
        group && !me && !cont && skin !== 'whatsapp' ? h('div', { class: 'cs-sender' }, m.from) : null,
        h('div', { class: 'cs-line' }, group && !me && skin !== 'whatsapp' ? av(m.from, 'mini' + (skin === 'imessage' ? ' grey' : '')) : null, bubble));
      if (cont) { prev.bubble.classList.remove('tail'); prev.bubble.classList.add('gnext'); bubble.classList.add('gprev'); prev.row.classList.add('hasnext'); }
      if (skin !== 'whatsapp' || !cont) bubble.classList.add('tail');
      if (skin === 'whatsapp' && cont) bubble.classList.remove('tail');
      if (skin === 'imessage') {
        if (delivered) { delivered.remove(); delivered = null; }
        list.append(row);
        if (me) { delivered = h('div', { class: 'cs-delivered' }, 'Delivered'); list.append(delivered); }
      } else list.append(row);
      prev = { from: m.from, bubble, row };
      list.scrollTop = 1e6;
    }
    function typing(m) {
      if (m.side === 'right') { input.textContent = m.text; input.classList.add('on'); return { stop() { input.textContent = 'Message'; input.classList.remove('on'); } }; }
      const t = h('div', { class: 'cs-row them' }, h('div', { class: 'cs-line' }, group && skin !== 'whatsapp' ? av(m.from, 'mini' + (skin === 'imessage' ? ' grey' : '')) : null, h('div', { class: 'cs-b cs-typing tail' }, h('i'), h('i'), h('i'))));
      list.append(t); list.scrollTop = 1e6;
      const old = sub ? sub.textContent : null;
      if (sub) sub.textContent = group ? m.from + ' is typing...' : 'typing...';
      return { stop() { t.remove(); if (sub) sub.textContent = old; } };
    }
    return { el, add, typing, kind: 'phone', list };
  }

  function facetimeSkin(st) {
    const P = people(st);
    const remote = P.others.length ? P.others : P.all.slice(1);
    const self = P.me[0] || P.all[0] || 'You';
    const tiles = {};
    const grid = h('div', { class: 'cs-ftgrid n' + Math.min(4, remote.length) }, ...remote.map((p) => (tiles[p] = h('div', { class: 'cs-fttile' }, av(p, 'big'), h('span', { class: 'cs-ftname' }, p)))));
    tiles[self] = h('div', { class: 'cs-ftself' }, av(self, 'mid'));
    const cap = h('div', { class: 'cs-ftcap' });
    const timer = h('span', null, '00:00');
    const head = h('div', { class: 'cs-fthead' }, h('b', null, remote.length === 1 ? remote[0] : (!generic(P.meta.header) ? P.meta.header : remote.length + ' people')), h('small', null, 'Video call  ', timer));
    const controls = h('div', { class: 'cs-ftctl' }, h('span', null, svg(I.chat)), h('span', null, svg(I.mic)), h('span', null, svg(I.cam)), h('span', null, svg(I.share)), h('span', { class: 'end' }, svg(I.end)));
    const el = device('facetime', h('div', { class: 'cs-app' }, head, grid, tiles[self], cap, controls), true);
    let n = 0;
    function speaking(p) { Object.keys(tiles).forEach((k) => tiles[k].classList.toggle('speaking', k === p)); }
    function add(m, i) {
      speaking(m.from);
      cap.append(h('div', { class: 'cs-capl' }, h('b', null, m.from), ' ', text(m.text)));
      while (cap.children.length > 2) cap.firstChild.remove();
      n++; timer.textContent = '00:' + String(Math.min(59, 8 + n * 7)).padStart(2, '0');
    }
    function typing(m) { speaking(m.from); return { stop() {} }; }
    return { el, add, typing, kind: 'phone', list: cap };
  }

  /* ---------- desktop window ---------- */
  function win(skin, title, inner) {
    return h('div', { class: 'cs-win cs-' + skin },
      h('div', { class: 'cs-titlebar' }, h('span', { class: 'cs-lights' }, h('i'), h('i'), h('i')), h('span', { class: 'cs-wtitle' }, title)),
      inner);
  }

  function slackSkin(st) {
    const P = people(st);
    const ch = String(P.meta.header || 'lab-general').replace(/^#\s*/, '');
    const chans = [...new Set([ch, 'general', 'cleanroom-schedule', 'equipment-booking', 'random'])];
    const side = h('div', { class: 'cs-sk-side' },
      h('div', { class: 'cs-sk-ws' }, 'Five Sigma Photonics'),
      h('div', { class: 'cs-sk-sec' }, 'Channels'),
      ...chans.map((c) => h('div', { class: 'cs-sk-item' + (c === ch ? ' on' : '') }, '# ' + c)),
      h('div', { class: 'cs-sk-sec' }, 'Direct messages'),
      ...P.all.slice(0, 5).map((p) => h('div', { class: 'cs-sk-item dm' }, h('i', { class: 'cs-dot' }), p)));
    const list = h('div', { class: 'cs-sk-list' }, h('div', { class: 'cs-sk-day' }, h('span', null, 'Today')));
    const typingLine = h('div', { class: 'cs-sk-typing' }, ' ');
    const threadMode = !!P.meta.thread;
    const tlist = h('div', { class: 'cs-sk-tlist' });
    const tpane = threadMode ? h('div', { class: 'cs-sk-thread' }, h('div', { class: 'cs-sk-thead' }, h('b', null, 'Thread'), h('small', null, '# ' + ch)), tlist) : null;
    const main = h('div', { class: 'cs-sk-main' },
      h('div', { class: 'cs-sk-top' }, h('b', null, '# ' + ch), h('small', null, (P.all.length || 1) + ' members')),
      list,
      h('div', { class: 'cs-sk-comp' }, h('div', { class: 'cs-sk-box' }, 'Message # ' + ch), typingLine));
    const el = win('slack', 'Workspace', h('div', { class: 'cs-sk-body' + (threadMode ? ' threaded' : '') }, side, main, tpane));
    let prevFrom = null, parentLink = null, replies = 0, tprev = null;
    function row(m, i, compact) {
      return compact ? h('div', { class: 'cs-sk-msg compact' }, h('span', { class: 'cs-sk-ht' }, short(i)), h('div', null, text(m.text)))
        : h('div', { class: 'cs-sk-msg' }, av(m.from, 'sq'), h('div', null, h('div', { class: 'cs-sk-name' }, h('b', null, m.from), h('small', null, clock(i))), text(m.text)));
    }
    function add(m, i) {
      if (threadMode && i > 0) {
        tlist.append(row(m, i, tprev === m.from)); tprev = m.from; replies++;
        parentLink.textContent = replies + (replies === 1 ? ' reply' : ' replies');
        tlist.scrollTop = 1e6; return;
      }
      list.append(row(m, i, prevFrom === m.from));
      if (threadMode && i === 0) { parentLink = h('span', null, ''); list.lastChild.lastChild.append(h('div', { class: 'cs-sk-replies' }, parentLink)); tlist.append(row(m, i, false)); tlist.append(h('div', { class: 'cs-sk-div' }, h('span', null, 'Replies'))); }
      prevFrom = m.from; list.scrollTop = 1e6;
    }
    function typing(m) { typingLine.innerHTML = ''; typingLine.append(h('b', null, m.from), ' is typing...'); return { stop() { typingLine.textContent = ' '; } }; }
    return { el, add, typing, kind: 'desk', list };
  }

  function teamsSkin(st) {
    const P = people(st);
    const name = !generic(P.meta.header) ? String(P.meta.header).replace(/^#\s*/, '') : P.all.join(', ');
    const rail = h('div', { class: 'cs-tm-rail' }, ...['Activity', 'Chat', 'Teams', 'Calendar', 'Files'].map((w) => h('div', { class: 'cs-tm-ri' + (w === 'Chat' ? ' on' : '') }, h('i'), w)));
    const preview = h('small', null, '');
    const chats = h('div', { class: 'cs-tm-chats' }, h('div', { class: 'cs-tm-ch' }, 'Chat'),
      h('div', { class: 'cs-tm-ci on' }, av(name, 'grp'), h('div', null, h('b', null, name), preview)),
      ...P.all.slice(0, 4).map((p) => h('div', { class: 'cs-tm-ci' }, av(p), h('div', null, h('b', null, p), h('small', null, 'Yesterday')))));
    const list = h('div', { class: 'cs-tm-list' });
    const typingLine = h('div', { class: 'cs-tm-typing' }, ' ');
    const main = h('div', { class: 'cs-tm-main' },
      h('div', { class: 'cs-tm-top' }, av(name, 'grp'), h('b', null, name), h('span', { class: 'cs-tm-tabs' }, h('u', null, 'Chat'), h('span', null, 'Files'), h('span', null, 'Notes'))),
      list, typingLine, h('div', { class: 'cs-tm-comp' }, 'Type a message', svg(I.send)));
    const el = win('teams', 'Chat', h('div', { class: 'cs-tm-body' }, h('div', { class: 'cs-tm-search' }, svg(I.search), 'Search'), h('div', { class: 'cs-tm-cols' }, rail, chats, main)));
    let prevFrom = null;
    function add(m, i) {
      const me = m.side === 'right';
      const cont = prevFrom === m.from;
      const card = h('div', { class: 'cs-tm-card' }, cont ? null : h('div', { class: 'cs-tm-meta' }, me ? null : h('b', null, m.from), h('small', null, clock(i))), text(m.text));
      list.append(h('div', { class: 'cs-tm-row ' + (me ? 'me' : 'them') + (cont ? ' cont' : '') }, me ? null : (cont ? h('span', { class: 'cs-av-sp' }) : av(m.from)), card));
      preview.textContent = m.from + ': ' + m.text;
      prevFrom = m.from; list.scrollTop = 1e6;
    }
    function typing(m) { typingLine.innerHTML = ''; typingLine.append(h('span', { class: 'cs-dots' }, h('i'), h('i'), h('i')), ' ' + m.from + ' is typing'); return { stop() { typingLine.textContent = ' '; } }; }
    return { el, add, typing, kind: 'desk', list };
  }

  function meetingSkin(st) {
    const P = people(st);
    const tiles = {};
    const grid = h('div', { class: 'cs-mt-grid n' + Math.min(6, P.all.length) }, ...P.all.map((p) => (tiles[p] = h('div', { class: 'cs-mt-tile' }, av(p, 'big'), h('span', { class: 'cs-mt-name' }, svg(I.micOff), p)))));
    const log = h('div', { class: 'cs-mt-log' });
    const timer = h('span', null, '00:12:04');
    const title = P.meta.header || 'Meeting';
    const bar = h('div', { class: 'cs-mt-bar' }, h('span', { class: 'cs-mt-time' }, timer),
      h('span', { class: 'cs-mt-btns' }, ...[[I.mic, 'Mute'], [I.cam, 'Video'], [I.share, 'Share'], [I.chat, 'Chat']].map(([ic, l]) => h('span', { class: 'cs-mt-b' }, svg(ic), l)), h('span', { class: 'cs-mt-b leave' }, 'Leave')),
      h('span', { class: 'cs-mt-rec' }, h('i'), 'Recording'));
    const el = win('meeting', title, h('div', { class: 'cs-mt-body' }, h('div', { class: 'cs-mt-stage' }, grid, bar), h('div', { class: 'cs-mt-side' }, h('div', { class: 'cs-mt-sh' }, 'Meeting transcript'), log)));
    let n = 0;
    function speaking(p) {
      Object.keys(tiles).forEach((k) => { const on = k === p; tiles[k].classList.toggle('speaking', on); const ic = tiles[k].querySelector('.cs-mt-name .cs-ico'); ic.innerHTML = on ? I.mic : I.micOff; });
    }
    function add(m, i) {
      speaking(m.from);
      log.append(h('div', { class: 'cs-mt-l' }, h('div', null, h('b', null, m.from), h('small', null, short(i))), text(m.text)));
      log.scrollTop = 1e6; n++;
      timer.textContent = '00:12:' + String(Math.min(59, 4 + n * 9)).padStart(2, '0');
    }
    function typing(m) { speaking(m.from); return { stop() {} }; }
    return { el, add, typing, kind: 'desk', list: log };
  }

  /* ---------- mail: identities from LMS_CAST (email, title, org); derived firstname.lastname@fivesigma.photonics otherwise ---------- */
  const MAIL_DOMAIN = 'fivesigma.photonics', MAIL_ORG = 'Five Sigma Photonics, Inc.';
  function castEntry(name) {
    const C = window.LMS_CAST || {};
    const n = String(name == null ? '' : name).trim(), l = n.toLowerCase();
    if (Object.prototype.hasOwnProperty.call(C, n)) return [n, C[n]];
    for (const k in C) {
      if (!Object.prototype.hasOwnProperty.call(C, k)) continue;
      if (k.toLowerCase() === l || (C[k].aliases || []).some((a) => String(a).toLowerCase() === l)) return [k, C[k]];
    }
    return [n, null];
  }
  function ident(name) {
    const [key, c] = castEntry(name);
    const display = String(key).replace(/\s*\([^)]*\)\s*/g, ' ').replace(/\s+/g, ' ').trim() || String(name);
    const email = (c && c.email) || (display.toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, '.').replace(/^\.+|\.+$/g, '') || 'staff') + '@' + MAIL_DOMAIN;
    return { name: display, email, title: (c && c.title) || '', org: (c && c.org) || (email.endsWith('@' + MAIL_DOMAIN) ? MAIL_ORG : ''), key: name };
  }
  const addr = (p) => p.name + ' <' + p.email + '>';
  const PAPERCLIP = '<svg viewBox="0 0 20 20"><path d="M13.5 6.5l-6 6a1.8 1.8 0 0 0 2.5 2.5l6.5-6.5a3.3 3.3 0 0 0-4.7-4.7L5.2 10.4a4.8 4.8 0 0 0 6.8 6.8l5-5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
  const REPLY = '<svg viewBox="0 0 20 20"><path d="M8 5L3 10l5 5M3 10h8a6 6 0 0 1 6 6" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const REPLYALL = '<svg viewBox="0 0 22 20"><path d="M9 5L4 10l5 5M5 5L0.8 10 5 15M4 10h9a6 6 0 0 1 6 6" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const FWD = '<svg viewBox="0 0 20 20"><path d="M12 5l5 5-5 5M17 10H9a6 6 0 0 0-6 6" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const escH = (t) => String(t == null ? '' : t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  /* fixed fictional mail clock: 2026-11-16 09:12, two minutes apart */
  const mailDate = (i) => { const m = 9 * 60 + 12 + 2 * i; return '2026-11-16 ' + String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0'); };

  function mailSkin(st, ctx) {
    const P = people(st);
    const toast = (ctx && ctx.toast) || (() => {});
    const thread = String(P.meta.subject || P.meta.header || '(no subject)');
    const subjOf = (m, i) => m.subject || (i === 0 || /^(re|fw|fwd):/i.test(thread) ? thread : 'RE: ' + thread);
    /* older unrelated mail, dated the days before this thread starts so the list stays in date order */
    const first = (P.msgs[0] && /^\d{4}-\d{2}-\d{2}/.test(P.msgs[0].date || '')) ? P.msgs[0].date.slice(0, 10) : '2026-11-16';
    const dayBefore = (k) => { const d = new Date(first + 'T12:00:00Z'); d.setUTCDate(d.getUTCDate() - k); return d.toISOString().slice(0, 10); };
    const fillers = [
      ['Facilities', 'Scheduled power maintenance, Building 2', 'Cleanroom exhaust will be offline Saturday from 06:00. Please plan accordingly.', dayBefore(1)],
      ['Office of Photonic Conduct', 'Reminder: required training due 2026-12-31', 'Our records show you have required training outstanding. Completion is tracked.', dayBefore(2)],
      ['Lab Operations', 'Lunch and learn: fiber cleaving', 'Bring your own cleaver. Snacks provided. Safety glasses required.', dayBefore(4)],
    ];
    const folders = h('div', { class: 'cs-ml-folders' }, h('div', { class: 'cs-ml-new' }, 'New message'),
      ...[['Inbox', ''], ['Flagged', ''], ['Sent', ''], ['Drafts', '1'], ['Archive', ''], ['Junk', '']].map(([f, c]) => h('div', { class: 'cs-ml-f' + (f === 'Inbox' ? ' on' : '') }, f, h('small', { class: f === 'Inbox' ? 'cs-ml-unread' : '' }, c))));
    const unread = folders.querySelector('.cs-ml-unread');
    const items = h('div', { class: 'cs-ml-items' }, ...fillers.map(([a, b, c, d]) => {
      const p = ident(a);
      return h('div', { class: 'cs-ml-it' }, h('div', { class: 'cs-ml-r1' }, h('span', null, p.name), h('small', null, d.slice(5))), h('div', { class: 'cs-ml-subj' }, b), h('small', { class: 'cs-ml-prev' }, c));
    }));
    const titleEl = h('div', { class: 'cs-ml-rh' }, thread);
    const btn = (ic, label, msg) => h('button', { class: 'cs-ml-tb', onclick: () => toast(msg) }, svg(ic), label);
    const tools = h('div', { class: 'cs-ml-tools' },
      btn(REPLY, 'Reply', 'Replying is disabled during training. Your intent to reply has been noted.'),
      btn(REPLYALL, 'Reply all', 'Reply all is restricted by the Office of Photonic Conduct. Consider whether everyone needs to know.'),
      btn(FWD, 'Forward', 'Forwarding is disabled. This thread is retained for training purposes.'));
    const conv = h('div', { class: 'cs-ml-conv' });
    const read = h('div', { class: 'cs-ml-read' }, h('div', { class: 'cs-ml-rtop' }, titleEl, tools), conv);
    const el = win('mail', 'Mail', h('div', { class: 'cs-ml-body' }, folders, h('div', { class: 'cs-ml-listp' }, h('div', { class: 'cs-ml-lh' }, 'Inbox', h('span', null, svg(I.search))), items), read));
    const msgsEls = [], listEls = [];
    function expand(k) {
      msgsEls.forEach((e, j) => e.classList.toggle('open', j === k));
      listEls.forEach((e, j) => e.classList.toggle('on', j === k));
      titleEl.textContent = msgsEls[k] ? msgsEls[k].dataset.subject : thread;
    }
    function add(m, i) {
      const from = ident(m.from);
      const toNames = m.to && m.to.length ? m.to : P.all.filter((p) => p !== m.from);
      const to = (toNames.length ? toNames : ['team']).map(ident);
      const cc = (m.cc || []).map(ident);
      const date = m.date || mailDate(i);
      const subject = subjOf(m, i);
      const bodyHtml = escH(m.text).split(/\n{2,}/).map((para) => '<p>' + para.replace(/\n/g, '<br>') + '</p>').join('');
      const preview = String(m.text || '').replace(/\s+/g, ' ');
      const sig = h('div', { class: 'cs-ml-sig' }, h('b', null, from.name), from.title ? h('span', null, from.title) : null, from.org ? h('span', null, from.org) : null);
      const atts = (m.attachments || []).length ? h('div', { class: 'cs-ml-atts' }, ...m.attachments.map((a) => h('span', { class: 'cs-ml-att' }, svg(PAPERCLIP), h('b', null, a.name), a.size ? h('small', null, a.size) : null))) : null;
      const k = msgsEls.length;
      const msg = h('div', { class: 'cs-ml-msg' },
        h('div', { class: 'cs-ml-mh', onclick: () => expand(k) }, av(m.from),
          h('div', { class: 'cs-ml-who' },
            h('div', { class: 'cs-ml-from' }, h('b', null, from.name), h('span', { class: 'cs-ml-addr' }, ' <' + from.email + '>')),
            h('div', { class: 'cs-ml-snip' }, preview),
            h('div', { class: 'cs-ml-rcpt' }, h('span', null, 'To: ' + to.map(addr).join(', ')), cc.length ? h('span', null, 'Cc: ' + cc.map(addr).join(', ')) : null)),
          h('small', { class: 'cs-ml-t' }, date)),
        h('div', { class: 'cs-ml-full' }, atts, h('div', { class: 'cs-ml-mb', html: bodyHtml }), sig));
      msg.dataset.subject = subject;
      conv.append(msg); msgsEls.push(msg);
      const li = h('div', { class: 'cs-ml-it new', onclick: () => expand(k) }, h('div', { class: 'cs-ml-r1' }, h('b', null, from.name), h('small', null, date.slice(11))), h('div', { class: 'cs-ml-subj' }, subject, atts ? svg(PAPERCLIP) : null), h('small', { class: 'cs-ml-prev' }, preview));
      items.prepend(li); listEls.push(li);
      setTimeout(() => li.classList.remove('new'), 1200);
      unread.textContent = String(listEls.length);
      expand(k);
      requestAnimationFrame(() => { read.scrollTop = msg.offsetTop - 60; });
    }
    function typing() { return null; }
    return { el, add, typing, kind: 'desk', list: conv };
  }

  function build(st, ctx) {
    const app = st.app || 'phone';
    const allowed = SKINS[app] || SKINS.phone;
    const skin = allowed.indexOf(st.skin) >= 0 ? st.skin : allowed[0];
    if (app === 'thread') return skin === 'teams' ? teamsSkin(st) : slackSkin(st);
    if (app === 'call') return skin === 'meeting' ? meetingSkin(st) : facetimeSkin(st);
    if (app === 'email') return mailSkin(st, ctx);
    return phoneSkin(st, skin);
  }

  window.LMS_CHAT = { build, SKINS, avatar: av, people };
})();
