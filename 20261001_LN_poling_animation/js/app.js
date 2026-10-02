// Shared simulation state: pulse time t (ms). Every panel is a view of t.
(() => {
  const q = new URLSearchParams(location.search);
  const state = { t: Math.min(WF.TMAX, Math.max(0, parseFloat(q.get('t') || '0'))), playing: q.get('paused') === null,
    speed: 1, holdUntil: 0 };
  const $ = (id) => document.getElementById(id);
  const track = { x: 0.1, y: 1.15 };

  const graph = new Graph($('cGraph'), (t) => { state.t = t; state.playing = false; sync(); });
  const top = new TopView($('cTop'), track);
  const ov = new Overview($('ovBody'), track);
  const uc = new UnitCell($('ucBody'), $('well'), $('ucInfo'), track);

  const scrub = $('scrub'), readout = $('readout'), playIcon = $('playIcon');
  const ICON_PLAY = 'M3 2l11 6-11 6z', ICON_PAUSE = 'M3 2h4v12H3zM9 2h4v12H9z';
  function sync() { playIcon.firstElementChild.setAttribute('d', state.playing ? ICON_PAUSE : ICON_PLAY); }
  scrub.addEventListener('input', () => { state.t = parseFloat(scrub.value); state.playing = false; sync(); });
  $('play').addEventListener('click', () => { if (state.t >= WF.TMAX - 1e-3) state.t = 0; state.playing = !state.playing; sync(); });
  $('restart').addEventListener('click', () => { state.t = 0; state.playing = true; sync(); });
  $('speed').addEventListener('change', (e) => { state.speed = parseFloat(e.target.value); });
  window.addEventListener('keydown', (e) => { if (e.code === 'Space' && e.target.tagName !== 'SELECT') { e.preventDefault(); $('play').click(); } });

  const setOrbit = (on) => { ov.controls.autoRotate = on; uc.controls.autoRotate = on; $('orbit').classList.toggle('on', on); };
  $('orbit').addEventListener('click', () => setOrbit(!ov.controls.autoRotate));
  $('arrowsBtn').addEventListener('click', () => { ov.arrows.visible = !ov.arrows.visible; $('arrowsBtn').classList.toggle('on', ov.arrows.visible); });

  // tooltips
  const tip = $('tip');
  document.addEventListener('pointermove', (e) => {
    const el = e.target.closest && e.target.closest('[data-tip]');
    if (!el) { tip.style.display = 'none'; return; }
    tip.textContent = el.getAttribute('data-tip'); tip.style.display = 'block';
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(window.innerWidth - w - 6, e.clientX + 12) + 'px';
    tip.style.top = Math.min(window.innerHeight - h - 6, e.clientY + 16) + 'px';
  });

  let last = performance.now();
  function frame(now) {
    const dt = Math.max(0, Math.min(0.1, (now - last) / 1000)); last = now;
    if (state.playing) {
      if (now >= state.holdUntil) {
        if (state.t >= WF.TMAX) state.t = 0;
        else {
          state.t += WF.rate(state.t) * state.speed * dt;
          if (state.t >= WF.TMAX) { state.t = WF.TMAX; state.holdUntil = now + 1500; }
        }
      }
    }
    const t = state.t;
    scrub.value = t;
    readout.innerHTML = `t <b>${t.toFixed(2)}</b> ms &nbsp; V <b>${WF.V(t).toFixed(0)}</b> V &nbsp; I <b>${WF.I(t).toFixed(0)}</b> nA &nbsp; Q <b>${WF.Q(t).toFixed(1)}</b> pC`;
    graph.draw(t); top.draw(t); ov.draw(t); uc.draw(t);
    requestAnimationFrame(frame);
  }
  sync();
  requestAnimationFrame(frame);
  window.__sim = { state, WF, DOM, track, ov, uc };
})();
