/* Math rendering for displayed text: $...$ inline, $$...$$ display (KaTeX, vendored in engine/vendor/katex/).
   Classic script, works from file:// and http. Loaded after vendor/katex/katex.min.js and contrib/auto-render.min.js.
   - window.LMS_math(el): render math inside el (idempotent: rendered .katex output is skipped).
   - A MutationObserver on <body> renders math in anything inserted later (step bodies, reveal tabs, stages, popups, chats, widgets).
   - window.LMS_mathPlain(s): plain-text version of a string with math, for attributes (title, aria-label) and the document title.
   Never inside <svg>, <canvas>, <textarea>, <code>, <pre>, <script>, <style>, or elements with class "nomath".
   Narration stays plain speakable English; tools/validate.js rejects any spoken string containing "$". */
(function (G) {
  'use strict';
  var SKIP = 'script,style,noscript,textarea,code,pre,option,svg,canvas,input,select,.katex,.nomath,[contenteditable]';
  var OPTS = {
    delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }],
    ignoredTags: ['script', 'style', 'noscript', 'textarea', 'code', 'pre', 'option', 'svg', 'canvas', 'input', 'select'],
    ignoredClasses: ['katex', 'nomath'],
    throwOnError: false,
    errorColor: '#cc3333',
    strict: 'ignore',
    errorCallback: function (msg) { if (G.console) G.console.warn('Math markup: ' + msg); }
  };

  function render(el) {
    if (!el || typeof G.renderMathInElement !== 'function') return el;
    if (el.nodeType === 3) el = el.parentNode;
    if (!el || el.nodeType !== 1) return el;
    if ((el.textContent || '').indexOf('$') < 0) return el;
    if (el.closest && el.closest(SKIP)) return el;
    try { G.renderMathInElement(el, OPTS); } catch (e) { if (G.console) G.console.warn('Math render failed', e); }
    return el;
  }

  var SYM = {
    alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', varepsilon: 'ε', zeta: 'ζ', eta: 'η', theta: 'θ', kappa: 'κ', lambda: 'λ',
    mu: 'μ', nu: 'ν', xi: 'ξ', pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ', phi: 'φ', varphi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω',
    Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π', Sigma: 'Σ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω',
    times: '×', cdot: '·', approx: '≈', pm: '±', mp: '∓', le: '≤', leq: '≤', ge: '≥', geq: '≥', ne: '≠', neq: '≠', ll: '≪', gg: '≫',
    infty: '∞', partial: '∂', nabla: '∇', propto: '∝', sim: '~', circ: '°', degree: '°', to: '→', rightarrow: '→', leftarrow: '←',
    parallel: '∥', perp: '⊥', langle: '⟨', rangle: '⟩', ell: 'ℓ', hbar: 'ħ', quad: ' ', qquad: ' ', ',': ' ', ';': ' ', '!': ''
  };
  /* Plain text for places that cannot hold markup. Simple TeX only (Greek, common operators, \frac, \sqrt, \mathrm, \text). */
  function plain(s) {
    return String(s == null ? '' : s).replace(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g, function (m, a, b) {
      var t = a != null ? a : b;
      for (var k = 0; k < 4; k++) {
        t = t.replace(/\\(?:mathrm|text|textrm|mathit|mathbf|operatorname|boldsymbol|mathcal)\s*\{([^{}]*)\}/g, '$1')
          .replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, '($1)/($2)')
          .replace(/\\sqrt\s*\{([^{}]*)\}/g, '√($1)');
      }
      t = t.replace(/\\([A-Za-z]+|[,;!])/g, function (mm, name) { return Object.prototype.hasOwnProperty.call(SYM, name) ? SYM[name] : name; })
        .replace(/\\left|\\right/g, '').replace(/[{}]/g, '').replace(/\s+/g, ' ').trim();
      return t;
    });
  }

  var started = false;
  function start() {
    if (started || !G.document || !G.document.body) return;
    started = true;
    render(G.document.body);
    if (typeof G.MutationObserver !== 'function') return;
    var mo = new G.MutationObserver(function (recs) {
      var todo = [];
      for (var i = 0; i < recs.length; i++) {
        var r = recs[i];
        if (r.type === 'characterData') { todo.push(r.target.parentNode); continue; }
        for (var j = 0; j < r.addedNodes.length; j++) {
          var n = r.addedNodes[j];
          todo.push(n.nodeType === 3 ? n.parentNode : n);
        }
      }
      for (var k = 0; k < todo.length; k++) { var el = todo[k]; if (el && el.isConnected) render(el); }
    });
    mo.observe(G.document.body, { childList: true, subtree: true, characterData: true });
  }

  G.LMS_math = render;
  G.LMS_mathPlain = plain;
  if (G.document) {
    if (G.document.readyState === 'loading') G.document.addEventListener('DOMContentLoaded', start);
    else start();
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = { LMS_mathPlain: plain };
})(typeof window !== 'undefined' ? window : globalThis);
