/* Engine configuration. Loaded first by player.html and index.html. Contract: course/COURSE_SPEC.md (Tranche 2, Dev mode). */
window.LMS_CONFIG = window.LMS_CONFIG || { devHosts: ['localhost', '127.0.0.1'], fileDev: true };

/* True when the page is served from a host where dev mode may be enabled. `loc` defaults to window.location. */
window.LMS_isDevHost = function (loc) {
  var c = window.LMS_CONFIG || {};
  loc = loc || window.location;
  if (loc.protocol === 'file:') return !!c.fileDev;
  return (c.devHosts || []).indexOf(loc.hostname) >= 0;
};

/* Dev mode is ON only with ?dev=1 on a dev-allowed host. On any other host the flag is ignored. */
window.LMS_devMode = function (loc) {
  loc = loc || window.location;
  var on = false;
  try { on = new URLSearchParams(loc.search || '').get('dev') === '1'; } catch (e) { on = false; }
  return on && window.LMS_isDevHost(loc);
};
