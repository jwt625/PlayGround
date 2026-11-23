/* Text key for recorded narration clips (COURSE_SPEC.md, "Recorded voice (Kokoro)").
   LMS_normText(t): collapse whitespace runs to one space, trim.
   LMS_hash(t): FNV-1a 32-bit (prime 0x01000193) over the UTF-8 bytes of LMS_normText(t), run twice with offsets
   0x811c9dc5 and 0x050c5d1f; result = 8 + 8 lowercase hex chars (16 chars). Clip path: audio/<hash>.mp3.
   Python twin: tools/tts/speech_hash.py (must stay byte-identical; tools/tts/test_speech_hash.py checks it).
   Works as a classic browser script (globals) and as a node module (module.exports). */
(function (root) {
  'use strict';

  function LMS_normText(t) {
    return String(t).replace(/\s+/g, ' ').trim();
  }

  /* UTF-8 encode; lone surrogates become U+FFFD (same as TextEncoder). */
  function utf8(s) {
    var out = [];
    for (var i = 0; i < s.length; i++) {
      var c = s.charCodeAt(i);
      if (c >= 0xd800 && c <= 0xdbff && i + 1 < s.length) {
        var d = s.charCodeAt(i + 1);
        if (d >= 0xdc00 && d <= 0xdfff) { c = 0x10000 + ((c - 0xd800) << 10) + (d - 0xdc00); i++; }
        else c = 0xfffd;
      } else if (c >= 0xd800 && c <= 0xdfff) c = 0xfffd;
      if (c < 0x80) out.push(c);
      else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 63));
      else if (c < 0x10000) out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
      else out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
    }
    return out;
  }

  function fnv1a(bytes, h) {
    for (var i = 0; i < bytes.length; i++) { h ^= bytes[i]; h = Math.imul(h, 0x01000193); }
    return ('0000000' + (h >>> 0).toString(16)).slice(-8);
  }

  function LMS_hash(t) {
    var b = utf8(LMS_normText(t));
    return fnv1a(b, 0x811c9dc5 | 0) + fnv1a(b, 0x050c5d1f);
  }

  root.LMS_normText = LMS_normText;
  root.LMS_hash = LMS_hash;
  if (typeof module !== 'undefined' && module.exports) module.exports = { LMS_normText: LMS_normText, LMS_hash: LMS_hash };
})(typeof window !== 'undefined' ? window : typeof globalThis !== 'undefined' ? globalThis : this);
