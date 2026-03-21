# OFC Metadata and PDF Caching

## Summary

This log records the initial engineering work for extracting OFC schedule metadata and caching technical paper PDFs.

The core outcomes so far are:

- A schedule metadata exporter was built and tested.
- The OFC schedule API was confirmed to contain presentation/session metadata directly.
- Technical paper links were found to be embedded in presentation description HTML rather than in the structured `paperDownloadLink` field.
- Browser automation was tested across multiple configurations.
- End-to-end PDF caching was successfully completed for a small sample using visible Chrome plus exported cookies.

## Initial Findings

### Schedule app behavior

The OFC schedule page is a frontend application backed by:

- `https://www.ofcconference.org/api/schedule/`

The frontend bundle renders a paper button only when:

- `presentation.paperDownloadLink` is populated
- `presentation.paperIsReady` is true

However, live API data showed a mismatch:

- `paperIsReady` was often `true`
- `paperDownloadLink` was empty
- the actual `Download PDF` URL existed inside `presentation.description` HTML

This explains why frontend state alone was not enough to expose paper links consistently.

### DOM / expand-fold behavior

The presentation rows are already present in the schedule payload and rendered in the page. Expand/fold actions mainly toggle frontend visibility state. The paper link is not loaded lazily by a separate request on presentation expansion.

## Scripts Created and Updated

### `scripts/ofc_schedule_probe.py`

Purpose:

- Probe the schedule API
- Identify where paper links exist
- Sample-check paper URL behavior

Confirmed that Monday had many embedded paper links with no structured paper link field.

### `scripts/export_ofc_metadata.py`

Purpose:

- Export normalized schedule metadata to JSON and CSV
- Preserve both presentation rows and session-only rows
- Track explicit paper state
- Optionally validate/capture paper PDFs

The exporter now includes:

- presentation/session metadata normalization
- abstract parsing
- author parsing
- special session / short course speaker capture via `extra_people_json`
- explicit paper state fields
- cache-backed paper validation results

## Key Bugs and Fixes

### 1. Missing structured paper links

Observed behavior:

- `paperDownloadLink` was empty
- actual paper links were embedded in HTML descriptions

Fix:

- parse the `Access the Technical Paper` anchor directly from `description_html`
- populate `embedded_pdf_link`
- derive `best_pdf_link`

### 2. Weak paper state model

Initial exporter output did not distinguish enough states.

Fix:

Added explicit fields:

- `paper_link_source`
- `paper_link_tested`
- `paper_link_status`
- `paper_link_final_url`
- `paper_link_content_type`
- `paper_link_error`
- `paper_cached`
- `paper_cache_path`

### 3. Re-testing previously failed links unnecessarily

Initial cache logic only skipped already cached PDFs.

Fix:

- cache negative outcomes as well
- skip previously tested links by default
- make retries opt-in via `--retry-tested`

### 4. Playwright persistent Firefox launch hang

Observed behavior:

- Playwright with system Firefox hung during `launch_persistent_context`

Fix:

- switched browser validation work to Selenium for Firefox/Chrome experiments

### 5. Headless browser anti-bot failures

Observed behavior:

- headless Chrome and headless Firefox frequently landed on Radware / Perfdrive anti-bot pages
- fingerprints exposed `HeadlessChrome` and triggered blocking

Fix:

- moved the working PDF path to visible Chrome

### 6. Chrome profile clone alone was insufficient

Observed behavior:

- cloned `Default` Chrome profile did not consistently preserve the authenticated Optica state needed for direct PDF access

Fix:

- accepted explicit cookie export
- injected cookies into browser automation

### 7. Chrome opened PDFs in browser viewer instead of saving

Observed behavior:

- authenticated PDF URLs loaded in Chrome successfully
- no files were written to download directory

Fix:

- use browser automation only to resolve the authenticated final `directpdfaccess/...pdf` URL
- fetch and save the PDF bytes using Python with the same cookies

### 8. Cookie injection regression

Observed behavior:

- a generalized cookie payload caused Optica-domain cookies to fail during Selenium injection

Fix:

- simplified cookie injection payload to the minimal accepted fields
- verified successful authentication path with the simplified form

## Browser Testing Timeline

### API-only export

Result:

- successful
- metadata export completed cleanly
- paper links discovered from description HTML

### Firefox automation

#### Browserless / urllib checks

Result:

- mixed outcomes:
  - login page
  - anti-bot page

#### Selenium Firefox

Result:

- automation launched successfully
- schedule page loaded
- paper links still often classified as:
  - `login_required`
  - `blocked_by_bot`

Conclusion:

- Firefox path was useful for diagnostics but not the winning download path

### Chrome automation

#### Headless Chrome with cloned profile

Result:

- still blocked by Radware

#### Visible Chrome with cloned profile

Result:

- no anti-bot block
- redirected to `User Login`

Conclusion:

- visible mode reduced anti-bot issues
- profile clone alone did not fully recover authenticated paper access

#### Visible Chrome with injected cookies

Result:

- OFC redirect successfully resolved to authenticated `directpdfaccess/...pdf` URLs
- browser loaded PDF content successfully

This was the turning point that proved the cookies were sufficient.

## Working PDF Caching Strategy

Current working method:

1. Use visible Chrome.
2. Seed the session with an exported cookie set for the relevant OFC/Optica domains.
3. Open the OFC/Optica paper link.
4. Let the browser resolve the authenticated final `directpdfaccess/...pdf` URL.
5. Use Python with the same cookies to fetch the PDF bytes directly.
6. Save the PDF into cache.
7. Record the result in `.cache/ofc/paper_cache.json`.

This avoids fighting Chrome’s built-in PDF viewer and produces deterministic file output.

## Successful End-to-End Cached PDFs

Verified cached PDFs:

- `M1D.3` Silicon Nitride LiDAR Chip With Minimal Back-Reflection for Timing Jitter Compensation
- `M1D.4` Wafer-Scale Heterogeneously Integrated Self-Injection-Locked Lasers
- `M1D.5` Ultrafast FMCW LiDAR With Micrometer Resolution Enabled by a Quantum Walk Comb Laser

Verified files:

- `.cache/ofc/pdfs/M1D.3-ad8245e0-67d3-4030-9dca98a73f5853ca_so4440380.pdf`
- `.cache/ofc/pdfs/M1D.4-5f80a1e7-1fff-43f5-950be4c4dfdc4d32_so4441152.pdf`
- `.cache/ofc/pdfs/M1D.5-e45591fb-511d-49eb-a209d035579383de_so4440909.pdf`

These were confirmed as real PDF files via `file`.

## Current Export Status

Latest tested summary:

- `records`: 1003
- `presentations`: 830
- `sessions_without_presentations`: 173
- `paper_ready`: 677
- `paper_links_available`: 677
- `paper_links_tested`: 10
- `paper_cached`: 3
- `status_cached_pdf`: 3

### Full Metadata Export Totals

Latest full-conference metadata export:

- `records`: 1003
- `presentations`: 830
- `sessions_without_presentations`: 173
- `paper_ready`: 677
- `paper_links_structured`: 0
- `paper_links_description`: 677
- `paper_links_available`: 677

### PDF Coverage So Far

At the time of this log update:

- `677` records have paper links available in metadata
- `3` PDFs have been cached successfully end-to-end
- `674` linked papers remain uncached

Current cached-PDF count by status:

- `status_cached_pdf`: 3

Current known tested status counts from cache-bearing exports:

- `status_cached_pdf`: 3
- `status_blocked_by_bot`: 12
- `status_not_tested`: 662
- `status_not_available`: 326

## Learnings

### Data model learnings

- OFC schedule data is richer than the visible UI suggests.
- Technical paper access data is present in the API, but not always in the structured field the frontend uses.
- Short courses and special sessions require separate handling from standard presentations.

### Browser / anti-bot learnings

- Headless mode is materially different and is not acceptable for this site’s protected paper flow.
- Visible Chrome with valid cookies is the closest match to the user’s successful manual browser behavior.
- Cookie-based recovery is sufficient for at least part of the authenticated paper flow.
- Rate limiting and anti-bot escalation must be treated as a hard operational constraint, not as a soft optimization concern.

### Rate-Limit Incident

Observed behavior:

- After the authenticated visible-browser path was proven on a small sample, additional protected-link status checks were attempted too aggressively.
- Even without downloading PDFs, repeated protected navigations to `opg.optica.org` triggered Radware anti-bot escalation on the active network/session.

Correction:

- Protected Optica requests must remain in very small visible-browser batches only.
- No broad protected-link census or sweep should be run.
- Any anti-bot warning is a hard stop condition.

Standing rule for follow-on work:

- Do not run headless access against protected Optica paper endpoints.
- Do not run broad status sweeps against protected Optica paper endpoints.
- Use only low-rate visible-browser batches.
- Stop immediately on anti-bot or rate-limit signals.

### Download learnings

- Browser rendering success is not equivalent to file caching success.
- Resolving the final authenticated PDF URL in-browser, then downloading the bytes directly, is more reliable than trying to force browser-managed downloads.

## Known Constraints

- Full-scale downloading should still be rate-limited and batched carefully.
- Cookie validity is time-sensitive.
- Some papers may still fail depending on session expiry, access rules, or site protections.
- The active network reputation can degrade if protected endpoints are probed too aggressively, even when cookies are valid.

## Recommended Next Steps

1. Scale PDF caching in small batches rather than full-conference bulk fetches.
2. Continue recording negative states in cache to avoid unnecessary retesting.
3. Add a resumable batch runner for the remaining uncached paper links.
4. Consider separating:
   - metadata export
   - authenticated paper resolution
   - PDF byte download
   into distinct commands for easier recovery and retries.
5. Treat protected-endpoint rate limiting as a primary design constraint for all future runs.
