# Wi-Fi speed stats — CSV replay

A dependency-free frontend that rebuilds a [speed.cloudflare.com](https://speed.cloudflare.com)
result page from its CSV export alone. Groundwork for a community Wi-Fi speed map
where people upload their own test results.

## Run

```sh
python3 -m http.server 8777   # then open http://localhost:8777/index.html
```

A server is only needed so `fetch()` can read the CSV; the file picker in the
header works over `file://` too, and accepts any Cloudflare results CSV.

## Files

| File | Role |
| --- | --- |
| `index.html` | Layout: overview tiles, sparklines, quality score, map, measurement cards |
| `app.css` | Styling, light + dark, responsive |
| `app.js` | CSV parse → statistics → SVG charts, Leaflet map, hover tooltips |
| `speed-results-20260810.csv` | Sample export (46 samples, Reno-Tahoe Airport) |
| `tmp.html.html` | The saved original page, used as the reference to reproduce |

External deps: Leaflet + OpenStreetMap tiles from CDN, for the map card only.

## Metrics

Formulas were reverse-engineered by matching this CSV against the numbers printed
in the saved page:

| Value | Formula |
| --- | --- |
| Download / upload speed | **90th percentile** of `bps` for that direction |
| Unloaded latency | median of the `latency` column |
| Jitter | mean absolute difference between consecutive samples |
| Loaded latency / jitter | same statistics over `loadedLatencies`, split by direction |

The 90th-percentile rule reproduces the source page exactly — 70.1 and 95.5 Mbps
to the printed digit. Latency figures differ slightly because the export and the
saved page were captured at different moments in the run.

Network Quality Score thresholds are **approximate**: Cloudflare does not publish
its AIM cutoffs, so they were fitted to reproduce the three "Good" ratings. See
`scoreNetwork()` in `app.js`.

## What the CSV cannot tell you

Packet loss, server location, ASN/ISP, client IP, and anything geographic exist
only in the page, never in the export. They live in a hard-coded `META` object at
the top of `app.js`, copied from the saved page and labeled `not in CSV` in the UI.

This is the load-bearing constraint for the map idea: **a CSV alone cannot be
placed on a map.** The uploader has to name the place.

## Notes toward a community site

Frontend-only is enough to render one person's own file, but not to collect
submissions — a browser cannot write to storage other visitors can read. The
cheapest workable shape:

- **Pages** for the static site, **one Worker** for `POST /submit`, **D1** for rows.
- Skip R2 — it requires a payment method even inside its free allowance. A ~6 KB
  CSV fits in a D1 `TEXT` column; 5 GB free ≈ 800k submissions.
- Serve the map a cached aggregate JSON. Never query D1 on page load, and never
  trigger a Pages rebuild per submission (500 builds/month ≈ 16/day).
- Update a per-location summary row incrementally. Recomputing every row in one
  request is what would hit the free plan's 10 ms CPU ceiling.
- Cache geocoding results keyed by the place string so a repeated label never
  geocodes twice.
- Store `serverCity` per submission. A Cloudflare test measures the path to an
  edge PoP, so without it "Reno airport is slow" may really mean "that day it
  routed somewhere distant."

At community scale this all sits inside free tiers; the real costs are a domain
and a geocoding provider.
