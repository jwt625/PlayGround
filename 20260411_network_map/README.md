# X Verified Followers Scraper

This project contains a Selenium-based scraper for exporting follower profile metadata from an X followers page while reusing an already-authenticated local Firefox profile.

It was built for the verified followers page first:

- `https://x.com/<account>/verified_followers`

The scraper collects:

- display name
- handle
- profile description
- profile URL
- profile image URL
- optional downloaded profile image path

## Safety Notes

- The script does not store account passwords.
- It works by cloning a local Firefox profile into a temporary directory and launching Selenium against that temporary copy.
- The temporary profile is deleted after the run.
- The README intentionally does not include any local profile path, account-specific identifier, or scraped follower data.
- To avoid stressing the X session, run one scrape at a time and keep a reasonable pause between scrolls.

## Setup

Create a virtual environment with `uv` and install Selenium:

```bash
uv venv .venv
uv pip install --python .venv/bin/python selenium
```

## Usage

Basic smoke test:

```bash
./.venv/bin/python scrape_verified_followers.py --mode smoke
```

Inspect the page structure and save a screenshot:

```bash
./.venv/bin/python scrape_verified_followers.py --mode inspect
```

Run a scrape:

```bash
./.venv/bin/python scrape_verified_followers.py \
  --mode scrape \
  --pause 1.5 \
  --max-scrolls 250 \
  --stagnant-limit 8 \
  --scroll-fraction 0.8 \
  --min-overlap-ratio 0.15 \
  --output-prefix verified_followers
```

Download profile images too:

```bash
./.venv/bin/python scrape_verified_followers.py \
  --mode scrape \
  --download-images \
  --output-prefix verified_followers
```

## Overlap Coverage Check

The scraper now performs an overlap audit between adjacent scroll windows.

- Each step records the currently visible follower handles.
- It compares the current visible window with the previous one.
- It reports `overlap` and `overlap_ratio` in the console.
- It writes a per-scroll audit file to `output/<prefix>_audit.json`.
- If adjacent windows overlap less than the configured minimum, it prints an `overlap_warning`.

This helps detect cases where scrolling moves too far and risks skipping entries in a virtualized list.

## Outputs

The script writes:

- `output/<prefix>.json`
- `output/<prefix>.csv`
- `output/<prefix>_audit.json`
- optionally `output/profile_images/`

## Notes

- X uses lazy loading and virtualization, so complete scraping depends on careful scrolling and enough time for new items to render.
- The default behavior is tuned to prefer overlap and completeness over raw speed.
- If the site layout changes, selectors may need to be updated.
