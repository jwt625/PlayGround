# X Followers Scraper

This project contains a Selenium-based scraper for exporting profile metadata from X list pages while reusing an already-authenticated local Firefox profile.

Supported list types:

- `https://x.com/<account>/verified_followers`
- `https://x.com/<account>/followers`
- `https://x.com/<account>/following`

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
  --target-kind verified_followers \
  --pause 1.5 \
  --max-scrolls 250 \
  --stagnant-limit 8 \
  --scroll-fraction 0.8 \
  --min-overlap-ratio 0.15 \
  --output-prefix verified_followers
```

Scrape the full followers list:

```bash
./.venv/bin/python scrape_verified_followers.py \
  --mode scrape \
  --target-kind followers \
  --output-prefix followers
```

Scrape the following list:

```bash
./.venv/bin/python scrape_verified_followers.py \
  --mode scrape \
  --target-kind following \
  --output-prefix following
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

## Resume Support

The scraper supports interruption-safe checkpointing.

- It writes a checkpoint file during scraping.
- The checkpoint stores collected records, overlap audit entries, and the most recent scroll position metadata.
- You can restart the scrape with `--resume` to continue from the prior checkpoint instead of starting over.
- Final successful runs remove the checkpoint file after the main outputs are written.

Example:

```bash
./.venv/bin/python scrape_verified_followers.py \
  --mode scrape \
  --target-kind verified_followers \
  --output-prefix verified_followers_formal \
  --checkpoint-every 1
```

Resume later:

```bash
./.venv/bin/python scrape_verified_followers.py \
  --mode scrape \
  --target-kind verified_followers \
  --output-prefix verified_followers_formal \
  --resume
```

## Local Viewer

A lightweight local web UI is included for browsing scraped datasets with search, sort, and multiple layouts.

Start the viewer:

```bash
python3 serve_viewer.py
```

Then open:

```text
http://127.0.0.1:8000
```

Viewer features:

- dataset picker for available `output/*.json` files
- search by name, handle, or bio
- sort by name, handle, or bio length
- filter chips for bio and image presence
- three layouts: `Tiles`, `Cover Flow`, and `List`
- hover details with name, handle, bio, and a direct profile link

## Outputs

The script writes:

- `output/<prefix>.json`
- `output/<prefix>.csv`
- `output/<prefix>_audit.json`
- `output/<prefix>_checkpoint.json` while a scrape is in progress
- optionally `output/profile_images/`

## Notes

- X uses lazy loading and virtualization, so complete scraping depends on careful scrolling and enough time for new items to render.
- The default behavior is tuned to prefer overlap and completeness over raw speed.
- If the site layout changes, selectors may need to be updated.
