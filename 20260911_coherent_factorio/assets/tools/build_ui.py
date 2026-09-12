"""Build original vector UI assets; no raster image processing."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
icons = {
'metal':'<path d="m3 9 13-4 5 5-13 5Z M3 9v6l5 5 13-5v-5 M8 15v5"/>',
'ceramic':'<path d="m3 9 9-5 9 5-9 5Z M3 9v7l9 5 9-5V9 M12 14v7"/>',
'precision':'<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 1v4 M12 19v4 M1 12h4 M19 12h4"/>',
'build':'<path d="M4 10 12 5l8 5v10H4Z M9 20v-6h6v6"/>',
'power':'<path d="m14 2-9 12h6l-1 8 9-12h-6Z"/>',
'field':'<circle cx="12" cy="12" r="3"/><path d="M7 5a9 9 0 0 0 0 14 M17 5a9 9 0 0 1 0 14"/>',
'reference':'<path d="M2 12h4l3-7 6 14 3-7h4"/>',
'return':'<path d="M5 7h10a6 6 0 0 1 0 12h-5 M5 7l4-4 M5 7l4 4"/>',
'heat':'<path d="M7 20c-5-5 4-7 0-12 M12 20c-5-6 4-9 0-16 M17 20c-5-5 4-7 0-12"/>',
'phase':'<circle cx="12" cy="12" r="9"/><path d="M12 12 17 6 M12 12H5"/>',
'locked':'<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V6a4 4 0 0 1 8 0v4 M12 14v3"/>',
'unlocked':'<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V6a4 4 0 0 1 8 0"/>',
'warning':'<path d="m12 3 10 18H2Z M12 9v5 M12 17v1"/>',
'check':'<path d="m4 12 5 5L20 6"/>',
'inspect':'<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6 M7 10h6 M10 7v6"/>',
'ore':'<path d="m3 16 3-9 8-4 7 8-4 9H7Z M6 7l6 6 9-2 M12 13l-5 7"/>',
'component':'<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4 M15 2v4 M9 18v4 M15 18v4 M2 9h4 M2 15h4 M18 9h4 M18 15h4"/>',
'scrap':'<path d="m4 8 6-4 3 5 7-2-2 12-8 1-6-5Z M10 4l-1 9 9 6"/>',
'blueprint':'<rect x="4" y="3" width="16" height="18" rx="1"/><path d="M8 7h8v5H8Z M8 16h3 M14 16h2"/>',
'play':'<path d="m8 4 12 8-12 8Z"/>',
'stop':'<rect x="5" y="5" width="14" height="14" rx="1"/>',
'cooling':'<path d="M12 2v20 M3 7l18 10 M3 17 21 7 M9 4l3 3 3-3 M9 20l3-3 3 3"/>',
'route':'<path d="M3 5h8v14h10"/><circle cx="3" cy="5" r="2"/><circle cx="21" cy="19" r="2"/>',
}
folder=ROOT/'ui'/'icons'; folder.mkdir(parents=True,exist_ok=True)
for name, geometry in icons.items():
    (folder/f'{name}.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#d8e4e6" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><title>{name}</title>{geometry}</svg>\n')
print(f'Created {len(icons)} SVG icons')
