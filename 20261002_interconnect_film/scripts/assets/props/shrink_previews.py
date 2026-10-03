"""Post-process: quantise preview PNGs (256 colours, no alpha) to keep the category previews under 25 MB.

Run with the shell python: python3 scripts/assets/props/shrink_previews.py assets/components/props/previews
Needs ImageMagick (magick) on PATH; only touches *.png in the given folder.
"""
import glob
import subprocess
import sys

for f in sorted(glob.glob(sys.argv[1] + "/*.png")):
    subprocess.run(["magick", f, "-alpha", "off", "-colors", "256", "+dither", "-define", "png:compression-level=9", "-define", "png:format=png8", f],
                   check=True, stderr=subprocess.DEVNULL)
