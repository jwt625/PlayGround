"""Photo avatars: square crops resized to 256 px JPEG.
Usage: uv run --no-project --with pillow python photos.py COURSE_DIR DOWNLOAD_DIR OUT_DIR
DOWNLOAD_DIR holds the 960 px Commons thumbnails fetched with commons.py (1 request per second):
  corgi.jpg = File:Fawn and white Welsh Corgi puppy standing on rear legs and sticking out the tongue.jpg (CC0)
  moon.jpg  = File:FullMoon2010.jpg (Gregory H. Revera, CC BY-SA 3.0)
Gary is a crop of media/fe/orient_microscope_operator.jpg (public domain).
"""
import sys
from PIL import Image, ImageFilter, ImageEnhance
SRC = sys.argv[1]; DL = sys.argv[2]; OUT = sys.argv[3]
def save(im, out):
    im.resize((256, 256), Image.LANCZOS).save(out, quality=86, optimize=True, progressive=True)
def crop(src, box, out):
    save(Image.open(src).convert("RGB").crop(box), out)
crop(f"{SRC}/media/fe/orient_microscope_operator.jpg", (840, 0, 1640, 800), f"{OUT}/gary_bunnysuit.jpg")
crop(f"{DL}/corgi.jpg", (170, 320, 690, 840), f"{OUT}/newhire_corgi.jpg")
m = Image.open(f"{DL}/moon.jpg").convert("RGB")
S = 1180
canvas = Image.new("RGB", (S, S), (0, 0, 0))
canvas.paste(m, ((S - m.width) // 2, (S - m.height) // 2))
save(canvas, f"{OUT}/prevshift_moon.jpg")
