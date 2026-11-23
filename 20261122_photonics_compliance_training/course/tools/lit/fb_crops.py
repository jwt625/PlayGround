# Crops of CC BY / CC BY-SA licensed figures for the feedback course literature tabs.
# Sources (downloaded by hand, see research/lit_sources/fb/README.md) are read from research/lit_sources/fb/.
# Run from the project root:
#   uv run --no-project --with pillow python course/tools/lit/fb_crops.py
# Needs poppler (pdftoppm, pdfimages) on PATH. Boxes are fractions (x0, y0, x1, y1) of the source image.
import os
import subprocess
import tempfile

from PIL import Image

SRC = 'research/lit_sources/fb'
OUT = 'course/media/lit'
MAXSIDE = 1600


def flatten(rgb_path, mask_path=None):
    im = Image.open(rgb_path).convert('RGB')
    if mask_path:
        m = Image.open(mask_path).convert('L').resize(im.size)
        bg = Image.new('RGB', im.size, (255, 255, 255))
        bg.paste(im, (0, 0), m)
        im = bg
    return im


def pdf_page(pdf, dpi, page=1):
    tmp = tempfile.mkdtemp()
    subprocess.run(['pdftoppm', '-r', str(dpi), '-png', '-f', str(page), '-l', str(page), pdf, tmp + '/p'], check=True)
    f = sorted(os.listdir(tmp))[0]
    return Image.open(os.path.join(tmp, f)).convert('RGB')


def pdf_raster(pdf, page):
    """Embedded raster images of one PDF page, alpha mask applied onto white when present."""
    tmp = tempfile.mkdtemp()
    subprocess.run(['pdfimages', '-png', '-f', str(page), '-l', str(page), pdf, tmp + '/i'], check=True)
    return sorted(os.path.join(tmp, f) for f in os.listdir(tmp))


def crop(im, box, name, pad=8):
    W, H = im.size
    x0, y0, x1, y1 = box
    c = im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
    s = min(1.0, MAXSIDE / max(c.size))
    if s < 1:
        c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
    out = Image.new('RGB', (c.width + 2 * pad, c.height + 2 * pad), (255, 255, 255))
    out.paste(c, (pad, pad))
    out.save(os.path.join(OUT, name), optimize=True)
    print(name, out.size)


# Bao et al., arXiv:2405.16583 (CC BY-SA 4.0), page 9, Fig. 4(a,b)
f = pdf_raster(SRC + '/bao2024_arxiv2405.16583.pdf', 9)
crop(flatten(f[0], f[1]), (0.0, 0.0, 0.665, 1.0), 'fb_bao2024_fig4ab.png')

# De Koninck et al., arXiv:2309.04473 (CC BY 4.0), page 37, Fig. S5(a)
f = pdf_raster(SRC + '/dekoninck_arxiv2309.04473.pdf', 37)
crop(flatten(f[1], f[2]), (0.0, 0.0, 1.0, 0.385), 'fb_dekoninck2023_figS5a.png')

# Lapointe et al., Research Square rs-3157765 (CC BY 4.0), page 11, Fig. 3(g) and 3(k)
f = pdf_raster(SRC + '/lapointe_researchsquare_rs-3157765.pdf', 11)
im = flatten(f[0])
crop(im, (0.69, 0.0, 0.94, 0.56), 'fb_lapointe2023_fig3g.png')
crop(im, (0.535, 0.58, 1.0, 1.0), 'fb_lapointe2023_fig3k.png')

# Kramnik et al., arXiv:2411.05921 (CC BY 4.0), source figure fig2.pdf, Fig. 2(c)
crop(pdf_page(SRC + '/kramnik_arxiv2411.05921/figures/fig2.pdf', 300), (0.61, 0.49, 1.0, 1.0), 'fb_kramnik2024_fig2c.png')

# Franken et al., arXiv:2407.00269 (CC BY 4.0), source figures, Fig. 3(a) and Fig. 4(d)
crop(pdf_page(SRC + '/franken_arxiv2407.00269/Figure_3.pdf', 300), (0.03, 0.0, 0.495, 0.485), 'fb_franken2024_fig3a.png')
crop(pdf_page(SRC + '/franken_arxiv2407.00269/Figure_4.pdf', 300), (0.0, 0.714, 1.0, 1.0), 'fb_franken2024_fig4d.png')

# Xu et al., Sci. Rep. 11, 16426 (2021) (CC BY 4.0), Fig. 5(a)
crop(flatten(SRC + '/xu2021_scirep_fig5.png'), (0.0, 0.0, 0.196, 1.0), 'fb_xu2021_fig5a.png')

# Raja et al., Nat. Commun. 10, 680 (2019) (CC BY 4.0), Fig. 2(d) inset. The right edge keeps the 0.6 tick label
# (ends at x = 433 px); a spike of the main panel's spectrum under it (x >= 428 px, y >= 732 px) is blanked.
im = flatten(SRC + '/raja2019_ncomms_fig2.png')
im.paste((255, 255, 255), (420, 731, 460, 741))
crop(im, (0.027, 0.37, 0.222, 0.607), 'fb_raja2019_fig2d_inset.png')

# Chen et al., Light Sci. Appl. 15, 265 (2026) (CC BY 4.0), Fig. 2(d), top row
crop(flatten(SRC + '/chen2026_lsa_fig2.png'), (0.51, 0.0, 1.0, 0.49), 'fb_chen2026_fig2d.png')
