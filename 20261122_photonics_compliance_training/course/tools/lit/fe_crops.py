# Crops of CC BY 4.0 figures for the ferroelectric course literature tabs.
# Sources are read from research/lit_sources/fe/ (see the README there for URL, license, date).
# Run from the project root:
#   uv run --no-project --with pymupdf --with pillow python course/tools/lit/fe_crops.py
# PDF clips are in PDF points (x0, y0, x1, y1) on the given 0-based page; PNG boxes are pixels.
import io
import os

import pymupdf
from PIL import Image

SRC = 'research/lit_sources/fe'
OUT = 'course/media/lit'
MAXSIDE = 1600


def finish(im, name, pad=10):
    im = im.convert('RGB')
    s = min(1.0, MAXSIDE / max(im.size))
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    out = Image.new('RGB', (im.width + 2 * pad, im.height + 2 * pad), (255, 255, 255))
    out.paste(im, (pad, pad))
    if name.endswith('.jpg'):
        out.save(os.path.join(OUT, name), quality=86, optimize=True)
    else:
        out.save(os.path.join(OUT, name), optimize=True)
    print(name, out.size, os.path.getsize(os.path.join(OUT, name)) // 1024, 'KB')


def pdf_clip(pdf, page, clip, dpi, name):
    d = pymupdf.open(os.path.join(SRC, pdf))
    pix = d[page].get_pixmap(dpi=dpi, clip=pymupdf.Rect(*clip))
    finish(Image.open(io.BytesIO(pix.tobytes('png'))), name)


def png_box(png, box, name):
    finish(Image.open(os.path.join(SRC, png)).crop(box), name)


# Wang et al., Nature 629, 784 (2024), Fig. 2g. CC BY 4.0 (Springer Nature figure file). Bottom 742 px: axis label descenders end at 741, panel i starts at 742.
png_box('wang2024_nature_fig2.png', (982, 392, 1955, 742), 'fe_wang2024_fig2g.png')

# Wang, Chen, Dai, Liu, IEEE Photonics Journal 12(3), 2200310 (2020), Fig. 6(b,c). CC BY 4.0.
# Panels (b,c) end at x = 335.9 pt, the outer edge of the frame line shared with panel d. The legend row spans all four panels; its z-cut half
# (x 339 to 481 pt) is moved under the x-cut half so the dashed z-cut references stay labelled.
d = pymupdf.open(os.path.join(SRC, 'wang2020_ieeepj.pdf'))
def clip_im(rect, dpi=520):
    return Image.open(io.BytesIO(d[6].get_pixmap(dpi=dpi, clip=pymupdf.Rect(*rect)).tobytes('png'))).convert('RGB')
legx = clip_im((177, 255, 335.9, 266.8))
legz = clip_im((339, 260.0, 481, 265.8))
body = clip_im((177, 266.8, 335.9, 362))
k = 520 / 72
comp = Image.new('RGB', (body.width, legx.height + legz.height + body.height), (255, 255, 255))
comp.paste(legx, (0, 0))
comp.paste(legz, (int(round((192.5 - 177) * k)), legx.height))
comp.paste(body, (0, legx.height + legz.height))
finish(comp, 'fe_wang2020_fig6bc.png')

# Liu et al., Light: Advanced Manufacturing 4, 13 (2023), Fig. 3e. CC BY 4.0.
pdf_clip('liu2023_lam.pdf', 5, (396.2, 264.6, 507.5, 352.3), 600, 'fe_liu2023_fig3e.jpg')

# Xu et al., arXiv:2012.12671 (CC BY 4.0), published in Optics Express 29, 5497 (2021), Fig. 5.
pdf_clip('xu2021_arxiv_2012.12671.pdf', 5, (172, 383, 437, 614), 330, 'fe_xu2021_fig5.png')

# Shams-Ansari et al., arXiv:2203.17133 (CC BY 4.0), published in APL Photonics 7, 081301 (2022), Fig. 2.
pdf_clip('shams2022_arxiv_2203.17133.pdf', 1, (190, 410, 422, 575), 380, 'fe_shams2022_fig2.png')

# Zhang et al., arXiv:2606.06637 (CC BY 4.0), Fig. 2(d,e).
pdf_clip('zhang2026_arxiv_2606.06637.pdf', 6, (228, 181.4, 542, 322), 380, 'fe_zhang2026_fig2de.png')

# Liu et al., ACS Photonics 13, 3951 (2026), Fig. 5a. CC BY 4.0.
pdf_clip('liu2026_acsphotonics.pdf', 6, (58, 58, 302, 222), 400, 'fe_liu2026_fig5a.png')

# Zheng et al., arXiv:2402.16513 (CC BY 4.0), published in Laser and Photonics Reviews 18, 2400565 (2024), Fig. 2.
pdf_clip('zheng2024_arxiv_2402.16513.pdf', 16, (122.9, 414.5, 472.3, 580.3), 300, 'fe_zheng2024_fig2.jpg')
