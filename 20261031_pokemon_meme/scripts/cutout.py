"""Segment the CPO package from marvell-CPO.png (rembg BiRefNet) and write cutout + silhouette."""
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import new_session, remove

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CROP_BOX = (780, 560, 1930, 970)  # CPO region in source px, with margin
SIL_RGB = (20, 24, 52)  # dark navy silhouette fill

src = Image.open(ROOT / "marvell-CPO.png").convert("RGB")
cut = remove(src.crop(CROP_BOX), session=new_session("birefnet-general"))
a = np.array(cut)[..., 3]
ys, xs = np.nonzero(a > 8)
cut = cut.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
cut.save(OUT / "cpo_cutout.png")

sil = np.zeros((*np.array(cut).shape[:2], 4), np.uint8)
sil[..., :3] = SIL_RGB
sil[..., 3] = np.array(cut)[..., 3]
Image.fromarray(sil).save(OUT / "cpo_silhouette.png")
print("cutout size", cut.size)
