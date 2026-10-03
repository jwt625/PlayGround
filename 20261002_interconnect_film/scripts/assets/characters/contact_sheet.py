"""Blender -b --python contact_sheet.py -- <out.png> <in1.png> <in2.png> ...  (downscale 2x, centre-crop, tile in a row)"""
import sys
import bpy
import numpy as np
a = sys.argv[sys.argv.index("--") + 1:]
out, ins = a[0], a[1:]
tiles = []
for p in ins:
    im = bpy.data.images.load(p)
    w, h = im.size
    px = np.array(im.pixels[:], np.float32).reshape(h, w, 4)
    px = px[: h // 2 * 2]
    px = px.reshape(h // 2, 2, w // 2, 2, 4).mean(axis=(1, 3))
    cw = px.shape[1] // 2
    x0 = (px.shape[1] - cw) // 2
    tiles.append(px[:, x0:x0 + cw])
arr = np.concatenate(tiles, axis=1)
img = bpy.data.images.new("s", arr.shape[1], arr.shape[0])
img.pixels = arr.ravel().tolist()
img.filepath_raw = out
img.file_format = "PNG"
img.save()
