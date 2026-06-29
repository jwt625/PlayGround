from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw
from scipy import optimize
from skimage import measure, metrics, transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reproduce_cmn import build_figure  # noqa: E402


def render_pdf_page(pdf_path: Path, page_index: int, dpi: int) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def find_largest_dark_panel(image: Image.Image, threshold: int = 35) -> tuple[int, int, int, int]:
    arr = np.asarray(image.convert("RGB"))
    dark = np.all(arr < threshold, axis=2)
    labeled = measure.label(dark, connectivity=2)
    props = measure.regionprops(labeled)
    candidates = []
    for prop in props:
        y0, x0, y1, x1 = prop.bbox
        width = x1 - x0
        height = y1 - y0
        if width < 100 or height < 100:
            continue
        candidates.append((prop.area, x0, y0, x1, y1))
    if not candidates:
        raise RuntimeError("could not find a dark plot panel")
    _, x0, y0, x1, y1 = max(candidates, key=lambda item: item[0])
    return x0, y0, x1, y1


def pad_bbox(bbox: tuple[int, int, int, int], image: Image.Image, pad: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(image.width, x1 + pad),
        min(image.height, y1 + pad),
    )


def render_generated_plot(npz_path: Path, out_png: Path, width: int, height: int, scale: int) -> Image.Image:
    data = np.load(npz_path)
    fig = build_figure(data["m"], data["n"], data["cmn_db"])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(out_png, width=width, height=height, scale=scale)
    return Image.open(out_png).convert("RGB")


def resize_to(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.BICUBIC)


def warp_array(
    source: np.ndarray,
    output_shape: tuple[int, int],
    tx: float,
    ty: float,
    sx: float,
    sy: float,
) -> np.ndarray:
    height, width = output_shape
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    matrix = np.array(
        [
            [sx, 0.0, cx + tx - sx * cx],
            [0.0, sy, cy + ty - sy * cy],
            [0.0, 0.0, 1.0],
        ]
    )
    tform = transform.AffineTransform(matrix=matrix)
    warped = transform.warp(
        source,
        inverse_map=tform.inverse,
        output_shape=output_shape,
        order=3,
        mode="constant",
        cval=0.0,
        preserve_range=True,
    )
    return np.clip(warped, 0.0, 255.0)


def optimize_registration(
    source: Image.Image,
    generated: Image.Image,
    max_shift_frac: float = 0.18,
    scale_bounds: tuple[float, float] = (0.65, 1.45),
    work_width: int = 420,
) -> tuple[Image.Image, dict[str, float | int | list[float]]]:
    target_size = generated.size
    work_height = max(1, int(target_size[1] * work_width / target_size[0]))
    src_work = np.asarray(resize_to(source, (work_width, work_height))).astype(np.float32)
    gen_work = np.asarray(resize_to(generated, (work_width, work_height))).astype(np.float32)

    signal_weight = (np.any(gen_work > 40, axis=2)).astype(np.float32)
    signal_weight += 0.35 * (np.any(src_work > 40, axis=2)).astype(np.float32)
    signal_weight = np.maximum(signal_weight, 0.05)

    max_tx = max_shift_frac * work_width
    max_ty = max_shift_frac * work_height
    lo_scale, hi_scale = scale_bounds

    def objective(params: np.ndarray) -> float:
        tx, ty, sx, sy = params
        warped = warp_array(src_work, (work_height, work_width), tx, ty, sx, sy)
        err = np.mean((warped - gen_work) ** 2, axis=2)
        return float(np.sqrt(np.sum(err * signal_weight) / np.sum(signal_weight)))

    result = optimize.minimize(
        objective,
        x0=np.array([0.0, 0.0, 1.0, 1.0]),
        method="Powell",
        bounds=[(-max_tx, max_tx), (-max_ty, max_ty), (lo_scale, hi_scale), (lo_scale, hi_scale)],
        options={"maxiter": 160, "xtol": 0.02, "ftol": 0.02},
    )

    tx_work, ty_work, sx, sy = [float(v) for v in result.x]
    scale_x = target_size[0] / work_width
    scale_y = target_size[1] / work_height
    src_full = np.asarray(source).astype(np.float32)
    warped_full = warp_array(
        src_full,
        (target_size[1], target_size[0]),
        tx_work * scale_x,
        ty_work * scale_y,
        sx,
        sy,
    )
    registered = Image.fromarray(warped_full.astype(np.uint8), "RGB")
    reg_metrics = rgb_metrics(registered, generated)
    reg_metrics.update(
        {
            "optimizer_success": bool(result.success),
            "optimizer_fun_working_rgb_rmse": float(result.fun),
            "tx_px": float(tx_work * scale_x),
            "ty_px": float(ty_work * scale_y),
            "sx": sx,
            "sy": sy,
            "work_width": int(work_width),
            "work_height": int(work_height),
        }
    )
    return registered, reg_metrics


def transforms_for(image: Image.Image) -> dict[str, Image.Image]:
    return {
        "identity": image,
        "flip_lr": image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "flip_ud": image.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        "rot180": image.transpose(Image.Transpose.ROTATE_180),
        "transpose": image.transpose(Image.Transpose.TRANSPOSE),
        "transpose_flip_lr": image.transpose(Image.Transpose.TRANSPOSE).transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "transpose_flip_ud": image.transpose(Image.Transpose.TRANSPOSE).transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        "transpose_rot180": image.transpose(Image.Transpose.TRANSPOSE).transpose(Image.Transpose.ROTATE_180),
    }


def rgb_metrics(source: Image.Image, generated: Image.Image) -> dict[str, float | list[float] | int]:
    src = np.asarray(source).astype(np.float32)
    gen = np.asarray(generated).astype(np.float32)
    diff = src - gen
    absdiff = np.abs(diff)
    signal = (np.any(src > 40, axis=2) | np.any(gen > 40, axis=2))
    signal_count = int(signal.sum())

    out: dict[str, float | list[float] | int] = {
        "mae_rgb_full": float(absdiff.mean()),
        "rmse_rgb_full": float(np.sqrt(np.mean(diff**2))),
        "mae_by_channel_full": [float(v) for v in absdiff.mean(axis=(0, 1))],
        "rmse_by_channel_full": [float(v) for v in np.sqrt(np.mean(diff**2, axis=(0, 1)))],
        "signal_pixel_count": signal_count,
        "signal_pixel_fraction": float(signal_count / signal.size),
        "ssim_full_rgb": float(metrics.structural_similarity(src, gen, channel_axis=2, data_range=255)),
    }
    if signal_count:
        sd = diff[signal]
        sad = absdiff[signal]
        out.update(
            {
                "mae_rgb_signal": float(sad.mean()),
                "rmse_rgb_signal": float(np.sqrt(np.mean(sd**2))),
                "mae_by_channel_signal": [float(v) for v in sad.mean(axis=0)],
                "rmse_by_channel_signal": [float(v) for v in np.sqrt(np.mean(sd**2, axis=0))],
            }
        )
    return out


def make_diff_heatmap(source: Image.Image, generated: Image.Image) -> Image.Image:
    src = np.asarray(source).astype(np.int16)
    gen = np.asarray(generated).astype(np.int16)
    err = np.mean(np.abs(src - gen), axis=2)
    err = np.clip(err / 160.0, 0.0, 1.0)
    r = np.clip(255 * err, 0, 255)
    g = np.clip(255 * np.maximum(0, 1 - np.abs(err - 0.5) * 2), 0, 255)
    b = np.clip(255 * (1 - err), 0, 255)
    return Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8), "RGB")


def make_contact_sheet(images: list[tuple[str, Image.Image]], out_path: Path) -> None:
    thumb_w = 420
    label_h = 32
    thumbs = []
    for label, image in images:
        thumb_h = int(image.height * thumb_w / image.width)
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (thumb_w, thumb_h + label_h), "white")
        canvas.paste(thumb, (0, label_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), label, fill=(0, 0, 0))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (thumb_w * len(thumbs), max(t.height for t in thumbs)), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, (i * thumb_w, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pdf", default="references/cartledge_2017_dsp_fiber_nonlinearities.pdf")
    parser.add_argument("--source-page", type=int, default=8, help="1-based page number")
    parser.add_argument("--npz", default="outputs/cmn_3600km_data.npz")
    parser.add_argument("--out-dir", default="outputs/eval")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--generated-width", type=int, default=980)
    parser.add_argument("--generated-height", type=int, default=760)
    parser.add_argument("--generated-scale", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_page = render_pdf_page(Path(args.source_pdf), args.source_page - 1, args.dpi)
    source_page_path = out_dir / "source_page8.png"
    source_page.save(source_page_path)

    source_bbox = find_largest_dark_panel(source_page)
    source_crop = source_page.crop(pad_bbox(source_bbox, source_page, pad=0))
    source_crop_path = out_dir / "source_plot_crop.png"
    source_crop.save(source_crop_path)

    generated_png_path = out_dir / "generated_plot_full.png"
    generated_full = render_generated_plot(
        Path(args.npz),
        generated_png_path,
        args.generated_width,
        args.generated_height,
        args.generated_scale,
    )
    generated_bbox = find_largest_dark_panel(generated_full)
    generated_crop = generated_full.crop(pad_bbox(generated_bbox, generated_full, pad=0))
    generated_crop_path = out_dir / "generated_plot_crop.png"
    generated_crop.save(generated_crop_path)

    generated_size = generated_crop.size
    trials = {}
    registered_trials = {}
    best_name = None
    best_metric = float("inf")
    best_source = None
    best_registered = None
    best_registered_name = None
    best_registered_metric = float("inf")
    for name, transformed_source in transforms_for(source_crop).items():
        aligned_source = resize_to(transformed_source, generated_size)
        trial_metrics = rgb_metrics(aligned_source, generated_crop)
        trials[name] = trial_metrics
        metric = float(trial_metrics.get("rmse_rgb_signal", trial_metrics["rmse_rgb_full"]))
        if metric < best_metric:
            best_metric = metric
            best_name = name
            best_source = aligned_source

        registered_source, registered_metrics = optimize_registration(aligned_source, generated_crop)
        registered_trials[name] = registered_metrics
        registered_metric = float(registered_metrics.get("rmse_rgb_signal", registered_metrics["rmse_rgb_full"]))
        if registered_metric < best_registered_metric:
            best_registered_metric = registered_metric
            best_registered_name = name
            best_registered = registered_source

    assert best_name is not None and best_source is not None
    assert best_registered_name is not None and best_registered is not None
    aligned_source_path = out_dir / "source_plot_aligned_to_generated.png"
    best_source.save(aligned_source_path)
    registered_source_path = out_dir / "source_plot_registered_to_generated.png"
    best_registered.save(registered_source_path)
    generated_aligned_path = out_dir / "generated_plot_aligned.png"
    generated_crop.save(generated_aligned_path)

    diff = make_diff_heatmap(best_source, generated_crop)
    diff_path = out_dir / "rgb_absdiff_heatmap.png"
    diff.save(diff_path)
    registered_diff = make_diff_heatmap(best_registered, generated_crop)
    registered_diff_path = out_dir / "rgb_absdiff_heatmap_registered.png"
    registered_diff.save(registered_diff_path)

    contact_path = out_dir / "visual_eval_contact_sheet.png"
    make_contact_sheet(
        [
            ("source, resized/flipped", best_source),
            ("source, registered", best_registered),
            ("generated crop", generated_crop),
            ("RGB diff after registration", registered_diff),
        ],
        contact_path,
    )

    payload = {
        "source_pdf": args.source_pdf,
        "source_page_1_based": args.source_page,
        "source_page_png": str(source_page_path),
        "source_plot_crop": str(source_crop_path),
        "generated_plot_full": str(generated_png_path),
        "generated_plot_crop": str(generated_crop_path),
        "source_bbox_xyxy": list(source_bbox),
        "generated_bbox_xyxy": list(generated_bbox),
        "best_resize_only_axis_alignment": best_name,
        "best_resize_only_metrics": trials[best_name],
        "best_registered_axis_alignment": best_registered_name,
        "best_registered_metrics": registered_trials[best_registered_name],
        "all_alignment_trials": trials,
        "all_registered_trials": registered_trials,
        "artifacts": {
            "source_aligned": str(aligned_source_path),
            "source_registered": str(registered_source_path),
            "generated_aligned": str(generated_aligned_path),
            "rgb_absdiff_heatmap": str(diff_path),
            "rgb_absdiff_heatmap_registered": str(registered_diff_path),
            "contact_sheet": str(contact_path),
        },
    }
    metrics_path = out_dir / "visual_eval_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("resize-only best:")
    print(json.dumps(payload["best_resize_only_metrics"], indent=2))
    print(f"best_resize_only_axis_alignment: {best_name}")
    print("registered best:")
    print(json.dumps(payload["best_registered_metrics"], indent=2))
    print(f"best_registered_axis_alignment: {best_registered_name}")
    print(f"wrote {metrics_path}")
    print(f"wrote {contact_path}")


if __name__ == "__main__":
    main()
