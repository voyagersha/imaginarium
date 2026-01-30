#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _load_deps():
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
    except Exception as exc:
        raise SystemExit(
            "Missing dependencies. Run this with the ComfyUI venv:\n"
            "  vendor/ComfyUI/.venv/bin/python scripts/mask_refine.py --help\n"
            f"Error: {exc}"
        )
    return np, Image, ImageDraw, ImageFont, ImageFilter


def _overlay_preview(Image, ImageDraw, ImageFont, base_image, mask_binary, out_path: Path) -> None:
    base = base_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 120, 255, 0))
    alpha = Image.fromarray((mask_binary * 140).astype("uint8"), mode="L")
    overlay.putalpha(alpha)
    preview = Image.alpha_composite(base, overlay).convert("RGB")

    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    draw.rectangle((6, 6, 96, 24), fill=(0, 0, 0))
    draw.text((10, 8), "mask", fill=(255, 255, 255), font=font)
    preview.save(out_path)


def _apply_morph(ImageFilter, mask_img, erode, dilate):
    if erode > 0:
        size = max(1, int(erode) * 2 + 1)
        mask_img = mask_img.filter(ImageFilter.MinFilter(size=size))
    if dilate > 0:
        size = max(1, int(dilate) * 2 + 1)
        mask_img = mask_img.filter(ImageFilter.MaxFilter(size=size))
    return mask_img


def _apply_crop(mask_binary, crop):
    x, y, w, h = crop
    h_img, w_img = mask_binary.shape
    x = max(0, min(x, w_img))
    y = max(0, min(y, h_img))
    w = max(0, min(w, w_img - x))
    h = max(0, min(h, h_img - y))
    out = mask_binary * 0
    out[y : y + h, x : x + w] = mask_binary[y : y + h, x : x + w]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine a binary mask (crop/threshold/morphology).")
    parser.add_argument("--mask", required=True, help="Input mask image path.")
    parser.add_argument("--image", help="Optional base image for preview.")
    parser.add_argument("--out-dir", required=True, help="Output directory for mask + preview.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Mask threshold (0-1).")
    parser.add_argument("--invert", action="store_true", help="Invert mask.")
    parser.add_argument("--erode", type=int, default=0, help="Erode mask by N pixels.")
    parser.add_argument("--dilate", type=int, default=0, help="Dilate mask by N pixels.")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="Crop region in pixels.")
    parser.add_argument(
        "--crop-rel",
        nargs=4,
        type=float,
        metavar=("X", "Y", "W", "H"),
        help="Crop region as ratios (0-1) of width/height.",
    )
    parser.add_argument(
        "--subtract",
        action="append",
        default=[],
        help="Mask image to subtract (repeatable).",
    )
    parser.add_argument("--subtract-threshold", type=float, default=0.3, help="Threshold for subtract masks (0-1).")
    args = parser.parse_args()

    np_mod, Image, ImageDraw, ImageFont, ImageFilter = _load_deps()

    mask_path = Path(args.mask)
    if not mask_path.exists():
        raise SystemExit(f"Mask not found: {mask_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_img = Image.open(mask_path).convert("L")
    mask_img = _apply_morph(ImageFilter, mask_img, args.erode, args.dilate)
    mask_arr = np_mod.array(mask_img, dtype=np_mod.float32) / 255.0
    mask_binary = (mask_arr >= float(args.threshold)).astype(np_mod.uint8)

    if args.invert:
        mask_binary = 1 - mask_binary

    if args.subtract:
        for sub in args.subtract:
            sub_path = Path(sub)
            if not sub_path.exists():
                raise SystemExit(f"Subtract mask not found: {sub_path}")
            sub_img = Image.open(sub_path).convert("L")
            sub_arr = np_mod.array(sub_img, dtype=np_mod.float32) / 255.0
            sub_bin = (sub_arr >= float(args.subtract_threshold)).astype(np_mod.uint8)
            mask_binary = np_mod.clip(mask_binary - sub_bin, 0, 1)

    if args.crop_rel:
        x, y, w, h = args.crop_rel
        w_img, h_img = mask_img.size
        crop = (
            int(x * w_img),
            int(y * h_img),
            int(w * w_img),
            int(h * h_img),
        )
        mask_binary = _apply_crop(mask_binary, crop)
    elif args.crop:
        mask_binary = _apply_crop(mask_binary, tuple(args.crop))

    mask_out = out_dir / "mask.png"
    preview_out = out_dir / "preview.png"
    Image.fromarray(mask_binary * 255, mode="L").save(mask_out)

    if args.image:
        image_path = Path(args.image)
        if image_path.exists():
            base_image = Image.open(image_path).convert("RGB")
            _overlay_preview(Image, ImageDraw, ImageFont, base_image, mask_binary, preview_out)

    print(f"Saved mask to {mask_out}")
    if preview_out.exists():
        print(f"Saved preview to {preview_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
