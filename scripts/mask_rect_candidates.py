#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_deps():
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise SystemExit(
            "Missing dependencies. Run this with the ComfyUI venv:\n"
            "  vendor/ComfyUI/.venv/bin/python scripts/mask_rect_candidates.py --help\n"
            f"Error: {exc}"
        )
    return np, Image, ImageDraw, ImageFont


def _overlay_preview(Image, ImageDraw, ImageFont, base_image, mask_binary, idx: int, out_path: Path) -> None:
    base = base_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 120, 255, 0))
    alpha = Image.fromarray((mask_binary * 140).astype("uint8"), mode="L")
    overlay.putalpha(alpha)
    preview = Image.alpha_composite(base, overlay).convert("RGB")

    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    draw.rectangle((6, 6, 54, 24), fill=(0, 0, 0))
    draw.text((10, 8), f"{idx:02d}", fill=(255, 255, 255), font=font)
    preview.save(out_path)


def _rect_mask(np_mod, width, height, cx, cy, w, h):
    mask = np_mod.zeros((height, width), dtype=np_mod.uint8)
    x0 = int(max(0, (cx - w / 2) * width))
    x1 = int(min(width, (cx + w / 2) * width))
    y0 = int(max(0, (cy - h / 2) * height))
    y1 = int(min(height, (cy + h / 2) * height))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 1
    return mask


def _preset_candidates(preset: str):
    if preset == "top":
        return [
            (0.50, 0.46, 0.55, 0.16),
            (0.50, 0.50, 0.55, 0.16),
            (0.50, 0.54, 0.55, 0.16),
            (0.50, 0.58, 0.55, 0.16),
            (0.50, 0.52, 0.60, 0.18),
            (0.50, 0.56, 0.60, 0.18),
            (0.50, 0.60, 0.60, 0.18),
        ]
    if preset == "skirt":
        return [
            (0.50, 0.66, 0.60, 0.22),
            (0.50, 0.70, 0.60, 0.22),
            (0.50, 0.74, 0.60, 0.22),
            (0.50, 0.68, 0.55, 0.20),
            (0.50, 0.72, 0.55, 0.20),
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate simple rectangular mask candidates.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--out-dir", required=True, help="Output directory for masks/previews.")
    parser.add_argument("--preset", choices=["top", "skirt"], default="top")
    args = parser.parse_args()

    np_mod, Image, ImageDraw, ImageFont = _load_deps()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_image = Image.open(image_path).convert("RGB")
    width, height = base_image.size

    candidates = _preset_candidates(args.preset)
    if not candidates:
        raise SystemExit(f"No candidates for preset: {args.preset}")

    meta = []
    for idx, (cx, cy, w, h) in enumerate(candidates, start=1):
        mask = _rect_mask(np_mod, width, height, cx, cy, w, h)
        mask_path = out_dir / f"mask_{idx:02d}.png"
        preview_path = out_dir / f"preview_{idx:02d}.png"
        Image.fromarray(mask * 255, mode="L").save(mask_path)
        _overlay_preview(Image, ImageDraw, ImageFont, base_image, mask, idx, preview_path)
        meta.append({"index": idx, "cx": cx, "cy": cy, "w": w, "h": h, "mask": mask_path.name, "preview": preview_path.name})

    (out_dir / "masks.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {len(candidates)} masks to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
