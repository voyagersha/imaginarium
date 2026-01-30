#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_cc_aux_on_path() -> None:
    cc_aux = _repo_root() / "vendor" / "ComfyUI" / "custom_nodes" / "comfyui_controlnet_aux" / "src"
    if not cc_aux.exists():
        raise SystemExit(f"Missing comfyui_controlnet_aux at {cc_aux}. Run make bootstrap.")
    sys.path.insert(0, str(cc_aux))


def _load_deps() -> tuple[object, object, object, object, object, object]:
    try:
        import numpy as np
        import torch
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise SystemExit(
            "Missing dependencies. Run this with the ComfyUI venv:\n"
            "  vendor/ComfyUI/.venv/bin/python scripts/mask_sam.py --help\n"
            f"Error: {exc}"
        )

    return np, torch, Image, ImageDraw, ImageFont


def _load_sam() -> object:
    try:
        from custom_controlnet_aux.sam import SamDetector
    except Exception as exc:
        msg = (
            "Failed to import SAM from comfyui_controlnet_aux.\n"
            "Make sure the ComfyUI venv has required packages (transformers, torch, numpy, scipy, scikit-image).\n"
            "If you see a missing transformers error, install it into the ComfyUI venv:\n"
            "  uv pip install --python vendor/ComfyUI/.venv/bin/python transformers\n"
            f"Error: {exc}"
        )
        raise SystemExit(msg)
    return SamDetector


def _patch_sam_detector(np_mod, torch_mod, Image) -> None:
    try:
        from custom_controlnet_aux.sam.sam import SamDetector
    except Exception:
        return
    if getattr(SamDetector, "_persona_stack_float_patch", False):
        return

    def generate_automatic_masks(self, input_image):
        if isinstance(input_image, np_mod.ndarray):
            pil_image = Image.fromarray(input_image)
        else:
            pil_image = input_image

        height, width = pil_image.size[1], pil_image.size[0]

        points_per_side = max(8, min(24, width // 64, height // 64))

        grid_points = []
        for i in range(points_per_side):
            for j in range(points_per_side):
                x = int((j + 0.5) * width / points_per_side)
                y = int((i + 0.5) * height / points_per_side)
                x_offset = int((np_mod.random.random() - 0.5) * (width / points_per_side * 0.3))
                y_offset = int((np_mod.random.random() - 0.5) * (height / points_per_side * 0.3))
                x = max(5, min(width - 5, x + x_offset))
                y = max(5, min(height - 5, y + y_offset))
                grid_points.append([x, y])

        batch_size = 16
        all_masks = []

        for i in range(0, len(grid_points), batch_size):
            batch_points = grid_points[i : i + batch_size]
            input_points = [batch_points]

            inputs = self.processor(images=pil_image, input_points=input_points, return_tensors="pt")
            for key, value in inputs.items():
                if isinstance(value, torch_mod.Tensor) and value.dtype == torch_mod.float64:
                    inputs[key] = value.float()
            inputs = inputs.to(self.device)

            with torch_mod.no_grad():
                outputs = self.model(**inputs)

            masks = self.processor.post_process_masks(
                outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
            )[0]

            masks_np = masks.cpu().numpy()

            for j, mask in enumerate(masks_np):
                mask_2d = mask[0] if len(mask.shape) > 2 else mask
                area = int(mask_2d.sum())

                if area > 100:
                    cleaned_mask = self._postprocess_mask(mask_2d)
                    cleaned_area = int(cleaned_mask.sum())

                    mask_dict = {
                        "segmentation": cleaned_mask,
                        "area": cleaned_area,
                        "stability_score": 0.88,
                        "point_coords": batch_points[j % len(batch_points)],
                    }
                    all_masks.append(mask_dict)

        return all_masks

    SamDetector.generate_automatic_masks = generate_automatic_masks
    SamDetector._persona_stack_float_patch = True


def _pick_device(torch_mod, requested: str) -> str:
    if requested != "auto":
        return requested
    if getattr(torch_mod.cuda, "is_available", lambda: False)():
        return "cuda"
    mps = getattr(torch_mod.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _mask_to_bbox(np_mod, mask_bool) -> tuple[int, int, int, int] | None:
    ys, xs = np_mod.where(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _save_mask(Image, np_mod, mask_bool, out_path: Path) -> None:
    mask_img = Image.fromarray((mask_bool.astype(np_mod.uint8) * 255), mode="L")
    mask_img.save(out_path)


def _save_preview(Image, ImageDraw, ImageFont, np_mod, image, mask_bool, idx: int, out_path: Path) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    alpha = Image.fromarray((mask_bool.astype(np_mod.uint8) * 140), mode="L")
    overlay.putalpha(alpha)
    preview = Image.alpha_composite(base, overlay).convert("RGB")

    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    draw.rectangle((6, 6, 54, 24), fill=(0, 0, 0))
    draw.text((10, 8), f"{idx:02d}", fill=(255, 255, 255), font=font)
    preview.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate candidate inpaint masks using SAM (auto-segmentation).")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--out-dir", required=True, help="Output directory for masks/previews.")
    parser.add_argument("--max-masks", type=int, default=12, help="Max number of masks to save.")
    parser.add_argument("--min-area", type=int, default=5000, help="Minimum mask area in pixels.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for SAM sampling.")
    parser.add_argument("--device", default="auto", help="cpu|cuda|mps|auto.")
    parser.add_argument("--pick", type=int, default=None, help="Copy selected mask (1-based index) to mask.png.")
    parser.add_argument("--invert", action="store_true", help="Invert masks before saving (swap keep/edit).")
    args = parser.parse_args()

    _ensure_cc_aux_on_path()
    np_mod, torch_mod, Image, ImageDraw, ImageFont = _load_deps()
    SamDetector = _load_sam()
    _patch_sam_detector(np_mod, torch_mod, Image)

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np_mod.random.seed(args.seed)
    torch_mod.manual_seed(args.seed)

    device = _pick_device(torch_mod, args.device)

    image = Image.open(image_path).convert("RGB")
    sam = SamDetector.from_pretrained().to(device)
    masks = sam.generate_automatic_masks(image)

    items = []
    for mask in masks:
        seg = mask.get("segmentation")
        if seg is None:
            continue
        mask_bool = seg > 0.5
        area = int(mask_bool.sum())
        if area < args.min_area:
            continue
        items.append(
            {
                "area": area,
                "mask_bool": mask_bool,
                "bbox": _mask_to_bbox(np_mod, mask_bool),
            }
        )

    if not items:
        raise SystemExit("No masks found. Try lowering --min-area or using a different image.")

    items.sort(key=lambda x: x["area"], reverse=True)
    items = items[: args.max_masks]

    meta = []
    saved_masks = []
    for idx, item in enumerate(items, start=1):
        mask_path = out_dir / f"mask_{idx:02d}.png"
        preview_path = out_dir / f"preview_{idx:02d}.png"
        mask_bool = item["mask_bool"]
        if args.invert:
            mask_bool = ~mask_bool
        _save_mask(Image, np_mod, mask_bool, mask_path)
        _save_preview(Image, ImageDraw, ImageFont, np_mod, image, mask_bool, idx, preview_path)
        saved_masks.append(mask_path)
        meta.append({"index": idx, "area": item["area"], "bbox": item["bbox"], "mask": mask_path.name, "preview": preview_path.name})

    (out_dir / "masks.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.pick is not None:
        pick_idx = args.pick - 1
        if pick_idx < 0 or pick_idx >= len(saved_masks):
            raise SystemExit(f"--pick {args.pick} is out of range (1..{len(saved_masks)}).")
        target = out_dir / "mask.png"
        target.write_bytes(saved_masks[pick_idx].read_bytes())

    print(f"Saved {len(saved_masks)} masks to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
