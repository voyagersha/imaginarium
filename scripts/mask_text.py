#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_deps():
    try:
        import numpy as np
        import torch
        from PIL import Image, ImageDraw, ImageFont
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    except Exception as exc:
        raise SystemExit(
            "Missing dependencies. Run this with the ComfyUI venv:\n"
            "  vendor/ComfyUI/.venv/bin/python scripts/mask_text.py --help\n"
            f"Error: {exc}"
        )
    return np, torch, Image, ImageDraw, ImageFont, CLIPSegForImageSegmentation, CLIPSegProcessor


def _pick_device(torch_mod, requested: str) -> str:
    if requested != "auto":
        return requested
    if getattr(torch_mod.cuda, "is_available", lambda: False)():
        return "cuda"
    mps = getattr(torch_mod.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _patch_clipseg_decoder() -> None:
    # Work around non-contiguous view in CLIPSeg decoder on some PyTorch builds.
    try:
        from transformers.models.clipseg import modeling_clipseg as mc
    except Exception:
        return
    if getattr(mc, "_persona_stack_clipseg_patch", False):
        return

    import math

    CLIPSegDecoder = mc.CLIPSegDecoder
    CLIPSegDecoderOutput = mc.CLIPSegDecoderOutput

    def forward(
        self,
        hidden_states,
        conditional_embeddings,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        activations = hidden_states[::-1]

        output = None
        for i, (activation, layer, reduce) in enumerate(zip(activations, self.layers, self.reduces)):
            if output is not None:
                output = reduce(activation) + output
            else:
                output = reduce(activation)

            if i == self.conditional_layer:
                output = self.film_mul(conditional_embeddings) * output.permute(1, 0, 2) + self.film_add(
                    conditional_embeddings
                )
                output = output.permute(1, 0, 2)

            layer_outputs = layer(
                output, attention_mask=None, causal_attention_mask=None, output_attentions=output_attentions
            )

            output = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states += (output,)

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        output = output[:, 1:, :].permute(0, 2, 1).contiguous()
        size = int(math.sqrt(output.shape[2]))
        batch_size = conditional_embeddings.shape[0]
        output = output.reshape(batch_size, output.shape[1], size, size)

        logits = self.transposed_convolution(output).squeeze(1)

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_attentions] if v is not None)

        return CLIPSegDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    CLIPSegDecoder.forward = forward
    mc._persona_stack_clipseg_patch = True


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a text-guided mask using CLIPSeg.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Text prompt for the region to segment (repeatable).",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for mask + preview.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold (0-1).")
    parser.add_argument("--invert", action="store_true", help="Invert mask (swap keep/edit).")
    parser.add_argument(
        "--subtract-prompt",
        action="append",
        default=[],
        help="Text prompt to subtract from the mask (repeatable).",
    )
    parser.add_argument(
        "--subtract-strength",
        type=float,
        default=1.0,
        help="How strongly to subtract the negative mask (0-1).",
    )
    parser.add_argument("--erode", type=int, default=0, help="Erode mask by N pixels (tighten edges).")
    parser.add_argument("--dilate", type=int, default=0, help="Dilate mask by N pixels (expand edges).")
    parser.add_argument("--device", default="auto", help="cpu|cuda|mps|auto.")
    parser.add_argument("--model", default="CIDAS/clipseg-rd64-refined", help="CLIPSeg model id.")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE"), help="HF cache directory.")
    parser.add_argument("--offline", action="store_true", help="Require local files only.")
    args = parser.parse_args()

    np_mod, torch_mod, Image, ImageDraw, ImageFont, CLIPSegForImageSegmentation, CLIPSegProcessor = _load_deps()
    _patch_clipseg_decoder()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    offline = args.offline or bool(os.environ.get("HF_HUB_OFFLINE")) or bool(os.environ.get("TRANSFORMERS_OFFLINE"))
    cache_dir = args.cache_dir or None
    device = _pick_device(torch_mod, args.device)

    processor = CLIPSegProcessor.from_pretrained(args.model, cache_dir=cache_dir, local_files_only=offline)
    model = CLIPSegForImageSegmentation.from_pretrained(args.model, cache_dir=cache_dir, local_files_only=offline)
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    prompts = [p.strip() for p in args.prompt if p.strip()]
    if not prompts:
        raise SystemExit("No prompts provided (use --prompt).")
    subtract_prompts = [p.strip() for p in args.subtract_prompt if p.strip()]

    inputs = processor(text=prompts, images=[image] * len(prompts), padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch_mod.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (batch, h, w)
    include_probs = torch_mod.sigmoid(logits).max(dim=0).values

    if subtract_prompts:
        sub_inputs = processor(text=subtract_prompts, images=[image] * len(subtract_prompts), padding=True, return_tensors="pt")
        sub_inputs = {k: v.to(device) for k, v in sub_inputs.items()}
        with torch_mod.no_grad():
            sub_outputs = model(**sub_inputs)
        sub_logits = sub_outputs.logits
        subtract_probs = torch_mod.sigmoid(sub_logits).max(dim=0).values
        include_probs = torch_mod.clamp(include_probs - subtract_probs * float(args.subtract_strength), 0.0, 1.0)

    mask_raw = include_probs.cpu().numpy()

    mask_img = Image.fromarray((mask_raw * 255).astype("uint8"), mode="L").resize(image.size, resample=Image.BILINEAR)
    if args.erode > 0 or args.dilate > 0:
        from PIL import ImageFilter
        if args.erode > 0:
            size = max(1, int(args.erode) * 2 + 1)
            mask_img = mask_img.filter(ImageFilter.MinFilter(size=size))
        if args.dilate > 0:
            size = max(1, int(args.dilate) * 2 + 1)
            mask_img = mask_img.filter(ImageFilter.MaxFilter(size=size))
    mask_arr = np_mod.array(mask_img, dtype=np_mod.float32) / 255.0
    mask_binary = (mask_arr >= args.threshold).astype(np_mod.uint8)
    if args.invert:
        mask_binary = 1 - mask_binary

    mask_path = out_dir / "mask.png"
    preview_path = out_dir / "preview.png"
    Image.fromarray(mask_binary * 255, mode="L").save(mask_path)
    _overlay_preview(Image, ImageDraw, ImageFont, image, mask_binary, preview_path)

    meta = {
        "image": str(image_path),
        "prompts": prompts,
        "model": args.model,
        "threshold": args.threshold,
        "invert": args.invert,
        "device": device,
    }
    (out_dir / "mask.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved mask to {mask_path}")
    print(f"Saved preview to {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
