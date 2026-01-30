# Workflows Guide

This folder contains two kinds of workflows:

- `workflows/gui/` are ComfyUI GUI graphs (for the Workflows tab).
- `workflows/cli/` are API templates used by `persona-stack run ...`.

The tracked set is the core/SFW bundle.

Regenerating CLI templates:

- Core only (tracked): `make export-workflows`

If you are unsure which to use:

- Use `workflows/gui` when working inside ComfyUI.
- Use `workflows/cli` when running the CLI.

## Quick chooser by use case

Pick the workflow type first, then pick the model variant you want (sdxl, flux2_klein).

Use case -> Workflow

- Prompt only (no reference images): `txt2img_*`
- Edit an existing image (keep pose/body): `img2img_*`
- Edit an existing image + lock face identity: `img2img_identity_*`
- New image from identity anchors only: `character_*`
- Identity + style refs (consistent look): `scene_*`
- Style transfer only (no identity): `style_*`
- Pose control (uses OpenPose): `character_pose_*` or `scene_pose_*`
- Fix small areas with a mask (clothing, accessories, small edits): `inpaint_*`
- For reliable apparel edits (replace clothing on a reference image): `inpaint_sdxl_inpaint`

## Model variants (suffixes)

Tracked suffixes:

- No suffix: SDXL base (default).
- `_flux2_klein`: Flux 2 Klein distilled 4B.

## Flux 2 Klein (distilled 4B)

Workflows:

- `txt2img_flux2_klein`
- `img2img_flux2_klein`
- `img2img_identity_flux2_klein` (identity anchors via reference latents)

Recommended starting points (distilled):

- Steps: 4 to 8
- CFG: ~1.0
- Sampler: `euler`
- Scheduler: `normal`
- img2img denoise: 0.4 to 0.6
- Optional: `--flux-guidance 1.0` to override the guidance embed value (default is 1.0 in the workflow).

Note: CLI defaults are SDXL-like (20 steps, cfg 7). Pass explicit `--steps` and `--cfg` for Flux.

Identity workflow notes:

- Use `persona-stack run img2img-identity --workflow workflows/cli/img2img_identity_flux2_klein.json`.
- Anchors are injected as reference latents; anchor weights are not used in this workflow.

## Which workflow should I choose?

If you want the best identity lock on face + body:

1) Use `img2img_identity_*`.
2) Choose a clean base image as `--image` (no face occlusions).
3) Provide 1 to 3 clean face anchors.
4) Keep denoise low to preserve identity.

If you want to change pose:

- Use `character_pose_*` or `scene_pose_*` and provide a pose image.

If you want to change clothing but keep the same body pose:

- Use `inpaint_sdxl_inpaint` with a clothing mask (best reliability).
- If you need more global changes, use `img2img_identity_*` and raise denoise slightly.

If you want to use a reference image and change clothing or body shape:

- Use `inpaint_sdxl_inpaint` when you can mask the clothing area.
- Use `img2img_identity_*` for full-body changes while keeping identity.
- Use `img2img_*` if identity does not need to match.
- Increase denoise for bigger changes (expect more drift).

## Parameter guide (best starting points)

General:

- SDXL base: `clip_skip=1`.
- Resolution: 1024x1024 for portraits, 832x1216 or 896x1152 for full body.

txt2img:

- Steps: 25 to 35
- CFG: 6 to 7.5

img2img:

- Denoise: 0.25 to 0.4 for small edits
- Denoise: 0.45 to 0.6 for larger changes

img2img_identity (best for face and body lock):

- Denoise: 0.25 to 0.4
- Anchors: 1 to 3 clean face images
- If identity drifts: lower denoise or add a close-up anchor

character:

- Anchors: 3 to 5 clean face images
- Steps: 28 to 35
- CFG: 6 to 7
- If identity is weak: raise anchor weight (CLI) or increase IPAdapter weight (GUI)

scene:

- Same as `character`, plus style refs if you want a consistent look

pose workflows:

- Pose strength: 0.6 to 1.0
- Start/end: 0.0 / 1.0 (default)
- Use a clear full-body pose image when possible

inpaint:

- Denoise: 0.4 to 0.7
- Mask only the area you want to change
- If you see flat/gray fills, switch to `inpaint_sdxl_inpaint` (requires the SDXL inpainting checkpoint).
- For clothing replacement on a reference image, start with `inpaint_sdxl_inpaint` and a clean binary mask.

## Mask helper (auto, SAM)

If your manual mask is off, you can auto-generate candidate masks using SAM.
This uses ComfyUI's venv to avoid installing heavy deps into the repo env.

Example:

```bash
vendor/ComfyUI/.venv/bin/python scripts/mask_sam.py \
  --image anchors_identity/vi/candidates/IMG_0218.JPG \
  --out-dir anchors_identity/vi/masks/IMG_0218 \
  --max-masks 8
```

Open the `preview_*.png` files, pick the best mask, and pass it to inpaint:

```bash
uv run python -m persona_stack.cli run inpaint \
  --image anchors_identity/vi/candidates/IMG_0218.JPG \
  --mask anchors_identity/vi/masks/IMG_0218/mask_02.png \
  --prompt "new outfit, detailed fabric, realistic folds" \
  --denoise 0.55
```

Notes:

- SAM needs the HF weights cached (facebook/sam-vit-base). If it tries to download, run this on a machine with network access or pre-seed the HF cache.
- If SAM import fails, install transformers into the ComfyUI venv:
  `uv pip install --python vendor/ComfyUI/.venv/bin/python transformers`

## Mask helper (text-guided, CLIPSeg)

If you want to target clothing or other objects with text (e.g. "dress", "skirt"),
use the CLIPSeg helper. It produces a single `mask.png` plus a `preview.png`.

Download the model once:

```bash
hf download CIDAS/clipseg-rd64-refined --repo-type model
```

Then generate a mask:

```bash
HF_HUB_CACHE=/path/to/hf/cache \
vendor/ComfyUI/.venv/bin/python scripts/mask_text.py \
  --image anchors_identity/vi/candidates/IMG_0215.JPG \
  --out-dir anchors_identity/vi/masks/IMG_0215_text \
  --prompt "dress" \
  --prompt "skirt" \
  --threshold 0.45 \
  --subtract-prompt "skin" \
  --subtract-prompt "stomach" \
  --subtract-prompt "belly" \
  --subtract-strength 0.7 \
  --erode 2
```

Use `anchors_identity/vi/masks/IMG_0215_text/mask.png` in your inpaint run.

## Anchor guidance (identity images)

Good anchors are more important than quantity.

- Use clear, front-facing or 3/4 face shots.
- Avoid occlusions (hands, hair over eyes, sunglasses, props).
- Avoid extreme lighting or heavy filters.
- Non-square images are OK (the CLI pads them).

Recommended:

- 1 to 3 anchors for `img2img_identity_*`.
- 3 to 5 anchors for `character_*` and `scene_*`.

## Sampler and scheduler overrides (CLI)

By default, the CLI uses whatever sampler/scheduler are embedded in the workflow.

If you want to override:

- `--sampler-name dpmpp_sde`
- `--scheduler sgm_uniform`

## Troubleshooting

If the face looks deformed:

- Lower denoise in img2img flows.
- Add a clean close-up anchor.
- Reduce prompt complexity (shorter prompt).

If there are extra people:

- Add "single subject" and "solo" to the prompt.
- Add "multiple people" to the negative prompt.

If the identity drifts:

- Use `img2img_identity_*` instead of `character_*`.
- Lower denoise to 0.25 to 0.35.
- Use 2 to 3 clean anchors.
