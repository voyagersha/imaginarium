# AI Persona Stack (SDXL + ComfyUI + Python)

This repo automates local SDXL image generation via a structured “character / style / scene / repair” workflow using ComfyUI as the backend.

License: MIT. See `LICENSE`.

## Quick start

1) Bootstrap dependencies (creates a repo-local `uv` env, clones ComfyUI, installs custom nodes):

```bash
make bootstrap
```

Optional: copy `.env.example` to `.env` and set `HF_TOKEN` if you need access to gated models.

2) Start ComfyUI (in its own terminal):

```bash
make start
```

If port `8188` is already in use, choose a different port:

```bash
COMFY_PORT=8189 make start
```

3) Download models (core, SFW set):

```bash
make models
```

4) Validate setup:

```bash
make validate
```

5) Run the end-to-end smoke test:

```bash
make smoke-test
```

If ComfyUI is running on a different port:

```bash
COMFY_URL=http://127.0.0.1:8189 make smoke-test
```

## Layout

- `src/persona_stack/` — Python automation library + CLI
- `scripts/` — bootstrap/start/download/export helpers
- `workflows/cli/` — ComfyUI API workflow templates (with `persona_stack.patch_points`)
- `workflows/gui/` — ComfyUI GUI workflow exports (used for the Workflows tab)
- `manifests/models.json` — pinned HF artifacts (filenames + optional sha256)
- `anchors_identity/` — identity anchor images (see `anchors_identity/README.md`)
- `runs/` — generated outputs + metadata

## Z-Image Turbo (ComfyUI)

Minimal model download (no extras):

```bash
make models MODEL_IDS="flux2_klein_text_encoder_qwen_3_4b z_image_turbo_diffusion_bf16 z_image_turbo_vae"
```

Optional pixel-art LoRA (referenced by the official workflow):

```bash
make models MODEL_IDS="z_image_turbo_lora_pixel_art"
```

Load `workflows/imported/image_z_image_turbo.json` in ComfyUI. The Qwen text encoder is shared with Flux 2 Klein, so the `flux2_klein_text_encoder_qwen_3_4b` entry satisfies the workflow's `qwen_3_4b.safetensors` requirement.

## Model licensing

This repo does not redistribute model weights. It downloads models from the
original publishers listed in `manifests/models.json`. You are responsible for
complying with each model's license and terms.

## Prompt-only generation

Run plain SDXL text-to-image (no persona/style refs):

```bash
uv run persona-stack run txt2img --prompt "a portrait photo" --steps 25 --seed 123 --batch-size 4
```

## Image-to-image (img2img)

Use an input image as the starting point (keeps the image aspect ratio):

```bash
uv run persona-stack run img2img --image path/to/input.png --prompt "a portrait photo" --denoise 0.5
```

## Image-to-image + identity anchors

Use an input image for body/pose and identity anchors for face consistency:

```bash
uv run persona-stack run img2img-identity \
  --image path/to/input.png \
  --anchors anchors_identity/vi/curated \
  --prompt "score_9, raw photo, portrait" \
  --clip-skip 2 --denoise 0.35 \
  --workflow workflows/cli/img2img_identity.json
```

CLI example:

```bash
uv run persona-stack run character --prompt "..." --anchors anchors_identity/alina/curated --workflow workflows/cli/character.json
```

Pose example (uses OpenPose preprocessor + ControlNet):

```bash
uv run persona-stack run character \
  --prompt "..." \
  --anchors anchors_identity/vi/curated \
  --pose-image path/to/pose_ref.jpg \
  --pose-strength 0.8 \
  --workflow workflows/cli/character_pose.json
```

Tip: if you pass `--pose-image` and omit `--workflow`, the CLI defaults to the pose workflow for you.

## ComfyUI GUI workflows

The repo’s `workflows/cli/*.json` files are API templates for the CLI. If you want ready-made workflows to show up in ComfyUI’s **Workflows** tab, install the GUI versions from `workflows/gui`:

```bash
make install-gui-workflows
```

Then refresh the ComfyUI UI and load:
- `persona_stack_txt2img.json`
- `persona_stack_img2img.json`
- `persona_stack_character.json`
- `persona_stack_style.json`
- `persona_stack_scene.json`
- `persona_stack_inpaint.json`


### Multi-anchor identity (5 slots)

`character` and `scene` workflows include 5 identity anchor slots by default. In the GUI, pick a file for each `LoadImage` node. In the CLI, use `--anchor-count` to control how many anchors are used.

Notes:
- CLI automatically distributes anchor weights across the anchors you provide and sets unused slots to weight 0 (no repeats).
- Anchor images are padded to square before upload by default; use `--no-anchor-pad` to disable.
