# Parameter Mental Model

This doc explains how the CLI parameters map to ComfyUI workflows and how to
reason about them. A parameter only has an effect if the workflow has a matching
patch point. If a workflow does not expose a patch point, the parameter is
ignored.

## Common Parameters (All Models)

These show up across most workflows and models in `manifests/models.json`
(SDXL, Flux.2 klein, Z-Image Turbo).

- prompt: What you want. Think "target description." More specific = more
  controlled, but also less flexible.
- negative_prompt: What to avoid. Works by pushing the model away from those
  concepts. Overusing it can reduce quality or fight the positive prompt.
- seed: Controls determinism. Same inputs + same seed -> same output.
- steps: Number of denoising iterations. More steps = more compute and usually
  higher fidelity (up to a point).
- cfg: Classifier-free guidance scale. Higher values push the output to match
  the prompt more literally, but can create artifacts or harsh lighting.
- sampler_name: The sampling algorithm (euler, dpmpp_sde, etc). Think "how the
  model walks down the noise schedule."
- scheduler: The noise schedule (normal, karras, sgm_uniform, etc). Think
  "how step sizes are distributed over time."
- width/height: Output resolution (txt2img, character, style, scene, inpaint).
  Larger sizes cost more memory and compute.
- batch_size: Number of images generated per run. Larger batch = more memory.
- clip_skip: How many final CLIP layers to skip.

## Img2img / Inpaint Parameters

- image: The init image. Think "what to preserve."
- denoise: How far to drift from the init image.
  - 0.1-0.3: small edits, strong identity preservation
  - 0.3-0.6: moderate edits (outfit, lighting, background)
  - 0.6-1.0: large changes, can fully reimagine the image
- mask (inpaint): White = replace, black = keep. Inpaint only touches masked
  regions, so good masks are the biggest quality lever.

## Identity / Anchor Parameters

Used by character, scene, and img2img-identity workflows.

- anchors: Reference images. Think "identity anchor set."
- anchor_count: Max number of anchors to use (1-5). More anchors = stronger
  identity signal, but also more compute.
- anchor_pad: Pad to square before upload. Helps avoid face cropping when a
  model expects square inputs.
- anchor_pad_color: Background color for padding (name or R,G,B).
- anchor_max_side: Resize anchors so the longest side is <= this size. Lowers
  memory and speeds uploads at the cost of fine detail.
- anchor_weight_scale: Multiplies the anchor weights. >1 = stronger identity,
  <1 = looser identity.

## Style / Scene Parameters

- stylepack: A directory that contains a `refs/` folder of style references.
- style_ref: Explicit style reference image path (repeatable). More refs bias
  style more strongly.
- style_strength / cuteness_strength: Strength of a style LoRA if the workflow
  includes one. Use cuteness_strength as an alias.

## Pose Parameters (ControlNet)

Only used by pose-enabled workflows.

- pose_image: Reference pose image.
- pose_strength: How strongly to enforce the pose (0-1).
- pose_start / pose_end: Portion of the denoising schedule where pose is active.
  Short ranges reduce pose influence.

## Model-Specific Parameters

These are only effective if the workflow includes the matching nodes.

- flux_guidance: Flux.2 guidance value. Flux workflows use both cfg and
  flux_guidance; in practice keep cfg around 1.0 and adjust flux_guidance
  (1.0-2.0) for prompt adherence.
- dmd2_strength: Strength of the DMD2 LoRA (model + clip). Best for fast,
  high-quality runs at low steps.
- lightning_strength: Strength of the SDXL Lightning LoRA. Optimized for very
  low steps; can add contrast and sharpness.
- subtle_strength: Strength of Subtle Styles LoRAs. Adds style details and
  contrast; reduce for cleaner, more neutral results.

## Execution Parameters

- workflow: Overrides the default workflow JSON path. Use this to switch between
  SDXL, Flux.2, or Z-Image Turbo templates.
- comfy_url: ComfyUI server URL (default: http://127.0.0.1:8188).

## Quick Mental Model Summary

- Resolution and steps drive compute the most.
- denoise controls "how much the input image can change."
- cfg controls "how literal the prompt is."
- anchors and pose are stronger than prompt alone; use them when identity/pose
  must stay stable.
