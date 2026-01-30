# Model Tuning Ranges

This doc complements `docs/parameters.md` with model-specific starting ranges.
Use workflow defaults first, then adjust within these ranges.

## SDXL Base (sd_xl_base_1.0)

- txt2img: steps 20-30, cfg 5-8, sampler euler or dpmpp_sde, scheduler normal
  or karras, clip_skip 1.
- img2img: denoise 0.3-0.6; keep steps/cfg similar to txt2img.
- inpaint: steps 20-30, cfg 4.5-6, denoise 0.5-0.8, sampler dpmpp_sde,
  scheduler karras. Higher denoise replaces more of the masked region.
- Resolution: 1024x1024 or 832x1216 for faster iteration.

## SDXL Refiner (sd_xl_refiner_1.0)

- Use as a second pass after latent upscale.
- Steps 10-15, cfg 4.5-6, denoise 0.25-0.35, sampler dpmpp_sde, scheduler karras.

## Flux.2 Klein 4B (flux-2-klein-4b)

- Distilled txt2img: steps 4-8, cfg 1.0, flux_guidance 1.0-2.0, sampler euler,
  scheduler normal.
- img2img: denoise 0.2-0.5; lower denoise to preserve identity.
- Inpaint workflows are not included yet; use SDXL-based inpaint if needed.
- On macOS/MPS, use non-FP8 weights only.

## Z-Image Turbo (z_image_turbo_bf16)

- txt2img: steps 4-8, cfg 1.0-1.2, sampler res_multistep, scheduler simple.
- img2img: denoise 0.2-0.5; high cfg or many steps often degrade quality.
- Inpaint workflows are not included yet; use SDXL-based inpaint if needed.

## Inpaint Notes (All SDXL-Based Models)

- Mask quality dominates results; expand masks slightly to avoid seams.
- Lower cfg if edges look harsh or the fill looks over-processed.
- For subtle edits, drop denoise closer to 0.4-0.5.

## Character / Scene / Style Workflows

- Use the same step/cfg ranges as txt2img for the underlying model.
- Identity anchors and style refs are stronger than prompt text; lower cfg
  if prompts start fighting the references.

## ControlNet OpenPose (OpenPoseXL2)

- pose_strength 0.6-0.9, pose_start 0.0, pose_end 0.8.
- If poses look too rigid, lower strength or shorten the active window.

## Identity (IP-Adapter workflows)

- anchors: 3-5 images; anchor_max_side 512-768 for speed.
- anchor_weight_scale 0.8-1.2 to tune identity strength.

## Style LoRAs

- Typical strength range: 0.3-0.8.
- Reduce strength during upscaling if textures get harsh.
