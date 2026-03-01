# Parameter Mental Model

This doc explains what each CLI parameter does and how to tune it in practice.
It is guidance-first, but kept aligned with current code/workflows.

Source of truth:

- `src/persona_stack/cli.py`
- `workflows/cli/*.json` patch points

Core rule: a CLI option only has an effect if the selected workflow exposes a
matching `persona_stack.patch_points` key. If not, the option is ignored.

## Quick Command Map

- `persona-stack version`
- `persona-stack bootstrap-check`
- `persona-stack validate`
- `persona-stack grids`
- `persona-stack run character`
- `persona-stack run txt2img`
- `persona-stack run img2img`
- `persona-stack run sam3-mask`
- `persona-stack run img2img-identity`
- `persona-stack run style`
- `persona-stack run scene`
- `persona-stack run inpaint`

## Common Generation Parameters

These are the main image-quality/control levers across most run modes.

- `--prompt`: the target description.
  - More specific prompt = tighter control.
  - Overly long prompts can overconstrain and reduce naturalness.
- `--negative-prompt`: what to avoid.
  - Useful for persistent artifacts.
  - Too aggressive negatives can fight the main prompt.
- `--seed`: determinism.
  - Same seed + same inputs + same workflow = reproducible output.
  - Change seed when stuck in a bad local minimum.
- `--steps`: denoising iterations.
  - More steps usually improves detail, up to diminishing returns.
  - If speed matters, lower steps before lowering resolution.
- `--cfg`: prompt adherence strength.
  - Lower cfg = more natural but less literal.
  - Higher cfg = stricter prompt but easier to overcook.
- `--sampler-name`: sampling algorithm.
  - Influences texture, sharpness, and convergence behavior.
- `--scheduler`: step-distribution schedule.
  - Changes how sampler effort is allocated over denoise trajectory.
- `--clip-skip`: CLIP truncation amount (internally clamped and negated).
  - Higher skip can shift style/prompt interpretation.
- `--workflow`: choose workflow template.
  - This is often the biggest behavior switch (model family + node graph).
- `--comfy-url`: ComfyUI endpoint.

## Resolution and Batch Controls

Used by text-first layout workflows (`txt2img`, `character`, `style`, `scene`).

- `--width` and `--height`: output resolution.
  - Bigger = more detail and composition room.
  - Bigger = much higher VRAM + runtime cost.
- `--batch-size`: images per run.
  - Higher batch improves throughput if VRAM allows.

Note: `run inpaint` currently exposes `--width` and `--height` in CLI, but
shipped inpaint workflows do not expose width/height patch points, so they are
usually ignored unless your custom workflow adds those keys.

## Img2img and Inpaint Parameters

- `--image`: init image / base image.
  - Controls structure, pose, and composition prior.
- `--denoise`: edit intensity.
  - `0.1-0.3`: subtle edits, strong preservation.
  - `0.3-0.6`: medium edits (clothes, lighting, local scene).
  - `0.6-1.0`: heavy rewrites, weaker preservation.
- `--mask` (inpaint): white edits, black preserves.
  - Cleaner masks produce cleaner boundaries.
- `--mask-channel`: channel used by `LoadImageMask` (default `red`).
- `--mask-threshold`: binarize mask before grow.
  - Lower threshold includes more gray edge pixels.
  - Higher threshold tightens edited region.
- `--mask-grow`: expand mask in pixels after threshold.
  - Helps avoid hard seams.
  - Too high can spill edits outside intended area.

Inpaint crop/stitch options (only if workflow supports them):

- `--crop-mask-blend-pixels`
  - Blending width around crop boundary.
  - Higher = softer transitions, lower seam risk, but can bleed edits.
- `--crop-context-from-mask-extend-factor`
  - How much extra context around mask is included in crop.
  - Higher = better context coherence, but larger edited neighborhood.

## Prompt Input Behavior (Inpaint)

`run inpaint` accepts either prompt text or a prompt file:

- `--prompt "text"`
- `--prompt-file path/to/prompt.md`

Behavior:

- If `--prompt-file` is provided, file contents are used.
- If `--prompt` points to an existing file path, that file is read.
- Passing both `--prompt` and `--prompt-file` is rejected.
- One of them is required for inpaint.

## Identity Anchor Parameters

Used by `character`, `scene`, and `img2img-identity`.

- `--anchors`: reference images (file/dir, repeatable).
- `--anchor-count`: max anchors to use (clamped `1..5`).
  - More anchors = stronger identity consistency.
  - Too many diverse anchors can blur identity.
- `--anchor-pad` / `--no-anchor-pad`: square pad before upload.
  - Padding improves fit for square-oriented pipelines.
- `--anchor-pad-color`: pad color (`black`, `white`, `R,G,B`, etc.).
- `--anchor-max-side`: optional anchor resize cap.
  - Lower values reduce upload/memory cost.
- `--anchor-weight-scale`: global identity strength scale.
  - `>1` stronger identity pull.
  - `<1` looser identity pull.

## Style Reference Parameters

Used by `style` and `scene`.

- `--stylepack`: directory with `refs/`.
- `--style-ref`: explicit style reference image(s), repeatable.

Practical note: style references bias color/texture/composition language more
than wording alone. Keep references consistent if you want stable style.

## Pose Parameters

Used by pose-capable variants of `character` and `scene`.

- `--pose-image`: pose reference.
- `--pose-strength`: pose enforcement amount.
- `--pose-start` and `--pose-end`: denoise schedule window where pose is active.
  - Narrow window can preserve style/identity while still constraining pose.

Default workflow selection:

- `run character`: uses `character_pose` if `--pose-image` is set, otherwise
  `character`.
- `run scene`: uses `scene_pose` if `--pose-image` is set, otherwise `scene`.

## Model-Specific Controls

These only work when workflow patch points exist.

- `--flux-guidance`
  - Flux-only guidance value (mapped to positive/negative flux guidance keys).
  - Typical practical band is around `1.0-2.0` with cfg near `1.0`.
- `--dmd2-strength`
  - LoRA strength for DMD2 (model + clip).
  - Useful for low-step quality/speed tradeoff.
- `--lightning-strength`
  - LoRA strength for SDXL Lightning (model + clip).
  - Useful for very low-step fast workflows.
- `--subtle-strength`
  - LoRA strength for Subtle Styles (model + clip).
  - Adds stylization/contrast; lower for cleaner look.
- `--style-strength` and `--cuteness-strength`
  - Available in `txt2img` and `img2img`.
  - `--cuteness-strength` is an alias for `--style-strength`.

## SAM3 Mask Command Guidance

`run sam3-mask` generates segmentation-style masks from text grounding.

- `--prompt`: what to segment (`face`, `hair`, `person`, etc.).
- `--confidence-threshold`: confidence gate (`0..1`).
  - Higher = fewer, cleaner detections.
  - Lower = more detections, more false positives.
- `--max-detections`: cap number of detections (minimum 1).
- `--offload-model`: reduce VRAM pressure between runs.
- `--model-path`: SAM3 checkpoint path in ComfyUI model tree.

Output:

- Standard run output under `runs/single/...`.
- Convenience mask copy to `../masks/mask_<source_filename>` relative to input.

## Workflow-Gated Parameters (Current Repo)

Flux guidance patch keys currently appear in:

- `workflows/cli/txt2img_flux2_klein.json`
- `workflows/cli/img2img_flux2_klein.json`
- `workflows/cli/img2img_identity_flux2_klein.json`

Inpaint crop controls currently appear in:

- `workflows/cli/img2img_headswap_z_image_turbo_v1_mask_input.json`
- Patch keys:
  - `crop_mask_blend_pixels`
  - `crop_context_from_mask_extend_factor`

SAM3 patch keys currently appear in:

- `workflows/cli/sam3_mask.json`
- Patch keys:
  - `sam3_prompt`
  - `sam3_confidence_threshold`
  - `sam3_max_detections`
  - `sam3_offload_model`
  - `sam3_model_path`

## Quick Tuning Heuristics

- If output is off-topic: raise `cfg` modestly or improve prompt specificity.
- If output looks overcooked: lower `cfg` or reduce LoRA strengths.
- If not enough change in img2img/inpaint: increase `denoise`.
- If identity drifts: add/clean anchors and raise `anchor-weight-scale`.
- If inpaint seams appear: slightly raise `mask-grow`, or use crop blend/context
  controls on supporting workflows.
