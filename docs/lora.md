# LoRA Guide

This repo uses LoRAs only when a workflow explicitly loads them (via a
`LoraLoader` node). Setting a LoRA strength in the CLI or sweep config does
nothing if the workflow does not expose the matching patch points.

## Where LoRAs Live

- ComfyUI LoRAs: `vendor/ComfyUI/models/loras/`
- Style packs (if downloaded separately): typically under `stylepacks/` and
  then copied into `vendor/ComfyUI/models/loras/`.

## How LoRAs Are Applied

LoRAs are tied to workflow variants; check the workflow filename and graph for
the `LoraLoader` node(s) it includes.

To enable a LoRA:
- Pick a workflow that already includes it.
- Pass the strength parameter (CLI or sweep config).

If the workflow does not include the LoRA, strength params are ignored.

## Z-Image Turbo LoRA (pixel art)

Purpose: stylize Z-Image Turbo output into pixel art.
Use when: you want pixel-art output from Z-Image Turbo.

Use with:
- a workflow that loads `pixel_art_style_z_image_turbo.safetensors`

Notes:
- Not applied by default in Z-Image Turbo workflows.
- Add a dedicated workflow if you want this style.

## Flux.2 Klein

There are no Flux LoRAs wired in this repo. Use `flux_guidance` instead.

## How to Set LoRA Strengths

CLI flags (only work if the workflow exposes these patch points):
- `--dmd2-strength`
- `--lightning-strength`
- `--subtle-strength`
- `--style-strength` (alias: `--cuteness-strength`)

## Quick Selection Guide

- Want baseline quality: use non-LoRA workflows.
- Want a specific LoRA: use a workflow that already loads it.
