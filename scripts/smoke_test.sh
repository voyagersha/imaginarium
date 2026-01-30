#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"

COMFY_URL="${COMFY_URL:-http://127.0.0.1:8188}"
SMOKE_WIDTH="${SMOKE_WIDTH:-256}"
SMOKE_HEIGHT="${SMOKE_HEIGHT:-256}"
SMOKE_STEPS="${SMOKE_STEPS:-1}"
SMOKE_BATCH="${SMOKE_BATCH:-1}"

echo "==> Checking ComfyUI: ${COMFY_URL}/system_stats"
curl -fsS "${COMFY_URL}/system_stats" >/dev/null

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

REF_PNG="${TMP_DIR}/ref.png"
MASK_PNG="${TMP_DIR}/mask.png"

PYTHON_FOR_IMAGES="${ROOT_DIR}/vendor/ComfyUI/.venv/bin/python"
if [[ ! -x "${PYTHON_FOR_IMAGES}" ]]; then
  PYTHON_FOR_IMAGES="python3"
fi

"${PYTHON_FOR_IMAGES}" - <<PY
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw

ref_path = Path("${REF_PNG}")
mask_path = Path("${MASK_PNG}")

w = int(os.environ.get("SMOKE_WIDTH", "256"))
h = int(os.environ.get("SMOKE_HEIGHT", "256"))

img = Image.new("RGB", (w, h), (120, 120, 120))
draw = ImageDraw.Draw(img)
draw.ellipse((w * 0.2, h * 0.2, w * 0.8, h * 0.8), fill=(200, 200, 200))
img.save(ref_path)

mask = Image.new("L", (w, h), 0)
mask_draw = ImageDraw.Draw(mask)
mask_draw.rectangle((w * 0.35, h * 0.35, w * 0.65, h * 0.65), fill=255)
mask.save(mask_path)
PY

echo "==> Running character"
uv run python -m persona_stack.cli run character --prompt "a portrait photo" --anchors "${REF_PNG}" --steps "${SMOKE_STEPS}" --width "${SMOKE_WIDTH}" --height "${SMOKE_HEIGHT}" --batch-size "${SMOKE_BATCH}" --comfy-url "${COMFY_URL}"

echo "==> Running txt2img"
uv run python -m persona_stack.cli run txt2img --prompt "a portrait photo" --steps "${SMOKE_STEPS}" --width "${SMOKE_WIDTH}" --height "${SMOKE_HEIGHT}" --batch-size "${SMOKE_BATCH}" --comfy-url "${COMFY_URL}"

echo "==> Running style"
uv run python -m persona_stack.cli run style --prompt "a portrait photo, cinematic lighting" --style-ref "${REF_PNG}" --steps "${SMOKE_STEPS}" --width "${SMOKE_WIDTH}" --height "${SMOKE_HEIGHT}" --batch-size "${SMOKE_BATCH}" --comfy-url "${COMFY_URL}"

echo "==> Running scene"
uv run python -m persona_stack.cli run scene --prompt "a portrait photo, indoor studio" --anchors "${REF_PNG}" --style-ref "${REF_PNG}" --steps "${SMOKE_STEPS}" --width "${SMOKE_WIDTH}" --height "${SMOKE_HEIGHT}" --batch-size "${SMOKE_BATCH}" --comfy-url "${COMFY_URL}"

echo "==> Running inpaint"
uv run python -m persona_stack.cli run inpaint --prompt "fix artifacts" --image "${REF_PNG}" --mask "${MASK_PNG}" --steps "${SMOKE_STEPS}" --denoise 0.5 --comfy-url "${COMFY_URL}"

echo "Smoke test completed (generation steps require models to be installed)."
