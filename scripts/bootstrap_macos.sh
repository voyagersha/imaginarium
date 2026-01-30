#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export PIP_CACHE_DIR="${ROOT_DIR}/.pip-cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

echo "Repo root: ${ROOT_DIR}"

if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode Command Line Tools not found. Run: xcode-select --install" >&2
  exit 1
fi

require_cmd git
require_cmd python3
require_cmd uv

cd "${ROOT_DIR}"

echo "==> Setting up repo-local uv environment (.venv)"
if [[ ! -d ".venv" ]]; then
  uv venv
fi

if [[ ! -f "uv.lock" ]]; then
  echo "==> Creating uv.lock (first run)"
  uv lock
fi

echo "==> Syncing python deps"
uv sync --frozen

echo "==> Generating default workflow templates (workflows/cli/*.json)"
uv run python scripts/export_workflow_templates.py

echo "==> Installing ComfyUI into vendor/ComfyUI (not committed)"
mkdir -p vendor
if [[ ! -d "vendor/ComfyUI/.git" ]]; then
  git clone https://github.com/comfyanonymous/ComfyUI.git vendor/ComfyUI
fi

if [[ -n "${COMFYUI_REF:-}" ]]; then
  echo "==> Pinning ComfyUI to ${COMFYUI_REF}"
  git -C vendor/ComfyUI fetch --all --tags
  git -C vendor/ComfyUI checkout --detach "${COMFYUI_REF}"
fi

echo "==> Creating ComfyUI venv (vendor/ComfyUI/.venv)"
COMFY_PYTHON="${COMFY_PYTHON:-}"
if [[ -z "${COMFY_PYTHON}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    COMFY_PYTHON="$(command -v python3.11)"
  elif command -v python3.12 >/dev/null 2>&1; then
    COMFY_PYTHON="$(command -v python3.12)"
  else
    COMFY_PYTHON="$(command -v python3)"
  fi
fi

uv venv vendor/ComfyUI/.venv -p "${COMFY_PYTHON}" --seed --allow-existing

echo "==> Installing ComfyUI requirements into ComfyUI venv"
uv pip install -r vendor/ComfyUI/requirements.txt -p vendor/ComfyUI/.venv/bin/python

mkdir -p vendor/ComfyUI/custom_nodes

echo "==> Installing ComfyUI-Manager"
MANAGER_DIR="vendor/ComfyUI/custom_nodes/ComfyUI-Manager"
if [[ ! -d "${MANAGER_DIR}/.git" ]]; then
  if git ls-remote https://github.com/ltdrdata/ComfyUI-Manager.git >/dev/null 2>&1; then
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git "${MANAGER_DIR}"
  else
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git "${MANAGER_DIR}"
  fi
fi
if [[ -n "${COMFYUI_MANAGER_REF:-}" ]]; then
  git -C "${MANAGER_DIR}" fetch --all --tags
  git -C "${MANAGER_DIR}" checkout --detach "${COMFYUI_MANAGER_REF}"
fi
if [[ -f "${MANAGER_DIR}/requirements.txt" ]]; then
  uv pip install -r "${MANAGER_DIR}/requirements.txt" -p vendor/ComfyUI/.venv/bin/python
fi

echo "==> Installing IP-Adapter+ custom nodes"
IPADAPTER_DIR="vendor/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus"
if [[ ! -d "${IPADAPTER_DIR}/.git" ]]; then
  git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git "${IPADAPTER_DIR}"
fi
if [[ -n "${IPADAPTER_NODE_REF:-}" ]]; then
  git -C "${IPADAPTER_DIR}" fetch --all --tags
  git -C "${IPADAPTER_DIR}" checkout --detach "${IPADAPTER_NODE_REF}"
fi
if [[ -f "${IPADAPTER_DIR}/requirements.txt" ]]; then
  uv pip install -r "${IPADAPTER_DIR}/requirements.txt" -p vendor/ComfyUI/.venv/bin/python
fi

echo "==> Installing comfyui_controlnet_aux (optional but recommended)"
CN_AUX_DIR="vendor/ComfyUI/custom_nodes/comfyui_controlnet_aux"
if [[ ! -d "${CN_AUX_DIR}/.git" ]]; then
  git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git "${CN_AUX_DIR}"
fi
if [[ -n "${CONTROLNET_AUX_REF:-}" ]]; then
  git -C "${CN_AUX_DIR}" fetch --all --tags
  git -C "${CN_AUX_DIR}" checkout --detach "${CONTROLNET_AUX_REF}"
fi
if [[ -f "${CN_AUX_DIR}/requirements.txt" ]]; then
  uv pip install -r "${CN_AUX_DIR}/requirements.txt" -p vendor/ComfyUI/.venv/bin/python
fi

echo "==> Ensuring model directories exist"
mkdir -p vendor/ComfyUI/models/clip_vision vendor/ComfyUI/models/ipadapter

echo
echo "Bootstrap complete."
echo "- Start ComfyUI: make start"
echo "- Download models: make models (requires HF_TOKEN for gated models)"
echo "- Validate: make validate"
