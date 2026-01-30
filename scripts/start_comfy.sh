#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_DIR="${ROOT_DIR}/vendor/ComfyUI"
LOG_FILE="${COMFY_DIR}/comfy.log"

LISTEN="0"
PORT="${COMFY_PORT:-8188}"
FORCE_CPU="${COMFY_FORCE_CPU:-0}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}" || true

if command -v lsof >/dev/null 2>&1; then
  if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port ${PORT} is already in use. Stop the existing process or choose a different port (e.g. COMFY_PORT=8189 make start)." >&2
    exit 1
  fi
fi

for arg in "$@"; do
  case "${arg}" in
    --listen)
      LISTEN="1"
      ;;
    --port=*)
      PORT="${arg#*=}"
      ;;
  esac
done

if [[ "${LISTEN}" == "1" ]]; then
  echo "WARNING: --listen exposes an unauthenticated ComfyUI API on your LAN." >&2
fi

if [[ ! -d "${COMFY_DIR}/.venv" ]]; then
  echo "Missing ComfyUI venv at ${COMFY_DIR}/.venv. Run: make bootstrap" >&2
  exit 1
fi

cd "${COMFY_DIR}"
source .venv/bin/activate

CPU_FLAG=""
if [[ "${FORCE_CPU}" == "1" ]]; then
  CPU_FLAG="--cpu"
else
  MPS_AVAIL="$(python -c 'import torch; print(int(getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()))' 2>/dev/null || echo 0)"
  CUDA_AVAIL="$(python -c 'import torch; print(int(torch.cuda.is_available()))' 2>/dev/null || echo 0)"
  if [[ "${MPS_AVAIL}" != "1" && "${CUDA_AVAIL}" != "1" ]]; then
    CPU_FLAG="--cpu"
    echo "No MPS/CUDA detected; starting ComfyUI with --cpu."
  fi
fi

echo "Starting ComfyUI on http://127.0.0.1:${PORT} (logs: ${LOG_FILE}) ${CPU_FLAG}"
if [[ "${LISTEN}" == "1" ]]; then
  python main.py --listen --port "${PORT}" ${CPU_FLAG} 2>&1 | tee "${LOG_FILE}"
else
  python main.py --port "${PORT}" ${CPU_FLAG} 2>&1 | tee "${LOG_FILE}"
fi
