# ai-persona-stack

Local, reproducible SDXL persona/image pipeline driven by ComfyUI plus a Python automation layer (`persona_stack`).

## Commands

- `make bootstrap` — install repo `uv` env + clone/install ComfyUI + custom nodes
- `make start` — start ComfyUI on `http://127.0.0.1:8188`
- `make models` — download required model artifacts (reads `manifests/models.json`)
- `make validate` — validate models + workflows and optionally run a tiny job
- `make smoke-test` — end-to-end test (requires ComfyUI running + models present)

## Editing rules

- Do not edit vendored ComfyUI code under `vendor/ComfyUI/`.
- Safe edit zones: `src/`, `scripts/`, `workflows/`, `manifests/`, and `*.md` docs.
- Keep workflow templates deterministic: patch only known fields via `persona_stack.patch_points`.

