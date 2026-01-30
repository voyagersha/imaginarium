.PHONY: bootstrap start models validate smoke-test export-workflows install-gui-workflows

# Auto-load local env vars (HF_TOKEN, etc) if present.
ifneq (,$(wildcard .env))
include .env
export
endif

UV_CACHE_DIR ?= .uv-cache
HF_HUB_CACHE ?= .hf-cache/hub
COMFY_PORT ?= 8188
COMFY_URL ?= http://127.0.0.1:$(COMFY_PORT)
MODEL_IDS ?=

bootstrap:
	bash scripts/bootstrap_macos.sh

start:
	COMFY_PORT=$(COMFY_PORT) bash scripts/start_comfy.sh

export-workflows:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python scripts/export_workflow_templates.py

install-gui-workflows:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python scripts/install_gui_workflows.py

models:
	UV_CACHE_DIR=$(UV_CACHE_DIR) HF_HUB_CACHE=$(HF_HUB_CACHE) uv run python scripts/download_models.py --update-manifest --continue-on-error --cache-dir $(HF_HUB_CACHE) $(foreach id,$(MODEL_IDS),--id $(id))

validate:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python -m persona_stack.cli validate --comfy-url $(COMFY_URL)

smoke-test:
	COMFY_URL=$(COMFY_URL) bash scripts/smoke_test.sh
