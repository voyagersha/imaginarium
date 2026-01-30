#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from decimal import Decimal, getcontext
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

from persona_stack.cli import (
    _apply_flux_guidance,
    _apply_lora_strengths,
    _apply_sampler_settings,
    _apply_style_strength,
    _clip_stop_value,
)
from persona_stack.comfy_client import ComfyUIError
from persona_stack.hashing import sha256_json
from persona_stack.media import list_image_files
from persona_stack.runner import run_workflow


SUPPORTED_MODES = {"txt2img", "img2img", "inpaint", "img2img_identity"}


def _load_config(path: Path) -> dict[str, Any]:
    if path.is_dir():
        raise SystemExit(f"Config path is a directory: {path}. Pass a file (e.g. docs/sweep.zit.yaml).")
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise SystemExit(
                "YAML config requested but PyYAML is not installed.\n"
                "Install it with:\n"
                "  uv pip install pyyaml\n"
                f"Error: {exc}"
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return data


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def _range_values(spec: dict[str, Any]) -> list[Any]:
    if not {"start", "end", "step"} <= set(spec.keys()):
        raise SystemExit("Range spec must include start/end/step")
    start = spec.get("start")
    end = spec.get("end")
    step = spec.get("step")
    if step is None or start is None or end is None:
        raise SystemExit("Range spec missing start/end/step")
    if isinstance(step, bool) or isinstance(start, bool) or isinstance(end, bool):
        raise SystemExit("Range spec start/end/step must be numeric")

    getcontext().prec = 28
    s = Decimal(str(start))
    e = Decimal(str(end))
    st = Decimal(str(step))
    if st == 0:
        raise SystemExit("Range spec step must be non-zero")
    direction = 1 if e >= s else -1
    if st < 0:
        st = -st
    st = st * direction

    values: list[Decimal] = []
    current = s
    if direction > 0:
        while current <= e:
            values.append(current)
            current += st
    else:
        while current >= e:
            values.append(current)
            current += st

    all_ints = all(isinstance(v, int) and not isinstance(v, bool) for v in (start, end, step))
    if all_ints:
        return [int(v) for v in values]
    return [float(v) for v in values]


def _normalize_grid(grid: dict[str, Any] | None) -> dict[str, list[Any]]:
    if not grid:
        return {}
    normalized: dict[str, list[Any]] = {}
    for key, value in grid.items():
        if isinstance(value, dict):
            normalized[key] = _range_values(value)
            continue
        if isinstance(value, list):
            normalized[key] = value
        else:
            normalized[key] = [value]
    return normalized


def _grid_product(grid: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def _resolve_prompt_section(section: dict[str, Any], key: str) -> str | None:
    file_key = f"{key}_file"
    if file_key in section and section[file_key]:
        return _read_text(section[file_key])
    value = section.get(key)
    if value is None:
        return None
    return str(value)


def _resolve_prompts_config(prompt_cfg: dict[str, Any], mode: str) -> dict[str, str]:
    mode_overrides = (prompt_cfg.get("by_mode") or {}).get(mode) or {}
    if mode_overrides and not isinstance(mode_overrides, dict):
        raise SystemExit(f"prompts.by_mode.{mode} must be a mapping")

    resolved: dict[str, str] = {}
    for key in ("positive", "negative", "positive_refiner", "negative_refiner"):
        value = _resolve_prompt_section(prompt_cfg, key)
        override = _resolve_prompt_section(mode_overrides, key) if mode_overrides else None
        if override is not None:
            value = override
        if value is not None:
            resolved[key] = value
    return resolved


def _merge_prompts(base: dict[str, str], override: dict[str, str]) -> dict[str, str]:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = value
    return merged


def _resolve_mask_path(mask_spec: str | Path, image_path: Path) -> str:
    spec = str(mask_spec)
    if "{stem}" in spec or "{name}" in spec:
        resolved = spec.format(stem=image_path.stem, name=image_path.name)
        mask_path = Path(resolved)
        if not mask_path.exists():
            raise SystemExit(f"Mask not found: {mask_path}")
        return str(mask_path)

    mask_path = Path(spec)
    if mask_path.is_file():
        return str(mask_path)
    if mask_path.is_dir():
        candidate = mask_path / image_path.stem / "mask.png"
        if candidate.exists():
            return str(candidate)
        candidate = mask_path / f"{image_path.stem}.png"
        if candidate.exists():
            return str(candidate)
        raise SystemExit(f"Mask not found for {image_path.name} in {mask_path}")
    raise SystemExit(f"Mask path not found: {mask_path}")


def _select_images(
    image_input: str | Path | Sequence[str | Path],
    *,
    sample: int | None,
    sample_seed: int,
    max_images: int | None,
) -> list[str]:
    sources: list[str | Path] = []
    if isinstance(image_input, (list, tuple)):
        sources = list(image_input)
    else:
        sources = [image_input]
    images = list_image_files(sources)
    if not images:
        raise SystemExit(f"No images found at {image_input}")
    if sample:
        rng = random.Random(sample_seed)
        count = min(sample, len(images))
        images = rng.sample(images, count)
    if max_images:
        images = images[: max(1, int(max_images))]
    return images


def _normalize_workflow_entries(mode: str, value: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def _add_entry(item: Any, idx: int) -> None:
        if isinstance(item, str):
            entry = {"path": item}
        elif isinstance(item, dict):
            entry = dict(item)
        else:
            raise SystemExit(f"workflows.{mode} entries must be a string or mapping")

        enabled = bool(entry.get("enabled", True))
        path = entry.get("path")
        if not path:
            if enabled:
                raise SystemExit(f"workflows.{mode} entry missing path")
            return

        name = entry.get("name") or Path(str(path)).stem or f"{mode}_{idx}"
        params = entry.get("params") or {}
        grid = entry.get("grid") or {}
        prompts = entry.get("prompts") or {}

        if params and not isinstance(params, dict):
            raise SystemExit(f"workflows.{mode}.{name}.params must be a mapping")
        if grid and not isinstance(grid, dict):
            raise SystemExit(f"workflows.{mode}.{name}.grid must be a mapping")
        if prompts and not isinstance(prompts, dict):
            raise SystemExit(f"workflows.{mode}.{name}.prompts must be a mapping")

        if enabled:
            entries.append(
                {
                    "name": str(name),
                    "path": str(path),
                    "params": params,
                    "grid": grid,
                    "prompts": prompts,
                }
            )

    if isinstance(value, list):
        for idx, item in enumerate(value, start=1):
            _add_entry(item, idx)
    else:
        _add_entry(value, 1)

    return entries


def _resolve_workflows(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    workflows = config.get("workflows") or {}
    if not isinstance(workflows, dict):
        raise SystemExit("workflows must be a mapping")
    resolved: dict[str, list[dict[str, Any]]] = {}
    for mode, value in workflows.items():
        if mode not in SUPPORTED_MODES:
            continue
        entries = _normalize_workflow_entries(mode, value)
        if entries:
            resolved[mode] = entries
    return resolved


def _resolve_modes(config: dict[str, Any], workflows: dict[str, list[dict[str, Any]]]) -> list[str]:
    modes = config.get("modes")
    if modes is None:
        modes = (config.get("settings") or {}).get("modes")
    if modes is None:
        return sorted(workflows.keys())
    if not isinstance(modes, list):
        raise SystemExit("modes must be a list if provided")
    selected = [m for m in modes if m in SUPPORTED_MODES]
    if not selected:
        raise SystemExit("No valid modes selected")
    return selected


def _apply_special_params(patch_values: dict[str, Any], params: dict[str, Any]) -> None:
    clip_skip = params.pop("clip_skip", None)
    if clip_skip is not None:
        patch_values["clip_skip"] = _clip_stop_value(int(clip_skip))

    sampler_name = params.pop("sampler_name", None)
    scheduler = params.pop("scheduler", None)
    _apply_sampler_settings(patch_values, sampler_name, scheduler)

    flux_guidance = params.pop("flux_guidance", None)
    _apply_flux_guidance(patch_values, flux_guidance)

    dmd2_strength = params.pop("dmd2_strength", None)
    lightning_strength = params.pop("lightning_strength", None)
    subtle_strength = params.pop("subtle_strength", None)
    _apply_lora_strengths(patch_values, dmd2_strength, lightning_strength, subtle_strength)

    style_strength = params.pop("style_strength", None)
    cuteness_strength = params.pop("cuteness_strength", None)
    effective_style = style_strength if style_strength is not None else cuteness_strength
    _apply_style_strength(patch_values, effective_style)


def _build_patch_values(
    *,
    mode: str,
    base_params: dict[str, Any],
    grid_params: dict[str, Any],
    prompts: dict[str, str],
) -> dict[str, Any]:
    patch_values: dict[str, Any] = {}

    # prompts
    if "positive" in prompts:
        patch_values["positive_prompt"] = prompts["positive"]
    if "negative" in prompts:
        patch_values["negative_prompt"] = prompts["negative"]
    if "positive_refiner" in prompts:
        patch_values["positive_prompt_refiner"] = prompts["positive_refiner"]
    if "negative_refiner" in prompts:
        patch_values["negative_prompt_refiner"] = prompts["negative_refiner"]

    params = dict(base_params)
    params.update(grid_params)

    # mode defaults
    if mode == "inpaint":
        params.setdefault("filename_prefix", "persona_stack/{run_id}/repaired")
        params.setdefault("init_image", "init_image.png")
        params.setdefault("mask_image", "mask.png")
    elif mode in {"img2img", "img2img_identity"}:
        params.setdefault("filename_prefix", "persona_stack/{run_id}/raw")
        params.setdefault("init_image", "init.png")
    else:
        params.setdefault("filename_prefix", "persona_stack/{run_id}/raw")

    _apply_special_params(patch_values, params)

    for key, value in params.items():
        if value is None:
            continue
        patch_values[key] = value

    return patch_values


def _spec_hash(spec: dict[str, Any]) -> str:
    return sha256_json(spec)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sequential parameter sweeps against ComfyUI workflows.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config file.")
    parser.add_argument("--image", help="Image file or directory for img2img/inpaint modes.")
    parser.add_argument("--mask", help="Mask file or directory for inpaint (optional if set in config).")
    parser.add_argument("--max-images", type=int, default=None, help="Max number of images to run.")
    parser.add_argument("--sample", type=int, default=None, help="Randomly sample N images.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for image sampling.")
    parser.add_argument("--timeout", type=float, default=None, help="Override workflow timeout in seconds.")
    parser.add_argument("--log", help="Path to JSONL log file.")
    parser.add_argument("--resume", action="store_true", help="Skip runs already recorded as ok in log.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config(config_path)

    settings = config.get("settings") or {}
    if settings and not isinstance(settings, dict):
        raise SystemExit("settings must be a mapping if provided")

    workflows = _resolve_workflows(config)
    modes = _resolve_modes(config, workflows)
    for mode in modes:
        if mode not in workflows:
            raise SystemExit(f"Workflow path missing for mode: {mode}")

    sweep_ts = time.strftime("%Y%m%d-%H%M%S")
    comfy_url = str(settings.get("comfy_url") or "http://127.0.0.1:8188")
    out_root_setting = settings.get("out_root")
    if out_root_setting:
        out_root_path = Path(str(out_root_setting))
        out_root = str(out_root_path)
        sweep_root = out_root_path.parent if out_root_path.name == "runs" else out_root_path
    else:
        sweep_root = Path("runs") / "sweeps" / sweep_ts
        out_root = str(sweep_root / "runs")
    timeout_s = float(args.timeout if args.timeout is not None else settings.get("timeout_s", 600.0))

    inputs = config.get("inputs") or {}
    if inputs and not isinstance(inputs, dict):
        raise SystemExit("inputs must be a mapping if provided")
    image_input = args.image or inputs.get("image")
    mask_input = args.mask or inputs.get("mask")

    anchors = inputs.get("anchors") or []
    if not isinstance(anchors, list):
        anchors = [anchors]

    anchor_count = inputs.get("anchor_count")
    anchor_pad = inputs.get("anchor_pad", True)
    anchor_pad_color = inputs.get("anchor_pad_color", "black")
    anchor_max_side = inputs.get("anchor_max_side")
    anchor_weight_scale = inputs.get("anchor_weight_scale", 1.0)

    base_params = config.get("params") or {}
    if base_params and not isinstance(base_params, dict):
        raise SystemExit("params must be a mapping if provided")
    params_by_mode = config.get("params_by_mode") or {}
    if params_by_mode and not isinstance(params_by_mode, dict):
        raise SystemExit("params_by_mode must be a mapping if provided")

    grid_common = _normalize_grid(config.get("grid"))
    grid_by_mode = config.get("grid_by_mode") or {}
    if grid_by_mode and not isinstance(grid_by_mode, dict):
        raise SystemExit("grid_by_mode must be a mapping if provided")

    prompt_cfg = config.get("prompts") or {}
    if prompt_cfg and not isinstance(prompt_cfg, dict):
        raise SystemExit("prompts must be a mapping if provided")

    only_workflows = settings.get("only_workflows")
    if only_workflows is not None and not isinstance(only_workflows, list):
        raise SystemExit("settings.only_workflows must be a list when provided")
    skip_workflows = settings.get("skip_workflows")
    if skip_workflows is not None and not isinstance(skip_workflows, list):
        raise SystemExit("settings.skip_workflows must be a list when provided")

    # Input images
    images: list[str] = []
    if any(m in {"img2img", "inpaint", "img2img_identity"} for m in modes):
        if image_input is None:
            raise SystemExit("image input is required for img2img/inpaint/img2img_identity modes")
        images = _select_images(
            image_input,
            sample=args.sample,
            sample_seed=args.sample_seed,
            max_images=args.max_images,
        )
    if "img2img_identity" in modes and not anchors:
        raise SystemExit("anchors are required for img2img_identity mode")

    log_path: Path
    if args.log:
        log_path = Path(args.log)
    else:
        log_path = sweep_root / "sweep.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    completed: set[str] = set()
    if args.resume and log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok" and row.get("spec_id"):
                completed.add(row["spec_id"])

    total_runs = 0
    ok_runs = 0
    error_runs = 0

    with log_path.open("a", encoding="utf-8") as log_file:
        for mode in modes:
            grid_mode = _normalize_grid(grid_by_mode.get(mode, {}))
            mode_params = params_by_mode.get(mode) or {}
            if mode_params and not isinstance(mode_params, dict):
                raise SystemExit(f"params_by_mode.{mode} must be a mapping")

            for wf in workflows.get(mode, []):
                wf_name = wf["name"]
                if only_workflows and wf_name not in only_workflows:
                    continue
                if skip_workflows and wf_name in skip_workflows:
                    continue

                merged_grid = dict(grid_common)
                merged_grid.update(grid_mode)
                merged_grid.update(_normalize_grid(wf.get("grid") or {}))

                prompts = _resolve_prompts_config(prompt_cfg, mode)
                wf_prompts = wf.get("prompts") or {}
                if wf_prompts:
                    prompts = _merge_prompts(prompts, _resolve_prompts_config(wf_prompts, mode))

                params = dict(base_params)
                params.update(mode_params)
                params.update(wf.get("params") or {})

                has_prompt = bool(prompts.get("positive"))
                if not has_prompt:
                    has_prompt = "positive_prompt" in params or "positive_prompt" in merged_grid
                if not has_prompt:
                    raise SystemExit(f"Missing positive prompt for mode {mode} ({wf_name})")

                for grid_params in _grid_product(merged_grid):
                    patch_values = _build_patch_values(
                        mode=mode,
                        base_params=params,
                        grid_params=grid_params,
                        prompts=prompts,
                    )

                    target_images = [None]
                    if mode in {"img2img", "inpaint", "img2img_identity"}:
                        target_images = images

                    for image_path in target_images:
                        mask_path = None
                        if mode == "inpaint":
                            if not mask_input:
                                raise SystemExit("mask input is required for inpaint mode")
                            mask_path = _resolve_mask_path(mask_input, Path(str(image_path)))

                        spec = {
                            "mode": mode,
                            "workflow_name": wf_name,
                            "workflow": wf["path"],
                            "image": image_path,
                            "mask": mask_path,
                            "anchors": anchors,
                            "grid_params": grid_params,
                            "base_params": params,
                            "prompts": prompts,
                        }
                        spec_id = _spec_hash(spec)
                        if args.resume and spec_id in completed:
                            continue

                        total_runs += 1
                        started = time.time()
                        entry = {
                            "spec_id": spec_id,
                            "mode": mode,
                            "workflow_name": wf_name,
                            "workflow": wf["path"],
                            "image": image_path,
                            "mask": mask_path,
                            "anchors": anchors,
                            "grid_params": grid_params,
                            "patch_values": patch_values,
                            "started_at": started,
                        }

                        try:
                            result = run_workflow(
                                base_url=comfy_url,
                                workflow_path=wf["path"],
                                out_root=out_root,
                                patch_values=patch_values,
                                init_image=str(image_path) if image_path else None,
                                mask_image=str(mask_path) if mask_path else None,
                                identity_refs=anchors,
                                identity_limit=int(anchor_count) if anchor_count else None,
                                anchor_pad=bool(anchor_pad),
                                anchor_pad_color=str(anchor_pad_color),
                                anchor_max_side=int(anchor_max_side) if anchor_max_side else None,
                                anchor_weight_scale=float(anchor_weight_scale),
                                timeout_s=timeout_s,
                            )
                            entry.update(
                                {
                                    "status": "ok",
                                    "run_id": result.run_id,
                                    "prompt_id": result.prompt_id,
                                    "images": result.images,
                                    "finished_at": time.time(),
                                }
                            )
                            ok_runs += 1
                        except (ComfyUIError, Exception) as exc:
                            entry.update(
                                {
                                    "status": "error",
                                    "error": str(exc),
                                    "finished_at": time.time(),
                                }
                            )
                            error_runs += 1

                        log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        log_file.flush()

    print(f"Completed sweep: total={total_runs} ok={ok_runs} errors={error_runs}")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
