from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .comfy_client import ComfyUIClient, ComfyUIError
from .hashing import sha256_json
from .media import (
    ensure_dir,
    list_image_files,
    parse_rgb,
    pad_image_to_square,
    read_json,
    write_json,
    write_placeholder_image,
)
from .workflow_patch import apply_patch_values


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: str
    images: list[str]
    prompt_id: str


def _new_run_id() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def _is_sam3_mps_runtime_error(message: str) -> bool:
    msg = message.lower()
    return (
        "operationutils.mm" in msg
        and "placeholder tensor is empty" in msg
        and "mps" in msg
    )


def run_workflow(
    *,
    base_url: str,
    workflow_path: str,
    out_root: str,
    patch_values: dict[str, Any],
    identity_refs: list[str] | None = None,
    style_refs: list[str] | None = None,
    init_image: str | None = None,
    mask_image: str | None = None,
    pose_image: str | None = None,
    identity_limit: int | None = None,
    timeout_s: float = 600.0,
    anchor_pad: bool = True,
    anchor_pad_color: str = "black",
    anchor_max_side: int | None = None,
    anchor_weight_scale: float = 1.0,
) -> RunResult:
    identity_refs = identity_refs or []
    style_refs = style_refs or []

    run_id = _new_run_id()
    run_dir = str(Path(out_root) / run_id)
    raw_dir = str(Path(run_dir) / "raw")
    repaired_dir = str(Path(run_dir) / "repaired")
    meta_dir = str(Path(run_dir) / "meta")
    ensure_dir(raw_dir)
    ensure_dir(repaired_dir)
    ensure_dir(meta_dir)

    workflow_wrapper = read_json(workflow_path)

    comfy = ComfyUIClient(base_url)
    client_id = comfy.new_client_id()

    uploaded_identity: list[dict[str, Any]] = []
    pad_color = parse_rgb(anchor_pad_color)
    identity_paths = list_image_files(identity_refs)
    if identity_limit is not None:
        identity_limit = max(0, int(identity_limit))
        identity_paths = identity_paths[:identity_limit]

    if identity_paths and anchor_pad:
        anchor_dir = Path(run_dir) / "inputs" / "anchors"
        padded_paths: list[str] = []
        for idx, path in enumerate(identity_paths, start=1):
            dest_path = anchor_dir / f"anchor_{idx}.png"
            padded_paths.append(
                pad_image_to_square(
                    path,
                    dest_path,
                    pad_color=pad_color,
                    max_side=anchor_max_side,
                )
            )
        identity_paths = padded_paths
    for p in identity_paths:
        up = comfy.upload_image(p)
        uploaded_identity.append(up.__dict__)

    uploaded_style: list[dict[str, Any]] = []
    for p in list_image_files(style_refs):
        up = comfy.upload_image(p)
        uploaded_style.append(up.__dict__)

    uploaded_init: dict[str, Any] | None = None
    uploaded_mask: dict[str, Any] | None = None
    uploaded_pose: dict[str, Any] | None = None
    if init_image:
        uploaded_init = comfy.upload_image(init_image).__dict__
    if mask_image:
        uploaded_mask = comfy.upload_image(mask_image).__dict__
    if pose_image:
        uploaded_pose = comfy.upload_image(pose_image).__dict__

    resolved_patch_values: dict[str, Any] = dict(patch_values)
    filename_prefix = resolved_patch_values.get("filename_prefix")
    if isinstance(filename_prefix, str):
        resolved_patch_values["filename_prefix"] = filename_prefix.replace("{run_id}", run_id)
    patch_points = {}
    if isinstance(workflow_wrapper, dict):
        patch_points = workflow_wrapper.get("persona_stack", {}).get("patch_points", {}) or {}

    identity_keys = []
    identity_weight_keys = []
    for key in patch_points:
        if key.startswith("identity_image_"):
            identity_keys.append(key)
        elif key.startswith("identity_weight_"):
            identity_weight_keys.append(key)
    def _sort_key(value: str) -> int:
        suffix = value.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0
    identity_keys = sorted(identity_keys, key=_sort_key)
    identity_weight_keys = sorted(identity_weight_keys, key=_sort_key)

    if identity_keys and not uploaded_identity:
        raise ComfyUIError(f"Workflow requires identity anchors but none were provided (workflow={workflow_path}).")

    placeholder_name: str | None = None
    if uploaded_identity:
        if "identity_image" in patch_points and "identity_image" not in resolved_patch_values:
            resolved_patch_values["identity_image"] = uploaded_identity[0]["name"]

        if identity_keys and len(uploaded_identity) < len(identity_keys):
            placeholder_path = write_placeholder_image(
                Path(run_dir) / "inputs" / "anchors" / "placeholder.png",
                color=pad_color,
            )
            placeholder_upload = comfy.upload_image(placeholder_path)
            placeholder_name = placeholder_upload.name

        for idx, key in enumerate(identity_keys):
            if key in resolved_patch_values:
                continue
            if idx < len(uploaded_identity):
                resolved_patch_values[key] = uploaded_identity[idx]["name"]
            else:
                resolved_patch_values[key] = placeholder_name or uploaded_identity[-1]["name"]

        if identity_weight_keys:
            anchor_count = max(1, len(uploaded_identity))
            per_weight = (1.0 / anchor_count) * float(anchor_weight_scale)
            for idx, key in enumerate(identity_weight_keys):
                if key in resolved_patch_values:
                    continue
                resolved_patch_values[key] = per_weight if idx < len(uploaded_identity) else 0.0
    if uploaded_style and "style_image" not in resolved_patch_values:
        resolved_patch_values["style_image"] = uploaded_style[0]["name"]
    if uploaded_init and "init_image" in resolved_patch_values:
        resolved_patch_values["init_image"] = uploaded_init["name"]
    if uploaded_mask and "mask_image" in resolved_patch_values:
        resolved_patch_values["mask_image"] = uploaded_mask["name"]
    if uploaded_pose and "pose_image" in resolved_patch_values:
        resolved_patch_values["pose_image"] = uploaded_pose["name"]

    patched = apply_patch_values(workflow_wrapper, resolved_patch_values)
    prompt_graph = patched["prompt"]

    workflow_hash = sha256_json(prompt_graph)
    prompt_id = comfy.submit_prompt(prompt_graph, client_id=client_id)
    comfy.wait_for_completion(prompt_id=prompt_id, client_id=client_id, timeout_s=timeout_s)
    history_map = comfy.get_history(prompt_id)
    write_json(Path(meta_dir) / "history.json", history_map)

    history = history_map.get(prompt_id) if isinstance(history_map, dict) else None
    if not isinstance(history, dict):
        raise ComfyUIError(f"ComfyUI history missing for prompt_id={prompt_id} (see {meta_dir}/history.json)")

    status = history.get("status") or {}
    if isinstance(status, dict) and status.get("status_str") == "error":
        messages = status.get("messages") or []
        msg = "; ".join(str(m) for m in messages) if isinstance(messages, list) else str(messages)
        hint = ""
        if _is_sam3_mps_runtime_error(msg):
            hint = " Hint: SAM3 hit a PyTorch MPS backend bug. Restart ComfyUI in CPU mode: COMFY_FORCE_CPU=1 make start."
        raise ComfyUIError(f"ComfyUI execution failed: {msg}.{hint} (run_dir={run_dir})")

    images: list[str] = []
    outputs = history.get("outputs") or {}
    for node_id, out in (outputs.items() if isinstance(outputs, dict) else []):
        for img in (out.get("images") or []):
            filename = img.get("filename")
            subfolder = img.get("subfolder", "")
            type_ = img.get("type", "output")
            if not filename:
                continue
            data = comfy.download_view(filename=filename, subfolder=subfolder, type_=type_)
            dest_dir = repaired_dir if "repaired" in (patch_values.get("filename_prefix") or "") else raw_dir
            dest_path = str(Path(dest_dir) / filename)
            Path(dest_path).write_bytes(data)
            images.append(dest_path)

            meta = {
                "run_id": run_id,
                "prompt_id": prompt_id,
                "workflow_path": workflow_path,
                "workflow_hash": workflow_hash,
                "comfyui_base_url": base_url,
                "patch_values": resolved_patch_values,
                "identity_refs": uploaded_identity,
                "style_refs": uploaded_style,
                "init_image": uploaded_init,
                "mask_image": uploaded_mask,
                "comfyui_image": {"filename": filename, "subfolder": subfolder, "type": type_, "node_id": node_id},
                "time_unix": int(time.time()),
                "pins": {
                    "COMFYUI_REF": os.environ.get("COMFYUI_REF"),
                    "COMFYUI_MANAGER_REF": os.environ.get("COMFYUI_MANAGER_REF"),
                    "IPADAPTER_NODE_REF": os.environ.get("IPADAPTER_NODE_REF"),
                    "CONTROLNET_AUX_REF": os.environ.get("CONTROLNET_AUX_REF"),
                },
            }
            write_json(Path(meta_dir) / f"{Path(filename).stem}.meta.json", meta)

    if not images:
        raise ComfyUIError(f"No output images were produced (run_dir={run_dir}). Check {meta_dir}/history.json and vendor/ComfyUI/comfy.log.")

    run_summary = {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "workflow_path": workflow_path,
        "workflow_hash": workflow_hash,
        "images": images,
        "meta_dir": meta_dir,
    }
    write_json(Path(run_dir) / "run.json", run_summary)
    return RunResult(run_id=run_id, run_dir=run_dir, images=images, prompt_id=prompt_id)
