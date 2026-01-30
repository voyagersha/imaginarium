from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PatchPoint:
    node_id: str
    input_key: str


def _load_patch_points(workflow_wrapper: dict[str, Any]) -> dict[str, PatchPoint]:
    ps = workflow_wrapper.get("persona_stack") or {}
    raw = ps.get("patch_points") or {}
    patch_points: dict[str, PatchPoint] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Invalid patch point for {key}: {value!r}")
        node_id = str(value.get("node_id"))
        input_key = str(value.get("input_key"))
        patch_points[key] = PatchPoint(node_id=node_id, input_key=input_key)
    return patch_points


def apply_patch_values(workflow_wrapper: dict[str, Any], patch_values: dict[str, Any]) -> dict[str, Any]:
    patched = copy.deepcopy(workflow_wrapper)
    prompt = patched.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError("workflow wrapper missing 'prompt' dict")

    patch_points = _load_patch_points(patched)
    for key, value in patch_values.items():
        if key not in patch_points:
            continue
        pp = patch_points[key]
        node = prompt.get(pp.node_id)
        if not isinstance(node, dict):
            raise ValueError(f"Missing node_id {pp.node_id} for patch key {key}")
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            raise ValueError(f"Node {pp.node_id} has non-dict inputs")
        inputs[pp.input_key] = value
    return patched


def ensure_patch_points(workflow_wrapper: dict[str, Any], required_keys: set[str]) -> None:
    patch_points = _load_patch_points(workflow_wrapper)
    missing = sorted(required_keys - set(patch_points.keys()))
    if missing:
        raise ValueError(f"Workflow missing required patch keys: {missing}")

