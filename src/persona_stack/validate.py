from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .comfy_client import ComfyUIClient
from .hashing import sha256_file
from .media import read_json
from .workflow_patch import ensure_patch_points


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    details: list[str]

def _read_safetensors_header(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        return header if isinstance(header, dict) else None
    except Exception:
        return None


def validate_comfy_running(base_url: str) -> ValidationResult:
    client = ComfyUIClient(base_url)
    try:
        stats = client.get_system_stats()
    except Exception as e:
        return ValidationResult(ok=False, details=[f"ComfyUI not reachable at {base_url}: {e}"])
    return ValidationResult(ok=True, details=[f"ComfyUI reachable: {stats.get('system', 'ok')}"])


def validate_models(manifest_path: str) -> ValidationResult:
    details: list[str] = []
    manifest = read_json(manifest_path)
    entries = manifest.get("artifacts") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        return ValidationResult(ok=False, details=[f"Invalid manifest format (expected artifacts list): {manifest_path}"])

    ok = True
    for entry in entries:
        if not isinstance(entry, dict):
            ok = False
            details.append(f"Invalid manifest entry: {entry!r}")
            continue
        entry_id = str(entry.get("id") or entry.get("filename") or "unknown")
        dest_dir = entry.get("dest_dir")
        dest_filename = entry.get("dest_filename") or entry.get("filename")
        if not dest_dir or not dest_filename:
            ok = False
            details.append(f"Manifest entry missing dest_dir/dest_filename: {entry}")
            continue
        dest_path = Path(dest_dir) / dest_filename
        if not dest_path.exists():
            ok = False
            details.append(f"Missing: {entry_id}: {dest_path}")
            continue
        expected_sha = (entry.get("sha256") or "").strip().lower()
        if expected_sha:
            actual = sha256_file(str(dest_path))
            if actual != expected_sha:
                ok = False
                details.append(f"SHA256 mismatch: {entry_id}: {dest_path} expected={expected_sha} actual={actual}")
            else:
                details.append(f"OK: {entry_id}: sha256 OK: {dest_path}")
        else:
            details.append(f"SHA256 not pinned (ok for first run): {entry_id}: {dest_path}")

        if entry_id == "clip_vision_vit_h" and dest_path.suffix.lower() == ".safetensors":
            header = _read_safetensors_header(dest_path)
            if header is None:
                ok = False
                details.append(f"Could not read safetensors header: {entry_id}: {dest_path}")
            else:
                embed_shape = None
                if "vision_model.embeddings.class_embedding" in header:
                    embed_shape = header["vision_model.embeddings.class_embedding"].get("shape")
                elif "visual.class_embedding" in header:
                    embed_shape = header["visual.class_embedding"].get("shape")
                embed_dim = embed_shape[0] if isinstance(embed_shape, list) and embed_shape else None
                if embed_dim is not None and embed_dim != 1280:
                    ok = False
                    details.append(f"Unexpected CLIP vision embed dim for {entry_id}: expected=1280 actual={embed_dim} ({dest_path})")
    return ValidationResult(ok=ok, details=details)


def validate_workflow(workflow_path: str, required_keys: set[str]) -> ValidationResult:
    try:
        workflow_wrapper: dict[str, Any] = read_json(workflow_path)
        ensure_patch_points(workflow_wrapper, required_keys)
        return ValidationResult(ok=True, details=[f"Workflow patch points OK: {workflow_path}"])
    except Exception as e:
        return ValidationResult(ok=False, details=[f"Workflow invalid: {workflow_path}: {e}"])
