#!/usr/bin/env python3
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


GUI_DIR = Path("workflows/gui")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _next_ids(wf: dict) -> tuple[int, int, int]:
    max_node_id = max(node["id"] for node in wf["nodes"])
    max_link_id = max(link[0] for link in wf["links"]) if wf["links"] else 0
    max_order = max(node.get("order", 0) for node in wf["nodes"])
    return max_node_id, max_link_id, max_order


def _find_node(wf: dict, node_type: str) -> dict:
    return next(node for node in wf["nodes"] if node["type"] == node_type)


def _find_nodes(wf: dict, node_type: str) -> list[dict]:
    return [node for node in wf["nodes"] if node["type"] == node_type]


def _set_save_prefix(wf: dict, prefix: str) -> None:
    save = _find_node(wf, "SaveImage")
    save["widgets_values"] = [prefix]


def _add_link(wf: dict, origin_id: int, origin_slot: int, target_id: int, target_slot: int, link_type: str) -> int:
    _, max_link_id, _ = _next_ids(wf)
    link_id = max_link_id + 1
    wf["links"].append([link_id, origin_id, origin_slot, target_id, target_slot, link_type])
    return link_id


def _append_output_link(node: dict, slot_index: int, link_id: int) -> None:
    outputs = node.get("outputs", [])
    for out in outputs:
        if out.get("slot_index", 0) == slot_index:
            links = out.setdefault("links", []) or []
            if link_id not in links:
                links.append(link_id)
            out["links"] = links
            return
    raise ValueError(f"Missing output slot {slot_index} on node {node.get('id')}")


def _clear_output_link(node: dict, link_id: int) -> None:
    for out in node.get("outputs", []) or []:
        links = out.get("links") or []
        if link_id in links:
            out["links"] = [l for l in links if l != link_id]


def _update_link_target(wf: dict, link_id: int, target_id: int, target_slot: int, link_type: str | None = None) -> None:
    for link in wf["links"]:
        if link[0] == link_id:
            link[3] = target_id
            link[4] = target_slot
            if link_type is not None:
                link[5] = link_type
            return
    raise ValueError(f"Missing link id {link_id}")


def _update_link_origin(wf: dict, link_id: int, origin_id: int, origin_slot: int, link_type: str | None = None) -> None:
    for link in wf["links"]:
        if link[0] == link_id:
            link[1] = origin_id
            link[2] = origin_slot
            if link_type is not None:
                link[5] = link_type
            return
    raise ValueError(f"Missing link id {link_id}")


def _find_ksampler_inputs(node: dict) -> dict[str, dict]:
    return {inp["name"]: inp for inp in node.get("inputs", [])}


def _add_pose_nodes(wf: dict) -> dict:
    wf = deepcopy(wf)
    max_node_id, _, max_order = _next_ids(wf)

    clip_nodes = _find_nodes(wf, "CLIPTextEncode")
    if len(clip_nodes) != 2:
        raise ValueError("Expected exactly 2 CLIPTextEncode nodes for pose workflows.")
    positive = next(n for n in clip_nodes if (n.get("widgets_values") or [""])[0] == "PROMPT")
    negative = next(n for n in clip_nodes if n is not positive)
    ksampler = _find_node(wf, "KSampler")

    ksampler_inputs = _find_ksampler_inputs(ksampler)
    pos_link_id = ksampler_inputs["positive"]["link"]
    neg_link_id = ksampler_inputs["negative"]["link"]

    pose_load_id = max_node_id + 1
    openpose_id = max_node_id + 2
    control_loader_id = max_node_id + 3
    control_apply_id = max_node_id + 4
    order = max_order + 1

    pose_load = {
        "id": pose_load_id,
        "type": "LoadImage",
        "pos": [440, 1660],
        "size": [315, 314],
        "flags": {},
        "order": order,
        "mode": 0,
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [], "shape": 3},
            {"name": "MASK", "type": "MASK", "links": None, "shape": 3},
        ],
        "properties": {"Node name for S&R": "LoadImage"},
        "widgets_values": ["pose.png", "image"],
    }
    openpose = {
        "id": openpose_id,
        "type": "OpenposePreprocessor",
        "pos": [820, 1680],
        "size": {"0": 315, "1": 106},
        "flags": {},
        "order": order + 1,
        "mode": 0,
        "inputs": [{"name": "image", "type": "IMAGE", "link": None}],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [], "shape": 3, "slot_index": 0}
        ],
        "properties": {"Node name for S&R": "OpenposePreprocessor"},
        "widgets_values": ["enable", "enable", "enable"],
    }
    control_loader = {
        "id": control_loader_id,
        "type": "ControlNetLoader",
        "pos": [440, 1880],
        "size": {"0": 315, "1": 58},
        "flags": {},
        "order": order + 2,
        "mode": 0,
        "outputs": [
            {"name": "CONTROL_NET", "type": "CONTROL_NET", "links": [], "slot_index": 0}
        ],
        "properties": {"Node name for S&R": "ControlNetLoader"},
        "widgets_values": ["OpenPoseXL2.safetensors"],
    }
    control_apply = {
        "id": control_apply_id,
        "type": "ControlNetApplyAdvanced",
        "pos": [1210, 1680],
        "size": {"0": 315, "1": 186},
        "flags": {},
        "order": order + 3,
        "mode": 0,
        "inputs": [
            {"name": "positive", "type": "CONDITIONING", "link": None},
            {"name": "negative", "type": "CONDITIONING", "link": None},
            {"name": "control_net", "type": "CONTROL_NET", "link": None},
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "vae", "type": "VAE", "link": None},
        ],
        "outputs": [
            {"name": "positive", "type": "CONDITIONING", "links": [], "shape": 3, "slot_index": 0},
            {"name": "negative", "type": "CONDITIONING", "links": [], "shape": 3, "slot_index": 1},
        ],
        "properties": {"Node name for S&R": "ControlNetApplyAdvanced"},
        "widgets_values": [1.0, 0.0, 1.0],
    }

    wf["nodes"].extend([pose_load, openpose, control_loader, control_apply])

    # Rewire CLIP conditioning to ControlNetApplyAdvanced inputs.
    _update_link_target(wf, pos_link_id, control_apply_id, 0, "CONDITIONING")
    _update_link_target(wf, neg_link_id, control_apply_id, 1, "CONDITIONING")

    for inp in control_apply["inputs"]:
        if inp["name"] == "positive":
            inp["link"] = pos_link_id
        elif inp["name"] == "negative":
            inp["link"] = neg_link_id

    # Pose image -> OpenposePreprocessor.
    pose_link_id = _add_link(wf, pose_load_id, 0, openpose_id, 0, "IMAGE")
    _append_output_link(pose_load, 0, pose_link_id)
    openpose["inputs"][0]["link"] = pose_link_id

    # OpenposePreprocessor -> ControlNetApplyAdvanced.
    pose_img_link_id = _add_link(wf, openpose_id, 0, control_apply_id, 3, "IMAGE")
    _append_output_link(openpose, 0, pose_img_link_id)
    for inp in control_apply["inputs"]:
        if inp["name"] == "image":
            inp["link"] = pose_img_link_id

    # ControlNetLoader -> ControlNetApplyAdvanced.
    control_link_id = _add_link(wf, control_loader_id, 0, control_apply_id, 2, "CONTROL_NET")
    _append_output_link(control_loader, 0, control_link_id)
    for inp in control_apply["inputs"]:
        if inp["name"] == "control_net":
            inp["link"] = control_link_id

    # ControlNetApplyAdvanced -> KSampler (positive/negative).
    pos_out_link_id = _add_link(wf, control_apply_id, 0, ksampler["id"], 1, "CONDITIONING")
    neg_out_link_id = _add_link(wf, control_apply_id, 1, ksampler["id"], 2, "CONDITIONING")
    _append_output_link(control_apply, 0, pos_out_link_id)
    _append_output_link(control_apply, 1, neg_out_link_id)

    ksampler_inputs["positive"]["link"] = pos_out_link_id
    ksampler_inputs["negative"]["link"] = neg_out_link_id

    wf["last_node_id"] = max(node["id"] for node in wf["nodes"])
    wf["last_link_id"] = max(link[0] for link in wf["links"])
    return wf


def _add_img2img_identity_nodes(wf: dict) -> dict:
    wf = deepcopy(wf)
    max_node_id, _, max_order = _next_ids(wf)
    order = max_order + 1

    load_id = max_node_id + 1
    encode_id = max_node_id + 2

    checkpoint = _find_node(wf, "CheckpointLoaderSimple")
    empty_latent = _find_node(wf, "EmptyLatentImage")
    ksampler = _find_node(wf, "KSampler")

    ksampler_inputs = _find_ksampler_inputs(ksampler)
    latent_link_id = ksampler_inputs["latent_image"]["link"]

    load_node = {
        "id": load_id,
        "type": "LoadImage",
        "pos": [40, -40],
        "size": [315, 314],
        "flags": {},
        "order": order,
        "mode": 0,
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [], "shape": 3},
            {"name": "MASK", "type": "MASK", "links": None, "shape": 3},
        ],
        "properties": {"Node name for S&R": "LoadImage"},
        "widgets_values": ["init.png", "image"],
    }

    encode_node = {
        "id": encode_id,
        "type": "VAEEncode",
        "pos": [360, 140],
        "size": [315, 82],
        "flags": {},
        "order": order + 1,
        "mode": 0,
        "inputs": [
            {"name": "pixels", "type": "IMAGE", "link": None},
            {"name": "vae", "type": "VAE", "link": None},
        ],
        "outputs": [{"name": "LATENT", "type": "LATENT", "links": [], "slot_index": 0}],
        "properties": {"Node name for S&R": "VAEEncode"},
        "widgets_values": [],
    }

    wf["nodes"].extend([load_node, encode_node])

    # LoadImage -> VAEEncode
    img_link_id = _add_link(wf, load_id, 0, encode_id, 0, "IMAGE")
    _append_output_link(load_node, 0, img_link_id)
    encode_node["inputs"][0]["link"] = img_link_id

    # Checkpoint VAE -> VAEEncode
    vae_link_id = _add_link(wf, checkpoint["id"], 2, encode_id, 1, "VAE")
    _append_output_link(checkpoint, 2, vae_link_id)
    encode_node["inputs"][1]["link"] = vae_link_id

    # Rewire latent: VAEEncode -> KSampler (reuse existing latent link id).
    _update_link_origin(wf, latent_link_id, encode_id, 0, "LATENT")
    _clear_output_link(empty_latent, latent_link_id)
    _append_output_link(encode_node, 0, latent_link_id)

    # Default denoise for img2img.
    if ksampler.get("widgets_values"):
        ksampler["widgets_values"][-1] = 0.5

    wf["last_node_id"] = max(node["id"] for node in wf["nodes"])
    wf["last_link_id"] = max(link[0] for link in wf["links"])
    return wf


def main() -> None:
    if not GUI_DIR.exists():
        raise SystemExit("workflows/gui not found")

    character_paths = sorted(p for p in GUI_DIR.glob("character*.json") if "pose" not in p.stem)
    scene_paths = sorted(p for p in GUI_DIR.glob("scene*.json") if "pose" not in p.stem)

    for path in character_paths:
        base = _read_json(path)

        pose = _add_pose_nodes(base)
        _set_save_prefix(pose, "persona_stack/gui/character_pose")
        pose_path = GUI_DIR / path.name.replace("character", "character_pose", 1)
        _write_json(pose_path, pose)

        img2img_identity = _add_img2img_identity_nodes(base)
        _set_save_prefix(img2img_identity, "persona_stack/gui/img2img_identity")
        img2img_path = GUI_DIR / path.name.replace("character", "img2img_identity", 1)
        _write_json(img2img_path, img2img_identity)

    for path in scene_paths:
        base = _read_json(path)
        pose = _add_pose_nodes(base)
        _set_save_prefix(pose, "persona_stack/gui/scene_pose")
        pose_path = GUI_DIR / path.name.replace("scene", "scene_pose", 1)
        _write_json(pose_path, pose)


if __name__ == "__main__":
    main()
