from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sampler_node(
    *,
    node_id: str,
    model_ref: list[Any],
    positive_ref: list[Any],
    negative_ref: list[Any],
    latent_ref: list[Any],
    steps: int = 20,
    cfg: float = 7.0,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    denoise: float = 1.0,
) -> dict[str, Any]:
    return {
        node_id: {
            "class_type": "KSampler",
            "inputs": {
                "seed": 0,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "model": model_ref,
                "positive": positive_ref,
                "negative": negative_ref,
                "latent_image": latent_ref,
            },
        }
    }


def _next_node_id(prompt: dict[str, Any]) -> int:
    if not prompt:
        return 1
    return max(int(k) for k in prompt.keys()) + 1


def _style_suffix(lora_id: str) -> str:
    name = lora_id
    if name.startswith("subtle_"):
        name = name[len("subtle_") :]
    for suffix in ("_xl1", "_xl2", "_xl25"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _load_style_loras(registry_path: Path) -> list[dict[str, Any]]:
    if not registry_path.exists():
        return []
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return payload.get("loras", [])


def _apply_loras(
    prompt: dict[str, Any],
    loras: list[dict[str, Any]] | None,
    model_targets: list[tuple[str, str]],
    clip_targets: list[tuple[str, str]],
) -> dict[str, Any]:
    if not loras:
        return {}

    next_node_id = _next_node_id(prompt)
    model_ref: list[Any] = ["1", 0]
    clip_ref: list[Any] = ["1", 1]
    patch_points: dict[str, Any] = {}

    for spec in loras:
        node_id = str(next_node_id)
        prompt[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": spec["lora_name"],
                "strength_model": spec["strength_model"],
                "strength_clip": spec["strength_clip"],
            },
        }
        patch_points[f"{spec['patch_prefix']}_strength_model"] = {"node_id": node_id, "input_key": "strength_model"}
        patch_points[f"{spec['patch_prefix']}_strength_clip"] = {"node_id": node_id, "input_key": "strength_clip"}
        model_ref = [node_id, 0]
        clip_ref = [node_id, 1]
        next_node_id += 1

    for node_id, input_key in model_targets:
        prompt[node_id]["inputs"][input_key] = model_ref
    for node_id, input_key in clip_targets:
        prompt[node_id]["inputs"][input_key] = clip_ref

    return patch_points


def _t2i_character_prompt(
    ckpt_name: str,
    anchor_count: int,
    clip_skip: int,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Requires ComfyUI_IPAdapter_plus custom nodes.
    anchor_count = max(1, int(anchor_count))
    sampler = sampler or {}
    prompt: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "identity_1.png"}},
        "3": {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS FACE (portraits)"}},
    }

    next_node_id = 4
    anchor_node_ids = ["2"]
    for idx in range(2, anchor_count + 1):
        node_id = str(next_node_id)
        prompt[node_id] = {"class_type": "LoadImage", "inputs": {"image": f"identity_{idx}.png"}}
        anchor_node_ids.append(node_id)
        next_node_id += 1

    first_ipadapter_id = str(next_node_id)
    prompt[first_ipadapter_id] = {
        "class_type": "IPAdapter",
        "inputs": {
            "model": ["3", 0],
            "ipadapter": ["3", 1],
            "image": [anchor_node_ids[0], 0],
            "weight": 1.0,
            "start_at": 0.0,
            "end_at": 1.0,
            "weight_type": "standard",
        },
    }
    previous_model_ref = [first_ipadapter_id, 0]
    next_node_id += 1

    for anchor_id in anchor_node_ids[1:]:
        node_id = str(next_node_id)
        prompt[node_id] = {
            "class_type": "IPAdapter",
            "inputs": {
                "model": previous_model_ref,
                "ipadapter": ["3", 1],
                "image": [anchor_id, 0],
                "weight": 1.0,
                "start_at": 0.0,
                "end_at": 1.0,
                "weight_type": "standard",
            },
        }
        previous_model_ref = [node_id, 0]
        next_node_id += 1

    latent_id = str(next_node_id)
    prompt[latent_id] = {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
    next_node_id += 1

    clip_id = str(next_node_id)
    prompt[clip_id] = {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": clip_skip}}
    next_node_id += 1

    pos_id = str(next_node_id)
    prompt[pos_id] = {"class_type": "CLIPTextEncode", "inputs": {"text": "PROMPT", "clip": [clip_id, 0]}}
    next_node_id += 1

    neg_id = str(next_node_id)
    prompt[neg_id] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": [clip_id, 0]}}
    next_node_id += 1

    sampler_id = str(next_node_id)
    prompt.update(
        _sampler_node(
            node_id=sampler_id,
            model_ref=previous_model_ref,
            positive_ref=[pos_id, 0],
            negative_ref=[neg_id, 0],
            latent_ref=[latent_id, 0],
            steps=int(sampler.get("steps", 20)),
            cfg=float(sampler.get("cfg", 7.0)),
            sampler_name=str(sampler.get("sampler_name", "euler")),
            scheduler=str(sampler.get("scheduler", "normal")),
            denoise=float(sampler.get("denoise", 1.0)),
        )
    )
    next_node_id += 1

    decode_id = str(next_node_id)
    prompt[decode_id] = {"class_type": "VAEDecode", "inputs": {"samples": [sampler_id, 0], "vae": ["1", 2]}}
    next_node_id += 1

    save_id = str(next_node_id)
    prompt[save_id] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "persona_stack/template/raw", "images": [decode_id, 0]}}
    return prompt


def _t2i_character_pose_prompt(
    ckpt_name: str,
    anchor_count: int,
    clip_skip: int,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Requires ComfyUI_IPAdapter_plus and comfyui_controlnet_aux custom nodes.
    anchor_count = max(1, int(anchor_count))
    sampler = sampler or {}
    prompt: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "identity_1.png"}},
        "3": {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS FACE (portraits)"}},
    }

    next_node_id = 4
    anchor_node_ids = ["2"]
    for idx in range(2, anchor_count + 1):
        node_id = str(next_node_id)
        prompt[node_id] = {"class_type": "LoadImage", "inputs": {"image": f"identity_{idx}.png"}}
        anchor_node_ids.append(node_id)
        next_node_id += 1

    first_ipadapter_id = str(next_node_id)
    prompt[first_ipadapter_id] = {
        "class_type": "IPAdapter",
        "inputs": {
            "model": ["3", 0],
            "ipadapter": ["3", 1],
            "image": [anchor_node_ids[0], 0],
            "weight": 1.0,
            "start_at": 0.0,
            "end_at": 1.0,
            "weight_type": "standard",
        },
    }
    previous_model_ref = [first_ipadapter_id, 0]
    next_node_id += 1

    for anchor_id in anchor_node_ids[1:]:
        node_id = str(next_node_id)
        prompt[node_id] = {
            "class_type": "IPAdapter",
            "inputs": {
                "model": previous_model_ref,
                "ipadapter": ["3", 1],
                "image": [anchor_id, 0],
                "weight": 1.0,
                "start_at": 0.0,
                "end_at": 1.0,
                "weight_type": "standard",
            },
        }
        previous_model_ref = [node_id, 0]
        next_node_id += 1

    latent_id = str(next_node_id)
    prompt[latent_id] = {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
    next_node_id += 1

    clip_id = str(next_node_id)
    prompt[clip_id] = {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": clip_skip}}
    next_node_id += 1

    pos_id = str(next_node_id)
    prompt[pos_id] = {"class_type": "CLIPTextEncode", "inputs": {"text": "PROMPT", "clip": [clip_id, 0]}}
    next_node_id += 1

    neg_id = str(next_node_id)
    prompt[neg_id] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": [clip_id, 0]}}
    next_node_id += 1

    pose_load_id = str(next_node_id)
    prompt[pose_load_id] = {"class_type": "LoadImage", "inputs": {"image": "pose.png"}}
    next_node_id += 1

    pose_pre_id = str(next_node_id)
    prompt[pose_pre_id] = {
        "class_type": "OpenposePreprocessor",
        "inputs": {
            "image": [pose_load_id, 0],
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "resolution": 512,
            "scale_stick_for_xinsr_cn": "disable",
        },
    }
    next_node_id += 1

    control_id = str(next_node_id)
    prompt[control_id] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "OpenPoseXL2.safetensors"}}
    next_node_id += 1

    control_apply_id = str(next_node_id)
    prompt[control_apply_id] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {
            "positive": [pos_id, 0],
            "negative": [neg_id, 0],
            "control_net": [control_id, 0],
            "image": [pose_pre_id, 0],
            "strength": 1.0,
            "start_percent": 0.0,
            "end_percent": 1.0,
        },
    }
    next_node_id += 1

    sampler_id = str(next_node_id)
    prompt.update(
        _sampler_node(
            node_id=sampler_id,
            model_ref=previous_model_ref,
            positive_ref=[control_apply_id, 0],
            negative_ref=[control_apply_id, 1],
            latent_ref=[latent_id, 0],
            steps=int(sampler.get("steps", 20)),
            cfg=float(sampler.get("cfg", 7.0)),
            sampler_name=str(sampler.get("sampler_name", "euler")),
            scheduler=str(sampler.get("scheduler", "normal")),
            denoise=float(sampler.get("denoise", 1.0)),
        )
    )
    next_node_id += 1

    decode_id = str(next_node_id)
    prompt[decode_id] = {"class_type": "VAEDecode", "inputs": {"samples": [sampler_id, 0], "vae": ["1", 2]}}
    next_node_id += 1

    save_id = str(next_node_id)
    prompt[save_id] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "persona_stack/template/raw", "images": [decode_id, 0]}}
    return prompt


def _t2i_style_prompt(
    ckpt_name: str,
    clip_skip: int,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Requires ComfyUI_IPAdapter_plus custom nodes.
    sampler = sampler or {}
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "style.png"}},
        "3": {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}},
        "4": {
            "class_type": "IPAdapter",
            "inputs": {
                "model": ["3", 0],
                "ipadapter": ["3", 1],
                "image": ["2", 0],
                "weight": 1.0,
                "start_at": 0.0,
                "end_at": 1.0,
                "weight_type": "style transfer",
            },
        },
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "PROMPT", "clip": ["11", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["11", 0]}},
        **_sampler_node(
            node_id="8",
            model_ref=["4", 0],
            positive_ref=["6", 0],
            negative_ref=["7", 0],
            latent_ref=["5", 0],
            steps=int(sampler.get("steps", 20)),
            cfg=float(sampler.get("cfg", 7.0)),
            sampler_name=str(sampler.get("sampler_name", "euler")),
            scheduler=str(sampler.get("scheduler", "normal")),
            denoise=float(sampler.get("denoise", 1.0)),
        ),
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": "persona_stack/template/raw", "images": ["9", 0]}},
        "11": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": clip_skip}},
    }


def _t2i_txt2img_prompt(
    ckpt_name: str,
    clip_skip: int,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sampler = sampler or {}
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "PROMPT", "clip": ["8", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["8", 0]}},
        **_sampler_node(
            node_id="5",
            model_ref=["1", 0],
            positive_ref=["3", 0],
            negative_ref=["4", 0],
            latent_ref=["2", 0],
            steps=int(sampler.get("steps", 20)),
            cfg=float(sampler.get("cfg", 7.0)),
            sampler_name=str(sampler.get("sampler_name", "euler")),
            scheduler=str(sampler.get("scheduler", "normal")),
            denoise=float(sampler.get("denoise", 1.0)),
        ),
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "persona_stack/template/raw", "images": ["6", 0]}},
        "8": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": clip_skip}},
    }


def _t2i_txt2img_refine_prompt(
    ckpt_name: str,
    refiner_ckpt_name: str,
    clip_skip: int,
    *,
    refiner_clip_skip: int = -1,
    upscale_by: float = 1.5,
    upscale_method: str = "bislerp",
    sampler: dict[str, Any] | None = None,
    refiner: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sampler = sampler or {}
    refiner = refiner or {}
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "PROMPT", "clip": ["8", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["8", 0]}},
        **_sampler_node(
            node_id="5",
            model_ref=["1", 0],
            positive_ref=["3", 0],
            negative_ref=["4", 0],
            latent_ref=["2", 0],
            steps=int(sampler.get("steps", 28)),
            cfg=float(sampler.get("cfg", 5.0)),
            sampler_name=str(sampler.get("sampler_name", "dpmpp_sde")),
            scheduler=str(sampler.get("scheduler", "karras")),
            denoise=float(sampler.get("denoise", 1.0)),
        ),
        "6": {
            "class_type": "LatentUpscaleBy",
            "inputs": {"samples": ["5", 0], "upscale_method": upscale_method, "scale_by": upscale_by},
        },
        "7": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": refiner_ckpt_name}},
        "8": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": clip_skip}},
        "9": {
            "class_type": "CLIPSetLastLayer",
            "inputs": {"clip": ["7", 1], "stop_at_clip_layer": refiner_clip_skip},
        },
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "PROMPT", "clip": ["9", 0]}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["9", 0]}},
        **_sampler_node(
            node_id="12",
            model_ref=["7", 0],
            positive_ref=["10", 0],
            negative_ref=["11", 0],
            latent_ref=["6", 0],
            steps=int(refiner.get("steps", 10)),
            cfg=float(refiner.get("cfg", 5.0)),
            sampler_name=str(refiner.get("sampler_name", "dpmpp_sde")),
            scheduler=str(refiner.get("scheduler", "karras")),
            denoise=float(refiner.get("denoise", 0.25)),
        ),
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["7", 2]}},
        "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": "persona_stack/template/raw", "images": ["13", 0]}},
    }


def _t2i_img2img_prompt(
    ckpt_name: str,
    clip_skip: int,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sampler = sampler or {}
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "init.png"}},
        "3": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
        "4": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": clip_skip}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "PROMPT", "clip": ["4", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 0]}},
        **_sampler_node(
            node_id="7",
            model_ref=["1", 0],
            positive_ref=["5", 0],
            negative_ref=["6", 0],
            latent_ref=["3", 0],
            steps=int(sampler.get("steps", 20)),
            cfg=float(sampler.get("cfg", 7.0)),
            sampler_name=str(sampler.get("sampler_name", "euler")),
            scheduler=str(sampler.get("scheduler", "normal")),
            denoise=float(sampler.get("denoise", 0.5)),
        ),
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "persona_stack/template/raw", "images": ["8", 0]}},
    }


def _t2i_img2img_identity_prompt(
    ckpt_name: str,
    anchor_count: int,
    clip_skip: int,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Requires ComfyUI_IPAdapter_plus custom nodes.
    anchor_count = max(1, int(anchor_count))
    sampler = sampler or {}
    prompt: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "init.png"}},
        "3": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "identity_1.png"}},
        "5": {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS FACE (portraits)"}},
    }

    next_node_id = 6
    anchor_node_ids = ["4"]
    for idx in range(2, anchor_count + 1):
        node_id = str(next_node_id)
        prompt[node_id] = {"class_type": "LoadImage", "inputs": {"image": f"identity_{idx}.png"}}
        anchor_node_ids.append(node_id)
        next_node_id += 1

    first_ipadapter_id = str(next_node_id)
    prompt[first_ipadapter_id] = {
        "class_type": "IPAdapter",
        "inputs": {
            "model": ["5", 0],
            "ipadapter": ["5", 1],
            "image": [anchor_node_ids[0], 0],
            "weight": 1.0,
            "start_at": 0.0,
            "end_at": 1.0,
            "weight_type": "standard",
        },
    }
    previous_model_ref = [first_ipadapter_id, 0]
    next_node_id += 1

    for anchor_id in anchor_node_ids[1:]:
        node_id = str(next_node_id)
        prompt[node_id] = {
            "class_type": "IPAdapter",
            "inputs": {
                "model": previous_model_ref,
                "ipadapter": ["5", 1],
                "image": [anchor_id, 0],
                "weight": 1.0,
                "start_at": 0.0,
                "end_at": 1.0,
                "weight_type": "standard",
            },
        }
        previous_model_ref = [node_id, 0]
        next_node_id += 1

    clip_id = str(next_node_id)
    prompt[clip_id] = {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": clip_skip}}
    next_node_id += 1

    pos_id = str(next_node_id)
    prompt[pos_id] = {"class_type": "CLIPTextEncode", "inputs": {"text": "PROMPT", "clip": [clip_id, 0]}}
    next_node_id += 1

    neg_id = str(next_node_id)
    prompt[neg_id] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": [clip_id, 0]}}
    next_node_id += 1

    sampler_id = str(next_node_id)
    prompt.update(
        _sampler_node(
            node_id=sampler_id,
            model_ref=previous_model_ref,
            positive_ref=[pos_id, 0],
            negative_ref=[neg_id, 0],
            latent_ref=["3", 0],
            steps=int(sampler.get("steps", 20)),
            cfg=float(sampler.get("cfg", 7.0)),
            sampler_name=str(sampler.get("sampler_name", "euler")),
            scheduler=str(sampler.get("scheduler", "normal")),
            denoise=float(sampler.get("denoise", 0.5)),
        )
    )
    next_node_id += 1

    decode_id = str(next_node_id)
    prompt[decode_id] = {"class_type": "VAEDecode", "inputs": {"samples": [sampler_id, 0], "vae": ["1", 2]}}
    next_node_id += 1

    save_id = str(next_node_id)
    prompt[save_id] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "persona_stack/template/raw", "images": [decode_id, 0]}}
    return prompt


def _t2i_scene_prompt(
    ckpt_name: str,
    anchor_count: int,
    clip_skip: int,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Requires ComfyUI_IPAdapter_plus custom nodes.
    anchor_count = max(1, int(anchor_count))
    sampler = sampler or {}
    prompt: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "identity_1.png"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "style.png"}},
        "4": {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS FACE (portraits)"}},
    }

    next_node_id = 5
    anchor_node_ids = ["2"]
    for idx in range(2, anchor_count + 1):
        node_id = str(next_node_id)
        prompt[node_id] = {"class_type": "LoadImage", "inputs": {"image": f"identity_{idx}.png"}}
        anchor_node_ids.append(node_id)
        next_node_id += 1

    first_ipadapter_id = str(next_node_id)
    prompt[first_ipadapter_id] = {
        "class_type": "IPAdapter",
        "inputs": {
            "model": ["4", 0],
            "ipadapter": ["4", 1],
            "image": [anchor_node_ids[0], 0],
            "weight": 1.0,
            "start_at": 0.0,
            "end_at": 1.0,
            "weight_type": "standard",
        },
    }
    previous_model_ref = [first_ipadapter_id, 0]
    next_node_id += 1

    for anchor_id in anchor_node_ids[1:]:
        node_id = str(next_node_id)
        prompt[node_id] = {
            "class_type": "IPAdapter",
            "inputs": {
                "model": previous_model_ref,
                "ipadapter": ["4", 1],
                "image": [anchor_id, 0],
                "weight": 1.0,
                "start_at": 0.0,
                "end_at": 1.0,
                "weight_type": "standard",
            },
        }
        previous_model_ref = [node_id, 0]
        next_node_id += 1

    style_loader_id = str(next_node_id)
    prompt[style_loader_id] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": previous_model_ref, "preset": "PLUS (high strength)"}}
    next_node_id += 1

    style_ipadapter_id = str(next_node_id)
    prompt[style_ipadapter_id] = {
        "class_type": "IPAdapter",
        "inputs": {
            "model": [style_loader_id, 0],
            "ipadapter": [style_loader_id, 1],
            "image": ["3", 0],
            "weight": 1.0,
            "start_at": 0.0,
            "end_at": 1.0,
            "weight_type": "style transfer",
        },
    }
    next_node_id += 1

    latent_id = str(next_node_id)
    prompt[latent_id] = {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
    next_node_id += 1

    clip_id = str(next_node_id)
    prompt[clip_id] = {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": clip_skip}}
    next_node_id += 1

    pos_id = str(next_node_id)
    prompt[pos_id] = {"class_type": "CLIPTextEncode", "inputs": {"text": "PROMPT", "clip": [clip_id, 0]}}
    next_node_id += 1

    neg_id = str(next_node_id)
    prompt[neg_id] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": [clip_id, 0]}}
    next_node_id += 1

    sampler_id = str(next_node_id)
    prompt.update(
        _sampler_node(
            node_id=sampler_id,
            model_ref=[style_ipadapter_id, 0],
            positive_ref=[pos_id, 0],
            negative_ref=[neg_id, 0],
            latent_ref=[latent_id, 0],
            steps=int(sampler.get("steps", 20)),
            cfg=float(sampler.get("cfg", 7.0)),
            sampler_name=str(sampler.get("sampler_name", "euler")),
            scheduler=str(sampler.get("scheduler", "normal")),
            denoise=float(sampler.get("denoise", 1.0)),
        )
    )
    next_node_id += 1

    decode_id = str(next_node_id)
    prompt[decode_id] = {"class_type": "VAEDecode", "inputs": {"samples": [sampler_id, 0], "vae": ["1", 2]}}
    next_node_id += 1

    save_id = str(next_node_id)
    prompt[save_id] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "persona_stack/template/raw", "images": [decode_id, 0]}}
    return prompt


def _t2i_scene_pose_prompt(
    ckpt_name: str,
    anchor_count: int,
    clip_skip: int,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Requires ComfyUI_IPAdapter_plus and comfyui_controlnet_aux custom nodes.
    anchor_count = max(1, int(anchor_count))
    sampler = sampler or {}
    prompt: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "identity_1.png"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "style.png"}},
        "4": {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS FACE (portraits)"}},
    }

    next_node_id = 5
    anchor_node_ids = ["2"]
    for idx in range(2, anchor_count + 1):
        node_id = str(next_node_id)
        prompt[node_id] = {"class_type": "LoadImage", "inputs": {"image": f"identity_{idx}.png"}}
        anchor_node_ids.append(node_id)
        next_node_id += 1

    first_ipadapter_id = str(next_node_id)
    prompt[first_ipadapter_id] = {
        "class_type": "IPAdapter",
        "inputs": {
            "model": ["4", 0],
            "ipadapter": ["4", 1],
            "image": [anchor_node_ids[0], 0],
            "weight": 1.0,
            "start_at": 0.0,
            "end_at": 1.0,
            "weight_type": "standard",
        },
    }
    previous_model_ref = [first_ipadapter_id, 0]
    next_node_id += 1

    for anchor_id in anchor_node_ids[1:]:
        node_id = str(next_node_id)
        prompt[node_id] = {
            "class_type": "IPAdapter",
            "inputs": {
                "model": previous_model_ref,
                "ipadapter": ["4", 1],
                "image": [anchor_id, 0],
                "weight": 1.0,
                "start_at": 0.0,
                "end_at": 1.0,
                "weight_type": "standard",
            },
        }
        previous_model_ref = [node_id, 0]
        next_node_id += 1

    style_loader_id = str(next_node_id)
    prompt[style_loader_id] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": previous_model_ref, "preset": "PLUS (high strength)"}}
    next_node_id += 1

    style_ipadapter_id = str(next_node_id)
    prompt[style_ipadapter_id] = {
        "class_type": "IPAdapter",
        "inputs": {
            "model": [style_loader_id, 0],
            "ipadapter": [style_loader_id, 1],
            "image": ["3", 0],
            "weight": 1.0,
            "start_at": 0.0,
            "end_at": 1.0,
            "weight_type": "style transfer",
        },
    }
    next_node_id += 1

    latent_id = str(next_node_id)
    prompt[latent_id] = {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
    next_node_id += 1

    clip_id = str(next_node_id)
    prompt[clip_id] = {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": clip_skip}}
    next_node_id += 1

    pos_id = str(next_node_id)
    prompt[pos_id] = {"class_type": "CLIPTextEncode", "inputs": {"text": "PROMPT", "clip": [clip_id, 0]}}
    next_node_id += 1

    neg_id = str(next_node_id)
    prompt[neg_id] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": [clip_id, 0]}}
    next_node_id += 1

    pose_load_id = str(next_node_id)
    prompt[pose_load_id] = {"class_type": "LoadImage", "inputs": {"image": "pose.png"}}
    next_node_id += 1

    pose_pre_id = str(next_node_id)
    prompt[pose_pre_id] = {
        "class_type": "OpenposePreprocessor",
        "inputs": {
            "image": [pose_load_id, 0],
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "resolution": 512,
            "scale_stick_for_xinsr_cn": "disable",
        },
    }
    next_node_id += 1

    control_id = str(next_node_id)
    prompt[control_id] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "OpenPoseXL2.safetensors"}}
    next_node_id += 1

    control_apply_id = str(next_node_id)
    prompt[control_apply_id] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {
            "positive": [pos_id, 0],
            "negative": [neg_id, 0],
            "control_net": [control_id, 0],
            "image": [pose_pre_id, 0],
            "strength": 1.0,
            "start_percent": 0.0,
            "end_percent": 1.0,
        },
    }
    next_node_id += 1

    sampler_id = str(next_node_id)
    prompt.update(
        _sampler_node(
            node_id=sampler_id,
            model_ref=[style_ipadapter_id, 0],
            positive_ref=[control_apply_id, 0],
            negative_ref=[control_apply_id, 1],
            latent_ref=[latent_id, 0],
            steps=int(sampler.get("steps", 20)),
            cfg=float(sampler.get("cfg", 7.0)),
            sampler_name=str(sampler.get("sampler_name", "euler")),
            scheduler=str(sampler.get("scheduler", "normal")),
            denoise=float(sampler.get("denoise", 1.0)),
        )
    )
    next_node_id += 1

    decode_id = str(next_node_id)
    prompt[decode_id] = {"class_type": "VAEDecode", "inputs": {"samples": [sampler_id, 0], "vae": ["1", 2]}}
    next_node_id += 1

    save_id = str(next_node_id)
    prompt[save_id] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "persona_stack/template/raw", "images": [decode_id, 0]}}
    return prompt


def _character_wrapper(
    name: str,
    ckpt_name: str,
    anchor_count: int,
    clip_skip: int,
    *,
    loras: list[dict[str, Any]] | None = None,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    anchor_count = max(1, int(anchor_count))
    identity_ids = ["2"] + [str(idx + 2) for idx in range(2, anchor_count + 1)]
    identity_keys = {
        f"identity_image_{idx}": {"node_id": identity_ids[idx - 1], "input_key": "image"}
        for idx in range(1, anchor_count + 1)
    }
    first_ipadapter_id = anchor_count + 3
    identity_weight_keys = {
        f"identity_weight_{idx}": {"node_id": str(first_ipadapter_id + idx - 1), "input_key": "weight"}
        for idx in range(1, anchor_count + 1)
    }
    latent_id = first_ipadapter_id + anchor_count
    clip_id = latent_id + 1
    pos_id = clip_id + 1
    neg_id = pos_id + 1
    sampler_id = neg_id + 1
    save_id = sampler_id + 2
    prompt = _t2i_character_prompt(ckpt_name=ckpt_name, anchor_count=anchor_count, clip_skip=clip_skip, sampler=sampler)
    patch_points = {
        **identity_keys,
        **identity_weight_keys,
        "positive_prompt": {"node_id": str(pos_id), "input_key": "text"},
        "negative_prompt": {"node_id": str(neg_id), "input_key": "text"},
        "clip_skip": {"node_id": str(clip_id), "input_key": "stop_at_clip_layer"},
        "seed": {"node_id": str(sampler_id), "input_key": "seed"},
        "steps": {"node_id": str(sampler_id), "input_key": "steps"},
        "cfg": {"node_id": str(sampler_id), "input_key": "cfg"},
        "sampler_name": {"node_id": str(sampler_id), "input_key": "sampler_name"},
        "scheduler": {"node_id": str(sampler_id), "input_key": "scheduler"},
        "width": {"node_id": str(latent_id), "input_key": "width"},
        "height": {"node_id": str(latent_id), "input_key": "height"},
        "batch_size": {"node_id": str(latent_id), "input_key": "batch_size"},
        "filename_prefix": {"node_id": str(save_id), "input_key": "filename_prefix"},
    }
    patch_points.update(
        _apply_loras(
            prompt,
            loras,
            model_targets=[("3", "model")],
            clip_targets=[(str(clip_id), "clip")],
        )
    )
    return {
        "persona_stack": {
            "name": name,
            "version": 1,
            "patch_points": patch_points,
        },
        "prompt": prompt,
    }


def _character_pose_wrapper(
    name: str,
    ckpt_name: str,
    anchor_count: int,
    clip_skip: int,
    *,
    loras: list[dict[str, Any]] | None = None,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    anchor_count = max(1, int(anchor_count))
    identity_ids = ["2"] + [str(idx + 2) for idx in range(2, anchor_count + 1)]
    identity_keys = {
        f"identity_image_{idx}": {"node_id": identity_ids[idx - 1], "input_key": "image"}
        for idx in range(1, anchor_count + 1)
    }
    first_ipadapter_id = anchor_count + 3
    identity_weight_keys = {
        f"identity_weight_{idx}": {"node_id": str(first_ipadapter_id + idx - 1), "input_key": "weight"}
        for idx in range(1, anchor_count + 1)
    }
    latent_id = first_ipadapter_id + anchor_count
    clip_id = latent_id + 1
    pos_id = clip_id + 1
    neg_id = pos_id + 1
    pose_load_id = neg_id + 1
    pose_pre_id = pose_load_id + 1
    control_id = pose_pre_id + 1
    control_apply_id = control_id + 1
    sampler_id = control_apply_id + 1
    save_id = sampler_id + 2
    prompt = _t2i_character_pose_prompt(
        ckpt_name=ckpt_name,
        anchor_count=anchor_count,
        clip_skip=clip_skip,
        sampler=sampler,
    )
    patch_points = {
        **identity_keys,
        **identity_weight_keys,
        "pose_image": {"node_id": str(pose_load_id), "input_key": "image"},
        "pose_strength": {"node_id": str(control_apply_id), "input_key": "strength"},
        "pose_start": {"node_id": str(control_apply_id), "input_key": "start_percent"},
        "pose_end": {"node_id": str(control_apply_id), "input_key": "end_percent"},
        "positive_prompt": {"node_id": str(pos_id), "input_key": "text"},
        "negative_prompt": {"node_id": str(neg_id), "input_key": "text"},
        "clip_skip": {"node_id": str(clip_id), "input_key": "stop_at_clip_layer"},
        "seed": {"node_id": str(sampler_id), "input_key": "seed"},
        "steps": {"node_id": str(sampler_id), "input_key": "steps"},
        "cfg": {"node_id": str(sampler_id), "input_key": "cfg"},
        "sampler_name": {"node_id": str(sampler_id), "input_key": "sampler_name"},
        "scheduler": {"node_id": str(sampler_id), "input_key": "scheduler"},
        "width": {"node_id": str(latent_id), "input_key": "width"},
        "height": {"node_id": str(latent_id), "input_key": "height"},
        "batch_size": {"node_id": str(latent_id), "input_key": "batch_size"},
        "filename_prefix": {"node_id": str(save_id), "input_key": "filename_prefix"},
    }
    patch_points.update(
        _apply_loras(
            prompt,
            loras,
            model_targets=[("3", "model")],
            clip_targets=[(str(clip_id), "clip")],
        )
    )
    return {
        "persona_stack": {
            "name": name,
            "version": 1,
            "patch_points": patch_points,
        },
        "prompt": prompt,
    }


def _style_wrapper(
    name: str,
    ckpt_name: str,
    clip_skip: int,
    *,
    loras: list[dict[str, Any]] | None = None,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = _t2i_style_prompt(ckpt_name=ckpt_name, clip_skip=clip_skip, sampler=sampler)
    patch_points = {
        "style_image": {"node_id": "2", "input_key": "image"},
        "positive_prompt": {"node_id": "6", "input_key": "text"},
        "negative_prompt": {"node_id": "7", "input_key": "text"},
        "clip_skip": {"node_id": "11", "input_key": "stop_at_clip_layer"},
        "seed": {"node_id": "8", "input_key": "seed"},
        "steps": {"node_id": "8", "input_key": "steps"},
        "cfg": {"node_id": "8", "input_key": "cfg"},
        "sampler_name": {"node_id": "8", "input_key": "sampler_name"},
        "scheduler": {"node_id": "8", "input_key": "scheduler"},
        "width": {"node_id": "5", "input_key": "width"},
        "height": {"node_id": "5", "input_key": "height"},
        "batch_size": {"node_id": "5", "input_key": "batch_size"},
        "filename_prefix": {"node_id": "10", "input_key": "filename_prefix"},
    }
    patch_points.update(
        _apply_loras(
            prompt,
            loras,
            model_targets=[("3", "model")],
            clip_targets=[("11", "clip")],
        )
    )
    return {
        "persona_stack": {
            "name": name,
            "version": 1,
            "patch_points": patch_points,
        },
        "prompt": prompt,
    }


def _scene_wrapper(
    name: str,
    ckpt_name: str,
    anchor_count: int,
    clip_skip: int,
    *,
    loras: list[dict[str, Any]] | None = None,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    anchor_count = max(1, int(anchor_count))
    identity_ids = ["2"] + [str(idx + 3) for idx in range(2, anchor_count + 1)]
    identity_keys = {
        f"identity_image_{idx}": {"node_id": identity_ids[idx - 1], "input_key": "image"}
        for idx in range(1, anchor_count + 1)
    }
    style_node_id = "3"
    first_ipadapter_id = anchor_count + 4
    identity_weight_keys = {
        f"identity_weight_{idx}": {"node_id": str(first_ipadapter_id + idx - 1), "input_key": "weight"}
        for idx in range(1, anchor_count + 1)
    }
    style_loader_id = first_ipadapter_id + anchor_count
    style_ipadapter_id = style_loader_id + 1
    latent_id = style_ipadapter_id + 1
    clip_id = latent_id + 1
    pos_id = clip_id + 1
    neg_id = pos_id + 1
    sampler_id = neg_id + 1
    save_id = sampler_id + 2
    prompt = _t2i_scene_prompt(ckpt_name=ckpt_name, anchor_count=anchor_count, clip_skip=clip_skip, sampler=sampler)
    patch_points = {
        **identity_keys,
        **identity_weight_keys,
        "style_image": {"node_id": style_node_id, "input_key": "image"},
        "positive_prompt": {"node_id": str(pos_id), "input_key": "text"},
        "negative_prompt": {"node_id": str(neg_id), "input_key": "text"},
        "clip_skip": {"node_id": str(clip_id), "input_key": "stop_at_clip_layer"},
        "seed": {"node_id": str(sampler_id), "input_key": "seed"},
        "steps": {"node_id": str(sampler_id), "input_key": "steps"},
        "cfg": {"node_id": str(sampler_id), "input_key": "cfg"},
        "sampler_name": {"node_id": str(sampler_id), "input_key": "sampler_name"},
        "scheduler": {"node_id": str(sampler_id), "input_key": "scheduler"},
        "width": {"node_id": str(latent_id), "input_key": "width"},
        "height": {"node_id": str(latent_id), "input_key": "height"},
        "batch_size": {"node_id": str(latent_id), "input_key": "batch_size"},
        "filename_prefix": {"node_id": str(save_id), "input_key": "filename_prefix"},
    }
    patch_points.update(
        _apply_loras(
            prompt,
            loras,
            model_targets=[("4", "model")],
            clip_targets=[(str(clip_id), "clip")],
        )
    )
    return {
        "persona_stack": {
            "name": name,
            "version": 1,
            "patch_points": patch_points,
        },
        "prompt": prompt,
    }


def _scene_pose_wrapper(
    name: str,
    ckpt_name: str,
    anchor_count: int,
    clip_skip: int,
    *,
    loras: list[dict[str, Any]] | None = None,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    anchor_count = max(1, int(anchor_count))
    identity_ids = ["2"] + [str(idx + 3) for idx in range(2, anchor_count + 1)]
    identity_keys = {
        f"identity_image_{idx}": {"node_id": identity_ids[idx - 1], "input_key": "image"}
        for idx in range(1, anchor_count + 1)
    }
    style_node_id = "3"
    first_ipadapter_id = anchor_count + 4
    identity_weight_keys = {
        f"identity_weight_{idx}": {"node_id": str(first_ipadapter_id + idx - 1), "input_key": "weight"}
        for idx in range(1, anchor_count + 1)
    }
    style_loader_id = first_ipadapter_id + anchor_count
    style_ipadapter_id = style_loader_id + 1
    latent_id = style_ipadapter_id + 1
    clip_id = latent_id + 1
    pos_id = clip_id + 1
    neg_id = pos_id + 1
    pose_load_id = neg_id + 1
    pose_pre_id = pose_load_id + 1
    control_id = pose_pre_id + 1
    control_apply_id = control_id + 1
    sampler_id = control_apply_id + 1
    save_id = sampler_id + 2
    prompt = _t2i_scene_pose_prompt(
        ckpt_name=ckpt_name,
        anchor_count=anchor_count,
        clip_skip=clip_skip,
        sampler=sampler,
    )
    patch_points = {
        **identity_keys,
        **identity_weight_keys,
        "style_image": {"node_id": style_node_id, "input_key": "image"},
        "pose_image": {"node_id": str(pose_load_id), "input_key": "image"},
        "pose_strength": {"node_id": str(control_apply_id), "input_key": "strength"},
        "pose_start": {"node_id": str(control_apply_id), "input_key": "start_percent"},
        "pose_end": {"node_id": str(control_apply_id), "input_key": "end_percent"},
        "positive_prompt": {"node_id": str(pos_id), "input_key": "text"},
        "negative_prompt": {"node_id": str(neg_id), "input_key": "text"},
        "clip_skip": {"node_id": str(clip_id), "input_key": "stop_at_clip_layer"},
        "seed": {"node_id": str(sampler_id), "input_key": "seed"},
        "steps": {"node_id": str(sampler_id), "input_key": "steps"},
        "cfg": {"node_id": str(sampler_id), "input_key": "cfg"},
        "sampler_name": {"node_id": str(sampler_id), "input_key": "sampler_name"},
        "scheduler": {"node_id": str(sampler_id), "input_key": "scheduler"},
        "width": {"node_id": str(latent_id), "input_key": "width"},
        "height": {"node_id": str(latent_id), "input_key": "height"},
        "batch_size": {"node_id": str(latent_id), "input_key": "batch_size"},
        "filename_prefix": {"node_id": str(save_id), "input_key": "filename_prefix"},
    }
    patch_points.update(
        _apply_loras(
            prompt,
            loras,
            model_targets=[("4", "model")],
            clip_targets=[(str(clip_id), "clip")],
        )
    )
    return {
        "persona_stack": {
            "name": name,
            "version": 1,
            "patch_points": patch_points,
        },
        "prompt": prompt,
    }


def _txt2img_wrapper(
    name: str,
    ckpt_name: str,
    clip_skip: int,
    *,
    loras: list[dict[str, Any]] | None = None,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = _t2i_txt2img_prompt(ckpt_name=ckpt_name, clip_skip=clip_skip, sampler=sampler)
    patch_points = {
        "positive_prompt": {"node_id": "3", "input_key": "text"},
        "negative_prompt": {"node_id": "4", "input_key": "text"},
        "clip_skip": {"node_id": "8", "input_key": "stop_at_clip_layer"},
        "seed": {"node_id": "5", "input_key": "seed"},
        "steps": {"node_id": "5", "input_key": "steps"},
        "cfg": {"node_id": "5", "input_key": "cfg"},
        "sampler_name": {"node_id": "5", "input_key": "sampler_name"},
        "scheduler": {"node_id": "5", "input_key": "scheduler"},
        "width": {"node_id": "2", "input_key": "width"},
        "height": {"node_id": "2", "input_key": "height"},
        "batch_size": {"node_id": "2", "input_key": "batch_size"},
        "filename_prefix": {"node_id": "7", "input_key": "filename_prefix"},
    }
    patch_points.update(
        _apply_loras(
            prompt,
            loras,
            model_targets=[("5", "model")],
            clip_targets=[("8", "clip")],
        )
    )
    return {
        "persona_stack": {
            "name": name,
            "version": 1,
            "patch_points": patch_points,
        },
        "prompt": prompt,
    }


def _txt2img_refine_wrapper(
    name: str,
    ckpt_name: str,
    refiner_ckpt_name: str,
    clip_skip: int,
    *,
    loras: list[dict[str, Any]] | None = None,
    sampler: dict[str, Any] | None = None,
    refiner: dict[str, Any] | None = None,
    upscale_by: float = 1.5,
    upscale_method: str = "bislerp",
) -> dict[str, Any]:
    prompt = _t2i_txt2img_refine_prompt(
        ckpt_name=ckpt_name,
        refiner_ckpt_name=refiner_ckpt_name,
        clip_skip=clip_skip,
        sampler=sampler,
        refiner=refiner,
        upscale_by=upscale_by,
        upscale_method=upscale_method,
    )
    patch_points = {
        "positive_prompt": {"node_id": "3", "input_key": "text"},
        "negative_prompt": {"node_id": "4", "input_key": "text"},
        "positive_prompt_refiner": {"node_id": "10", "input_key": "text"},
        "negative_prompt_refiner": {"node_id": "11", "input_key": "text"},
        "clip_skip": {"node_id": "8", "input_key": "stop_at_clip_layer"},
        "seed": {"node_id": "5", "input_key": "seed"},
        "steps": {"node_id": "5", "input_key": "steps"},
        "cfg": {"node_id": "5", "input_key": "cfg"},
        "sampler_name": {"node_id": "5", "input_key": "sampler_name"},
        "scheduler": {"node_id": "5", "input_key": "scheduler"},
        "width": {"node_id": "2", "input_key": "width"},
        "height": {"node_id": "2", "input_key": "height"},
        "batch_size": {"node_id": "2", "input_key": "batch_size"},
        "filename_prefix": {"node_id": "14", "input_key": "filename_prefix"},
    }
    patch_points.update(
        _apply_loras(
            prompt,
            loras,
            model_targets=[("5", "model")],
            clip_targets=[("8", "clip")],
        )
    )
    return {
        "persona_stack": {
            "name": name,
            "version": 1,
            "patch_points": patch_points,
        },
        "prompt": prompt,
    }


def _img2img_wrapper(
    name: str,
    ckpt_name: str,
    clip_skip: int,
    *,
    loras: list[dict[str, Any]] | None = None,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = _t2i_img2img_prompt(ckpt_name=ckpt_name, clip_skip=clip_skip, sampler=sampler)
    patch_points = {
        "init_image": {"node_id": "2", "input_key": "image"},
        "positive_prompt": {"node_id": "5", "input_key": "text"},
        "negative_prompt": {"node_id": "6", "input_key": "text"},
        "clip_skip": {"node_id": "4", "input_key": "stop_at_clip_layer"},
        "seed": {"node_id": "7", "input_key": "seed"},
        "steps": {"node_id": "7", "input_key": "steps"},
        "cfg": {"node_id": "7", "input_key": "cfg"},
        "sampler_name": {"node_id": "7", "input_key": "sampler_name"},
        "scheduler": {"node_id": "7", "input_key": "scheduler"},
        "denoise": {"node_id": "7", "input_key": "denoise"},
        "filename_prefix": {"node_id": "9", "input_key": "filename_prefix"},
    }
    patch_points.update(
        _apply_loras(
            prompt,
            loras,
            model_targets=[("7", "model")],
            clip_targets=[("4", "clip")],
        )
    )
    return {
        "persona_stack": {
            "name": name,
            "version": 1,
            "patch_points": patch_points,
        },
        "prompt": prompt,
    }


def _img2img_identity_wrapper(
    name: str,
    ckpt_name: str,
    anchor_count: int,
    clip_skip: int,
    *,
    loras: list[dict[str, Any]] | None = None,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    anchor_count = max(1, int(anchor_count))
    identity_ids = ["4"] + [str(idx + 4) for idx in range(2, anchor_count + 1)]
    identity_keys = {
        f"identity_image_{idx}": {"node_id": identity_ids[idx - 1], "input_key": "image"}
        for idx in range(1, anchor_count + 1)
    }
    first_ipadapter_id = anchor_count + 5
    identity_weight_keys = {
        f"identity_weight_{idx}": {"node_id": str(first_ipadapter_id + idx - 1), "input_key": "weight"}
        for idx in range(1, anchor_count + 1)
    }
    clip_id = (2 * anchor_count) + 5
    pos_id = clip_id + 1
    neg_id = pos_id + 1
    sampler_id = neg_id + 1
    save_id = sampler_id + 2
    prompt = _t2i_img2img_identity_prompt(
        ckpt_name=ckpt_name,
        anchor_count=anchor_count,
        clip_skip=clip_skip,
        sampler=sampler,
    )
    patch_points = {
        "init_image": {"node_id": "2", "input_key": "image"},
        **identity_keys,
        **identity_weight_keys,
        "positive_prompt": {"node_id": str(pos_id), "input_key": "text"},
        "negative_prompt": {"node_id": str(neg_id), "input_key": "text"},
        "clip_skip": {"node_id": str(clip_id), "input_key": "stop_at_clip_layer"},
        "seed": {"node_id": str(sampler_id), "input_key": "seed"},
        "steps": {"node_id": str(sampler_id), "input_key": "steps"},
        "cfg": {"node_id": str(sampler_id), "input_key": "cfg"},
        "sampler_name": {"node_id": str(sampler_id), "input_key": "sampler_name"},
        "scheduler": {"node_id": str(sampler_id), "input_key": "scheduler"},
        "denoise": {"node_id": str(sampler_id), "input_key": "denoise"},
        "filename_prefix": {"node_id": str(save_id), "input_key": "filename_prefix"},
    }
    patch_points.update(
        _apply_loras(
            prompt,
            loras,
            model_targets=[(str(sampler_id), "model")],
            clip_targets=[(str(clip_id), "clip")],
        )
    )
    return {
        "persona_stack": {
            "name": name,
            "version": 1,
            "patch_points": patch_points,
        },
        "prompt": prompt,
    }


def _inpaint_wrapper(
    name: str,
    ckpt_name: str,
    clip_skip: int,
    *,
    loras: list[dict[str, Any]] | None = None,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Masked img2img inpaint graph using latent noise mask (more reliable than VAEEncodeForInpaint).
    sampler = sampler or {}
    prompt: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "init.png"}},
        "3": {"class_type": "LoadImageMask", "inputs": {"image": "mask.png", "channel": "red"}},
        "4": {"class_type": "ThresholdMask", "inputs": {"mask": ["3", 0], "value": 0.2}},
        "5": {"class_type": "GrowMask", "inputs": {"mask": ["4", 0], "expand": 4, "tapered_corners": True}},
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["6", 0], "mask": ["5", 0]}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": "PROMPT", "clip": ["14", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["14", 0]}},
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 0,
                "steps": int(sampler.get("steps", 20)),
                "cfg": float(sampler.get("cfg", 7.0)),
                "sampler_name": str(sampler.get("sampler_name", "euler")),
                "scheduler": str(sampler.get("scheduler", "normal")),
                "denoise": float(sampler.get("denoise", 0.5)),
                "model": ["1", 0],
                "positive": ["8", 0],
                "negative": ["9", 0],
                "latent_image": ["7", 0],
            },
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": ["2", 0],
                "source": ["11", 0],
                "x": 0,
                "y": 0,
                "resize_source": False,
                "mask": ["5", 0],
            },
        },
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": "persona_stack/template/repaired", "images": ["12", 0]}},
        "14": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": clip_skip}},
    }
    patch_points = {
        "positive_prompt": {"node_id": "8", "input_key": "text"},
        "negative_prompt": {"node_id": "9", "input_key": "text"},
        "clip_skip": {"node_id": "14", "input_key": "stop_at_clip_layer"},
        "seed": {"node_id": "10", "input_key": "seed"},
        "steps": {"node_id": "10", "input_key": "steps"},
        "cfg": {"node_id": "10", "input_key": "cfg"},
        "sampler_name": {"node_id": "10", "input_key": "sampler_name"},
        "scheduler": {"node_id": "10", "input_key": "scheduler"},
        "denoise": {"node_id": "10", "input_key": "denoise"},
        "filename_prefix": {"node_id": "13", "input_key": "filename_prefix"},
        "init_image": {"node_id": "2", "input_key": "image"},
        "mask_image": {"node_id": "3", "input_key": "image"},
        "mask_channel": {"node_id": "3", "input_key": "channel"},
        "mask_threshold": {"node_id": "4", "input_key": "value"},
        "mask_grow": {"node_id": "5", "input_key": "expand"},
    }
    patch_points.update(
        _apply_loras(
            prompt,
            loras,
            model_targets=[("10", "model")],
            clip_targets=[("14", "clip")],
        )
    )
    return {
        "persona_stack": {
            "name": name,
            "version": 1,
            "patch_points": patch_points,
        },
        "prompt": prompt,
    }


def _inpaint_model_wrapper(
    name: str,
    ckpt_name: str,
    clip_skip: int,
    *,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sampler = sampler or {}
    prompt: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "init.png"}},
        "3": {"class_type": "LoadImageMask", "inputs": {"image": "mask.png", "channel": "red"}},
        "4": {"class_type": "ThresholdMask", "inputs": {"mask": ["3", 0], "value": 0.2}},
        "5": {"class_type": "GrowMask", "inputs": {"mask": ["4", 0], "expand": 4, "tapered_corners": True}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "PROMPT", "clip": ["10", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["10", 0]}},
        "8": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive": ["6", 0],
                "negative": ["7", 0],
                "pixels": ["2", 0],
                "vae": ["1", 2],
                "mask": ["5", 0],
                "noise_mask": True,
            },
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 0,
                "steps": int(sampler.get("steps", 25)),
                "cfg": float(sampler.get("cfg", 5.5)),
                "sampler_name": str(sampler.get("sampler_name", "dpmpp_sde")),
                "scheduler": str(sampler.get("scheduler", "karras")),
        "denoise": float(sampler.get("denoise", 0.8)),
        "model": ["1", 0],
        "positive": ["8", 0],
        "negative": ["8", 1],
        "latent_image": ["8", 2],
            },
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "12": {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": ["2", 0],
                "source": ["11", 0],
                "x": 0,
                "y": 0,
                "resize_source": False,
                "mask": ["5", 0],
            },
        },
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": "persona_stack/template/repaired", "images": ["12", 0]}},
        "10": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": clip_skip}},
    }
    patch_points = {
        "positive_prompt": {"node_id": "6", "input_key": "text"},
        "negative_prompt": {"node_id": "7", "input_key": "text"},
        "clip_skip": {"node_id": "10", "input_key": "stop_at_clip_layer"},
        "seed": {"node_id": "9", "input_key": "seed"},
        "steps": {"node_id": "9", "input_key": "steps"},
        "cfg": {"node_id": "9", "input_key": "cfg"},
        "sampler_name": {"node_id": "9", "input_key": "sampler_name"},
        "scheduler": {"node_id": "9", "input_key": "scheduler"},
        "denoise": {"node_id": "9", "input_key": "denoise"},
        "filename_prefix": {"node_id": "13", "input_key": "filename_prefix"},
        "init_image": {"node_id": "2", "input_key": "image"},
        "mask_image": {"node_id": "3", "input_key": "image"},
        "mask_channel": {"node_id": "3", "input_key": "channel"},
        "mask_threshold": {"node_id": "4", "input_key": "value"},
        "mask_grow": {"node_id": "5", "input_key": "expand"},
    }
    return {
        "persona_stack": {
            "name": name,
            "version": 1,
            "patch_points": patch_points,
        },
        "prompt": prompt,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate default ComfyUI API workflow templates.")
    parser.add_argument("--out-dir", default="workflows/cli")
    parser.add_argument("--anchor-count", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    anchor_count = args.anchor_count

    core_variants = [
        {
            "suffix": "",
            "ckpt": "sd_xl_base_1.0.safetensors",
            "clip_skip": -1,
        },
    ]

    # Core variants are always written to the tracked out_dir.
    for variant in core_variants:
        suffix = variant["suffix"]
        ckpt = variant["ckpt"]
        clip_skip = variant["clip_skip"]
        sampler = variant.get("sampler")
        loras = variant.get("loras")
        name_suffix = suffix.lstrip("_") or "sdxl"
        _write(
            out_dir / f"txt2img{suffix}.json",
            _txt2img_wrapper(f"txt2img_{name_suffix}", ckpt, clip_skip, loras=loras, sampler=sampler),
        )
        _write(
            out_dir / f"img2img{suffix}.json",
            _img2img_wrapper(f"img2img_{name_suffix}", ckpt, clip_skip, loras=loras, sampler=sampler),
        )
        _write(
            out_dir / f"img2img_identity{suffix}.json",
            _img2img_identity_wrapper(
                f"img2img_identity_{name_suffix}",
                ckpt,
                anchor_count,
                clip_skip,
                loras=loras,
                sampler=sampler,
            ),
        )
        _write(
            out_dir / f"character{suffix}.json",
            _character_wrapper(f"character_{name_suffix}", ckpt, anchor_count, clip_skip, loras=loras, sampler=sampler),
        )
        _write(
            out_dir / f"character_pose{suffix}.json",
            _character_pose_wrapper(
                f"character_pose_{name_suffix}",
                ckpt,
                anchor_count,
                clip_skip,
                loras=loras,
                sampler=sampler,
            ),
        )
        _write(
            out_dir / f"style{suffix}.json",
            _style_wrapper(f"style_{name_suffix}", ckpt, clip_skip, loras=loras, sampler=sampler),
        )
        _write(
            out_dir / f"scene{suffix}.json",
            _scene_wrapper(f"scene_{name_suffix}", ckpt, anchor_count, clip_skip, loras=loras, sampler=sampler),
        )
        _write(
            out_dir / f"scene_pose{suffix}.json",
            _scene_pose_wrapper(
                f"scene_pose_{name_suffix}",
                ckpt,
                anchor_count,
                clip_skip,
                loras=loras,
                sampler=sampler,
            ),
        )
        _write(
            out_dir / f"inpaint{suffix}.json",
            _inpaint_wrapper(f"inpaint_{name_suffix}", ckpt, clip_skip, loras=loras, sampler=sampler),
        )

    # Dedicated inpaint model (SDXL inpainting checkpoint).
    _write(
        out_dir / "inpaint_sdxl_inpaint.json",
        _inpaint_model_wrapper(
            "inpaint_sdxl_inpaint",
            "sd_xl_base_1.0_inpainting.safetensors",
            -1,
            sampler={"steps": 25, "cfg": 5.5, "sampler_name": "dpmpp_sde", "scheduler": "karras", "denoise": 0.6},
        ),
    )
    print(f"Wrote workflow templates to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
