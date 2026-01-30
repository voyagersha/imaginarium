from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WorkflowKind(str, Enum):
    txt2img = "txt2img"
    img2img = "img2img"
    img2img_identity = "img2img_identity"
    character = "character"
    character_pose = "character_pose"
    style = "style"
    scene = "scene"
    scene_pose = "scene_pose"
    inpaint = "inpaint"


@dataclass(frozen=True)
class ComfyUIConfig:
    base_url: str = "http://127.0.0.1:8188"


@dataclass(frozen=True)
class GenerationParams:
    prompt: str
    negative_prompt: str = ""
    clip_skip: int = 1
    seed: int = 0
    steps: int = 20
    cfg: float = 7.0
    sampler_name: str = "euler"
    scheduler: str = "normal"
    width: int = 1024
    height: int = 1024
    batch_size: int = 1
    denoise: float = 1.0


PATCH_KEY_PROMPT = "positive_prompt"
PATCH_KEY_NEGATIVE_PROMPT = "negative_prompt"
PATCH_KEY_CLIP_SKIP = "clip_skip"
PATCH_KEY_SEED = "seed"
PATCH_KEY_STEPS = "steps"
PATCH_KEY_CFG = "cfg"
PATCH_KEY_SAMPLER = "sampler_name"
PATCH_KEY_SCHEDULER = "scheduler"
PATCH_KEY_WIDTH = "width"
PATCH_KEY_HEIGHT = "height"
PATCH_KEY_BATCH = "batch_size"
PATCH_KEY_DENOISE = "denoise"
PATCH_KEY_FILENAME_PREFIX = "filename_prefix"
PATCH_KEY_INIT_IMAGE = "init_image"
PATCH_KEY_MASK_IMAGE = "mask_image"
PATCH_KEY_POSE_IMAGE = "pose_image"
PATCH_KEY_POSE_STRENGTH = "pose_strength"
PATCH_KEY_POSE_START = "pose_start"
PATCH_KEY_POSE_END = "pose_end"
IDENTITY_IMAGE_KEYS = {f"identity_image_{idx}" for idx in range(1, 6)}
IDENTITY_WEIGHT_KEYS = {f"identity_weight_{idx}" for idx in range(1, 6)}


REQUIRED_PATCH_KEYS_T2I = {
    PATCH_KEY_PROMPT,
    PATCH_KEY_NEGATIVE_PROMPT,
    PATCH_KEY_CLIP_SKIP,
    PATCH_KEY_SEED,
    PATCH_KEY_STEPS,
    PATCH_KEY_CFG,
    PATCH_KEY_SAMPLER,
    PATCH_KEY_SCHEDULER,
    PATCH_KEY_WIDTH,
    PATCH_KEY_HEIGHT,
    PATCH_KEY_BATCH,
    PATCH_KEY_FILENAME_PREFIX,
}

REQUIRED_PATCH_KEYS_CHARACTER = REQUIRED_PATCH_KEYS_T2I | IDENTITY_IMAGE_KEYS | IDENTITY_WEIGHT_KEYS
REQUIRED_PATCH_KEYS_STYLE = REQUIRED_PATCH_KEYS_T2I | {"style_image"}
REQUIRED_PATCH_KEYS_SCENE = REQUIRED_PATCH_KEYS_T2I | IDENTITY_IMAGE_KEYS | IDENTITY_WEIGHT_KEYS | {"style_image"}

REQUIRED_PATCH_KEYS_IMG2IMG = {
    PATCH_KEY_PROMPT,
    PATCH_KEY_NEGATIVE_PROMPT,
    PATCH_KEY_CLIP_SKIP,
    PATCH_KEY_SEED,
    PATCH_KEY_STEPS,
    PATCH_KEY_CFG,
    PATCH_KEY_SAMPLER,
    PATCH_KEY_SCHEDULER,
    PATCH_KEY_DENOISE,
    PATCH_KEY_FILENAME_PREFIX,
    PATCH_KEY_INIT_IMAGE,
}

REQUIRED_PATCH_KEYS_IMG2IMG_IDENTITY = REQUIRED_PATCH_KEYS_IMG2IMG | IDENTITY_IMAGE_KEYS | IDENTITY_WEIGHT_KEYS

REQUIRED_PATCH_KEYS_INPAINT = {
    PATCH_KEY_PROMPT,
    PATCH_KEY_NEGATIVE_PROMPT,
    PATCH_KEY_CLIP_SKIP,
    PATCH_KEY_SEED,
    PATCH_KEY_STEPS,
    PATCH_KEY_CFG,
    PATCH_KEY_SAMPLER,
    PATCH_KEY_SCHEDULER,
    PATCH_KEY_DENOISE,
    PATCH_KEY_FILENAME_PREFIX,
    PATCH_KEY_INIT_IMAGE,
    PATCH_KEY_MASK_IMAGE,
}

REQUIRED_PATCH_KEYS_POSE = {
    PATCH_KEY_POSE_IMAGE,
    PATCH_KEY_POSE_STRENGTH,
    PATCH_KEY_POSE_START,
    PATCH_KEY_POSE_END,
}

REQUIRED_PATCH_KEYS_CHARACTER_POSE = REQUIRED_PATCH_KEYS_CHARACTER | REQUIRED_PATCH_KEYS_POSE
REQUIRED_PATCH_KEYS_SCENE_POSE = REQUIRED_PATCH_KEYS_SCENE | REQUIRED_PATCH_KEYS_POSE
