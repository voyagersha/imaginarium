from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import (
    REQUIRED_PATCH_KEYS_T2I,
    REQUIRED_PATCH_KEYS_IMG2IMG,
    REQUIRED_PATCH_KEYS_IMG2IMG_IDENTITY,
    REQUIRED_PATCH_KEYS_INPAINT,
    REQUIRED_PATCH_KEYS_CHARACTER,
    REQUIRED_PATCH_KEYS_CHARACTER_POSE,
    REQUIRED_PATCH_KEYS_SCENE,
    REQUIRED_PATCH_KEYS_SCENE_POSE,
    REQUIRED_PATCH_KEYS_STYLE,
    WorkflowKind,
)
from .grids import build_default_grids
from .media import list_image_files
from .runner import run_workflow
from .validate import validate_comfy_running, validate_models, validate_workflow


app = typer.Typer(add_completion=False, help="persona_stack CLI")
run_app = typer.Typer(add_completion=False, help="Run workflows")
app.add_typer(run_app, name="run")

def _clip_stop_value(clip_skip: int) -> int:
    clip_skip = abs(int(clip_skip)) if clip_skip else 1
    clip_skip = max(1, min(clip_skip, 24))
    return -clip_skip


def _wf_path(kind: WorkflowKind) -> str:
    return str(Path("workflows") / "cli" / f"{kind.value}.json")


def _apply_lora_strengths(
    patch_values: dict[str, object],
    dmd2_strength: Optional[float],
    lightning_strength: Optional[float],
    subtle_strength: Optional[float],
) -> None:
    if dmd2_strength is not None:
        value = float(dmd2_strength)
        patch_values["lora_dmd2_strength_model"] = value
        patch_values["lora_dmd2_strength_clip"] = value
    if lightning_strength is not None:
        value = float(lightning_strength)
        patch_values["lora_lightning_strength_model"] = value
        patch_values["lora_lightning_strength_clip"] = value
    if subtle_strength is not None:
        value = float(subtle_strength)
        patch_values["lora_subtle_strength_model"] = value
        patch_values["lora_subtle_strength_clip"] = value


def _apply_style_strength(
    patch_values: dict[str, object],
    style_strength: Optional[float],
) -> None:
    if style_strength is None:
        return
    value = float(style_strength)
    patch_values["lora_style_strength_model"] = value
    patch_values["lora_style_strength_clip"] = value


def _apply_sampler_settings(
    patch_values: dict[str, object],
    sampler_name: Optional[str],
    scheduler: Optional[str],
) -> None:
    if sampler_name:
        patch_values["sampler_name"] = sampler_name
    if scheduler:
        patch_values["scheduler"] = scheduler


def _apply_flux_guidance(
    patch_values: dict[str, object],
    flux_guidance: Optional[float],
) -> None:
    if flux_guidance is None:
        return
    value = float(flux_guidance)
    patch_values["flux_guidance_positive"] = value
    patch_values["flux_guidance_negative"] = value


@app.command()
def version() -> None:
    typer.echo(__version__)


@app.command("grids")
def grids(
    runs_root: Path = typer.Option(Path("runs"), "--runs-root", help="Root directory containing run outputs."),
    out_dir: Path = typer.Option(Path("runs") / "grids", "--out-dir", help="Directory to write grid images."),
    tile_size: int = typer.Option(384, "--tile-size", help="Square tile size (pixels) for each image."),
    header_height: int = typer.Option(24, "--header-height", help="Header height (pixels) for run/model/cfg/den text."),
    footer_height: int = typer.Option(32, "--footer-height", help="Footer height (pixels) for per-image metadata."),
    max_images: int | None = typer.Option(None, "--max-images", help="Optional cap on how many meta files to read."),
    max_axis_values: int | None = typer.Option(
        None,
        "--max-axis-values",
        help="Optional cap per axis to keep grids reasonable when there are many unique values.",
    ),
) -> None:
    outputs = build_default_grids(
        runs_root=runs_root,
        out_dir=out_dir,
        tile_size=tile_size,
        header_height=header_height,
        footer_height=footer_height,
        max_images=max_images,
        max_axis_values=max_axis_values,
    )
    if not outputs:
        typer.echo("No grids created (no tiles found or missing axis values).")
        raise typer.Exit(code=1)
    for path in outputs:
        typer.echo(f"Wrote {path}")


@app.command("bootstrap-check")
def bootstrap_check(comfy_url: str = typer.Option("http://127.0.0.1:8188")) -> None:
    result = validate_comfy_running(comfy_url)
    for line in result.details:
        typer.echo(line)
    raise typer.Exit(code=0 if result.ok else 1)


@app.command()
def validate(
    comfy_url: str = typer.Option("http://127.0.0.1:8188"),
    manifest: str = typer.Option("manifests/models.json"),
    workflows_dir: str = typer.Option("workflows/cli"),
) -> None:
    results = []
    results.append(validate_comfy_running(comfy_url))
    results.append(validate_models(manifest))

    results.append(validate_workflow(str(Path(workflows_dir) / "txt2img.json"), REQUIRED_PATCH_KEYS_T2I))
    results.append(validate_workflow(str(Path(workflows_dir) / "img2img.json"), REQUIRED_PATCH_KEYS_IMG2IMG))
    results.append(validate_workflow(str(Path(workflows_dir) / "img2img_identity.json"), REQUIRED_PATCH_KEYS_IMG2IMG_IDENTITY))
    results.append(validate_workflow(str(Path(workflows_dir) / "character.json"), REQUIRED_PATCH_KEYS_CHARACTER))
    results.append(validate_workflow(str(Path(workflows_dir) / "character_pose.json"), REQUIRED_PATCH_KEYS_CHARACTER_POSE))
    results.append(validate_workflow(str(Path(workflows_dir) / "style.json"), REQUIRED_PATCH_KEYS_STYLE))
    results.append(validate_workflow(str(Path(workflows_dir) / "scene.json"), REQUIRED_PATCH_KEYS_SCENE))
    results.append(validate_workflow(str(Path(workflows_dir) / "scene_pose.json"), REQUIRED_PATCH_KEYS_SCENE_POSE))
    results.append(validate_workflow(str(Path(workflows_dir) / "inpaint.json"), REQUIRED_PATCH_KEYS_INPAINT))

    ok = all(r.ok for r in results)
    for r in results:
        for line in r.details:
            typer.echo(line)
    raise typer.Exit(code=0 if ok else 1)


@run_app.command("character")
def run_character(
    prompt: str = typer.Option(..., "--prompt"),
    negative_prompt: str = typer.Option("", "--negative-prompt"),
    anchors: list[Path] = typer.Option([], "--anchors", help="Anchor image file/dir (repeatable)"),
    anchor_count: int = typer.Option(5, "--anchor-count", help="Number of identity anchors to use (max 5)."),
    anchor_pad: bool = typer.Option(True, "--anchor-pad/--no-anchor-pad", help="Pad anchors to square before upload."),
    anchor_pad_color: str = typer.Option("black", "--anchor-pad-color", help="Padding color (name or R,G,B)."),
    anchor_max_side: Optional[int] = typer.Option(None, "--anchor-max-side", help="Resize anchors so longest side is <= this size."),
    anchor_weight_scale: float = typer.Option(1.0, "--anchor-weight-scale", help="Scale factor for distributed anchor weights."),
    clip_skip: int = typer.Option(1, "--clip-skip", help="CLIP skip."),
    seed: int = typer.Option(0, "--seed"),
    steps: int = typer.Option(20, "--steps"),
    cfg: float = typer.Option(7.0, "--cfg"),
    sampler_name: Optional[str] = typer.Option(None, "--sampler-name", help="Override sampler (default from workflow)."),
    scheduler: Optional[str] = typer.Option(None, "--scheduler", help="Override scheduler (default from workflow)."),
    dmd2_strength: Optional[float] = typer.Option(None, "--dmd2-strength", help="DMD2 LoRA strength (model+clip)."),
    lightning_strength: Optional[float] = typer.Option(None, "--lightning-strength", help="Lightning LoRA strength (model+clip)."),
    subtle_strength: Optional[float] = typer.Option(None, "--subtle-strength", help="Subtle Styles LoRA strength (model+clip)."),
    width: int = typer.Option(1024, "--width"),
    height: int = typer.Option(1024, "--height"),
    batch_size: int = typer.Option(1, "--batch-size"),
    comfy_url: str = typer.Option("http://127.0.0.1:8188"),
    pose_image: Optional[Path] = typer.Option(None, "--pose-image", help="Pose reference image for pose workflows."),
    pose_strength: Optional[float] = typer.Option(None, "--pose-strength", help="ControlNet pose strength."),
    pose_start: Optional[float] = typer.Option(None, "--pose-start", help="ControlNet pose start percent."),
    pose_end: Optional[float] = typer.Option(None, "--pose-end", help="ControlNet pose end percent."),
    workflow: Optional[Path] = typer.Option(None, "--workflow", help="Override workflow JSON path."),
) -> None:
    anchor_count = max(1, min(anchor_count, 5))
    run_id_prefix = "persona_stack"
    patch_values = {
        "positive_prompt": prompt,
        "negative_prompt": negative_prompt,
        "positive_prompt_refiner": prompt,
        "negative_prompt_refiner": negative_prompt,
        "clip_skip": _clip_stop_value(clip_skip),
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "filename_prefix": f"{run_id_prefix}/{{run_id}}/raw",
    }
    _apply_lora_strengths(patch_values, dmd2_strength, lightning_strength, subtle_strength)
    _apply_sampler_settings(patch_values, sampler_name, scheduler)
    if pose_strength is not None:
        patch_values["pose_strength"] = float(pose_strength)
    if pose_start is not None:
        patch_values["pose_start"] = float(pose_start)
    if pose_end is not None:
        patch_values["pose_end"] = float(pose_end)
    if pose_image is not None:
        patch_values["pose_image"] = "pose.png"
    # runner will compute run_id; patch filename_prefix post-hoc
    if workflow:
        wf_path = str(workflow)
    else:
        wf_kind = WorkflowKind.character_pose if pose_image else WorkflowKind.character
        wf_path = _wf_path(wf_kind)
    res = run_workflow(
        base_url=comfy_url,
        workflow_path=wf_path,
        out_root=str(Path("runs") / "single"),
        patch_values=patch_values,
        identity_refs=[str(p) for p in anchors],
        identity_limit=anchor_count,
        pose_image=str(pose_image) if pose_image else None,
        anchor_pad=anchor_pad,
        anchor_pad_color=anchor_pad_color,
        anchor_max_side=anchor_max_side,
        anchor_weight_scale=anchor_weight_scale,
    )
    typer.echo(json.dumps(res.__dict__, indent=2))

@run_app.command("txt2img")
def run_txt2img(
    prompt: str = typer.Option(..., "--prompt"),
    negative_prompt: str = typer.Option("", "--negative-prompt"),
    clip_skip: int = typer.Option(1, "--clip-skip", help="CLIP skip."),
    seed: int = typer.Option(0, "--seed"),
    steps: int = typer.Option(20, "--steps"),
    cfg: float = typer.Option(7.0, "--cfg"),
    sampler_name: Optional[str] = typer.Option(None, "--sampler-name", help="Override sampler (default from workflow)."),
    scheduler: Optional[str] = typer.Option(None, "--scheduler", help="Override scheduler (default from workflow)."),
    flux_guidance: Optional[float] = typer.Option(None, "--flux-guidance", help="Flux guidance value (applies to positive/negative conditioning)."),
    dmd2_strength: Optional[float] = typer.Option(None, "--dmd2-strength", help="DMD2 LoRA strength (model+clip)."),
    style_strength: Optional[float] = typer.Option(None, "--style-strength", help="Style LoRA strength (model+clip)."),
    cuteness_strength: Optional[float] = typer.Option(None, "--cuteness-strength", help="Alias for --style-strength."),
    lightning_strength: Optional[float] = typer.Option(None, "--lightning-strength", help="Lightning LoRA strength (model+clip)."),
    subtle_strength: Optional[float] = typer.Option(None, "--subtle-strength", help="Subtle Styles LoRA strength (model+clip)."),
    width: int = typer.Option(1024, "--width"),
    height: int = typer.Option(1024, "--height"),
    batch_size: int = typer.Option(1, "--batch-size"),
    comfy_url: str = typer.Option("http://127.0.0.1:8188"),
    workflow: Optional[Path] = typer.Option(None, "--workflow", help="Override workflow JSON path."),
) -> None:
    patch_values = {
        "positive_prompt": prompt,
        "negative_prompt": negative_prompt,
        "positive_prompt_refiner": prompt,
        "negative_prompt_refiner": negative_prompt,
        "clip_skip": _clip_stop_value(clip_skip),
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "filename_prefix": "persona_stack/{run_id}/raw",
    }
    _apply_lora_strengths(patch_values, dmd2_strength, lightning_strength, subtle_strength)
    _apply_sampler_settings(patch_values, sampler_name, scheduler)
    _apply_flux_guidance(patch_values, flux_guidance)
    effective_style_strength = style_strength if style_strength is not None else cuteness_strength
    _apply_style_strength(patch_values, effective_style_strength)
    res = run_workflow(
        base_url=comfy_url,
        workflow_path=str(workflow) if workflow else _wf_path(WorkflowKind.txt2img),
        out_root=str(Path("runs") / "single"),
        patch_values=patch_values,
    )
    typer.echo(json.dumps(res.__dict__, indent=2))


@run_app.command("img2img")
def run_img2img(
    prompt: str = typer.Option(..., "--prompt"),
    image: Path = typer.Option(..., "--image"),
    negative_prompt: str = typer.Option("", "--negative-prompt"),
    clip_skip: int = typer.Option(1, "--clip-skip", help="CLIP skip."),
    seed: int = typer.Option(0, "--seed"),
    steps: int = typer.Option(20, "--steps"),
    cfg: float = typer.Option(7.0, "--cfg"),
    denoise: float = typer.Option(0.5, "--denoise"),
    sampler_name: Optional[str] = typer.Option(None, "--sampler-name", help="Override sampler (default from workflow)."),
    scheduler: Optional[str] = typer.Option(None, "--scheduler", help="Override scheduler (default from workflow)."),
    flux_guidance: Optional[float] = typer.Option(None, "--flux-guidance", help="Flux guidance value (applies to positive/negative conditioning)."),
    dmd2_strength: Optional[float] = typer.Option(None, "--dmd2-strength", help="DMD2 LoRA strength (model+clip)."),
    style_strength: Optional[float] = typer.Option(None, "--style-strength", help="Style LoRA strength (model+clip)."),
    cuteness_strength: Optional[float] = typer.Option(None, "--cuteness-strength", help="Alias for --style-strength."),
    lightning_strength: Optional[float] = typer.Option(None, "--lightning-strength", help="Lightning LoRA strength (model+clip)."),
    subtle_strength: Optional[float] = typer.Option(None, "--subtle-strength", help="Subtle Styles LoRA strength (model+clip)."),
    comfy_url: str = typer.Option("http://127.0.0.1:8188"),
    workflow: Optional[Path] = typer.Option(None, "--workflow", help="Override workflow JSON path."),
) -> None:
    patch_values = {
        "positive_prompt": prompt,
        "negative_prompt": negative_prompt,
        "clip_skip": _clip_stop_value(clip_skip),
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "denoise": denoise,
        "filename_prefix": "persona_stack/{run_id}/raw",
        "init_image": "init.png",
    }
    _apply_lora_strengths(patch_values, dmd2_strength, lightning_strength, subtle_strength)
    _apply_sampler_settings(patch_values, sampler_name, scheduler)
    _apply_flux_guidance(patch_values, flux_guidance)
    effective_style_strength = style_strength if style_strength is not None else cuteness_strength
    _apply_style_strength(patch_values, effective_style_strength)
    res = run_workflow(
        base_url=comfy_url,
        workflow_path=str(workflow) if workflow else _wf_path(WorkflowKind.img2img),
        out_root=str(Path("runs") / "single"),
        patch_values=patch_values,
        init_image=str(image),
    )
    typer.echo(json.dumps(res.__dict__, indent=2))


@run_app.command("img2img-identity")
def run_img2img_identity(
    prompt: str = typer.Option(..., "--prompt"),
    image: Path = typer.Option(..., "--image"),
    anchors: list[Path] = typer.Option([], "--anchors", help="Anchor image file/dir (repeatable)"),
    anchor_count: int = typer.Option(5, "--anchor-count", help="Number of identity anchors to use (max 5)."),
    anchor_pad: bool = typer.Option(True, "--anchor-pad/--no-anchor-pad", help="Pad anchors to square before upload."),
    anchor_pad_color: str = typer.Option("black", "--anchor-pad-color", help="Padding color (name or R,G,B)."),
    anchor_max_side: Optional[int] = typer.Option(None, "--anchor-max-side", help="Resize anchors so longest side is <= this size."),
    anchor_weight_scale: float = typer.Option(1.0, "--anchor-weight-scale", help="Scale factor for distributed anchor weights."),
    negative_prompt: str = typer.Option("", "--negative-prompt"),
    clip_skip: int = typer.Option(1, "--clip-skip", help="CLIP skip."),
    seed: int = typer.Option(0, "--seed"),
    steps: int = typer.Option(20, "--steps"),
    cfg: float = typer.Option(7.0, "--cfg"),
    denoise: float = typer.Option(0.5, "--denoise"),
    sampler_name: Optional[str] = typer.Option(None, "--sampler-name", help="Override sampler (default from workflow)."),
    scheduler: Optional[str] = typer.Option(None, "--scheduler", help="Override scheduler (default from workflow)."),
    flux_guidance: Optional[float] = typer.Option(None, "--flux-guidance", help="Flux guidance value (applies to positive/negative conditioning)."),
    dmd2_strength: Optional[float] = typer.Option(None, "--dmd2-strength", help="DMD2 LoRA strength (model+clip)."),
    lightning_strength: Optional[float] = typer.Option(None, "--lightning-strength", help="Lightning LoRA strength (model+clip)."),
    subtle_strength: Optional[float] = typer.Option(None, "--subtle-strength", help="Subtle Styles LoRA strength (model+clip)."),
    comfy_url: str = typer.Option("http://127.0.0.1:8188"),
    workflow: Optional[Path] = typer.Option(None, "--workflow", help="Override workflow JSON path."),
) -> None:
    anchor_count = max(1, min(anchor_count, 5))
    patch_values = {
        "positive_prompt": prompt,
        "negative_prompt": negative_prompt,
        "clip_skip": _clip_stop_value(clip_skip),
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "denoise": denoise,
        "filename_prefix": "persona_stack/{run_id}/raw",
        "init_image": "init.png",
    }
    _apply_lora_strengths(patch_values, dmd2_strength, lightning_strength, subtle_strength)
    _apply_sampler_settings(patch_values, sampler_name, scheduler)
    _apply_flux_guidance(patch_values, flux_guidance)
    res = run_workflow(
        base_url=comfy_url,
        workflow_path=str(workflow) if workflow else _wf_path(WorkflowKind.img2img_identity),
        out_root=str(Path("runs") / "single"),
        patch_values=patch_values,
        init_image=str(image),
        identity_refs=[str(p) for p in anchors],
        identity_limit=anchor_count,
        anchor_pad=anchor_pad,
        anchor_pad_color=anchor_pad_color,
        anchor_max_side=anchor_max_side,
        anchor_weight_scale=anchor_weight_scale,
    )
    typer.echo(json.dumps(res.__dict__, indent=2))


@run_app.command("style")
def run_style(
    prompt: str = typer.Option(..., "--prompt"),
    negative_prompt: str = typer.Option("", "--negative-prompt"),
    stylepack: Optional[Path] = typer.Option(None, "--stylepack", help="Stylepack dir (reads <dir>/refs)"),
    style_refs: list[Path] = typer.Option([], "--style-ref", help="Style reference image (repeatable)"),
    clip_skip: int = typer.Option(1, "--clip-skip", help="CLIP skip."),
    seed: int = typer.Option(0, "--seed"),
    steps: int = typer.Option(20, "--steps"),
    cfg: float = typer.Option(7.0, "--cfg"),
    sampler_name: Optional[str] = typer.Option(None, "--sampler-name", help="Override sampler (default from workflow)."),
    scheduler: Optional[str] = typer.Option(None, "--scheduler", help="Override scheduler (default from workflow)."),
    dmd2_strength: Optional[float] = typer.Option(None, "--dmd2-strength", help="DMD2 LoRA strength (model+clip)."),
    lightning_strength: Optional[float] = typer.Option(None, "--lightning-strength", help="Lightning LoRA strength (model+clip)."),
    subtle_strength: Optional[float] = typer.Option(None, "--subtle-strength", help="Subtle Styles LoRA strength (model+clip)."),
    width: int = typer.Option(1024, "--width"),
    height: int = typer.Option(1024, "--height"),
    batch_size: int = typer.Option(1, "--batch-size"),
    comfy_url: str = typer.Option("http://127.0.0.1:8188"),
    workflow: Optional[Path] = typer.Option(None, "--workflow", help="Override workflow JSON path."),
) -> None:
    refs = [str(p) for p in style_refs]
    if stylepack:
        refs.append(str(stylepack / "refs"))
    patch_values = {
        "positive_prompt": prompt,
        "negative_prompt": negative_prompt,
        "clip_skip": _clip_stop_value(clip_skip),
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "filename_prefix": "persona_stack/{run_id}/raw",
    }
    _apply_lora_strengths(patch_values, dmd2_strength, lightning_strength, subtle_strength)
    _apply_sampler_settings(patch_values, sampler_name, scheduler)
    res = run_workflow(
        base_url=comfy_url,
        workflow_path=str(workflow) if workflow else _wf_path(WorkflowKind.style),
        out_root=str(Path("runs") / "single"),
        patch_values=patch_values,
        style_refs=refs,
    )
    typer.echo(json.dumps(res.__dict__, indent=2))


@run_app.command("scene")
def run_scene(
    prompt: str = typer.Option(..., "--prompt"),
    negative_prompt: str = typer.Option("", "--negative-prompt"),
    anchors: list[Path] = typer.Option([], "--anchors", help="Anchor image file/dir (repeatable)"),
    stylepack: Optional[Path] = typer.Option(None, "--stylepack", help="Stylepack dir (reads <dir>/refs)"),
    style_refs: list[Path] = typer.Option([], "--style-ref", help="Style reference image (repeatable)"),
    anchor_count: int = typer.Option(5, "--anchor-count", help="Number of identity anchors to use (max 5)."),
    anchor_pad: bool = typer.Option(True, "--anchor-pad/--no-anchor-pad", help="Pad anchors to square before upload."),
    anchor_pad_color: str = typer.Option("black", "--anchor-pad-color", help="Padding color (name or R,G,B)."),
    anchor_max_side: Optional[int] = typer.Option(None, "--anchor-max-side", help="Resize anchors so longest side is <= this size."),
    anchor_weight_scale: float = typer.Option(1.0, "--anchor-weight-scale", help="Scale factor for distributed anchor weights."),
    clip_skip: int = typer.Option(1, "--clip-skip", help="CLIP skip."),
    seed: int = typer.Option(0, "--seed"),
    steps: int = typer.Option(20, "--steps"),
    cfg: float = typer.Option(7.0, "--cfg"),
    sampler_name: Optional[str] = typer.Option(None, "--sampler-name", help="Override sampler (default from workflow)."),
    scheduler: Optional[str] = typer.Option(None, "--scheduler", help="Override scheduler (default from workflow)."),
    dmd2_strength: Optional[float] = typer.Option(None, "--dmd2-strength", help="DMD2 LoRA strength (model+clip)."),
    lightning_strength: Optional[float] = typer.Option(None, "--lightning-strength", help="Lightning LoRA strength (model+clip)."),
    subtle_strength: Optional[float] = typer.Option(None, "--subtle-strength", help="Subtle Styles LoRA strength (model+clip)."),
    width: int = typer.Option(1024, "--width"),
    height: int = typer.Option(1024, "--height"),
    batch_size: int = typer.Option(1, "--batch-size"),
    comfy_url: str = typer.Option("http://127.0.0.1:8188"),
    pose_image: Optional[Path] = typer.Option(None, "--pose-image", help="Pose reference image for pose workflows."),
    pose_strength: Optional[float] = typer.Option(None, "--pose-strength", help="ControlNet pose strength."),
    pose_start: Optional[float] = typer.Option(None, "--pose-start", help="ControlNet pose start percent."),
    pose_end: Optional[float] = typer.Option(None, "--pose-end", help="ControlNet pose end percent."),
    workflow: Optional[Path] = typer.Option(None, "--workflow", help="Override workflow JSON path."),
) -> None:
    anchor_count = max(1, min(anchor_count, 5))
    style = [str(p) for p in style_refs]
    if stylepack:
        style.append(str(stylepack / "refs"))
    patch_values = {
        "positive_prompt": prompt,
        "negative_prompt": negative_prompt,
        "clip_skip": _clip_stop_value(clip_skip),
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "filename_prefix": "persona_stack/{run_id}/raw",
    }
    _apply_lora_strengths(patch_values, dmd2_strength, lightning_strength, subtle_strength)
    _apply_sampler_settings(patch_values, sampler_name, scheduler)
    if pose_strength is not None:
        patch_values["pose_strength"] = float(pose_strength)
    if pose_start is not None:
        patch_values["pose_start"] = float(pose_start)
    if pose_end is not None:
        patch_values["pose_end"] = float(pose_end)
    if pose_image is not None:
        patch_values["pose_image"] = "pose.png"
    if workflow:
        wf_path = str(workflow)
    else:
        wf_kind = WorkflowKind.scene_pose if pose_image else WorkflowKind.scene
        wf_path = _wf_path(wf_kind)
    res = run_workflow(
        base_url=comfy_url,
        workflow_path=wf_path,
        out_root=str(Path("runs") / "single"),
        patch_values=patch_values,
        identity_refs=[str(p) for p in anchors],
        style_refs=style,
        identity_limit=anchor_count,
        pose_image=str(pose_image) if pose_image else None,
        anchor_pad=anchor_pad,
        anchor_pad_color=anchor_pad_color,
        anchor_max_side=anchor_max_side,
        anchor_weight_scale=anchor_weight_scale,
    )
    typer.echo(json.dumps(res.__dict__, indent=2))


@run_app.command("inpaint")
def run_inpaint(
    prompt: str = typer.Option(..., "--prompt"),
    negative_prompt: str = typer.Option("", "--negative-prompt"),
    image: Path = typer.Option(..., "--image"),
    mask: Path = typer.Option(..., "--mask"),
    clip_skip: int = typer.Option(1, "--clip-skip", help="CLIP skip."),
    seed: int = typer.Option(0, "--seed"),
    steps: int = typer.Option(25, "--steps"),
    cfg: float = typer.Option(5.5, "--cfg"),
    width: int = typer.Option(1024, "--width"),
    height: int = typer.Option(1024, "--height"),
    denoise: float = typer.Option(0.6, "--denoise"),
    sampler_name: Optional[str] = typer.Option(None, "--sampler-name", help="Override sampler (default from workflow)."),
    scheduler: Optional[str] = typer.Option(None, "--scheduler", help="Override scheduler (default from workflow)."),
    mask_channel: str = typer.Option("red", "--mask-channel", help="Mask channel for LoadImageMask."),
    mask_threshold: float = typer.Option(0.2, "--mask-threshold", help="Threshold before grow."),
    mask_grow: int = typer.Option(4, "--mask-grow", help="Grow mask pixels before sampling."),
    dmd2_strength: Optional[float] = typer.Option(None, "--dmd2-strength", help="DMD2 LoRA strength (model+clip)."),
    lightning_strength: Optional[float] = typer.Option(None, "--lightning-strength", help="Lightning LoRA strength (model+clip)."),
    subtle_strength: Optional[float] = typer.Option(None, "--subtle-strength", help="Subtle Styles LoRA strength (model+clip)."),
    comfy_url: str = typer.Option("http://127.0.0.1:8188"),
    workflow: Optional[Path] = typer.Option(None, "--workflow", help="Override workflow JSON path."),
) -> None:
    patch_values = {
        "positive_prompt": prompt,
        "negative_prompt": negative_prompt,
        "clip_skip": _clip_stop_value(clip_skip),
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "batch_size": 1,
        "denoise": denoise,
        "filename_prefix": "persona_stack/{run_id}/repaired",
        "init_image": "init_image.png",
        "mask_image": "mask.png",
        "mask_channel": mask_channel,
        "mask_threshold": mask_threshold,
        "mask_grow": mask_grow,
    }
    _apply_lora_strengths(patch_values, dmd2_strength, lightning_strength, subtle_strength)
    _apply_sampler_settings(patch_values, sampler_name, scheduler)
    default_workflow = Path("workflows") / "cli" / "inpaint_sdxl_inpaint.json"
    res = run_workflow(
        base_url=comfy_url,
        workflow_path=str(workflow) if workflow else str(default_workflow),
        out_root=str(Path("runs") / "single"),
        patch_values=patch_values,
        init_image=str(image),
        mask_image=str(mask),
    )
    typer.echo(json.dumps(res.__dict__, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
