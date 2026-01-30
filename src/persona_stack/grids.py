from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class TileEntry:
    image_path: Path
    patch_values: dict[str, object]
    run_id: str
    workflow_path: str | None


def _strip_ext(value: str) -> str:
    name = Path(value).name
    stem = Path(name).stem
    return stem or name


def _sanitize_model_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return _strip_ext(value)


def _model_label(tile: TileEntry, workflow_models: dict[str, str]) -> str:
    patch_values = tile.patch_values
    keys = (
        "model",
        "checkpoint",
        "checkpoint_name",
        "model_name",
        "base_model",
    )
    for key in keys:
        if key in patch_values:
            candidate = _sanitize_model_name(patch_values.get(key))
            if candidate:
                return candidate
    if tile.workflow_path:
        cached = workflow_models.get(tile.workflow_path)
        if cached:
            return cached
    return "n/a"


def _abbr(key: str) -> str:
    mapping = {
        "sampler_name": "sam",
        "scheduler": "sch",
        "steps": "stp",
        "cfg": "cfg",
        "denoise": "den",
        "model": "mod",
        "run_id": "run",
    }
    return mapping.get(key, key[:3].lower())


def _format_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4g}".rstrip("0").rstrip(".")
    return str(value)


def _sort_key(value: object) -> tuple[int, object]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    try:
        return (1, float(str(value)))
    except (TypeError, ValueError):
        return (2, str(value))


def load_tiles(runs_root: Path, limit: int | None = None) -> list[TileEntry]:
    runs_root = runs_root.expanduser()
    meta_paths = sorted(runs_root.rglob("meta/raw_*.meta.json"))
    tiles: list[TileEntry] = []
    for meta_path in meta_paths:
        if limit is not None and len(tiles) >= limit:
            break
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        patch_values = meta.get("patch_values") or {}
        run_id = str(meta.get("run_id") or meta_path.parent.parent.name)
        workflow_path = meta.get("workflow_path")
        filename = (meta.get("comfyui_image") or {}).get("filename") or f"{meta_path.stem}.png"
        run_dir = meta_path.parent.parent
        image_path = run_dir / "raw" / filename
        if not image_path.exists():
            continue
        tiles.append(TileEntry(image_path=image_path, patch_values=patch_values, run_id=run_id, workflow_path=workflow_path))
    return tiles


def _unique_labels(tiles: Iterable[TileEntry], key: str) -> list[str]:
    seen: dict[str, tuple[int, object]] = {}
    for tile in tiles:
        value = tile.patch_values.get(key)
        label = _format_value(value)
        sort_key = _sort_key(value)
        if label not in seen:
            seen[label] = sort_key
    return [label for label, _ in sorted(seen.items(), key=lambda item: item[1])]


def _text_size(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> tuple[int, int]:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_bold_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> None:
    x, y = position
    draw.text((x, y), text, fill=fill, font=font)
    # small offset for a thicker look
    draw.text((x + 1, y + 1), text, fill=fill, font=font)


def _resolve_workflow_path(path_str: str) -> Path | None:
    path = Path(path_str)
    if path.exists():
        return path
    alt = Path.cwd() / path
    if alt.exists():
        return alt
    return None


def _extract_workflow_model(workflow_path: Path) -> str | None:
    try:
        data = json.loads(workflow_path.read_text())
    except Exception:
        return None
    prompt = data.get("prompt")
    if not isinstance(prompt, dict):
        return None
    suffixes = (".safetensors", ".ckpt", ".pt", ".bin")
    preferred_keys = {
        "unet_name",
        "ckpt_name",
        "checkpoint",
        "checkpoint_name",
        "model_name",
        "base_model",
    }
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, value in inputs.items():
            if isinstance(value, str):
                if key in preferred_keys or value.lower().endswith(suffixes):
                    return _sanitize_model_name(value)
            # skip non-str (likely node refs)
    return None


def _build_workflow_model_cache(tiles: Sequence[TileEntry]) -> dict[str, str]:
    cache: dict[str, str] = {}
    for tile in tiles:
        path_str = tile.workflow_path
        if not path_str or path_str in cache:
            continue
        resolved = _resolve_workflow_path(path_str)
        if not resolved:
            continue
        model_name = _extract_workflow_model(resolved)
        if model_name:
            cache[path_str] = model_name
    return cache


def _make_footer_text(tile: TileEntry, footer_keys: Sequence[str]) -> str:
    parts: list[str] = []
    for key in footer_keys:
        if key == "run_id":
            parts.append(f"{_abbr('run_id')}={tile.run_id}")
            continue
        value = tile.patch_values.get(key)
        if value is None:
            continue
        parts.append(f"{_abbr(key)}={_format_value(value)}")
    return " | ".join(parts) if parts else ""


def _render_tile(
    tile: TileEntry | None,
    tile_size: int,
    header_height: int,
    footer_height: int,
    footer_keys: Sequence[str],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    workflow_models: dict[str, str],
) -> Image.Image:
    width = tile_size
    height = header_height + tile_size + footer_height
    canvas = Image.new("RGB", (width, height), "#000000")
    draw = ImageDraw.Draw(canvas)

    header_text = "missing"
    if tile is not None:
        cfg_val = _format_value(tile.patch_values.get("cfg"))
        den_val = _format_value(tile.patch_values.get("denoise"))
        header_text = " | ".join(
            [
                f"{_abbr('model')}={_model_label(tile, workflow_models)}",
                f"{_abbr('cfg')}={cfg_val}",
                f"{_abbr('denoise')}={den_val}",
            ]
        )

    # header strip
    if header_height > 0:
        text_w, text_h = _text_size(font, header_text)
        draw.text(
            (8, (header_height - text_h) / 2),
            header_text,
            fill="#ffffff",
            font=font,
        )

    # image area
    if tile is None:
        draw.rectangle((0, header_height, width, header_height + tile_size), fill="#111111")
        missing_text = "missing"
        text_w, text_h = _text_size(font, missing_text)
        draw.text(
            ((width - text_w) / 2, header_height + (tile_size - text_h) / 2),
            missing_text,
            fill="#ffffff",
            font=font,
        )
    else:
        try:
            with Image.open(tile.image_path) as img:
                img = img.convert("RGB")
                img.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
                offset = ((tile_size - img.width) // 2, header_height + (tile_size - img.height) // 2)
                canvas.paste(img, offset)
        except Exception:
            draw.rectangle((0, header_height, width, header_height + tile_size), fill="#222222")
            draw.text((8, header_height + 8), "load error", fill="#ffffff", font=font)

    footer_top = header_height + tile_size
    if footer_height > 0:
        footer_text = _make_footer_text(tile, footer_keys) if tile else "missing"
        draw.rectangle((0, footer_top, width, height), fill="#000000")
        if footer_text:
            draw.text((8, footer_top + 4), footer_text, fill="#ffffff", font=font)
    return canvas


def build_grid_image(
    tiles: Sequence[TileEntry],
    axis_x: str,
    axis_y: str,
    footer_keys: Sequence[str],
    output_path: Path,
    tile_size: int = 384,
    header_height: int = 24,
    footer_height: int = 40,
    max_x: int | None = None,
    max_y: int | None = None,
) -> Path | None:
    if not tiles:
        return None
    font = ImageFont.load_default()
    workflow_models = _build_workflow_model_cache(tiles)
    x_labels = _unique_labels(tiles, axis_x)
    y_labels = _unique_labels(tiles, axis_y)

    if max_x is not None and len(x_labels) > max_x:
        x_labels = x_labels[:max_x]
    if max_y is not None and len(y_labels) > max_y:
        y_labels = y_labels[:max_y]
    if not x_labels or not y_labels:
        return None

    tile_map: dict[tuple[str, str], list[TileEntry]] = {}
    x_repeat: dict[str, int] = {}
    for tile in sorted(tiles, key=lambda t: (t.run_id, t.image_path.name)):
        x_val = _format_value(tile.patch_values.get(axis_x))
        y_val = _format_value(tile.patch_values.get(axis_y))
        if x_val not in x_labels or y_val not in y_labels:
            continue
        key = (x_val, y_val)
        lst = tile_map.setdefault(key, [])
        lst.append(tile)
        x_repeat[x_val] = max(x_repeat.get(x_val, 1), len(lst))

    expanded_x: list[tuple[str, int, str]] = []
    for x_label in x_labels:
        repeat = x_repeat.get(x_label, 1)
        for idx in range(repeat):
            display = x_label if idx == 0 else f"{x_label}#{idx+1}"
            expanded_x.append((x_label, idx, display))

    label_padding = 8
    row_label_width = max((_text_size(font, y)[0] for y in y_labels), default=0) + (label_padding * 2)
    col_label_height = max((_text_size(font, x[2])[1] for x in expanded_x), default=0) + (label_padding * 2)

    cell_width = tile_size
    cell_height = header_height + tile_size + footer_height
    grid_width = row_label_width + (cell_width * len(expanded_x))
    grid_height = col_label_height + (cell_height * len(y_labels))

    canvas = Image.new("RGB", (grid_width, grid_height), "#000000")
    draw = ImageDraw.Draw(canvas)

    # corner axis labels in the 0,0 area (row/col label intersection)
    corner_lines = (f"x={axis_x}", f"y={axis_y}")
    corner_y = 4
    for line in corner_lines:
        _draw_bold_text(draw, (4, corner_y), line, font=font, fill="#ffffff")
        corner_y += _text_size(font, line)[1] + 2

    for idx, (x_label, _rep_idx, display) in enumerate(expanded_x):
        text_w, text_h = _text_size(font, display)
        x = row_label_width + idx * cell_width + (cell_width - text_w) / 2
        y = (col_label_height - text_h) / 2
        _draw_bold_text(draw, (x, y), display, font=font, fill="#ffffff")

    for idx, label in enumerate(y_labels):
        text_w, text_h = _text_size(font, label)
        x = row_label_width - text_w - label_padding
        y = col_label_height + idx * cell_height + (cell_height - text_h) / 2
        _draw_bold_text(draw, (x, y), label, font=font, fill="#ffffff")

    for row_idx, y_label in enumerate(y_labels):
        for col_idx, (x_label, rep_idx, _display) in enumerate(expanded_x):
            tile_list = tile_map.get((x_label, y_label), [])
            tile = tile_list[rep_idx] if rep_idx < len(tile_list) else None
            tile_img = _render_tile(tile, tile_size, header_height, footer_height, footer_keys, font, workflow_models)
            paste_x = row_label_width + col_idx * cell_width
            paste_y = col_label_height + row_idx * cell_height
            canvas.paste(tile_img, (paste_x, paste_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def build_default_grids(
    runs_root: Path,
    out_dir: Path,
    tile_size: int = 384,
    header_height: int = 24,
    footer_height: int = 40,
    max_images: int | None = None,
    max_axis_values: int | None = None,
) -> list[Path]:
    tiles = load_tiles(runs_root, limit=max_images)
    if not tiles:
        return []
    outputs: list[Path] = []

    cfg_denoise_path = out_dir / "grid_cfg_denoise.png"
    cfg_grid = build_grid_image(
        tiles=tiles,
        axis_x="denoise",
        axis_y="cfg",
        footer_keys=("run_id", "sampler_name", "scheduler", "steps"),
        output_path=cfg_denoise_path,
        tile_size=tile_size,
        header_height=header_height,
        footer_height=footer_height,
        max_x=max_axis_values,
        max_y=max_axis_values,
    )
    if cfg_grid:
        outputs.append(cfg_grid)

    sampler_sched_path = out_dir / "grid_sampler_scheduler.png"
    sampler_grid = build_grid_image(
        tiles=tiles,
        axis_x="scheduler",
        axis_y="sampler_name",
        footer_keys=("run_id", "cfg", "denoise", "steps"),
        output_path=sampler_sched_path,
        tile_size=tile_size,
        header_height=header_height,
        footer_height=footer_height,
        max_x=max_axis_values,
        max_y=max_axis_values,
    )
    if sampler_grid:
        outputs.append(sampler_grid)

    return outputs
