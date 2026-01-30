from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageColor


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def list_image_files(paths: Iterable[str | Path]) -> list[str]:
    files: list[str] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
                    files.append(str(child))
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(str(path))
    return files


def parse_rgb(value: str) -> tuple[int, int, int]:
    value = (value or "").strip()
    if not value:
        return (0, 0, 0)
    if "," in value:
        parts = [p.strip() for p in value.split(",")]
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return tuple(int(p) for p in parts)  # type: ignore[return-value]
    try:
        return ImageColor.getrgb(value)
    except ValueError:
        return (0, 0, 0)


def pad_image_to_square(
    src_path: str | Path,
    dest_path: str | Path,
    *,
    pad_color: tuple[int, int, int],
    max_side: int | None = None,
) -> str:
    image = Image.open(src_path).convert("RGB")
    width, height = image.size
    longest = max(width, height)

    if max_side and longest > max_side:
        scale = max_side / float(longest)
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
        image = image.resize((width, height), Image.LANCZOS)
        longest = max(width, height)

    if width == height and not max_side:
        return str(src_path)

    canvas = Image.new("RGB", (longest, longest), color=pad_color)
    offset = ((longest - width) // 2, (longest - height) // 2)
    canvas.paste(image, offset)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest_path)
    return str(dest_path)


def write_placeholder_image(
    dest_path: str | Path,
    *,
    size: int = 64,
    color: tuple[int, int, int] = (0, 0, 0),
) -> str:
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color=color).save(dest_path)
    return str(dest_path)


def env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}
