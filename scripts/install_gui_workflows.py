from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Install ComfyUI GUI workflows into ComfyUI's user workflows folder.")
    parser.add_argument("--src-dir", default="workflows/gui", help="Directory containing ComfyUI GUI workflow JSON files.")
    parser.add_argument(
        "--extra-src-dir",
        default="workflows/imported",
        help="Optional directory with additional GUI workflow JSON files to install.",
    )
    parser.add_argument("--dest-dir", default="vendor/ComfyUI/user/default/workflows", help="ComfyUI user workflows folder.")
    parser.add_argument("--prefix", default="persona_stack_", help="Filename prefix when installing into ComfyUI.")
    args = parser.parse_args()

    src_dirs = [Path(args.src_dir)]
    extra_dir = Path(args.extra_src_dir) if args.extra_src_dir else None
    if extra_dir is not None and extra_dir.exists():
        src_dirs.append(extra_dir)

    if not src_dirs[0].exists():
        raise SystemExit(f"Missing source directory: {src_dirs[0]}")

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    installed = 0
    for src_dir in src_dirs:
        for src_path in sorted(src_dir.glob("*.json")):
            dest_path = dest_dir / f"{args.prefix}{src_path.name}"
            shutil.copy2(src_path, dest_path)
            print(f"Installed: {dest_path}")
            installed += 1

    if installed == 0:
        print(f"No workflows found in {src_dir}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
