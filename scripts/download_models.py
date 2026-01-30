from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, utils as hf_utils

from persona_stack.hashing import sha256_file


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _string(value: Any) -> str:
    return str(value) if value is not None else ""


def _snapshot_revision(downloaded_path: str) -> str | None:
    parts = Path(downloaded_path).parts
    for i, p in enumerate(parts):
        if p == "snapshots" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model artifacts from Hugging Face (driven by manifests/models.json).")
    parser.add_argument("--manifest", default="manifests/models.json")
    parser.add_argument(
        "--id",
        action="append",
        dest="ids",
        default=[],
        help="Only download manifest entries with this id (repeatable).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN"),
        help="HF token; if omitted, will try the locally stored Hugging Face token.",
    )
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE"))
    parser.add_argument("--update-manifest", action="store_true", help="Fill missing sha256 values in the manifest after download.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue downloading other artifacts if one fails.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise SystemExit(f"Invalid manifest format (expected artifacts list): {manifest_path}")

    token_to_use = args.token or hf_utils.get_token()

    updated = False
    failures: list[str] = []
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise SystemExit(f"Invalid manifest entry: {entry!r}")

        entry_id = _string(entry.get("id"))
        if args.ids and entry_id not in set(args.ids):
            continue

        repo_id = _string(entry.get("repo_id"))
        repo_type = _string(entry.get("repo_type")) or None
        revision = _string(entry.get("revision")) or None
        filename = _string(entry.get("filename"))
        subfolder = _string(entry.get("subfolder")) or None
        dest_dir = Path(_string(entry.get("dest_dir")))
        dest_filename = _string(entry.get("dest_filename") or filename)
        expected_sha = _string(entry.get("sha256")).strip().lower()

        if not repo_id or not filename or not dest_dir or not dest_filename:
            raise SystemExit(f"Missing required keys in manifest entry: {entry}")

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / dest_filename

        print(f"==> {entry_id or filename}")

        if dest_path.exists() and expected_sha:
            actual = sha256_file(str(dest_path))
            if actual == expected_sha:
                print("Already present (sha256 OK); skipping.")
                continue

        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type if repo_type else None,
                filename=filename,
                subfolder=subfolder if subfolder else None,
                revision=revision if revision else None,
                token=token_to_use,
                cache_dir=args.cache_dir,
            )
        except Exception as e:
            msg = f"Download failed for {repo_id}/{filename}: {e}"
            print(msg)
            print("If this is a gated model, ensure you accepted the terms on Hugging Face and set HF_TOKEN.")
            failures.append(msg)
            if args.continue_on_error:
                continue
            return 1

        tmp_path = Path(downloaded)
        shutil.copy2(tmp_path, dest_path)

        actual_sha = sha256_file(str(dest_path))
        if expected_sha and actual_sha != expected_sha:
            raise SystemExit(f"SHA256 mismatch for {dest_path}: expected={expected_sha} actual={actual_sha}")

        if not expected_sha:
            print(f"sha256={actual_sha} (not pinned)")
            if args.update_manifest:
                entry["sha256"] = actual_sha
                updated = True
        else:
            print("sha256 OK")

        resolved = _snapshot_revision(downloaded)
        if resolved and args.update_manifest:
            current_revision = _string(entry.get("revision")).strip()
            if current_revision and current_revision != resolved:
                entry.setdefault("requested_revision", current_revision)
            if current_revision != resolved:
                entry["revision"] = resolved
                updated = True

    if updated:
        _save_manifest(manifest_path, manifest)
        print(f"Updated manifest: {manifest_path}")

    if failures:
        print("\nCompleted with failures:")
        for f in failures:
            print(f"- {f}")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
