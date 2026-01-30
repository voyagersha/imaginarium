# anchors_identity

Store identity anchor images for each persona here. Keep identity anchors separate
from style and scene references.

Repo note: real anchors are ignored by git via `.gitignore`. Only this README and
the `_template` placeholders are tracked.

Recommended layout:

```
anchors_identity/
  _template/
    curated/
    candidates/
    sets/
      example/
  <persona_name>/
    curated/       # best identity-consistent anchors (use these in runs)
    candidates/    # raw picks to review
    sets/
      <set_name>/  # optional focused subsets (angles, lighting, etc)
```

CLI usage (picks the first N images in sorted order):

```
uv run persona-stack run character --prompt \"...\" --anchors anchors_identity/<persona_name>/curated --anchor-count 5
```

Anchor tips:
- Larger images (>512px on the short side) with a clear, centered face work best.
- CLI runs pad anchors to square by default (no crop). Disable with `--no-anchor-pad`.

GUI usage (ComfyUI reads from `vendor/ComfyUI/input/`):
- Copy or symlink `anchors_identity/` into `vendor/ComfyUI/input/anchors_identity`.
- In the LoadImage nodes, pick the files you want as anchors.
