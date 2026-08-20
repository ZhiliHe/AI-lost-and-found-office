"""Draw indexed boxes back onto the photos.

RUN THIS ON DAY 1, before building anything else on top of the index.

If the boxes are not sitting on the objects, your coordinate conversion is
wrong, and every relation ("beside the laptop") computed from those boxes is
also wrong. Fix `vlm.coord_mode` in config.yaml first:

    norm1000     model reports 0..1000 normalised   <- what our prompt asks for
    abs_resized  model reports pixels of the resized image it was shown

    python -m src.visualize                 # every scene
    python -m src.visualize --scene office_01
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .config import load_config
from .retrieval import SceneIndex

PALETTE = [(255, 87, 51), (52, 168, 235), (76, 201, 118), (245, 183, 55),
           (168, 100, 235), (235, 100, 160)]


def render(scene, out_dir, project_root):
    image_path = Path(scene["image_path"])
    if not image_path.is_absolute():
        image_path = project_root / image_path
    if not image_path.exists():
        print(f"  missing image: {image_path}")
        return None

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for position, obj in enumerate(scene.get("objects", [])):
        color = PALETTE[position % len(PALETTE)]
        x1, y1, x2, y2 = obj["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        attributes = obj.get("attributes", {})
        label = f"{attributes.get('color') or ''} {obj.get('type')}".strip()
        draw.rectangle([x1, max(0, y1 - 20), x1 + 9 * len(label) + 8, y1], fill=color)
        draw.text((x1 + 4, max(0, y1 - 18)), label, fill=(255, 255, 255))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene['scene_id']}_boxes.jpg"
    image.save(out_path, quality=90)
    return out_path


def render_result(candidates, out_dir, project_root, focus=0):
    """Draw ONE search result: the found object in red, its neighbours faint.

    This exists so the demo works without Gradio. Installing Gradio pulls in
    forty packages, which is a bad bet on a slow network the day before a
    presentation. Pillow is already a dependency, so this always works.
    """
    if not candidates:
        return None

    target = candidates[focus]
    scene = target["scene"]
    image_path = Path(scene["image_path"])
    if not image_path.is_absolute():
        image_path = project_root / image_path
    if not image_path.exists():
        return None

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(4, image.width // 300)

    # every other object in this photo, faintly
    for obj in scene.get("objects", []):
        if obj["id"] == target["object"]["id"]:
            continue
        draw.rectangle(obj["bbox"], outline=(150, 150, 150), width=max(2, width // 2))

    x1, y1, x2, y2 = target["object"]["bbox"]
    draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 40), width=width * 2)

    attributes = target["object"].get("attributes", {})
    # The number matters: when the agent is asking "which one is yours?" the
    # user is looking at several photos at once and answers with a digit.
    label = (f" {focus + 1}. {attributes.get('color') or ''} "
             f"{target['object'].get('type')} ").upper()
    text_y = max(0, y1 - 34 - width)
    draw.rectangle([x1, text_y, x1 + 16 * len(label), text_y + 34], fill=(255, 60, 40))
    draw.text((x1 + 6, text_y + 10), label.strip(), fill=(255, 255, 255))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"result_{focus + 1}_{scene['scene_id']}.jpg"
    image.save(out_path, quality=92)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Overlay indexed bboxes for verification")
    parser.add_argument("--scene", default=None, help="one scene_id only")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(cfg["paths"]["index"]).parent.parent
    index = SceneIndex.load(cfg["paths"]["index"])

    scenes = index.scenes
    if args.scene:
        scenes = [s for s in scenes if s["scene_id"] == args.scene]
        if not scenes:
            print(f"No scene named {args.scene!r}. Available: "
                  f"{[s['scene_id'] for s in index.scenes]}")
            return

    print(f"coord_mode used at index time: {index.payload.get('coord_mode')}")
    for scene in scenes:
        out = render(scene, cfg["paths"]["debug"], project_root)
        if out:
            print(f"  {scene['scene_id']}: {len(scene['objects'])} boxes -> {out}")

    print("\nOpen those files. Every box should sit on its object.")
    print("If they are shifted or scaled, change vlm.coord_mode in config.yaml and re-index.")


if __name__ == "__main__":
    main()
