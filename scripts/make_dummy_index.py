"""Generate a fake scene_index.json AND matching placeholder images.

Run this FIRST, before anyone has downloaded a model or taken a photo:

    python scripts/make_dummy_index.py
    python -m src.cli "Where is my bottle?"

Why this exists: on a 5-day project the worst failure mode is four people
sitting idle while one person fights a model download. With this, retrieval,
the agent, the UI and the tests can all be built and demoed on day 1. When the
real photos arrive, delete data/scene_index.json and run the real indexer -
nothing else changes, because the schema is identical.

The fake scenes contain deliberately ambiguous sets (three black bottles across
three rooms, two black backpacks) so the clarification logic actually fires.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config          # noqa: E402
from src.spatial import compute_relations   # noqa: E402

W, H = 1200, 800

SWATCH = {
    "black": (40, 40, 44), "white": (240, 240, 238), "gray": (150, 150, 152),
    "silver": (198, 200, 205), "red": (200, 60, 55), "blue": (60, 110, 200),
    "green": (70, 160, 90), "yellow": (230, 195, 60), "orange": (230, 140, 55),
    "purple": (140, 90, 190), "pink": (225, 130, 175), "brown": (130, 95, 65),
    "beige": (215, 200, 175), "gold": (200, 170, 80), "transparent": (205, 225, 235),
    "multicolor": (120, 170, 170),
}

# scene_id -> (location, caption, [objects])
# bbox is in ORIGINAL pixels, exactly as the real indexer would write it.
SCENES = {
    "office_01": ("office", "A desk with an open laptop, two bottles and a mug.", [
        ("laptop", "silver", "metal", "large", [430, 300, 780, 560], 0.95),
        ("bottle", "black", "metal", "medium", [800, 300, 870, 520], 0.9),
        ("bottle", "blue", "plastic", "medium", [330, 330, 395, 530], 0.86),
        ("mug", "white", "glass", "small", [900, 440, 970, 520], 0.8),
        ("charger", "black", "plastic", "small", [200, 620, 280, 680], 0.7),
    ]),
    "meeting_02": ("meeting", "A meeting table with a backpack and a black bottle.", [
        ("backpack", "black", "fabric", "large", [180, 260, 470, 620], 0.92),
        ("bottle", "black", "metal", "medium", [560, 340, 630, 550], 0.88),
        ("notebook", "black", "paper", "medium", [700, 470, 950, 570], 0.75),
        ("pen", "blue", "plastic", "small", [740, 430, 830, 455], 0.6),
    ]),
    "classroom_01": ("classroom", "Student desks with bottles and a blue backpack.", [
        ("bottle", "black", "metal", "medium", [250, 320, 320, 520], 0.9),
        ("bottle", "black", "plastic", "small", [880, 380, 935, 520], 0.83),
        ("backpack", "blue", "fabric", "large", [480, 300, 760, 640], 0.93),
        ("book", "red", "paper", "medium", [330, 560, 560, 660], 0.78),
    ]),
    "lounge_01": ("lounge", "A sofa area with a backpack, an umbrella and headphones.", [
        ("backpack", "black", "fabric", "large", [200, 280, 500, 620], 0.9),
        ("umbrella", "red", "fabric", "medium", [820, 250, 890, 620], 0.85),
        ("headphones", "white", "plastic", "medium", [560, 480, 700, 590], 0.8),
        ("phone", "black", "glass", "small", [730, 520, 790, 600], 0.72),
    ]),
}


def draw_scene(path, objects):
    image = Image.new("RGB", (W, H), (232, 230, 226))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, int(H * 0.72), W, H], fill=(196, 176, 152))  # "table"
    for obj in objects:
        kind, color, _, _, bbox, _ = obj
        fill = SWATCH.get(color, (120, 120, 120))
        draw.rectangle(bbox, fill=fill, outline=(25, 25, 25), width=3)
        draw.text((bbox[0] + 8, bbox[1] + 8), kind,
                  fill=(255, 255, 255) if sum(fill) < 400 else (20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=92)


def main(config_path=None):
    cfg = load_config(config_path)
    images_root = Path(cfg["paths"]["images"])
    scenes = []

    for scene_id, (location, caption, raw_objects) in SCENES.items():
        # same layout the professor's slide shows: data/images/office/office_01.jpg
        image_path = images_root / location / f"{scene_id}.jpg"
        draw_scene(image_path, raw_objects)

        objects = []
        for position, (kind, color, material, size, bbox, conf) in enumerate(raw_objects):
            objects.append({
                "id": f"{scene_id}_o{position}",
                "type": kind,
                "attributes": {"color": color, "material": material, "size": size},
                "bbox": [float(v) for v in bbox],
                "confidence": conf,
            })

        scenes.append({
            "scene_id": scene_id,
            "location": location,
            # as_posix(), NOT str(): a Windows machine would otherwise write
            # "data\\images\\office\\x.jpg" into the shared index, which no Mac
            # or Linux teammate can then open. src/indexer.py does the same.
            "image_path": image_path.relative_to(
                Path(cfg["paths"]["images"]).parent.parent).as_posix(),
            "width": W, "height": H,
            "caption": caption,
            "objects": objects,
            "relations": compute_relations(objects, (W, H)),
        })

    payload = {
        "version": 1,
        "model": "DUMMY - replace by running: python -m src.indexer",
        "backend": "dummy",
        "coord_mode": "norm1000",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenes": scenes,
    }

    out = Path(cfg["paths"]["index"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {out}")
    print(f"  {len(scenes)} scenes, {sum(len(s['objects']) for s in scenes)} objects")
    print(f"  placeholder images under {images_root}")
    print("\nTry:  python -m src.cli \"Where is my bottle?\"")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None,
                        help="use a different config - tests point this at a "
                             "temp directory so they never touch real data")
    main(parser.parse_args().config)
