"""Offline scene indexer:  data/images/**  ->  data/scene_index.json

WHY OFFLINE: running the VLM at query time means every user question costs
10-30 seconds per image on a laptop, which is unusable in a live demo and makes
the multi-turn conversation non-reproducible. We pay that cost ONCE here, and
every query afterwards is a dictionary lookup.

Usage:
    python -m src.indexer                     # index everything
    python -m src.indexer --only office       # one location
    python -m src.indexer --force             # re-index images already done
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .config import load_config
from .prompts import SCENE_EXTRACTION_PROMPT
from .spatial import compute_relations, deduplicate
from .vlm import VLMError, extract_json, load_backend
from .vocab import normalize_color, normalize_object_type

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def iter_images(images_root, only=None):
    """data/images/<location>/<file>  ->  (location, path), sorted."""
    root = Path(images_root)
    if not root.exists():
        return
    for location_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if only and location_dir.name != only:
            continue
        for image_path in sorted(location_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                yield location_dir.name, image_path


def make_scene_id(location, image_path):
    """office/office_01.jpg -> 'office_01'   (not 'office_office_01')
    office/01.jpg        -> 'office_01'
    Both layouts work, so nobody has to rename their photos."""
    stem = Path(image_path).stem
    return stem if stem.startswith(f"{location}_") else f"{location}_{stem}"


def prepare_image(image_path, max_side, work_dir):
    """Downscale so the VLM sees a sane number of vision tokens.

    Returns (path_given_to_model, resized_wh, original_wh).
    Feeding a 4032px phone photo straight in is the single fastest way to make
    an 8GB machine swap and hang.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        original_wh = img.size
        scale = min(1.0, max_side / max(img.size))
        if scale >= 1.0:
            return str(image_path), original_wh, original_wh
        new_wh = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        resized = img.resize(new_wh, Image.LANCZOS)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / f"resized_{image_path.stem}.jpg"
    resized.save(out_path, quality=90)
    return str(out_path), new_wh, original_wh


def to_original_pixels(bbox, coord_mode, resized_wh, original_wh):
    """Convert whatever the model reported into ORIGINAL image pixels.

    This function is the #1 source of "my boxes are in the wrong place".
    Always sanity check with:  python -m src.visualize --scene <id>
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    ow, oh = original_wh

    if coord_mode == "norm1000":
        sx, sy = ow / 1000.0, oh / 1000.0
    elif coord_mode == "abs_resized":
        rw, rh = resized_wh
        sx, sy = ow / rw, oh / rh
    else:
        raise ValueError(f"unknown coord_mode {coord_mode!r}")

    box = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    # clamp into the image and fix inverted corners
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0.0, min(x1, ow)), max(0.0, min(x2, ow))))
    y1, y2 = sorted((max(0.0, min(y1, oh)), max(0.0, min(y2, oh))))
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def clean_object(raw, scene_id, position, coord_mode, resized_wh, original_wh):
    """Normalise one object dict from the model. Returns None if unusable."""
    bbox = raw.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        bbox = to_original_pixels(bbox, coord_mode, resized_wh, original_wh)
    except (TypeError, ValueError):
        return None
    if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
        return None  # degenerate box

    def clean_str(value):
        if value is None:
            return None
        text = str(value).strip().lower()
        return text or None

    return {
        "id": f"{scene_id}_o{position}",
        "type": normalize_object_type(raw.get("type")) or "unknown",
        "attributes": {
            "color": normalize_color(raw.get("color")),
            "material": clean_str(raw.get("material")),
            "size": clean_str(raw.get("size")),
        },
        "bbox": bbox,
        "confidence": float(raw.get("confidence", 0.5) or 0.5),
    }


def index_one(backend, image_path, location, cfg):
    vlm_cfg = cfg["vlm"]
    scene_id = make_scene_id(location, image_path)
    model_path, resized_wh, original_wh = prepare_image(
        image_path, vlm_cfg.get("max_image_side", 1024), cfg["paths"]["debug"])

    raw_text = backend.describe(model_path, SCENE_EXTRACTION_PROMPT)

    # Always keep the model's raw answer. When recall is bad you need to know
    # whether the model said little, or said a lot and our parser dropped it.
    debug_dir = Path(cfg["paths"]["debug"])
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / f"{scene_id}_raw.txt").write_text(raw_text or "", encoding="utf-8")

    parsed = extract_json(raw_text)

    # The prompt asks the model to count first. If the count and the list
    # disagree, the model ran out of steam - worth seeing.
    if parsed.get("truncated"):
        print("(output was cut off - salvaged what was complete; "
              "raise vlm.max_tokens) ", end="")
    claimed = parsed.get("n_objects")
    listed = len(parsed.get("objects", []))
    if isinstance(claimed, int) and claimed != listed:
        print(f"(model counted {claimed} but listed {listed}) ", end="")

    objects = []
    for position, raw in enumerate(parsed.get("objects", [])):
        obj = clean_object(raw, scene_id, position, vlm_cfg.get("coord_mode", "norm1000"),
                           resized_wh, original_wh)
        if obj:
            objects.append(obj)

    before = len(objects)
    objects = deduplicate(objects)
    if len(objects) < before:
        print(f"(dropped {before - len(objects)} duplicate) ", end="")
    # renumber so ids stay contiguous after dedup
    for position, obj in enumerate(objects):
        obj["id"] = f"{scene_id}_o{position}"

    return {
        "scene_id": scene_id,
        "location": location,
        "image_path": str(Path(image_path).as_posix()),
        "width": original_wh[0],
        "height": original_wh[1],
        "caption": str(parsed.get("caption", "")).strip(),
        "objects": objects,
        "relations": compute_relations(objects, original_wh),
    }


def write_index(index_path, cfg, scenes):
    payload = {
        "version": 1,
        "model": cfg["vlm"].get("model"),
        "backend": cfg["vlm"].get("backend"),
        "coord_mode": cfg["vlm"].get("coord_mode"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenes": scenes,
    }
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    # write to a temp file then move, so an interrupt can never leave a
    # half-written index behind
    temporary = index_path.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    temporary.replace(index_path)
    return payload


def build_index(cfg, only=None, force=False, limit=None):
    index_path = Path(cfg["paths"]["index"])
    existing = {}
    if index_path.exists() and not force:
        with open(index_path, "r", encoding="utf-8") as fh:
            for scene in json.load(fh).get("scenes", []):
                existing[scene["scene_id"]] = scene

    backend = load_backend(cfg["vlm"])
    scenes, failures = [], []
    todo = list(iter_images(cfg["paths"]["images"], only=only))
    if limit:
        todo = todo[:limit]

    if not todo:
        print(f"No images found under {cfg['paths']['images']}/<location>/")
        print("Expected layout:  data/images/office/office_01.jpg")

    for position, (location, image_path) in enumerate(todo, 1):
        scene_id = make_scene_id(location, image_path)
        if scene_id in existing:
            print(f"[{position}/{len(todo)}] skip {scene_id} (already indexed)")
            scenes.append(existing[scene_id])
            continue

        started = time.time()
        print(f"[{position}/{len(todo)}] {scene_id} ...", end=" ", flush=True)
        try:
            scene = index_one(backend, image_path, location, cfg)
        except (VLMError, OSError) as exc:
            print(f"FAILED: {exc}")
            failures.append((scene_id, str(exc)))
            continue
        scenes.append(scene)
        print(f"{len(scene['objects'])} objects, {time.time() - started:.1f}s")
        # Save after EVERY image. Each one costs ~40s; losing 20 of them to a
        # Ctrl+C is not acceptable. Re-running without --force resumes here.
        write_index(index_path, cfg, scenes)

    payload = write_index(index_path, cfg, scenes)

    total_objects = sum(len(s["objects"]) for s in scenes)
    print(f"\nWrote {index_path}")
    print(f"  {len(scenes)} scenes, {total_objects} objects")
    if failures:
        print(f"  {len(failures)} failed: {[f[0] for f in failures]}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Build scene_index.json from photos")
    parser.add_argument("--config", default=None)
    parser.add_argument("--only", default=None, help="index one location only, e.g. office")
    parser.add_argument("--force", action="store_true", help="re-index scenes already present")
    parser.add_argument("--limit", type=int, default=None, help="stop after N images (smoke test)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="stream the model's tokens, so you can see it is alive")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.verbose:
        cfg["vlm"]["verbose"] = True
    build_index(cfg, only=args.only, force=args.force, limit=args.limit)


if __name__ == "__main__":
    main()
