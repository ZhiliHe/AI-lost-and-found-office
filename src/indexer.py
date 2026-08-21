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

from .config import PROJECT_ROOT, load_config
from .prompts import SCENE_EXTRACTION_PROMPT
from .spatial import compute_relations, deduplicate, find_repeated_patterns
from .tiling import (crop_tile, is_truncated, make_tiles, merge_passes,
                     relative_size, tile_to_global)
from .vlm import VLMError, extract_json, load_backend
from .vocab import is_portable, normalize_color, normalize_object_type

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}

# The model anchors on the JSON example in SCENE_EXTRACTION_PROMPT (tuning log
# v1 -> v2 documented this): whenever a photo contains a laptop or a charger it
# tends to copy the example's ENTIRE entry, bbox included, so the same
# [470, 300, 660, 430] box appears on unrelated photos. A box that exactly
# matches an example is proof the model did NOT localise - drop it rather than
# index a box that sits on the wrong object. In 0-1000 normalised space.
EXAMPLE_BBOXES = (
    (470, 300, 660, 430),      # the prompt's example laptop
    (265, 335, 320, 375),      # the prompt's example charger
)


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


# Boxes that appear in the prompt's example. A 2B model under pressure will
# copy the example instead of looking at the photo - we caught it returning the
# example laptop and charger, to the pixel, as two of eight "found" objects.
# The example now uses these placeholder coordinates precisely so that a copy is
# recognisable. Anything matching them never came from the image.
PROMPT_EXAMPLE_BOXES = [(111, 222, 333, 444), (555, 666, 777, 888)]
ECHO_TOLERANCE = 2.0


def _is_prompt_echo(obj, resized_wh):
    """Did the model hand us back the example instead of an observation?"""
    x1, y1, x2, y2 = obj["bbox"]
    width, height = resized_wh
    for example in PROMPT_EXAMPLE_BOXES:
        ex1, ey1, ex2, ey2 = [v / 1000.0 for v in example]
        want = (ex1 * width, ey1 * height, ex2 * width, ey2 * height)
        if all(abs(a - b) <= ECHO_TOLERANCE for a, b in zip((x1, y1, x2, y2), want)):
            return True
    return False


def clean_object(raw, scene_id, position, coord_mode, resized_wh, original_wh):
    """Normalise one object dict from the model. Returns None if unusable."""
    bbox = raw.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        raw_box = [round(float(v)) for v in bbox]
        if any(all(abs(raw_box[i] - example[i]) <= 2 for i in range(4))
               for example in EXAMPLE_BBOXES):
            return None  # copied from the prompt example, not a localisation
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


def _prepare(cfg, location, image_path):
    """Resize the photo and decide the scene id. Shared by the single-image
    path and the CUDA batch path."""
    vlm_cfg = cfg["vlm"]
    scene_id = make_scene_id(location, image_path)
    model_path, resized_wh, original_wh = prepare_image(
        image_path, vlm_cfg.get("max_image_side", 1024), cfg["paths"]["debug"])
    return scene_id, model_path, resized_wh, original_wh


def _parse_pass(raw_text, cfg, scene_id, position_offset, resized_wh,
                original_wh, label, debug_dir):
    """Turn ONE model answer into cleaned objects, in ORIGINAL image pixels.

    Split out from scene_from_raw because tiling makes several calls per photo
    and has to parse each of them before any of them becomes a scene.
    """
    vlm_cfg = cfg["vlm"]

    # Always keep the model's raw answer. When recall is bad you need to know
    # whether the model said little, or said a lot and our parser dropped it.
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / f"{scene_id}_{label}_raw.txt").write_text(raw_text or "",
                                                          encoding="utf-8")
    parsed = extract_json(raw_text)

    if parsed.get("truncated"):
        print("(output was cut off - salvaged what was complete; "
              "raise vlm.max_tokens) ", end="")
    claimed = parsed.get("n_objects")
    listed = len(parsed.get("objects", []))
    if isinstance(claimed, int) and claimed != listed:
        print(f"(model counted {claimed} but listed {listed}) ", end="")

    objects = []
    for position, raw in enumerate(parsed.get("objects", [])):
        obj = clean_object(raw, scene_id, position_offset + position,
                           vlm_cfg.get("coord_mode", "norm1000"),
                           resized_wh, original_wh)
        if obj:
            objects.append(obj)

    before = len(objects)
    objects = [o for o in objects if not _is_prompt_echo(o, resized_wh)]
    if len(objects) < before:
        print(f"(dropped {before - len(objects)} copied from the prompt) ", end="")
    return objects, parsed.get("caption")


def _repo_relative(image_path):
    """Store "data/images/office/front.jpg", never "/Users/someone/...".

    The index is committed to git and shared - it IS the contract between
    teammates. An absolute path bakes in one person's home directory, so every
    photo becomes unopenable the moment anyone else clones the repo. Paths are
    written relative to the project root, POSIX-style, on every platform.
    """
    path = Path(image_path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # photo lives outside the repo - nothing better than the absolute path
        return path.as_posix()


def _finalise(objects, caption, location, image_path, scene_id, original_wh):
    """The last steps every path shares: drop what nobody can lose, drop
    duplicates, renumber, and build the scene dict."""
    before = len(objects)
    objects = [o for o in objects if is_portable(o["type"])]
    if len(objects) < before:
        print(f"(dropped {before - len(objects)} fixtures) ", end="")

    texture = set(find_repeated_patterns(objects, original_wh))
    if texture:
        objects = [o for o in objects if o["id"] not in texture]
        print(f"(dropped {len(texture)} as a repeated pattern) ", end="")

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
        "image_path": _repo_relative(image_path),
        "width": original_wh[0],
        "height": original_wh[1],
        "caption": str(caption or "").strip(),
        "objects": objects,
        "relations": compute_relations(objects, original_wh),
    }


def scene_from_raw(raw_text, cfg, location, image_path, scene_id,
                   resized_wh, original_wh):
    """Turn the model's raw answer into a scene dict (the parse pipeline).

    One answer, one scene - which is what the CUDA batch path needs. Tiling
    makes several calls per photo and therefore uses index_one_tiled instead.
    """
    objects, caption = _parse_pass(raw_text, cfg, scene_id, 0, resized_wh,
                                   original_wh, "full", cfg["paths"]["debug"])
    return _finalise(objects, caption, location, image_path, scene_id, original_wh)


def index_one_tiled(backend, image_path, location, cfg):
    """Full photo + overlapping crops. See src/tiling.py for why.

    Costs one model call per tile plus one for the whole frame, so it cannot
    share the batch path - the batch path is one answer per image by design.
    """
    vlm_cfg = cfg["vlm"]
    max_side = vlm_cfg.get("max_image_side", 1024)
    debug_dir = Path(cfg["paths"]["debug"])
    debug_dir.mkdir(parents=True, exist_ok=True)

    scene_id, model_path, resized_wh, original_wh = _prepare(cfg, location, image_path)

    # Pass 1: the whole photo. Reliable for anything big.
    raw_text = backend.describe(model_path, SCENE_EXTRACTION_PROMPT)
    found, caption = _parse_pass(raw_text, cfg, scene_id, 0, resized_wh,
                                 original_wh, "full", debug_dir)
    detections = [(obj, relative_size(obj["bbox"], original_wh)) for obj in found]

    # Passes 2..N: overlapping crops of the ORIGINAL image, so small objects
    # arrive at the model several times larger.
    grid = int(vlm_cfg.get("tiling", 0) or 0)
    tiles = make_tiles(original_wh[0], original_wh[1], grid=grid)
    print(f"\n      + {len(tiles)} tiles ", end="", flush=True)
    for number, tile in enumerate(tiles):
        tile_path = debug_dir / f"tile_{scene_id}_{number}.jpg"
        crop_tile(image_path, tile, tile_path)
        tile_wh = (tile[2] - tile[0], tile[3] - tile[1])
        tile_model_path, tile_resized_wh, _ = prepare_image(
            tile_path, max_side, cfg["paths"]["debug"])

        tile_raw = backend.describe(tile_model_path, SCENE_EXTRACTION_PROMPT)
        tile_objects, _ = _parse_pass(
            tile_raw, cfg, scene_id, 1000 * (number + 1), tile_resized_wh,
            tile_wh, f"tile{number}", debug_dir)

        cut = 0
        for obj in tile_objects:
            size = relative_size(obj["bbox"], tile_wh)
            obj["bbox"] = tile_to_global(obj["bbox"], tile)
            if is_truncated(obj["bbox"], tile, original_wh):
                cut += 1
                continue
            detections.append((obj, size))
        print(f"[{number}:{len(tile_objects) - cut}]", end="", flush=True)
    print(" ", end="")

    before = len(detections)
    objects = merge_passes(detections)
    print(f"(merged {before} detections -> {len(objects)}) ", end="")

    return _finalise(objects, caption, location, image_path, scene_id, original_wh)


def index_one(backend, image_path, location, cfg):
    """Index ONE image through any backend (used when batching is unavailable)."""
    if int(cfg["vlm"].get("tiling", 0) or 0) >= 2:
        return index_one_tiled(backend, image_path, location, cfg)
    scene_id, model_path, resized_wh, original_wh = _prepare(cfg, location, image_path)
    raw_text = backend.describe(model_path, SCENE_EXTRACTION_PROMPT)
    return scene_from_raw(raw_text, cfg, location, image_path, scene_id,
                          resized_wh, original_wh)


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


def build_index(cfg, only=None, force=False, limit=None, backend=None):
    index_path = Path(cfg["paths"]["index"])
    existing = {}
    if index_path.exists() and not force:
        with open(index_path, "r", encoding="utf-8") as fh:
            for scene in json.load(fh).get("scenes", []):
                existing[scene["scene_id"]] = scene

    if backend is None:
        backend = load_backend(cfg["vlm"])
    scenes, failures = [], []
    todo = list(iter_images(cfg["paths"]["images"], only=only))
    if limit:
        todo = todo[:limit]

    if not todo:
        print(f"No images found under {cfg['paths']['images']}/<location>/")
        print("Expected layout:  data/images/office/office_01.jpg")

    # keep the already-indexed scenes (and their progress numbering) intact
    for position, (location, image_path) in enumerate(todo, 1):
        scene_id = make_scene_id(location, image_path)
        if scene_id in existing:
            print(f"[{position}/{len(todo)}] skip {scene_id} (already indexed)")
            scenes.append(existing[scene_id])

    pending = [(location, image_path) for location, image_path in todo
               if make_scene_id(location, image_path) not in existing]

    # CUDA backends batch several images into ONE generate() call - generation
    # is GPU-bound, so 2-4 images per call cost barely more than one. Other
    # backends fall back to one image at a time.
    batch_size = int(cfg["vlm"].get("batch_size", 1))
    use_batch = getattr(backend, "supports_batch", False) and batch_size > 1

    # Tiling makes several calls per photo, so it cannot share the batch path -
    # batching is one answer per image by design. Tiling wins: recall matters
    # more than throughput, and this runs once.
    use_tiling = int(cfg["vlm"].get("tiling", 0) or 0) >= 2
    if use_tiling:
        if use_batch:
            print("Tiling is on, so batching is disabled (vlm.tiling)")
        use_batch = False
        batch_size = 1
    if use_batch:
        print(f"CUDA batching: {batch_size} images per generation call "
              f"(vlm.batch_size)")

    position = len(todo) - len(pending)
    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        prepared = [_prepare(cfg, location, image_path)
                    for location, image_path in chunk]

        # One generate() call for the whole chunk. If it fails (corrupt image,
        # out of memory, ...), fall back to one image at a time so a single
        # bad image cannot take the run down.
        raw_texts = None
        batch_elapsed = None
        if use_batch:
            try:
                batch_started = time.time()
                raw_texts = iter(backend.describe_batch(
                    [(model_path, SCENE_EXTRACTION_PROMPT)
                     for _, model_path, _, _ in prepared]))
                batch_elapsed = time.time() - batch_started
                print(f"(batch of {len(chunk)} imgs, {batch_elapsed:.0f}s) ", end="")
            except Exception as exc:                 # noqa: BLE001
                print(f"(batch of {len(chunk)} failed: {str(exc)[:160]}; "
                      f"retrying one by one) ", end="")
                raw_texts = None

        for (location, image_path), (scene_id, model_path, rw, oh) in zip(chunk, prepared):
            position += 1
            print(f"[{position}/{len(todo)}] {scene_id} ...", end=" ", flush=True)
            started = time.time()
            try:
                if use_tiling:
                    scene = index_one_tiled(backend, image_path, location, cfg)
                else:
                    raw = next(raw_texts) if raw_texts is not None else \
                        backend.describe(model_path, SCENE_EXTRACTION_PROMPT)
                    scene = scene_from_raw(raw, cfg, location, image_path,
                                           scene_id, rw, oh)
            except (VLMError, OSError) as exc:
                print(f"FAILED: {exc}")
                failures.append((scene_id, str(exc)))
                continue
            scenes.append(scene)
            # On the batch path the GPU time is shared across the chunk, so
            # report the amortised per-image cost instead of the ~0s it took
            # to parse an already-generated answer.
            per_image = (batch_elapsed / len(chunk)) if batch_elapsed else \
                time.time() - started
            print(f"{len(scene['objects'])} objects, {per_image:.1f}s")
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
