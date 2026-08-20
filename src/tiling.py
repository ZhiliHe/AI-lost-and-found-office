"""Look at the photo again, in pieces, so small objects stop being invisible.

THE PROBLEM

`max_image_side` shrinks a 4032px phone photo to 1280 before the model sees it.
That is not optional - a full-resolution photo explodes into tens of thousands of
vision tokens and stalls an 8GB machine. But it means a USB stick that was 120px
across is now 38px, and a 2B model does not see 38px. On our office desk the
model found 8 of ~18 objects, and every single miss was small or near an edge.

THE FIX

Run the model once on the whole photo, then again on overlapping tiles. A tile is
a crop of the ORIGINAL image, so an object inside it arrives at the model several
times larger than it would in the downscaled full view.

    full image     ->  reliable for big things (laptop, backpack)
    2x2 tiles      ->  reliable for small things (USB, keys, earbuds)

WHY THE TILES OVERLAP

Cutting a photo cuts objects. With tiles that overlap by `OVERLAP` of the image,
any object narrower than that overlap is fully inside at least one tile even if
another tile bisects it. We then throw away any detection touching a tile's
INTERIOR edge - it is probably a fragment, and a neighbouring tile has the whole
thing. Objects larger than the overlap are the big ones, and those are what the
full-image pass is for.

    ┌───────────┬─────┬───────────┐
    │  tile 1   │     │  tile 2   │
    │           │ overlap         │
    └───────────┴─────┴───────────┘
      a small object cut here is whole over there

COST

One pass becomes five. At ~45 s per pass that is ~4 minutes per photo, so this is
opt-in (`vlm.tiling` in config.yaml). Turn it on for the final index, leave it off
while iterating.
"""

from PIL import Image

# Fraction of the image width/height that neighbouring tiles share. Anything
# narrower than this is guaranteed to sit whole inside some tile.
OVERLAP = 0.30

# A detection whose box comes within this fraction of a tile's interior edge is
# treated as truncated and dropped.
EDGE_MARGIN = 0.02

# Two boxes of the same type overlapping by more than this are the same object.
MERGE_IOU = 0.45


def make_tiles(width, height, grid=2, overlap=OVERLAP):
    """Overlapping tile rectangles in original-image pixels.

    grid=2 gives 2x2 = 4 tiles. Each tile is (1 + overlap) / grid of the image,
    so neighbours share `overlap` of it.
    """
    if grid < 2:
        return []

    span_x = width * (1.0 + overlap) / grid
    span_y = height * (1.0 + overlap) / grid
    step_x = (width - span_x) / (grid - 1)
    step_y = (height - span_y) / (grid - 1)

    tiles = []
    for row in range(grid):
        for col in range(grid):
            x0 = col * step_x
            y0 = row * step_y
            tiles.append((round(x0), round(y0),
                          round(min(width, x0 + span_x)),
                          round(min(height, y0 + span_y))))
    return tiles


def crop_tile(image_path, tile, out_path):
    """Cut a tile out of the ORIGINAL image. No downscaling here - making the
    object bigger is the entire point."""
    with Image.open(image_path) as img:
        img.convert("RGB").crop(tile).save(out_path, quality=92)
    return out_path


def tile_to_global(bbox, tile):
    """A box measured inside a tile, expressed in whole-image pixels."""
    x1, y1, x2, y2 = bbox
    ox, oy = tile[0], tile[1]
    return [x1 + ox, y1 + oy, x2 + ox, y2 + oy]


def is_truncated(bbox, tile, image_wh):
    """Does this box run into an edge that is a CUT, not the photo's own border?

    A box touching the real edge of the photo is fine - the object really is at
    the edge of the scene. A box touching an edge we invented by cropping is
    probably half an object.
    """
    x1, y1, x2, y2 = bbox
    tx1, ty1, tx2, ty2 = tile
    width, height = image_wh
    margin_x = (tx2 - tx1) * EDGE_MARGIN
    margin_y = (ty2 - ty1) * EDGE_MARGIN

    if tx1 > 0 and x1 <= tx1 + margin_x:
        return True
    if ty1 > 0 and y1 <= ty1 + margin_y:
        return True
    if tx2 < width and x2 >= tx2 - margin_x:
        return True
    if ty2 < height and y2 >= ty2 - margin_y:
        return True
    return False


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_passes(detections, iou_threshold=MERGE_IOU):
    """Collapse the same object seen by the full pass and by one or more tiles.

    `detections` is a list of (object_dict, saw_it_this_big) where the second
    value is how much of ITS OWN pass the object filled. When two detections are
    the same object we keep the one that saw it bigger: for a USB stick that is
    the tile, for a desk-spanning laptop that is the full image. That is exactly
    the division of labour we want, and it falls out of one number.
    """
    ordered = sorted(detections, key=lambda pair: pair[1], reverse=True)

    kept = []
    for obj, relative_size in ordered:
        duplicate = False
        for other, _ in kept:
            if other["type"] != obj["type"]:
                continue
            if _iou(other["bbox"], obj["bbox"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append((obj, relative_size))
    return [obj for obj, _ in kept]


def relative_size(bbox, pass_wh):
    """How much of the frame the model was looking at did this object fill?"""
    x1, y1, x2, y2 = bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    total = float(pass_wh[0]) * float(pass_wh[1])
    return area / total if total > 0 else 0.0
