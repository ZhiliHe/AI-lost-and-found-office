from src.spatial import (Box, compute_relations, find_predicate, is_above,
                        is_beside, is_left_of, is_near, is_on)


def test_beside_true_for_side_by_side_objects():
    laptop = Box.of([400, 300, 700, 550])
    bottle = Box.of([720, 320, 780, 540])
    assert is_beside(laptop, bottle)
    assert is_beside(bottle, laptop)


def test_beside_false_when_far_apart():
    left = Box.of([50, 300, 110, 500])
    right = Box.of([1000, 300, 1060, 500])
    assert not is_beside(left, right)


def test_beside_false_when_stacked_vertically():
    top = Box.of([400, 100, 500, 200])
    bottom = Box.of([400, 400, 500, 500])
    assert not is_beside(top, bottom)


def test_on_requires_contact():
    book = Box.of([400, 480, 600, 520])
    table = Box.of([300, 520, 900, 700])
    assert is_on(book, table)

    floating = Box.of([400, 100, 600, 140])
    assert not is_on(floating, table)


def test_left_right_and_above():
    a = Box.of([100, 300, 200, 400])
    b = Box.of([500, 300, 600, 400])
    assert is_left_of(a, b)
    assert not is_left_of(b, a)

    high = Box.of([300, 100, 400, 200])
    low = Box.of([300, 600, 400, 700])
    assert is_above(high, low)


def test_near_scales_with_image_size():
    a = Box.of([100, 100, 150, 150])
    b = Box.of([200, 200, 250, 250])
    assert is_near(a, b, (1200, 800))
    far = Box.of([1150, 750, 1190, 790])
    assert not is_near(a, far, (1200, 800))


def test_compute_relations_emits_symmetric_beside():
    objects = [
        {"id": "s_o0", "bbox": [400, 300, 700, 550]},
        {"id": "s_o1", "bbox": [720, 320, 780, 540]},
    ]
    triples = compute_relations(objects, (1200, 800))
    beside = {(t["subject"], t["object"]) for t in triples if t["predicate"] == "beside"}
    assert ("s_o0", "s_o1") in beside
    assert ("s_o1", "s_o0") in beside


def test_compute_relations_skips_objects_without_boxes():
    objects = [{"id": "a", "bbox": [0, 0, 10, 10]}, {"id": "b"}]
    assert compute_relations(objects, (100, 100)) == []


def test_find_predicate_prefers_longest_match():
    predicate, phrase = find_predicate("the mug on top of the shelf")
    assert predicate == "on"
    assert phrase == "on top of"

    predicate, _ = find_predicate("the bottle next to the laptop")
    assert predicate == "beside"

    predicate, _ = find_predicate("just find my bottle")
    assert predicate is None


def test_box_tolerates_inverted_corners():
    box = Box.of([200, 400, 100, 300])
    assert box.x1 == 100 and box.y1 == 300
    assert box.w == 100 and box.h == 100


# --- which relations survive a moving camera -------------------------------- #

def test_view_dependent_predicates_are_classified():
    from src.spatial import CAMERA_INVARIANT, VIEW_DEPENDENT
    # these flip when you photograph the same desk from the other side
    assert VIEW_DEPENDENT == {"left_of", "right_of", "above", "below"}
    # these do not
    assert CAMERA_INVARIANT == {"near", "beside", "on", "inside"}


def test_left_of_query_falls_back_to_proximity():
    """A user saying "left of the laptop" means left from a viewpoint we do not
    know. We honour the part we can trust: the two things are near each other."""
    from src.spatial import alternatives_for
    assert set(alternatives_for("left_of")) == {"beside", "near"}
    assert set(alternatives_for("right_of")) == {"beside", "near"}
    # no query predicate is matched against a raw left/right relation
    for predicate in ("beside", "near", "left_of", "right_of", "on", "inside"):
        assert "left_of" not in alternatives_for(predicate)
        assert "right_of" not in alternatives_for(predicate)


# --- tiling ----------------------------------------------------------------- #

def test_tiles_overlap_enough_to_contain_a_small_object():
    """The whole point of overlapping: an object narrower than the overlap is
    whole inside SOME tile, even if another tile cuts it in half."""
    from src.tiling import OVERLAP, make_tiles

    width, height = 4000, 3000
    tiles = make_tiles(width, height, grid=2)
    assert len(tiles) == 4

    horizontal = sorted({(t[0], t[2]) for t in tiles})
    left, right = horizontal[0], horizontal[1]
    shared = left[1] - right[0]
    assert shared >= width * OVERLAP * 0.99

    # tiles must together cover the whole image
    assert min(t[0] for t in tiles) == 0
    assert max(t[2] for t in tiles) == width
    assert min(t[1] for t in tiles) == 0
    assert max(t[3] for t in tiles) == height


def test_a_box_cut_by_an_interior_edge_is_dropped():
    from src.tiling import is_truncated

    image = (4000, 3000)
    tile = (0, 0, 2600, 1950)

    cut = [2400, 900, 2600, 1100]          # runs into the interior right edge
    assert is_truncated(cut, tile, image)

    whole = [1000, 900, 1200, 1100]
    assert not is_truncated(whole, tile, image)


def test_a_box_on_the_real_photo_border_is_kept():
    """Touching the edge of the PHOTO is not truncation - the object really is
    at the edge of the scene."""
    from src.tiling import is_truncated

    image = (4000, 3000)
    tile = (1400, 1050, 4000, 3000)
    at_border = [3800, 2800, 4000, 3000]
    assert not is_truncated(at_border, tile, image)


def test_merge_keeps_whichever_pass_saw_the_object_bigger():
    """A USB stick should come from the tile that saw it large, a desk-spanning
    laptop from the full image."""
    from src.tiling import merge_passes

    from_full = {"type": "charger", "bbox": [1000, 1000, 1040, 1030], "confidence": 0.5}
    from_tile = {"type": "charger", "bbox": [1002, 1001, 1042, 1032], "confidence": 0.9}

    merged = merge_passes([(from_full, 0.0002), (from_tile, 0.01)])
    assert len(merged) == 1
    assert merged[0] is from_tile


def test_merge_keeps_different_objects_apart():
    from src.tiling import merge_passes

    a = {"type": "bottle", "bbox": [100, 100, 200, 400], "confidence": 0.9}
    b = {"type": "bottle", "bbox": [900, 100, 1000, 400], "confidence": 0.9}
    assert len(merge_passes([(a, 0.01), (b, 0.01)])) == 2


def test_tile_coordinates_map_back_to_the_whole_image(tmp_path):
    """The #1 risk with tiling is a coordinate bug: an object found in a tile
    must come back with its position in the ORIGINAL photo. We plant a red
    square at a known place and check we get that place back.
    """
    import json
    from PIL import Image
    from src.indexer import index_one

    width, height = 4000, 3000
    truth = (2900, 2050, 3000, 2130)          # a small object, bottom right

    images = tmp_path / "data" / "images" / "office"
    images.mkdir(parents=True)
    photo = Image.new("RGB", (width, height), (230, 228, 225))
    photo.paste(Image.new("RGB", (truth[2] - truth[0], truth[3] - truth[1]),
                          (220, 40, 40)), truth[:2])
    photo.save(images / "desk.jpg", quality=95)

    class Oracle:
        """Finds the red square in whatever frame it is shown, and reports it
        in 0-1000 coordinates - exactly what we ask a real model for."""

        def describe(self, path, prompt, **_):
            image = Image.open(path).convert("RGB")
            pixels = image.load()
            xs, ys = [], []
            for y in range(0, image.height, 2):
                for x in range(0, image.width, 2):
                    r, g, b = pixels[x, y]
                    if r > 180 and g < 90 and b < 90:
                        xs.append(x)
                        ys.append(y)
            if not xs:
                return json.dumps({"n_objects": 0, "objects": [], "caption": ""})
            box = [min(xs) / image.width * 1000, min(ys) / image.height * 1000,
                   max(xs) / image.width * 1000, max(ys) / image.height * 1000]
            return json.dumps({"n_objects": 1, "caption": "a desk", "objects": [
                {"type": "charger", "color": "red", "material": "plastic",
                 "size": "small", "bbox": [round(v) for v in box],
                 "confidence": 0.9}]})

    cfg = {"vlm": {"max_image_side": 1280, "coord_mode": "norm1000", "tiling": 2},
           "paths": {"debug": str(tmp_path / "debug")}}
    scene = index_one(Oracle(), images / "desk.jpg", "office", cfg)

    assert len(scene["objects"]) == 1, "the same square must not become two"
    found = scene["objects"][0]["bbox"]
    error = max(abs(a - b) for a, b in zip(found, truth))
    assert error <= 20, f"{found} vs {truth}, off by {error}px"


# --- keeping the relation graph small AND correct --------------------------- #

def test_relations_are_capped_per_object():
    """Unbounded, this is quadratic: 20 objects on one desk produced 500
    triples, 216 of them "near"."""
    from src.spatial import MAX_NEIGHBOURS, compute_relations

    objects = [{"id": f"o{i}", "bbox": [200 * i, 1000, 200 * i + 150, 1300]}
               for i in range(20)]
    relations = compute_relations(objects, (4000, 3000))

    for obj in objects:
        anchors = {r["object"] for r in relations if r["subject"] == obj["id"]}
        assert len(anchors) <= MAX_NEIGHBOURS, obj["id"]


def test_a_far_object_is_not_near():
    """"the bottle next to the laptop" must not match a bottle on the other
    side of the desk - which it did, because everything was "near" everything."""
    from src.spatial import compute_relations

    objects = [
        {"id": "laptop", "bbox": [1800, 1200, 2600, 1800]},
        {"id": "close_bottle", "bbox": [2650, 1250, 2800, 1750]},
        {"id": "far_bottle", "bbox": [200, 1250, 350, 1750]},
        {"id": "a", "bbox": [1700, 1900, 1850, 2050]},
        {"id": "b", "bbox": [2000, 1900, 2150, 2050]},
        {"id": "c", "bbox": [2300, 1900, 2450, 2050]},
    ]
    relations = compute_relations(objects, (4000, 3000))
    pairs = {(r["subject"], r["predicate"], r["object"]) for r in relations}

    assert ("close_bottle", "beside", "laptop") in pairs
    assert ("far_bottle", "near", "laptop") not in pairs
    assert ("far_bottle", "beside", "laptop") not in pairs


def test_near_is_dropped_when_a_more_specific_relation_holds():
    """"book on table" makes "book near table" redundant noise."""
    from src.spatial import compute_relations

    objects = [{"id": "table", "bbox": [1000, 1500, 3000, 2500]},
               {"id": "book", "bbox": [1500, 1200, 2000, 1500]}]
    relations = compute_relations(objects, (4000, 3000))
    pairs = {(r["subject"], r["predicate"], r["object"]) for r in relations}

    assert ("book", "on", "table") in pairs
    assert ("book", "near", "table") not in pairs
