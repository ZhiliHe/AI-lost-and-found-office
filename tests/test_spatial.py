from src.spatial import (Box, compute_relations, find_predicate, is_above,
                        is_beside, is_left_of, is_near, is_on, is_overlapping,
                        is_same_surface)


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


def test_overlapping_and_same_surface_are_geometry_only():
    a = Box.of([100, 100, 260, 260])
    b = Box.of([220, 140, 380, 300])
    assert is_overlapping(a, b)

    mug = Box.of([100, 300, 170, 520])
    laptop = Box.of([240, 330, 600, 520])
    assert is_same_surface(mug, laptop)


def test_compute_relations_emits_symmetric_beside():
    objects = [
        {"id": "s_o0", "bbox": [400, 300, 700, 550]},
        {"id": "s_o1", "bbox": [720, 320, 780, 540]},
    ]
    triples = compute_relations(objects, (1200, 800))
    beside = {(t["subject"], t["object"]) for t in triples if t["predicate"] == "beside"}
    assert ("s_o0", "s_o1") in beside
    assert ("s_o1", "s_o0") in beside


def test_compute_relations_emits_extended_geometry_predicates():
    objects = [
        {"id": "a", "bbox": [100, 100, 260, 260]},
        {"id": "b", "bbox": [220, 140, 380, 300]},
    ]
    triples = compute_relations(objects, (1200, 800))
    predicates = {t["predicate"] for t in triples}
    assert "overlapping" in predicates


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
    assert CAMERA_INVARIANT == {"near", "beside", "on", "inside",
                                "overlapping", "same_surface"}


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
