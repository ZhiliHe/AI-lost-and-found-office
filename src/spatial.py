"""Spatial relations computed from bounding boxes.

DESIGN DECISION - read this before "improving" it:

We do NOT ask the VLM "is the bottle beside the laptop?". Small VLMs are
confidently wrong about relations, and their answers change between runs, which
makes the multi-turn agent impossible to debug.

Instead we ask the VLM only for boxes (which it is good at) and compute every
relation here in plain geometry. That makes relations deterministic, unit
testable, and tunable - and it gives us something concrete to write in the
evaluation section of the report.

Box format everywhere: [x1, y1, x2, y2] in ORIGINAL image pixels.
"""

from dataclasses import dataclass

# Tuning knobs. If relations feel wrong on your photos, change these numbers,
# not the code. Then run: pytest tests/test_spatial.py
#
# TUNING LOG: on a real 4032x3024 desk photo with 11 objects, the original
# values produced 164 relations - nearly every object was "near" every other,
# which makes the spatial filter useless. Tightened to the values below, which
# give ~130 and still correctly separate "the bottle next to the laptop" (the
# lululemon bottle) from the other bottle further away.
NEAR_DIAGONAL_FRACTION = 0.22   # centres closer than 22% of image diagonal = "near"
BESIDE_GAP_FACTOR = 0.80        # horizontal gap < 0.8x mean width = "beside"
OVERLAP_MIN = 0.25              # min projection overlap to count as aligned
ON_TOP_TOLERANCE = 0.15         # vertical gap tolerance as fraction of lower box height
DUPLICATE_IOU = 0.70            # two boxes overlapping this much are the same object


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self):
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self):
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self):
        return self.w * self.h

    @classmethod
    def of(cls, seq):
        x1, y1, x2, y2 = [float(v) for v in seq]
        # tolerate models that emit the corners in the wrong order
        return cls(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _overlap_1d(a1, a2, b1, b2):
    """Length of the overlap between two intervals."""
    return max(0.0, min(a2, b2) - max(a1, b1))


def _overlap_ratio_x(a, b):
    ov = _overlap_1d(a.x1, a.x2, b.x1, b.x2)
    denom = min(a.w, b.w)
    return ov / denom if denom > 0 else 0.0


def _overlap_ratio_y(a, b):
    ov = _overlap_1d(a.y1, a.y2, b.y1, b.y2)
    denom = min(a.h, b.h)
    return ov / denom if denom > 0 else 0.0


def iou(a, b):
    inter = _overlap_1d(a.x1, a.x2, b.x1, b.x2) * _overlap_1d(a.y1, a.y2, b.y1, b.y2)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def center_distance(a, b):
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def is_near(a, b, image_wh):
    w, h = image_wh
    diagonal = (w ** 2 + h ** 2) ** 0.5
    return center_distance(a, b) < NEAR_DIAGONAL_FRACTION * diagonal


def is_beside(a, b):
    """Roughly the same height on screen, and horizontally close without
    one sitting on top of the other."""
    if _overlap_ratio_y(a, b) < OVERLAP_MIN:
        return False
    gap = max(0.0, max(a.x1, b.x1) - min(a.x2, b.x2))
    mean_w = (a.w + b.w) / 2.0
    return mean_w > 0 and gap < BESIDE_GAP_FACTOR * mean_w


def is_left_of(a, b):
    return _overlap_ratio_y(a, b) >= OVERLAP_MIN and a.cx < b.cx


def is_right_of(a, b):
    return _overlap_ratio_y(a, b) >= OVERLAP_MIN and a.cx > b.cx


def is_above(a, b):
    return _overlap_ratio_x(a, b) >= OVERLAP_MIN and a.cy < b.cy


def is_below(a, b):
    return _overlap_ratio_x(a, b) >= OVERLAP_MIN and a.cy > b.cy


def is_on(a, b):
    """a rests on b: a is above b, horizontally aligned, and the vertical gap
    between a's bottom and b's top is small."""
    if _overlap_ratio_x(a, b) < OVERLAP_MIN:
        return False
    gap = b.y1 - a.y2
    return -0.1 * a.h <= gap <= ON_TOP_TOLERANCE * max(b.h, 1.0)


def is_inside(a, b):
    """a is contained in b (e.g. a pen inside a pencil case)."""
    if a.area <= 0:
        return False
    inter = _overlap_1d(a.x1, a.x2, b.x1, b.x2) * _overlap_1d(a.y1, a.y2, b.y1, b.y2)
    return inter / a.area > 0.8 and b.area > a.area


# Predicates users actually say, mapped to the functions above.
# `near` needs the image size, the rest do not.
_BINARY = {
    "beside": is_beside,
    "left_of": is_left_of,
    "right_of": is_right_of,
    "above": is_above,
    "below": is_below,
    "on": is_on,
    "inside": is_inside,
}

# What a user might type -> canonical predicate
PREDICATE_SYNONYMS = {
    "beside": "beside", "next to": "beside", "by": "beside",
    "alongside": "beside", "adjacent to": "beside",
    "near": "near", "close to": "near", "around": "near",
    "on": "on", "on top of": "on", "atop": "on",
    "under": "below", "underneath": "below", "below": "below", "beneath": "below",
    "above": "above", "over": "above",
    "left of": "left_of", "to the left of": "left_of",
    "right of": "right_of", "to the right of": "right_of",
    "in": "inside", "inside": "inside", "within": "inside",
}


# How many neighbours each object keeps a relation to.
#
# Without a cap this is quadratic: 20 objects on a desk produced 500 triples,
# 216 of them "near". That is not just a big file - it is WRONG. "the bottle
# next to the laptop" matched a bottle on the far side of the desk, because
# everything on one desk is "near" everything else.
#
# Nobody describes a lost item by an anchor that is not its closest neighbour,
# so keeping the nearest few is both smaller and more accurate.
MAX_NEIGHBOURS = 4

# A more specific relation makes the vaguer one redundant: if the book is ON
# the table we do not also need "book near table".
_IMPLIES_NEAR = ("on", "inside", "beside")


def compute_relations(objects, image_wh):
    """objects: list of dicts with 'id' and 'bbox'.
    Returns a list of {subject, predicate, object} triples.

    Runs at INDEX time, so the result is baked into scene_index.json and the
    query path never recomputes geometry.

    Only each object's nearest MAX_NEIGHBOURS get relations - see the note
    above for why that is a correctness fix, not just a size one.
    """
    boxes = {}
    for obj in objects:
        bbox = obj.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        boxes[obj["id"]] = Box.of(bbox)

    triples = []
    ids = list(boxes)
    for sid in ids:
        a = boxes[sid]
        neighbours = sorted((oid for oid in ids if oid != sid),
                            key=lambda oid: center_distance(a, boxes[oid]))
        for oid in neighbours[:MAX_NEIGHBOURS]:
            b = boxes[oid]
            found = []
            for pred, fn in _BINARY.items():
                if fn(a, b):
                    found.append(pred)
            if is_near(a, b, image_wh) and not any(p in found for p in _IMPLIES_NEAR):
                found.append("near")
            for pred in found:
                triples.append({"subject": sid, "predicate": pred, "object": oid})
    return triples


# WHICH RELATIONS SURVIVE A MOVING CAMERA
#
# We compute all of these from 2D boxes, but they are not equally trustworthy
# once you photograph the same desk from another angle:
#
#   near, beside   proximity. Two things close together stay close together
#                  from any viewpoint. SAFE.
#   on, inside     gravity and containment. A book on a table is on the table
#                  no matter where you stand. SAFE.
#   left_of        pure camera artefact. The bottle left of the laptop from the
#   right_of       front is to its RIGHT from the other side of the desk.
#   above, below   in a top-down photo, image "below" means "in front of" in the
#                  real room - depth, not height.
#
# So view-dependent relations are still computed and still useful for
# DESCRIBING one specific photo, but a user's query is never matched against
# them: when someone says "left of the laptop" they mean left from a viewpoint
# we do not know, so we honour the part we can trust - that the two things are
# near each other.
CAMERA_INVARIANT = {"near", "beside", "on", "inside"}
VIEW_DEPENDENT = {"left_of", "right_of", "above", "below"}

PREDICATE_ALTERNATIVES = {
    "beside": ("beside", "near"),
    "near": ("near", "beside"),
    "left_of": ("beside", "near"),
    "right_of": ("beside", "near"),
    "above": ("on", "above", "near"),
    "below": ("below", "near"),
    "on": ("on", "above"),
    "inside": ("inside",),
}


def alternatives_for(predicate):
    """Stored predicates that should count as a match for this query predicate."""
    return PREDICATE_ALTERNATIVES.get(predicate, (predicate,))


def deduplicate(objects, threshold=None):
    """Drop objects whose box is essentially the same as an earlier one.

    A 2B model will happily list the same charger twice. Left in, the agent
    then says "I found 2 chargers" for a photo containing one, and the whole
    clarification loop is built on a lie.
    """
    threshold = DUPLICATE_IOU if threshold is None else threshold
    kept = []
    for obj in objects:
        box = Box.of(obj["bbox"])
        duplicate = False
        for other in kept:
            if iou(box, Box.of(other["bbox"])) > threshold:
                duplicate = True
                # keep whichever the model was more confident about
                if obj.get("confidence", 0) > other.get("confidence", 0):
                    other.update({k: v for k, v in obj.items() if k != "id"})
                break
        if not duplicate:
            kept.append(obj)
    return kept


def normalize_predicate(text):
    """'to the left of' -> 'left_of'. Returns None if nothing matches."""
    t = text.strip().lower()
    if t in PREDICATE_SYNONYMS:
        return PREDICATE_SYNONYMS[t]
    return None


def find_predicate(text):
    """Scan a free-text query for a relation word. Longest match wins so
    'on top of' is not swallowed by 'on'."""
    from .vocab import _padded
    t = _padded(text)
    for phrase in sorted(PREDICATE_SYNONYMS, key=len, reverse=True):
        if f" {phrase} " in t:
            return PREDICATE_SYNONYMS[phrase], phrase
    return None, None
