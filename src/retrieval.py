"""Stage 2 + 3: rank scenes, then collect matching object candidates.

We keep the professor's "retrieve Top-K scenes, then understand them" shape,
but because scene understanding already happened at index time, both stages are
pure dictionary work and run in milliseconds.

No CLIP in v1 on purpose. Matching against the indexed object list is more
accurate for attribute queries ("black bottle") than CLIP image-text similarity,
and it costs zero memory - which matters on an 8GB laptop. If you want CLIP
later, add it as an EXTRA scene score here; do not replace this.
"""

from .spatial import alternatives_for
from .vocab import FUZZY_TYPE_GROUPS, OBJECT_SYNONYMS, is_negated


class SceneIndex:
    def __init__(self, payload):
        self.payload = payload
        self.scenes = payload.get("scenes", [])
        self._by_id = {s["scene_id"]: s for s in self.scenes}

    @classmethod
    def load(cls, path):
        import json
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def scene(self, scene_id):
        return self._by_id.get(scene_id)

    def locations(self):
        seen = []
        for scene in self.scenes:
            if scene.get("location") and scene["location"] not in seen:
                seen.append(scene["location"])
        return seen

    def object(self, scene_id, object_id):
        scene = self.scene(scene_id)
        if not scene:
            return None
        for obj in scene.get("objects", []):
            if obj["id"] == object_id:
                return obj
        return None


def score_scene(scene, parsed):
    """Cheap lexical relevance so we can show a real Top-K retrieval stage."""
    score = 0.0
    types = [o.get("type") for o in scene.get("objects", [])]

    # Must use the same matcher as find_candidates, otherwise a scene can be
    # dropped at ranking time and silently never become a candidate.
    if parsed.target_type and any(_type_matches(t, parsed.target_type) for t in types):
        score += 3.0
    if parsed.anchor_type and any(_type_matches(t, parsed.anchor_type) for t in types):
        score += 2.0
    if parsed.location and scene.get("location") == parsed.location:
        score += 2.0

    wanted_color = parsed.attributes.get("color")
    if wanted_color:
        for obj in scene.get("objects", []):
            if obj.get("type") == parsed.target_type and \
                    obj.get("attributes", {}).get("color") == wanted_color:
                score += 1.5
                break

    # tiny nudge from the caption
    caption = (scene.get("caption") or "").lower()
    for word in (parsed.target_type, parsed.anchor_type):
        if word and word in caption:
            score += 0.5

    return score


def rank_scenes(index, parsed, top_k=5):
    scored = [(score_scene(s, parsed), s) for s in index.scenes]
    scored = [pair for pair in scored if pair[0] > 0]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [s for _, s in scored[:top_k]]


def _type_matches(obj_type, wanted):
    if obj_type == wanted:
        return True
    # "notebook" is genuinely ambiguous in English - it can mean a laptop or a
    # paper notebook - so a user asking for a notebook should see both, and the
    # agent will ask which they meant. The reverse is NOT true: a user who says
    # "laptop" does not mean a paper notebook.
    if wanted == "notebook" and obj_type in ("notebook", "laptop"):
        return True
    # A small VLM confuses categories that look alike - it labelled a tall
    # lululemon water bottle a "mug". Rather than lose the object entirely, we
    # accept the whole look-alike group and let the agent ask which one.
    for group in FUZZY_TYPE_GROUPS:
        if wanted in group and obj_type in group:
            return True
    return wanted in OBJECT_SYNONYMS.get(obj_type, [])


def _relation_holds(scene, subject_id, predicate, anchor_type):
    anchors = {o["id"] for o in scene.get("objects", [])
               if _type_matches(o.get("type"), anchor_type)}
    if not anchors:
        return False
    accepted = alternatives_for(predicate)
    for triple in scene.get("relations", []):
        if triple["subject"] == subject_id and triple["predicate"] in accepted \
                and triple["object"] in anchors:
            return True
    return False


def _view_quality(cand):
    """Bigger and more confident = the clearer look at the object, so that is
    the view we show the user."""
    x1, y1, x2, y2 = cand["object"]["bbox"]
    return abs(x2 - x1) * abs(y2 - y1) * float(cand["object"].get("confidence", 0.5))


def merge_views(candidates):
    """Collapse the same physical object seen from several camera angles.

    ASSUMPTION: every photo inside one location folder is a different VIEW of
    the same place. So `office/office_01.jpg` and `office/office_02.jpg` are two
    angles on one desk, not two different desks.

    Without this, three photos of one desk make the agent announce "I found 3
    bottles", then ask clarifying questions that can never separate them,
    because they are the same bottle. The loop stalls and looks broken.

    HOW IT WORKS - and why it is not a simple "same colour + material" match:

    Matching on attributes fails, because the attributes are exactly the thing
    the model is unreliable about. The same red bottle came back as "glass" from
    the front and "plastic" from the left, and a key that includes material then
    splits one bottle into two.

    So instead:
      1. The view that saw the MOST objects of this type becomes the reference.
         It decides how many objects exist. Two photos seeing 2 and 3 bottles
         means there are 3 bottles, never 5.
      2. Every other view's objects are matched one-to-one onto the reference
         objects by appearance similarity.
      3. Attributes are then decided by MAJORITY VOTE across views, and every
         value ever observed is kept, so the user's answer matches if ANY angle
         saw it that way.

    Two matching objects in the SAME photo are never merged - those really are
    two objects.

    If one location folder contains genuinely different places (two desks in
    one office), split them into separate folders instead - `office-deskA/`,
    `office-deskB/`.
    """
    if not candidates:
        return []

    groups = {}
    for cand in candidates:
        key = (cand["scene"].get("location"), cand["object"].get("type"))
        groups.setdefault(key, []).append(cand)

    merged = []
    for items in groups.values():
        merged.extend(_merge_one_group(items))
    return merged


def _merge_one_group(items):
    """All the objects of one type at one place, seen from several angles."""
    by_scene = {}
    for cand in items:
        by_scene.setdefault(cand["scene"]["scene_id"], []).append(cand)

    # The reference view is the one that saw the most; ties go to the clearest.
    # It decides HOW MANY objects exist. Other views only vote on what they look
    # like - which is why a bottle read as glass from the front and plastic from
    # the left stays one bottle.
    reference_id = max(
        by_scene,
        key=lambda sid: (len(by_scene[sid]),
                         sum(_view_quality(c) for c in by_scene[sid])))

    canonical = []
    for cand in by_scene[reference_id]:
        entry = dict(cand)
        entry["seen_in"] = [reference_id]
        entry["observed"] = {k: [v] for k, v in
                             (cand["object"].get("attributes") or {}).items() if v}
        canonical.append(entry)

    for scene_id, others in by_scene.items():
        if scene_id == reference_id:
            continue
        _absorb(canonical, others, scene_id)

    for entry in canonical:
        attributes = dict(entry["object"].get("attributes") or {})
        for key, values in entry["observed"].items():
            attributes[key] = _majority(values)
        entry["object"] = {**entry["object"], "attributes": attributes}
    return canonical


def _absorb(canonical, others, scene_id):
    """Match this view's objects one-to-one onto the reference objects."""
    pairs = []
    for other_index, other in enumerate(others):
        for canon_index, canon in enumerate(canonical):
            pairs.append((_similarity(canon, other), canon_index, other_index))
    pairs.sort(reverse=True)

    used_canon, used_other = set(), set()
    for _, canon_index, other_index in pairs:
        if canon_index in used_canon or other_index in used_other:
            continue
        used_canon.add(canon_index)
        used_other.add(other_index)

        canon, other = canonical[canon_index], others[other_index]
        canon["seen_in"].append(scene_id)
        for key, value in (other["object"].get("attributes") or {}).items():
            if value:
                canon["observed"].setdefault(key, []).append(value)
        # If this angle shows the object more clearly, display THAT photo -
        # but keep the accumulated votes and history.
        if _view_quality(other) > _view_quality(canon):
            seen, observed = canon["seen_in"], canon["observed"]
            canon.update(other)
            canon["seen_in"], canon["observed"] = seen, observed


def _similarity(a, b):
    """How many attributes two observations agree on. Position is useless here
    because the camera moved, so we compare appearance only."""
    left = a["object"].get("attributes") or {}
    right = b["object"].get("attributes") or {}
    score = 0.0
    for key in ("color", "material", "size"):
        if left.get(key) and left.get(key) == right.get(key):
            score += 2.0 if key == "color" else 1.0
    return score


def _majority(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=lambda v: (counts[v], values.index(v) * -1))


def _satisfies(cand, wanted):
    """Does this candidate meet every attribute the user has stated?

    Checks the values ACTUALLY OBSERVED across all camera angles, not just the
    winning view. A bottle the model called glass from the front and plastic
    from the left should match either answer - the user knows what their bottle
    is made of, our model does not.
    """
    attributes = cand["object"].get("attributes") or {}
    observed = cand.get("observed") or {}

    for key, value in wanted.items():
        if value is None:
            continue
        seen = observed.get(key) or ([attributes[key]] if attributes.get(key) else [])

        if is_negated(value):
            # "it is not metal" - only rule it out if EVERY angle said metal.
            # If one view said plastic, the user may well be right.
            rejected = value[1]
            if seen and all(v == rejected for v in seen):
                return False
            continue

        if not seen:
            return False        # unknown attribute cannot satisfy a stated one
        if value not in seen:
            return False
    return True


def find_candidates(index, parsed, constraints=None, top_k_scenes=5,
                    merge_across_views=True):
    """Return [{scene, object, score}] for every object that satisfies the
    query plus any constraints gathered during clarification."""
    constraints = constraints or {}
    scenes = rank_scenes(index, parsed, top_k=top_k_scenes) or index.scenes
    candidates = []

    # Pass 1: type, location and spatial relation only. Attributes are checked
    # AFTER merging, so that a value seen from any angle can satisfy the user.
    for scene in scenes:
        if parsed.location and scene.get("location") != parsed.location:
            continue
        for obj in scene.get("objects", []):
            if parsed.target_type and not _type_matches(obj.get("type"), parsed.target_type):
                continue
            if parsed.predicate and parsed.anchor_type:
                if not _relation_holds(scene, obj["id"], parsed.predicate, parsed.anchor_type):
                    continue
            candidates.append({
                "scene": scene,
                "object": obj,
                "score": float(obj.get("confidence", 0.5)),
            })

    if merge_across_views:
        candidates = merge_views(candidates)

    # Pass 2: now apply what the user told us.
    wanted = dict(parsed.attributes)
    wanted.update(constraints)
    if wanted:
        candidates = [c for c in candidates if _satisfies(c, wanted)]

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates
