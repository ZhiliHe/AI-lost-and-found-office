"""Resolve "the blue laptop in 703" against whatever the index currently holds.

WHY THIS FILE EXISTS

The first version of the evaluation set named its expected answers by object id
- "expect": "office_01_o2". Ids are assigned by the indexer in the order it
happens to emit objects, so the day the photos were re-indexed every single id
changed and the whole suite reported 10% accuracy while nothing was actually
broken. An evaluation that breaks when you re-run the system it evaluates is
worse than no evaluation: it trains you to ignore it.

So a case says WHAT the answer is - place, kind, colour - in exactly the terms
data/ground_truth.json uses, and this module looks up which object that is in
the index we are testing today. Re-index all you like; the cases stand.

It also gives us a distinction the old suite could not make. If the descriptor
matches nothing at all, the target is MISSING from the index - the indexer
never saw it. That is a recall failure, not a reasoning failure, and lumping
the two together hides the one number the report most needs.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vocab import FUZZY_TYPE_GROUPS, find_object_types   # noqa: E402

GROUND_TRUTH = Path(__file__).resolve().parent.parent / "data" / "ground_truth.json"


def load_ground_truth(path=GROUND_TRUTH):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def in_scope(ground_truth):
    """Every object we intend to find, as (location, object) pairs.

    `scope: out` is excluded on purpose - fixtures and things the building
    supplies. Counting them would quietly inflate recall.
    """
    pairs = []
    for location, body in ground_truth["locations"].items():
        for obj in body.get("objects", []):
            if obj.get("scope") == "in":
                pairs.append((location, obj))
    return pairs


def canonical(name):
    """The vocabulary's name for a thing a person wrote down.

    data/ground_truth.json is written in human words - "handheld fan", "tissue
    box" - because it was filled in by looking at photographs, not at
    OBJECT_SYNONYMS. The vocabulary calls those "fan" and "tissue". Comparing
    the two strings directly reports the object as missing from the index when
    it is sitting right there, which is the most misleading failure this
    evaluation could produce.
    """
    found = find_object_types(str(name))
    return found[0] if found else str(name)


def same_type(wanted, found):
    """Types that the indexer is allowed to confuse with each other.

    A 2B model called the same tall lululemon bottle a "mug" from one angle and
    a "bottle" from another. Marking that a miss would measure the label, not
    whether we found the object - the person asking is still shown their
    bottle. FUZZY_TYPE_GROUPS is the same list retrieval uses, so the evaluation
    forgives exactly what the search forgives and nothing more.
    """
    wanted, found = canonical(wanted), canonical(found)
    if wanted == found:
        return True
    for group in FUZZY_TYPE_GROUPS:
        if wanted in group and found in group:
            return True
    return False


def matches(descriptor, location, obj):
    """Is this indexed object the thing the descriptor is talking about?

    Colour is checked only when the descriptor names one, so a case can say
    "the keys in 711" without pinning a colour the model may reasonably see
    differently.
    """
    if descriptor.get("location") and descriptor["location"] != location:
        return False
    if not same_type(descriptor["type"], obj.get("type")):
        return False
    wanted_colour = descriptor.get("color")
    if wanted_colour:
        return (obj.get("attributes") or {}).get("color") == wanted_colour
    return True


def resolve(index, descriptor):
    """Descriptor -> every object id in the index that could be it.

    Several ids, not one: the same object appears once per view it was
    photographed in, and either view is a correct answer. Retrieval merges
    them; the evaluation only needs to know the agent landed on the right
    thing.
    """
    found = []
    for scene in index.scenes:
        location = scene.get("location")
        for obj in scene.get("objects", []):
            if matches(descriptor, location, obj):
                found.append((scene, obj))
    return found


def matches_candidate(descriptor, candidate):
    """Did the agent hand back the right physical object?

    Retrieval merges the views of one object and votes on its colour, so the
    blue MacBook can come back as the "gray" view when the side photo misread
    it. The vote is a labelling mistake; the object is still the one the person
    lost, and the answer even calls it blue - so counting it wrong would be
    measuring the indexer's adjective, not whether the search worked.

    `observed` records every value any view reported, which is exactly the
    evidence needed to say "some camera did see this as blue".
    """
    if candidate is None:
        return False
    scene, obj = candidate.get("scene") or {}, candidate.get("object") or {}
    if descriptor.get("location") and descriptor["location"] != scene.get("location"):
        return False
    if not same_type(descriptor["type"], obj.get("type")):
        return False
    wanted_colour = descriptor.get("color")
    if not wanted_colour:
        return True
    if (obj.get("attributes") or {}).get("color") == wanted_colour:
        return True
    seen = ((candidate.get("observed") or {}).get("color") or {})
    return wanted_colour in seen


def describe(descriptor):
    parts = [descriptor.get("color"), descriptor.get("type")]
    text = " ".join(part for part in parts if part)
    where = descriptor.get("location")
    return f"{text} in {where}" if where else text
