"""Turn "Find the black bottle beside my laptop" into a structured query.

Deliberately rule-based, not an LLM call. Three reasons:
  1. it is instant, so the demo never stalls on a second model
  2. it is deterministic, so tests are meaningful
  3. it fails visibly instead of hallucinating a target object

If it turns out to be too brittle on real user phrasing, the clean upgrade is to
swap ONLY this module for an LLM that emits the same dataclass. Nothing
downstream changes.
"""

from dataclasses import dataclass, field

from .spatial import find_predicate
from .vocab import find_colors, find_object_types


@dataclass
class ParsedQuery:
    raw: str
    target_type: str = None
    attributes: dict = field(default_factory=dict)
    predicate: str = None
    anchor_type: str = None
    location: str = None

    def describe(self):
        """Human-readable echo, used in agent replies."""
        bits = []
        if self.attributes.get("color"):
            bits.append(self.attributes["color"])
        bits.append(self.target_type or "object")
        text = " ".join(bits)
        if self.predicate and self.anchor_type:
            text += f" {self.predicate.replace('_', ' ')} the {self.anchor_type}"
        return text


def _squash(text):
    """Lowercase and throw away everything that is not a letter or digit.

    So "the meeting room", "meeting-room" and "meetingroom" all become the same
    string, and a folder called "6f_meetingroom" can be matched by someone who
    simply typed "meeting room".
    """
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


# Floor prefixes carry no meaning on their own - "7f" must not make a query
# about the 7th floor match the tearoom.
_FLOOR_PREFIX = {"1f", "2f", "3f", "4f", "5f", "6f", "7f", "8f", "9f", "10f", "b1", "b2"}


def _location_variants(name):
    """Ways a person might refer to a folder called "7f_tearoom"."""
    variants = {_squash(name)}
    for token in str(name).replace("-", "_").split("_"):
        squashed = _squash(token)
        if len(squashed) >= 3 and squashed not in _FLOOR_PREFIX:
            variants.add(squashed)
    return {v for v in variants if len(v) >= 3}


def _find_location(text, known_locations):
    """Longest match wins, so "8f_lobbytable" beats a stray "table"."""
    squashed_query = _squash(text)
    best = None
    for name in known_locations:
        for variant in _location_variants(name):
            if variant in squashed_query:
                if best is None or len(variant) > best[0]:
                    best = (len(variant), name)
    return best[1] if best else None


# locations are just the folder names under data/images
def parse(text, known_locations=()):
    predicate, phrase = find_predicate(text)

    if phrase:
        cut = text.lower().find(phrase)
        left, right = text[:cut], text[cut + len(phrase):]
    else:
        left, right = text, ""

    left_types = find_object_types(left)
    right_types = find_object_types(right)

    # Nothing on the left of the relation word ("beside the laptop, a bottle")
    # - fall back to scanning the whole string.
    if not left_types and right_types:
        all_types = find_object_types(text)
        target = all_types[0] if all_types else None
        anchor = right_types[0] if right_types[0] != target else (
            right_types[1] if len(right_types) > 1 else None)
    else:
        target = left_types[0] if left_types else None
        anchor = right_types[0] if right_types else None

    # colours mentioned before the relation word describe the target
    colors = find_colors(left if phrase else text)

    location = _find_location(text, known_locations)

    attributes = {}
    if colors:
        attributes["color"] = colors[0]

    return ParsedQuery(
        raw=text,
        target_type=target,
        attributes=attributes,
        predicate=predicate if anchor else None,
        anchor_type=anchor,
        location=location,
    )
