"""Turn natural language into the stable ParsedQuery schema.

This is a structured parser, not a retrieval model. It normalises user language
into deterministic constraints, then the rest of the pipeline still decides
matches from the offline index. An LLM can replace only this module later, but
it must emit the same ParsedQuery fields and must not choose results directly.
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


FILLER_PREFIXES = (
    "where is", "where's", "where did i put", "find", "find me", "look for",
    "locate", "is there", "do you see", "i lost", "i left", "i misplaced",
    "can you find", "please find", "show me", "help me find",
    "busca", "encuentra", "cherche", "trouve", "찾아", "어디",
)


def _normalise_text(text):
    text = str(text or "").strip()
    replacements = {
        "w/": " with ",
        "w/o": " without ",
        "next 2": " next to ",
        "nxt to": " next to ",
        "besids": " beside ",
        "besid": " beside ",
        "nearby": " near ",
    }
    lowered = text.lower()
    for old, new in replacements.items():
        lowered = lowered.replace(old, new)
    return lowered


def _remove_filler(text):
    cleaned = text
    for phrase in sorted(FILLER_PREFIXES, key=len, reverse=True):
        if cleaned.startswith(phrase):
            cleaned = cleaned[len(phrase):].strip(" ?!.,")
            break
    for word in (" my ", " the ", " a ", " an ", " that ", " this "):
        cleaned = cleaned.replace(word, " ")
    return " ".join(cleaned.split())


def _first_distinct(types):
    return types[0] if types else None


# locations are just the folder names under data/images
def parse(text, known_locations=()):
    normalised = _normalise_text(text)
    predicate, phrase = find_predicate(normalised)

    if phrase:
        cut = normalised.find(phrase)
        left, right = normalised[:cut], normalised[cut + len(phrase):]
    else:
        left, right = normalised, ""

    left_clean = _remove_filler(left)
    right_clean = _remove_filler(right)
    left_types = find_object_types(left_clean)
    right_types = find_object_types(right_clean)

    # Nothing on the left of the relation word ("beside the laptop, a bottle")
    # - fall back to scanning the whole string.
    if not left_types and right_types:
        all_types = find_object_types(normalised)
        target = _first_distinct(all_types)
        anchor = right_types[0] if right_types[0] != target else (
            right_types[1] if len(right_types) > 1 else None)
    else:
        target = _first_distinct(left_types) or _first_distinct(find_object_types(normalised))
        anchor = right_types[0] if right_types else None
        if anchor == target and len(right_types) > 1:
            anchor = right_types[1]

    # colours mentioned before the relation word describe the target
    colors = find_colors(left_clean if phrase else normalised)

    location = None
    lowered = normalised
    for name in known_locations:
        if name.lower() in lowered:
            location = name
            break

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
