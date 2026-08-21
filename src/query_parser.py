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


def _split_on_relation(text, phrase):
    """Cut the sentence at the relation word - at the WORD, not the letters.

    find_predicate matches on word boundaries, so "in" is only a relation when
    it stands alone. Locating it afterwards with a plain substring search threw
    that away and found the "in" inside "f-in-d": "can you help me to find my
    red bottle, it might be in the tearoom" was cut after "can you help me to
    f", so "red" ended up on the anchor side and the colour the person had just
    told us was dropped. We then asked them what colour it was.
    """
    if not phrase:
        return text, ""
    padded = f" {text} "
    at = padded.find(f" {phrase} ")
    if at < 0:
        return text, ""
    return padded[:at].strip(), padded[at + len(phrase) + 2:].strip()


# "it should be laptop, not skateboard". People correct themselves constantly,
# and a correction that is silently ignored is worse than a refusal: the
# original wrong search runs again and looks like the system is not listening.
NEGATIONS_BEFORE = ("not ", "no ", "isn't ", "is not ", "wasn't ", "was not ",
                    "instead of ", "rather than ", "不是", "不")
NEGATIONS_AFTER = ("말고", "아니라", "아니고", "이 아니", "가 아니")


def _earliest_position(text, object_type):
    """Where this type is first named, over every synonym in every language."""
    from .vocab import OBJECT_SYNONYMS, OBJECT_SYNONYMS_CJK
    words = list(OBJECT_SYNONYMS.get(object_type, []))
    words += list(OBJECT_SYNONYMS_CJK.get(object_type, []))
    words.append(object_type)
    found = [text.find(word.lower()) for word in words]
    hits = [at for at in found if at >= 0]
    return min(hits) if hits else None


def _drop_negated(text, types):
    """Remove the types the person just told us it is NOT."""
    if len(types) < 2:
        return types
    kept = []
    for object_type in types:
        at = _earliest_position(text, object_type)
        if at is None:
            kept.append(object_type)
            continue
        before = text[max(0, at - 14):at]
        after = text[at:at + 18]
        negated = (any(mark in before for mark in NEGATIONS_BEFORE)
                   or any(mark in after for mark in NEGATIONS_AFTER))
        if not negated:
            kept.append((at, object_type))
    ordered = [t for t in kept if isinstance(t, tuple)]
    ordered.sort()
    plain = [t for t in kept if not isinstance(t, tuple)]
    return [t for _, t in ordered] + plain or types


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
    normalised = _normalise_text(text)
    predicate, phrase = find_predicate(normalised)

    left, right = _split_on_relation(normalised, phrase)

    left_clean = _remove_filler(left)
    right_clean = _remove_filler(right)
    left_types = _drop_negated(left_clean, find_object_types(left_clean))
    right_types = _drop_negated(right_clean, find_object_types(right_clean))

    # Nothing on the left of the relation word ("beside the laptop, a bottle")
    # - fall back to scanning the whole string.
    if not left_types and right_types:
        all_types = _drop_negated(normalised, find_object_types(normalised))
        target = _first_distinct(all_types)
        anchor = right_types[0] if right_types[0] != target else (
            right_types[1] if len(right_types) > 1 else None)
    else:
        target = _first_distinct(left_types) or _first_distinct(
            _drop_negated(normalised, find_object_types(normalised)))
        anchor = right_types[0] if right_types else None
        if anchor == target and len(right_types) > 1:
            anchor = right_types[1]

    # colours mentioned before the relation word describe the target
    colors = find_colors(left_clean if phrase else normalised)

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
