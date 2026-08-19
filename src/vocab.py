"""Controlled vocabulary.

Why this file exists: a 2B model asked for free-text attributes will call the
same bottle "black", "dark", "charcoal" and "dark gray" across three runs, and
then no query ever matches. We pin the allowed values here, inject them into
the VLM prompt, and normalise anything the model returns anyway.

If you add an object type or colour, add it HERE and nowhere else.
"""

# --- colours the VLM is allowed to use --------------------------------------
COLORS = [
    "black", "white", "gray", "silver", "red", "orange", "yellow",
    "green", "blue", "purple", "pink", "brown", "beige", "gold",
    "transparent", "multicolor",
]

# map sloppy model output -> canonical colour
COLOR_ALIASES = {
    "dark": "black", "dark gray": "black", "dark grey": "black",
    "charcoal": "black", "grey": "gray", "light gray": "gray",
    "light grey": "gray", "navy": "blue", "sky blue": "blue",
    "light blue": "blue", "dark blue": "blue", "teal": "blue",
    "maroon": "red", "burgundy": "red", "crimson": "red",
    "clear": "transparent", "see-through": "transparent",
    "cream": "beige", "tan": "beige", "khaki": "beige",
    "golden": "gold", "yellowish": "yellow", "violet": "purple",
}

# --- object types -----------------------------------------------------------
# The key is canonical; the list is what a user might actually type.
OBJECT_SYNONYMS = {
    "bottle":     ["bottle", "water bottle", "flask", "thermos", "tumbler"],
    "backpack":   ["backpack", "bag", "rucksack", "knapsack", "schoolbag"],
    "laptop":     ["laptop", "macbook", "computer", "notebook computer"],
    "notebook":   ["notebook", "notepad", "note book", "writing pad"],
    "book":       ["book", "textbook"],
    "phone":      ["phone", "smartphone", "mobile", "cellphone", "iphone"],
    "charger":    ["charger", "adapter", "power adapter", "charging brick"],
    "cable":      ["cable", "cord", "wire", "usb cable", "charging cable"],
    "mug":        ["mug", "cup", "coffee cup", "tea cup"],
    "keys":       ["keys", "key", "keychain", "keyring"],
    "wallet":     ["wallet", "purse", "billfold"],
    "umbrella":   ["umbrella"],
    "headphones": ["headphones", "earphones", "earbuds", "headset", "airpods"],
    "glasses":    ["glasses", "eyeglasses", "spectacles", "sunglasses"],
    "mouse":      ["mouse", "computer mouse"],
    "pen":        ["pen", "pencil", "marker"],
    "card":       ["card", "id card", "student card", "badge", "lanyard"],
    "tumbler":    ["tumbler", "flask"],
    "power bank": ["power bank", "powerbank", "battery", "portable charger",
                   "external battery"],
    "tablet":     ["tablet", "ipad"],
    "watch":      ["watch", "smartwatch"],
    "skateboard": ["skateboard", "board"],
}

# Categories a small VLM genuinely mixes up. A query for one member matches any
# member of the group, and the agent then asks which one the user meant.
# Discovered from real output: Qwen3-VL-2B labelled a tall water bottle a "mug".
FUZZY_TYPE_GROUPS = [
    {"bottle", "tumbler", "mug", "cup"},
    {"charger", "cable", "power bank"},
]

# NOTE for the team: "notebook" is deliberately ambiguous (laptop vs paper
# notebook). Keep it that way - it is a free source of genuinely hard queries
# for the demo and the evaluation set.

OBJECT_TYPES = sorted(OBJECT_SYNONYMS.keys())

_SYNONYM_LOOKUP = {
    syn.lower(): canonical
    for canonical, syns in OBJECT_SYNONYMS.items()
    for syn in syns
}

# --- attribute keys the VLM should try to fill ------------------------------
ATTRIBUTE_KEYS = ["color", "material", "size"]

MATERIALS = ["metal", "plastic", "glass", "fabric", "leather", "paper", "wood", "unknown"]
SIZES = ["small", "medium", "large"]

# Constraint sentinel: the user ruled a value OUT. Stored as ("not", "metal")
# when someone answers "no" to "is it metal?" - which is real information, and
# used to be thrown away.
#
# There is deliberately no "this attribute is absent" sentinel any more. Every
# attribute we still index (colour, material, size) is a property every object
# HAS; a null means "the model could not tell", never "the object lacks it".
# The two attributes where null meant something - brand_or_logo and state -
# were both removed for being unreliable.
NOT = "not"


def negate(value):
    return (NOT, value)


def is_negated(value):
    return isinstance(value, tuple) and len(value) == 2 and value[0] == NOT


def normalize_color(value):
    """'Dark Grey ' -> 'black'.  Unknown values pass through lowercased."""
    if not value:
        return None
    v = str(value).strip().lower()
    v = COLOR_ALIASES.get(v, v)
    return v


def normalize_object_type(value):
    """'water bottle' -> 'bottle'.  Unknown values pass through lowercased."""
    if not value:
        return None
    v = str(value).strip().lower()
    if v in _SYNONYM_LOOKUP:
        return _SYNONYM_LOOKUP[v]
    # try the longest matching synonym contained in the string
    best = None
    for syn, canonical in _SYNONYM_LOOKUP.items():
        if syn in v and (best is None or len(syn) > len(best[0])):
            best = (syn, canonical)
    return best[1] if best else v


def _padded(text):
    """Lowercase, punctuation -> spaces, wrapped in spaces.

    Without this, "Where is my bottle?" never matches " bottle " and the parser
    silently finds no target. Learned the hard way; keep it.
    """
    cleaned = "".join(ch if ch.isalnum() else " " for ch in str(text).lower())
    return " " + " ".join(cleaned.split()) + " "


def find_object_types(text):
    """Return every canonical object type mentioned in a free-text string,
    longest match first so 'notebook computer' beats 'notebook'.

    Note what this does NOT do: it never parses grammar. "Where is my laptop?",
    "where my laptop" and "yo where'd my macbook go" all work for the same
    reason - everything except the known noun is thrown away. That is why
    phrasing is irrelevant and why an unknown word is fatal.
    """
    t = _padded(text)
    hits = []
    for syn in sorted(_SYNONYM_LOOKUP, key=len, reverse=True):
        if f" {syn} " in t or f" {syn}s " in t:
            canonical = _SYNONYM_LOOKUP[syn]
            if canonical not in hits:
                hits.append(canonical)
            t = t.replace(f" {syn} ", " ").replace(f" {syn}s ", " ")

    if hits:
        return hits

    # Nothing matched on word boundaries. If the user typed the whole thing as
    # one run-on token ("Whereismylaptop"), fall back to substring search.
    # ONLY in that case - doing this generally would match "pen" inside "open"
    # and "key" inside "keyboard".
    single_token = t.strip()
    if single_token and " " not in single_token and len(single_token) > 6:
        for syn in sorted(_SYNONYM_LOOKUP, key=len, reverse=True):
            if len(syn) >= 5 and syn in single_token:
                return [_SYNONYM_LOOKUP[syn]]
    return hits


def find_colors(text):
    """Return every colour mentioned in a free-text string."""
    t = _padded(text)
    hits = []
    for name in sorted(list(COLOR_ALIASES) + COLORS, key=len, reverse=True):
        if f" {name} " in t:
            c = normalize_color(name)
            if c not in hits:
                hits.append(c)
            t = t.replace(f" {name} ", " ")
    return hits
