"""The interactive agent: Search -> Reason -> Clarify -> Find.

KEY DESIGN DECISION - the ambiguity decision is NOT made by an LLM.

"Are these candidates ambiguous?" is a counting question, and "what should I ask
next?" is an information-gain question. Both are computed here in plain Python.
That means the clarification behaviour is deterministic, unit testable, and
explainable in the report - which is exactly the part of this project that
distinguishes us from plain CLIP retrieval.

The loop, matching our flowchart:

    query -> candidates -> ambiguous? --no--> answer
                              |
                             yes
                              v
                        ask best question
                              v
                      user reply -> add constraint
                              v
                      (back to ambiguous?)

with a hard cap on turns so we can never loop forever.
"""

from dataclasses import dataclass, field
import re

from .query_parser import parse
from .retrieval import confidence_band, find_candidates
from .spatial import CAMERA_INVARIANT, VIEW_DEPENDENT, find_predicate
from .vocab import MATERIALS, SIZES, find_colors, find_object_types, is_negated, negate

# Order matters: we prefer to ask about attributes a human can answer instantly.
# Removed over two rounds of looking at real output:
#   brand_or_logo - the 2B model hallucinated logos, so the agent asked "was it
#                   the one with the white logo?" about a bottle with no logo.
#   state         - meaningless for most objects. It reported a phone as
#                   "closed", and "was your backpack open?" is not a question
#                   anyone can answer about something they lost.
# An attribute we cannot trust is worse than no attribute: it costs a turn.
# Location is askable, but lower-priority than colour: people often remember a
# room, and sometimes two candidates share every visible attribute.
# "near" is a different kind of question again: unlike material, it is not a
# VLM attribute guess at all - it's pure box geometry computed once in
# spatial.py, so it is at least as trustworthy as colour, and (like location)
# something people genuinely remember ("it was next to my laptop").
ASKABLE_KEYS = ["color", "near", "size", "material", "location"]

# Tuning knob. Question choice = (how well the attribute splits the candidates)
# x (how reliably a human can answer it). People remember colour far better than
# they remember what something was made of, so material is discounted. "near"
# sits just under colour: concrete and memorable, but slightly harder to turn
# into a clean multiple-choice question than a colour name.
KEY_PRIOR = {"color": 1.3, "near": 1.1, "size": 0.9, "location": 0.8, "material": 0.7}

# Sentinel for "the pending question is a pick-from-photos", not an attribute.
PICK_KEY = "__pick__"

# More than this and a gallery stops helping - nobody scans nine photos.
MAX_TO_SHOW = 4

ORDINALS = ("first", "second", "third", "fourth")

# Openers that mark a sentence as a fresh search rather than an answer.
QUESTION_OPENERS = ("where", "find", "is there", "show", "which", "what about",
                    "어디", "찾아")

KEY_QUESTIONS = {
    "color": "What colour is it?",
    "near": "Was it near anything you remember?",
    "size": "Roughly how big is it?",
    "material": "What is it made of?",
    "location": "Which room or area was it in?",
}

QUESTION_VARIANTS = {
    "color": (
        "What colour is yours?",
        "Do you remember its colour?",
        "Which colour matches your item?",
    ),
    "near": (
        "Was it near anything you remember?",
        "Do you remember what it was next to?",
        "Was it close to something specific?",
    ),
    "size": (
        "Roughly how big was it?",
        "Would you call it small, medium, or large?",
        "Which size sounds right for it?",
    ),
    "material": (
        "What was it made of?",
        "Do you remember the material?",
        "Which material fits your item?",
    ),
    "location": (
        "Which location sounds right?",
        "Do you remember the room or area?",
        "Where do you think you left it?",
    ),
}


@dataclass
class Reply:
    kind: str                      # answer | question | none_found | giveup
    text: str
    candidates: list = field(default_factory=list)
    asked_key: str = None
    options: list = field(default_factory=list)
    confidence: str = None


def _value_of(cand, key):
    """Return the indexed object attribute used for clarification."""
    if key == "location":
        return cand["scene"].get("location")
    if key == "near":
        # Only SAFE_PREDICATES (camera-invariant) - a candidate question must
        # be answerable regardless of which angle the photo was taken from.
        # "left of the laptop in this photo" is not a fair thing to ask
        # someone to confirm from memory.
        anchor, _ = nearest_anchor_type(cand["scene"], cand["object"],
                                        predicate_order=SAFE_PREDICATES)
        return anchor
    return cand["object"].get("attributes", {}).get(key)


def _distinct(candidates, key):
    """Distinct non-null values of an attribute across candidates."""
    values = []
    for cand in candidates:
        value = _value_of(cand, key)
        if value and value not in values:
            values.append(value)
    return values


def _split_quality(candidates, key):
    """How well does this attribute divide the candidate set?

    1 - (largest group / total). 0 means the attribute is useless (everything
    shares one value); higher means asking about it eliminates more candidates.
    Candidates with a null value count against the attribute, because we cannot
    filter on something the indexer never saw.
    """
    counts = {}
    unknown = 0
    for cand in candidates:
        value = _value_of(cand, key)
        if not value:
            unknown += 1        # we cannot filter on what the model never saw
        else:
            counts[value] = counts.get(value, 0) + 1
    if len(counts) < 2:
        return 0.0
    total = len(candidates)
    largest = max(counts.values())
    return (1.0 - largest / total) * (1.0 - unknown / total)


def choose_question(candidates, already_asked):
    """Pick the attribute that eliminates the most candidates. Returns
    (key, options) or (None, []) if nothing useful is left to ask."""
    best_key, best_score = None, 0.0
    for key in ASKABLE_KEYS:
        if key in already_asked:
            continue
        score = _split_quality(candidates, key) * KEY_PRIOR.get(key, 1.0)
        if score > best_score:
            best_key, best_score = key, score

    if best_key is None:
        return None, []
    return best_key, _distinct(candidates, best_key)


def format_question(key, options, candidate_count, turn):
    """Build a varied clarification while keeping the indexed choices explicit."""
    choices = " or ".join(str(option) for option in options)
    prefix = f"I found {candidate_count} possible matches."
    if len(options) == 1:
        return f"{prefix} Was it the one with {options[0]}?"
    prompt = QUESTION_VARIANTS.get(key, (KEY_QUESTIONS.get(key, key),))
    return f"{prefix} {prompt[turn % len(prompt)]} ({choices})"


# Preference order when several relations could be reported for the same
# object: the most specific/informative predicate wins, falling back to the
# loosest ("near") only if nothing tighter is on record. Shared by
# describe_place() (building the answer sentence) and _value_of()'s "near"
# case (picking what to ask about) so both agree on which neighbour matters.
# These are ORDERED tuples (unlike spatial.py's CAMERA_INVARIANT/
# VIEW_DEPENDENT sets) because the search order is the whole point here - but
# they must contain the exact same predicates spatial.py classifies, so that
# invariant is asserted once at import time rather than left to silently
# drift if spatial.py's classification ever changes.
# Most specific/informative first: resting-on and containment are the
# strongest, unambiguous claims; "overlapping"/"same_surface" are visible but
# looser; "near" (pure distance) is the loosest and always comes last.
SAFE_PREDICATES = ("on", "inside", "beside", "overlapping", "same_surface", "near")
FALLBACK_PREDICATES = ("left_of", "right_of", "above", "below")  # view-dependent
assert set(SAFE_PREDICATES) == CAMERA_INVARIANT
assert set(FALLBACK_PREDICATES) == VIEW_DEPENDENT


def nearest_anchor_type(scene, obj, anchor_filter=None, predicate_order=None):
    """The type of the nearest OTHER object `obj` has a relation to in this
    scene, preferring SAFE_PREDICATES (true from any camera angle) before
    FALLBACK_PREDICATES. Returns (anchor_type, predicate) or (None, None).

    `anchor_filter` restricts the search to relations pointing at a specific
    object type (e.g. the user already said "beside the laptop").
    `predicate_order` overrides which predicates to search, in order - used
    to search only SAFE_PREDICATES when the result will be used to build a
    clarifying QUESTION (a "left of the laptop in this photo" answer is not a
    fair thing to ask someone to confirm, since it depends on a camera angle
    they never chose).
    """
    types = {o["id"]: o.get("type") for o in scene.get("objects", [])}
    mine = obj.get("type")
    orders = [predicate_order] if predicate_order else [SAFE_PREDICATES, FALLBACK_PREDICATES]

    for predicates in orders:
        for predicate in predicates:
            for triple in scene.get("relations", []):
                if triple["subject"] != obj["id"] or triple["predicate"] != predicate:
                    continue
                anchor = types.get(triple["object"])
                if not anchor or anchor == mine:
                    continue
                if anchor_filter and anchor != anchor_filter:
                    continue
                return anchor, predicate
    return None, None


def describe_place(scene, obj, prefer_anchor=None):
    """'in the office, beside the laptop' - built from indexed relations,
    never from the model's own words.

    If the user's query named an anchor ("next to the laptop"), we describe the
    result relative to THAT object. Answering "beside the bottle" when they
    asked about the laptop is technically true and completely unhelpful.
    """
    where = f"in the {scene.get('location', 'scene')}"

    def phrase(anchor, predicate):
        verb = "on top of" if predicate == "on" else predicate.replace("_", " ")
        if predicate in VIEW_DEPENDENT:
            # Only true of the photo we are about to show, so say so.
            # "left of the laptop" flips if you walk around the desk.
            return f"{where}, {verb} the {anchor} in this photo"
        return f"{where}, {verb} the {anchor}"

    if prefer_anchor:
        anchor, predicate = nearest_anchor_type(scene, obj, anchor_filter=prefer_anchor)
        if anchor:
            return phrase(anchor, predicate)

    anchor, predicate = nearest_anchor_type(scene, obj)
    return phrase(anchor, predicate) if anchor else where


def candidate_summary(cand, parsed=None):
    """Compact distinguishing text for comparison lists."""
    obj = cand["object"]
    attrs = obj.get("attributes", {})
    bits = [attrs.get("color"), attrs.get("material"), attrs.get("size"),
            obj.get("type"), cand["scene"].get("location")]
    place = describe_place(cand["scene"], obj, parsed.anchor_type if parsed else None)
    relation = place.split(",", 1)[1].strip() if "," in place else None
    if relation:
        bits.append(relation)
    unique = []
    for bit in bits:
        if bit and bit not in unique:
            unique.append(str(bit))
    return ", ".join(unique[:5])


def comparison_text(candidates, parsed=None, limit=3):
    """Short visual/text comparison for ambiguous candidate sets."""
    lines = []
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for idx, cand in enumerate(candidates[:limit]):
        lines.append(f"Candidate {labels[idx]}: {candidate_summary(cand, parsed)}")
    return "\n".join(lines)


def describe_object(obj, candidate=None, wanted=None):
    """"the blue laptop". Colour plus type, nothing else.

    Deliberately short. Every extra adjective is another chance to describe the
    object in a way the owner does not recognise, and the photo we show
    alongside carries far more information than any sentence could.

    USE THE OWNER'S WORD WHEN WE HAVE SEEN IT. The MacBook read as "blue" from
    the front and "gray" from the side; the vote picked gray, so someone who
    asked for a blue laptop got the right object described back to them as grey
    and reasonably concluded it was the wrong one. If the user says blue and
    some view agrees, we say blue - we have no grounds to correct them.
    """
    attributes = dict(obj.get("attributes") or {})
    observed = (candidate or {}).get("observed") or {}

    for key, value in (wanted or {}).items():
        if not value or isinstance(value, tuple):
            continue
        if value in (observed.get(key) or {}):
            attributes[key] = value

    bits = [attributes.get("color"), obj.get("type")]
    return " ".join(b for b in bits if b)


def _unindexed_clause(unindexed):
    """Turn an unindexed_photo_count() result into the trailing sentence of a
    not-found message. unindexed_photo_count() still tells the caller whether
    there are photos on disk the index never saw - but the reply to the user
    stays the same generic hedge either way; we don't expose indexing/ops
    state (counts, filenames) to the person searching for their item.
    """
    return "It may be outside the photographed area or hidden from view."


def describe_not_found(parsed, scene_count, unindexed=None):
    """Explain that absence from the indexed photos is not proof of loss.

    `scene_count` (a plain int, or a dict from SceneIndex.coverage()) is
    accepted for backward compatibility with existing callers but no longer
    appears in the message - keep the reply short instead of reciting photo/
    candidate counts.

    `unindexed` (SceneIndex.unindexed_photo_count()'s result) is accepted for
    backward compatibility but no longer changes the reply - see
    _unindexed_clause().
    """
    request = parsed.describe()
    return f"I couldn't find a {request}. {_unindexed_clause(unindexed)}"


class Session:
    """One conversation. Create it per user query."""

    def __init__(self, index, config=None, verifier=None):
        self.index = index
        # Optional query-time VLM re-check. See src/verify.py. Off unless the
        # caller builds one, so the agent stays instant and fully testable.
        self.verifier = verifier
        config = config or {}
        cfg = config.get("agent", {})
        self.max_turns = cfg.get("max_clarify_turns", 3)
        self.top_k_scenes = cfg.get("top_k_scenes", 5)
        # Photos in one location folder are treated as views of one place.
        self.merge_across_views = cfg.get("merge_across_views", True)
        # Used only to make a "not found" answer honest about whether there
        # are photos on disk that simply have not been indexed yet - see
        # describe_not_found(). None (e.g. in tests with a bare config dict)
        # just means that check is skipped, same as before this existed.
        self.images_root = config.get("paths", {}).get("images")
        self.reset()

    def reset(self):
        self.parsed = None
        self.constraints = {}
        self.asked = []
        self.pending_key = None
        self.pending_options = []
        self.picked = None
        self.rejected_shortlist = False
        self.offered_pick = False      # show the photos once, not in a loop
        self.turns = 0
        self.candidates = []
        self.verified_ids = set()      # only pay for each VLM check once
        self.rejected_ids = set()
        self.confirmed_low = False

    # ------------------------------------------------------------------ #
    def start(self, query):
        self.reset()
        self.parsed = parse(query, known_locations=self.index.locations())
        if not self.parsed.target_type:
            return Reply("none_found",
                         "I'm not sure what object you're looking for. "
                         "Could you name the item? For example: \"my black bottle\".")
        return self._advance()

    def reply(self, user_text):
        if self.parsed is None:
            return self.start(user_text)
        # A new question beats a pending one. Without this, typing "where is my
        # bottle" while the agent is asking "which laptop is yours?" gets eaten
        # as a failed answer, and the agent re-runs the OLD laptop search - so
        # the user asks about a bottle and gets shown laptops.
        if self._looks_like_a_new_question(user_text):
            return self.start(user_text)
        if self.pending_key:
            self._apply_answer(user_text)
        return self._advance()

    def _looks_like_a_new_question(self, text):
        """Is this a fresh search, or an answer to what we just asked?

        Answers are colours, materials, sizes, room names and numbers - none of
        which name an object... except the "near" question (and volunteering a
        neighbour while answering something else), whose answer NAMES an
        object on purpose ("near the laptop"). Naming a different object than
        the one being searched for is otherwise the signal for a fresh
        search, so two carve-outs are needed before that check:

          1. The mentioned object is literally one of the choices we just
             offered ("was it near the laptop or the backpack?" -> "laptop").
          2. The text is a relation phrase with the searched-for object left
             unstated ("near my laptop") - query_parser folds the anchor into
             target_type when there is nothing on the left of the relation
             word, since it cannot tell that apart from "find the bottle near
             the laptop" typed as a sentence fragment. Mid-conversation, a
             bare relation phrase is answering, not restarting.

        Both carve-outs are checked ONLY when the text does not itself open
        like a fresh search (QUESTION_OPENERS). Without that ordering, mid
        "was it near the laptop or the backpack?", a genuinely new and
        unrelated "where is my laptop" would match carve-out 1 (its
        target_type, "laptop", happens to be one of the offered near-options)
        and get silently eaten as an answer to the OLD bottle search instead
        of starting the new laptop one - an explicit "where"/"find" opener
        must always win.

        This does not fully resolve every ambiguous case (e.g. "is my charger
        near anything" has no recognised opener and no target-naming subject,
        so it still reads as an answer) - same known-limitation trade-off
        query_parser.py documents for itself: a keyword heuristic fails
        visibly sometimes rather than guessing with a real parser.

        The carve-outs apply for any pending_key, not just "near" - options
        for the other keys (colours, sizes, materials, or object ids for
        PICK_KEY) don't look like object-type strings, so this is a no-op for
        them in practice, but it is not hard-restricted to "near" on purpose:
        volunteering a neighbour while answering an unrelated question (e.g.
        "black, near my laptop" while asked about colour) must still land as
        an answer, not a restart.
        """
        asked = parse(text, known_locations=self.index.locations())
        if not asked.target_type:
            return False
        starts_like_a_question = text.strip().lower().startswith(QUESTION_OPENERS)
        if self.pending_key and not starts_like_a_question:
            if asked.target_type in self.pending_options:
                return False
            # We just asked which object it was next to. The options we listed
            # are the neighbours WE know about, not the list of things that can
            # be next to something - so "phone", when we offered "umbrella or
            # charger", is an answer, not a new search. Reading it as a new
            # search is the worst outcome available: the bottle the person is
            # looking for is silently abandoned and they are shown a phone.
            # _record_answer already accepts any known type here.
            if self.pending_key == "near":
                return False
            if asked.predicate is None:
                _, relation_phrase = find_predicate(text)
                if relation_phrase:
                    return False
        if asked.target_type != self.parsed.target_type:
            return True
        return starts_like_a_question

    # ------------------------------------------------------------------ #
    def _ask_to_pick(self):
        """Words have run out - show the photos and let the user point.

        This is the agent choosing a different ACTION, not a different question.
        Two laptops that the indexer described identically cannot be separated
        by anything we could ask; a human glancing at two pictures separates
        them instantly. Falling back to "they look the same to me, good luck"
        wastes the one channel that still works.
        """
        shortlist = self.candidates[:MAX_TO_SHOW]
        self.turns += 1
        self.offered_pick = True
        self.pending_key = PICK_KEY
        self.pending_options = [c["object"]["id"] for c in shortlist]
        return Reply(
            "choose",
            f"I found {len(self.candidates)} that I cannot tell apart from their "
            f"descriptions. Here they are - which one is yours? "
            f"(1-{len(shortlist)}, or \"none\")",
            candidates=shortlist,
            asked_key=PICK_KEY,
            options=self.pending_options)

    def _give_up(self):
        """Stop asking, show the best few, and say plainly that it is a guess."""
        return Reply(
            "giveup",
            f"I still see {len(self.candidates)} possible matches"
            f" Here are the most likely ones:\n"
            f"{comparison_text(self.candidates, self.parsed)}",
            candidates=self.candidates[:3],
            confidence="medium")

    def _pick_from_shortlist(self, text, options):
        """"2", "the second one", "none". Returns True if the session resolved."""
        cleaned = text.strip().lower()
        if cleaned.startswith(("none", "neither", "no ", "없", "아니")):
            # Not "I cannot tell" - "it is not one of these". Asking again would
            # show the same photos, so stop and be honest about what we checked.
            self.rejected_shortlist = True
            return False

        chosen = None
        digits = "".join(ch if ch.isdigit() else " " for ch in cleaned).split()
        if digits:
            index = int(digits[0]) - 1
            if 0 <= index < len(options):
                chosen = options[index]
        else:
            for position, word in enumerate(ORDINALS):
                if word in cleaned and position < len(options):
                    chosen = options[position]
                    break

        if chosen:
            self.picked = chosen
            return True
        return False

    def _wanted(self):
        """Everything the user has actually told us about the object."""
        merged = dict(self.parsed.attributes if self.parsed else {})
        merged.update(self.constraints)
        return merged

    def _verify(self):
        """Ask the VLM to look again, if the caller wired one up.

        Runs BEFORE we decide whether things are ambiguous, so a candidate the
        indexer got wrong never becomes a clarification question. Each object is
        checked at most once per conversation.
        """
        if not self.verifier:
            return
        from .verify import needs_verification

        fresh = [c for c in self.candidates
                 if c["object"]["id"] not in self.verified_ids and needs_verification(c)]
        if not fresh:
            return

        description = " ".join(filter(None, [
            self.parsed.attributes.get("color"), self.parsed.target_type]))
        survivors = self.verifier(fresh, description)
        for cand in fresh:
            self.verified_ids.add(cand["object"]["id"])

        kept = {c["object"]["id"] for c in survivors}
        self.candidates = [c for c in self.candidates if c["object"]["id"] in kept]

    def _volunteer(self, text, skip):
        """Grab anything ELSE the user mentioned while answering a question.

        People answer however they feel like: asked about the colour, they say
        "it's black, and it's made of metal". The metal is real information -
        take it for the attribute it names even though we did not ask, so it
        never costs the user a second turn. Deliberately limited to the
        askable attributes: location stays answer-only (see ASKABLE_KEYS).
        """
        if skip != "color" and "color" not in self.constraints:
            colors = find_colors(text)
            if colors:
                self.constraints["color"] = colors[0]

        words = set(re.findall(r"[a-z]+", text))
        if skip != "size" and "size" not in self.constraints:
            for size in SIZES:
                if size in words:
                    self.constraints["size"] = size
                    break
        if skip != "material" and "material" not in self.constraints:
            for material in MATERIALS:
                if material != "unknown" and material in words:
                    self.constraints["material"] = material
                    break

        # "near" is deliberately more conservative than the others: only take
        # it if the text has an explicit CAMERA-INVARIANT relation word
        # ("near my laptop"), not just any object type mentioned in passing
        # ("it's a metal charger" should not volunteer "near": "charger" -
        # that names the object itself, not a neighbour). Restricted to
        # CAMERA_INVARIANT specifically (not any relation word find_predicate
        # recognises): _near_type only ever checks camera-invariant relations,
        # so volunteering e.g. "left of my laptop" (a VIEW_DEPENDENT relation)
        # would record a constraint retrieval can never satisfy even when the
        # user's statement is true - a candidate that really is left of the
        # laptop in this photo would be filtered out for failing a check it
        # was never able to pass. Also excludes the target's own type, same
        # as the direct "near" answer handler below.
        if skip != "near" and "near" not in self.constraints:
            predicate, phrase = find_predicate(text)
            if phrase and predicate in CAMERA_INVARIANT:
                anchor_types = [t for t in find_object_types(text)
                                if t != self.parsed.target_type]
                if anchor_types:
                    self.constraints["near"] = anchor_types[0]

    def _apply_answer(self, user_text):
        key, options = self.pending_key, self.pending_options
        self.pending_key, self.pending_options = None, []
        self.asked.append(key)

        if key == PICK_KEY:
            self._pick_from_shortlist(user_text, options)
            return

        text = user_text.strip().lower()
        if key == "confirm_candidate":
            if text.startswith(("y", "yeah", "yep", "yes", "correct", "that's it",
                                "그", "네", "응", "맞")):
                self.confirmed_low = True
            elif self.candidates:
                self.rejected_ids.add(self.candidates[0]["object"]["id"])
            return

        if text in ("i don't know", "dont know", "no idea", "not sure", "idk",
                    "몰라", "모르겠어", ""):
            return  # attribute burned, no constraint added

        # Users answer whatever they feel like. Take any attribute they mention
        # that we were NOT asking about, instead of discarding a good clue.
        self._volunteer(text, skip=key)

        words = set(re.findall(r"[a-z]+", text))

        if key != "location" and not self.parsed.location:
            room = parse(text, known_locations=self.index.locations()).location
            if room:
                self.parsed.location = room

        if key == "color":
            colors = find_colors(text)
            for color in colors:
                if color in options:
                    self.constraints["color"] = color
                    return
            for option in options:            # bare "black"
                if option in text:
                    self.constraints["color"] = option
                    return
            if colors:
                # The user named a colour we never indexed. Record it anyway so
                # _advance hits the "nothing matches, here's the closest" path
                # instead of pretending the answer told us nothing.
                self.constraints["color"] = colors[0]
            return

        # Accept any KNOWN size/material, not just the values we offered:
        # "small" is a fine answer to "medium or large?".
        if key == "size":
            for size in SIZES:
                if size in words:
                    self.constraints["size"] = size
                    return
        if key == "material":
            for material in MATERIALS:
                if material != "unknown" and material in words:
                    self.constraints["material"] = material
                    return

        if key == "location":
            for option in options:
                if str(option).lower() in text or text in str(option).lower():
                    self.constraints["location"] = option
                    return
            return

        # Accept any known object type as the neighbour, not just the ones we
        # offered - "was it near the laptop or the backpack?" answered with
        # "it was near my charger" is still real information. Excludes the
        # searched-for object's own type: "my headphones were near the pen"
        # must pick "pen", not re-match "headphones" against itself just
        # because it was mentioned again.
        if key == "near":
            mentioned = [t for t in find_object_types(text) if t != self.parsed.target_type]
            for anchor_type in mentioned:
                if anchor_type in options:
                    self.constraints["near"] = anchor_type
                    return
            if mentioned:
                self.constraints["near"] = mentioned[0]
                return

        # Only one value to offer, so it became a yes/no question.
        # "No" is just as informative as "yes" - it rules that value out.
        if len(options) == 1:
            if text.startswith(("y", "yeah", "yep", "그", "네", "응", "맞")):
                self.constraints[key] = options[0]
            elif text.startswith(("n", "nope", "아니", "없")):
                self.constraints[key] = negate(options[0])
            return

        for option in options:
            if str(option).lower() in text or text in str(option).lower():
                self.constraints[key] = option
                return

    # ------------------------------------------------------------------ #
    def _advance(self):
        if self.rejected_shortlist:
            self.rejected_shortlist = False
            places = sorted({c["scene"].get("location") for c in self.candidates})
            return Reply(
                "none_found",
                f"Then I don't have it. I checked every photo I have of "
                f"{' and '.join(places)} and those were all the "
                f"{self.parsed.target_type}s in them.",
                candidates=[])

        if self.picked:
            chosen = [c for c in self.candidates if c["object"]["id"] == self.picked]
            self.picked = None
            if chosen:
                cand = chosen[0]
                return Reply(
                    "answer",
                    f"That one is {describe_place(cand['scene'], cand['object'], self.parsed.anchor_type)}.",
                    candidates=chosen)

        self.candidates = find_candidates(
            self.index, self.parsed, self.constraints, top_k_scenes=self.top_k_scenes,
            merge_across_views=self.merge_across_views)
        if self.rejected_ids:
            self.candidates = [c for c in self.candidates
                               if c["object"]["id"] not in self.rejected_ids]
        self._verify()

        if not self.candidates:
            if self.constraints:
                # We over-filtered. The newest constraint is usually the
                # culprit (the answer to the last question), but a volunteered
                # value can be inserted before it - so try dropping constraints
                # one at a time, newest first, until something survives again.
                for dropped in reversed(list(self.constraints)):
                    value = self.constraints[dropped]
                    trial = dict(self.constraints)
                    trial.pop(dropped)
                    refound = find_candidates(
                        self.index, self.parsed, trial,
                        top_k_scenes=self.top_k_scenes,
                        merge_across_views=self.merge_across_views)
                    if refound:
                        self.constraints = trial
                        self.candidates = refound
                        label = (f"not {value[1]}" if is_negated(value)
                                 else value)
                        return Reply(
                            "giveup",
                            f"I couldn't find anything matching \"{label}\". "
                            f"Here is the closest I have for \"{self.parsed.describe()}\":",
                            candidates=refound[:3])
            unindexed = (self.index.unindexed_photo_count(self.images_root)
                        if self.images_root else None)
            return Reply("none_found",
                         describe_not_found(self.parsed, self.index.coverage(), unindexed))

        if len(self.candidates) == 1:
            cand = self.candidates[0]
            band = confidence_band(cand.get("score", 0.5))
            if band == "low" and not self.confirmed_low:
                self.pending_key, self.pending_options = "confirm_candidate", ["yes", "no"]
                return Reply(
                    "question",
                    "I found one low-confidence possible match, so I don't want to "
                    f"guess. Is this your {self.parsed.describe()}?\n"
                    f"{comparison_text([cand], self.parsed)}",
                    candidates=self.candidates,
                    asked_key="confirm_candidate",
                    options=["yes", "no"],
                    confidence=band)
            prefix = "Found it" if band == "high" else "I think I found it"
            uncertainty = "" if band == "high" else " The match is medium-confidence, so please check the image."
            return Reply(
                "answer",
                f"{prefix} - the {describe_object(cand['object'])} is "
                f"{describe_place(cand['scene'], cand['object'], self.parsed.anchor_type)}."
                f"{uncertainty}",
                candidates=self.candidates,
                confidence=band)

        if self.turns >= self.max_turns:
            # Out of questions, but not out of options: the photos are already
            # on screen, so let the user point at one instead of ending on
            # "here are three, good luck". Offered once - if it is refused we
            # fall through to the honest give-up below.
            if not self.offered_pick:
                return self._ask_to_pick()
            return self._give_up()

        key, options = choose_question(self.candidates, self.asked)
        if key is None:
            return self._ask_to_pick() if not self.offered_pick else self._give_up()

        self.turns += 1
        self.pending_key, self.pending_options = key, options

        question = format_question(key, options, len(self.candidates), self.turns - 1)
        question = f"{question}\n{comparison_text(self.candidates, self.parsed)}"

        return Reply("question", question, candidates=self.candidates,
                     asked_key=key, options=options,
                     confidence="medium")
