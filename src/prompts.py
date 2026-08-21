"""Prompts for the visual understanding stage.

TUNING LOG - keep this updated, it is report material.

v1: long prompt, full type/colour lists inline, example showed a bottle.
    Qwen3-VL-2B returned only 2 objects (both bottles) on a desk photo
    containing ~10. It anchored on the example and stopped early.

v5: the example objects are now PLACEHOLDERS, not plausible ones.
    On a real desk photo the model copied the v4 example verbatim - same type,
    same colour, same confidence, same bounding box to the pixel - and did it
    for two of the eight objects it "found". A believable example is an
    invitation to reuse it instead of looking. Placeholder coordinates cannot
    be mistaken for a real answer, and src/indexer.py drops anything that still
    matches them.

v4: dropped "state" and redefined "size".
    "state" (open/closed/plugged in) is meaningless for most things a person
    loses - the model confidently reported a phone as "closed" - so it burned
    one of the agent's three questions on something the user cannot answer.
    "size" used to be defined relative to the other objects in the frame, which
    made the same bottle "medium" from one angle and "small" from another. It
    is now defined absolutely, in terms a person can actually answer.

v3: dropped "brand_or_logo". On real photos the model read a lululemon bottle
    as "lifeforce" and invented logos that were not there. A hallucinated
    attribute is worse than a missing one: it splits one real object into two
    across camera angles, and it makes the agent ask a question the user cannot
    answer.

v2: three changes, all aimed at RECALL.
    - "count first, then list exactly that many" - forcing the model to commit
      to a number before enumerating measurably increases how many it finds.
    - objects come BEFORE caption in the output, so if generation is ever cut
      short we lose the caption instead of the objects.
    - the example lists two DIFFERENT object types, so nothing anchors the
      model on one category.

Bounding boxes are requested normalised to 0..1000. Verified correct against a
4032x3024 photo, so keep `coord_mode: norm1000` in config.yaml.
"""

from .vocab import COLORS, MATERIALS, SIZES

# Short, concrete category list. The full OBJECT_TYPES list was too long and
# read as noise to a 2B model; vocab.normalize_object_type() maps whatever the
# model says onto our canonical types afterwards anyway.
_EXAMPLES_OF_OBJECTS = (
    "bottle, cup, mug, backpack, bag, laptop, tablet, phone, charger, power bank, "
    "cable, headphones, earbuds, keys, wallet, ID card, book, notebook, pen, "
    "glasses, umbrella, mouse, watch, fan, tissue"
)

SCENE_EXTRACTION_PROMPT = f"""You are the visual understanding module of a Lost & Found agent.

TASK: find EVERY portable personal object in this photo - anything a person could pick up, carry away, or lose.

Objects of interest include: {_EXAMPLES_OF_OBJECTS}.

IGNORE: furniture, chairs, desks, tables, walls, floors, ceilings, doors, windows, blinds, and people.

IMPORTANT - be exhaustive:
- A photo of a desk or room usually contains 6 to 15 such objects.
- First COUNT every object you can see, then list exactly that many.
- Look at the whole frame, including small items near the edges and in the background.
- Small dark objects on dark surfaces are easy to miss. Look again before finishing.
- If two objects are the same kind, list them SEPARATELY. Three bottles = three entries.
- Do NOT invent objects you cannot actually see.

For each object give:
- "type": a single lowercase noun
- "color": one of {COLORS}
- "material": one of {MATERIALS}
- "size": one of {SIZES}, judged in absolute terms, NOT relative to this photo:
    "small"  = fits in a trouser pocket (phone, keys, charger, wallet, pen, card)
    "medium" = carried comfortably in one hand (bottle, mug, book, tablet)
    "large"  = needs two hands, or is worn/carried on the body (laptop, backpack, umbrella)
- "bbox": [x1, y1, x2, y2] as INTEGERS on a 0-1000 scale, where 0,0 is the
  top-left corner of the image and 1000,1000 is the bottom-right corner
- "confidence": 0.0 to 1.0

Return ONLY valid JSON, no markdown fences, no commentary, in exactly this shape:

{{"n_objects": 8,
  "objects": [
    {{"type": "<noun>", "color": "<colour>", "material": "<material>",
      "size": "<size>", "bbox": [111, 222, 333, 444], "confidence": 0.9}},
    {{"type": "<noun>", "color": "<colour>", "material": "<material>",
      "size": "<size>", "bbox": [555, 666, 777, 888], "confidence": 0.9}}
  ],
  "caption": "one short sentence describing the scene"}}

The list must contain n_objects entries."""


VERIFY_CANDIDATE_PROMPT = """You are the visual verification module of a Lost & Found agent.

The user is looking for: "{description}"

Look at the highlighted region of this photo and answer honestly.

Return ONLY valid JSON:
{{"match": true or false, "reason": "one short sentence", "confidence": 0.0 to 1.0}}"""
