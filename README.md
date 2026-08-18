# AI Lost & Found

A multimodal agent that finds personal belongings in photos of rooms, and — this
is the point — **asks a question instead of guessing** when the request is
ambiguous.

```
You:   Where is my bottle?
Agent: I found 6 possible matches. Which room was it in - office or classroom or meeting?
You:   office
Agent: I found 3 possible matches. What colour is yours - black or blue or white?
You:   black
Agent: Found it - the black bottle is in the office, beside the laptop.
```

(That is a real transcript against the dummy index — reproduce it with the three
commands below.)

The photo opens automatically with the answer boxed in red.

---

## Start here (3 commands, no model required)

```bash
pip install -r requirements.txt
python scripts/make_dummy_index.py          # fake scenes + placeholder images
python -m src.cli "Where is my bottle?"
```

That works on any laptop with zero downloads. **Nobody on the team should be
blocked waiting for a model.** The fake index has exactly the same schema as the
real one, so retrieval, the agent, the UI and the tests can all be built on day 1
and keep working unchanged when the real photos arrive.

Then:

```bash
python -m pytest -q        # 65 tests
python eval/run_eval.py    # the evaluation table for the report
python -m src.app          # optional Gradio UI (the CLI already shows photos)
```

---

## Pipeline

```
              ┌─────────── done ONCE, offline ───────────┐
photos  ──►   VLM  ──►  objects + attributes + boxes  ──►  scene_index.json
              └──────────────────────────────────────────┘
                                                              │
user query ──► parse ──► rank scenes ──► merge views ◄────────┘
                                              │
                                       filter by constraints
                                              │
                                    (optional) VLM re-check
                                              │
                                       ambiguous?
                                        ╱        ╲
                                      no          yes
                                      │            │
                                   answer     ask best question
                                                   │
                                            user reply → constrain
                                                   │
                                            (back to ambiguous?)   ×3 max
```

---

## Five decisions worth defending in the presentation

### 1. The VLM runs offline, not per query

Calling a local VLM on Top-K scenes at query time costs 10–30 s *per image* on a
laptop. That is unusable live, and it makes multi-turn conversation
non-reproducible because the model's answers drift between runs. We pay that
cost once in `src/indexer.py`; every query afterwards is a dictionary lookup and
returns in milliseconds.

### 2. No embedding pre-filter — and that is a correctness argument, not a speed one

The standard design puts a cheap text-image embedding filter (CLIP) in front of
the VLM, to cut 50 photos down to 3–5 before paying for inference.

**For lost-and-found specifically, that filter is counterproductive.** People
search precisely *because* the object is somewhere they did not expect it. A
filter that ranks rooms by "where is a bottle likely to be" throws away the
stairwell and the car park first — which is exactly where a lost bottle
disproportionately is. Worse, the failure is silent: the user is told "not
found" and has no way to distinguish that from "never looked".

Because we index offline, we can afford to look at **every** object in **every**
photo, so the filter stage simply does not exist. Recall is not traded away.

If this scales to live video, offline indexing stops being possible and a filter
becomes necessary again — see *Roadmap* below.

### 3. Spatial relations are geometry, not model output

We never ask the VLM "is the bottle beside the laptop?" — small VLMs are
confidently wrong about relations. We ask only for boxes, which they are good
at, and compute relations from the coordinates in `src/spatial.py`.

Relations are then split by whether they survive a moving camera:

| Camera-invariant | View-dependent |
| --- | --- |
| `near`, `beside` — proximity holds from any angle | `left_of`, `right_of` — invert if you walk round the desk |
| `on`, `inside` — gravity and containment | `above`, `below` — in a top-down photo this is *depth*, not height |

**Only invariant relations are matched against queries.** When a user says "left
of the laptop" they mean left from a viewpoint we do not know, so we honour the
part we can trust — that the two things are near each other. View-dependent
relations are still used to describe a specific photo, and are labelled as such
("left of the laptop *in this photo*").

### 4. Multi-view merging is count-first, attributes-second

Several photos of one desk must not become "I found 5 bottles" when there are 3.
The obvious approach — merge objects that share colour and material — fails,
because attributes are exactly what the model is unreliable about. The same red
bottle came back as `glass` from the front and `plastic` from the left.

So `src/retrieval.py` does this instead:

1. The view that saw the **most** objects of a type becomes the reference. It
   decides how many objects exist. Views seeing 2 and 3 bottles means 3, never 5.
2. Other views' objects are matched one-to-one onto the reference by appearance.
3. Attributes are decided by **majority vote**, and every value ever observed is
   kept — so the user's answer matches if *any* angle saw it that way.

### 5. The ambiguity decision is Python, not an LLM

"Is this ambiguous?" is counting. "What should I ask next?" is information gain.
`src/agent.py` picks the attribute that best splits the remaining candidates,
weighted by how reliably a human can answer it (people remember colour and which
room far better than they remember material). That makes the clarification
behaviour deterministic, explainable and unit-testable.

---

## Attributes: what survived, and why

The schema was cut down twice after looking at real model output. Both removals
came from the same principle: **an attribute we cannot trust is worse than no
attribute, because it costs one of only three questions.**

| Attribute | Status | Reason |
| --- | --- | --- |
| `color` | kept | most reliable, and what people remember best |
| `material` | kept | moderately reliable |
| `size` | kept, **redefined** | was "relative to other objects in this photo", which made the same bottle `medium` from one angle and `small` from another. Now absolute: pocket / one hand / two hands |
| `brand_or_logo` | **removed** | the model read a lululemon bottle as "lifeforce" and invented logos that were not there. Hallucinated logos also split one object into two across angles |
| `state` | **removed** | meaningless for most objects — it reported a phone as "closed". "Was your backpack open?" is not answerable about something you lost |

`location` was promoted to a first-class question rather than a last resort.
Which room someone left something in is one of the things people remember best,
and it is often the single most discriminating thing we can ask. Doing this
dropped the average number of questions from 1.8 to 1.2.

---

## Optional: query-time VLM verification

The index is built once, offline — which freezes any mistake the model made that
day (a tall water bottle labelled "mug"). `src/verify.py` is the second look:
it crops each candidate and asks the VLM "is this what the user described?"
before the user is ever asked about it.

```bash
python -m src.cli --verify "where is my bottle"
```

**Off by default.** Each check costs a real model call (2–5 s on an 8 GB Mac),
and the fast path should feel instant. It is deliberately conservative: a
candidate survives unless the model rejects it *with confidence*, and
verification can never empty the result set — turning a findable object into
"not found" is the worst outcome a lost-and-found system can produce.

---

## Layout

```
config.yaml                model, paths, thresholds — the only place to edit
src/
  vlm.py                   backend adapter: mlx | transformers | dummy
  prompts.py               scene-extraction prompt + its tuning log
  indexer.py               photos  -> data/scene_index.json     (offline, slow)
  spatial.py               boxes   -> relations                 (pure geometry)
  vocab.py                 controlled colour / object vocabulary
  query_parser.py          text    -> structured query
  retrieval.py             query   -> ranked scenes -> merged candidates
  agent.py                 the clarification state machine
  verify.py                optional query-time VLM re-check
  visualize.py             draw boxes back onto photos          ← run this early
  cli.py                   terminal chat (opens result photos)
  app.py                   Gradio demo
scripts/make_dummy_index.py
eval/run_eval.py           the evaluation deliverable
tests/                     pytest
data/
  images/<location>/<name>.jpg
  scene_index.json         COMMIT THIS — it is the contract between teammates
  ground_truth.json        hand labels, for the evaluation
```

---

## The index schema

Everything downstream depends on this. Agree on it before writing code; change
it only by agreement.

```json
{
  "scene_id": "office_front",
  "location": "office",
  "image_path": "data/images/office/office_front.jpg",
  "width": 4032, "height": 3024,
  "caption": "A desk with a laptop and two bottles.",
  "objects": [
    {
      "id": "office_front_o1",
      "type": "bottle",
      "attributes": {"color": "black", "material": "metal", "size": "medium"},
      "bbox": [800.0, 300.0, 870.0, 520.0],
      "confidence": 0.9
    }
  ],
  "relations": [
    {"subject": "office_front_o1", "predicate": "beside", "object": "office_front_o0"}
  ]
}
```

`bbox` is `[x1, y1, x2, y2]` in **original image pixels**. `relations` are
computed at index time by `spatial.py`. There are exactly three attribute keys —
if you find yourself adding a fourth, read *Attributes* above first.

**One folder = one place.** Every photo in `data/images/office/` is treated as a
different *view* of the same office. If one folder really contains two different
places, split it: `office-deskA/`, `office-deskB/`.

---

## Running the real indexer

```bash
pip install -U mlx-vlm                       # Apple Silicon
# edit config.yaml: backend: mlx
python -m src.indexer --limit 1              # smoke test on ONE image first
python -m src.visualize                      # ← verify the boxes before anything else
python -m src.indexer                        # the whole set
```

`--limit 1` then `visualize` is not optional. If the boxes are not sitting on
the objects, every relation computed from them is wrong, and you will not notice
until day 3. Fix `vlm.coord_mode` in `config.yaml` first:

| value | meaning |
| --- | --- |
| `norm1000` | model reports 0–1000 normalised coords — what our prompt asks for |
| `abs_resized` | model reports pixels of the *resized* image it was shown |

### Swapping models

One line in `config.yaml`:

```yaml
vlm:
  backend: mlx
  model: mlx-community/Qwen3-VL-2B-Instruct-4bit   # 1.78 GB
# model: mlx-community/Qwen2.5-VL-3B-Instruct-4bit # 3.07 GB
# model: mlx-community/Qwen3-VL-8B-Instruct-4bit   # ~6 GB, needs 16 GB+ RAM
```

Then `python -m src.indexer --force`. No code changes anywhere.

**Whoever has the strongest machine should own indexing** and commit the
resulting `scene_index.json`. Everyone else then needs no model at all.
Use **one** model for the whole index — mixing models produces inconsistent
attributes and retrieval silently degrades.

---

## Platform notes

Everything except the VLM backend is pure Python and runs on macOS, Linux and
Windows. Two things to know:

- **Windows and Linux cannot use the `mlx` backend** — it is Apple Silicon only.
  Use `backend: transformers` (`pip install -U "transformers>=4.57" torch
  accelerate qwen-vl-utils jinja2`), or skip indexing entirely: `scene_index.json`
  is committed, so the agent, retrieval, UI and tests all run with no model.
- **Paths in the index are always POSIX-style** (`data/images/office/x.jpg`),
  written with `as_posix()` on every platform. A Windows machine writing
  `data\images\office\x.jpg` into the shared index would break every Mac and
  Linux teammate; a test enforces this.

---

## Suggested division of labour

| Area | Files | Needs a model? |
| --- | --- | --- |
| Photos + ground truth | `data/images/`, `data/ground_truth.json` | no — a phone |
| Indexing | `indexer.py`, `vlm.py`, `prompts.py` | yes — strongest laptop |
| Retrieval + spatial | `retrieval.py`, `spatial.py`, `query_parser.py` | no — dummy index |
| Agent + UI | `agent.py`, `app.py`, `cli.py` | no — dummy index |

---

## Photo shoot checklist

The demo lives or dies on the photos, so shoot deliberately:

- **Plant ambiguity on purpose.** Several bottles of *different colours* across
  different rooms is the single most useful thing you can stage — it makes the
  clarification loop fire. Two identical objects in the same room is a dead end:
  nothing can separate them, and the agent correctly gives up.
- **Shoot each place from 2–3 angles.** This is what exercises multi-view
  merging, and it is where the interesting failures live.
- 3–5 locations, 8–15 photos each, **one location per folder**.
- Include a few small objects (charger, keys, USB cable) — they are where a 2B
  model starts to fail, and that belongs in the report.
- Vary lighting a little, but not wildly: colour naming is the first thing to
  break under bad light.
- Fill in `data/ground_truth.json` **while you are still in the room**.

---

## Evaluation

```bash
python eval/run_eval.py
```

Cases live in `eval/test_queries.json`. Answers are **derived from the ground-truth
target object**, not hardcoded strings — so renaming a room or dropping an
attribute does not silently break the suite. Current numbers on the dummy index:

| metric | value |
| --- | --- |
| Resolution accuracy | 10/10 |
| Clarification rate | 3/3 ambiguous queries triggered a question |
| False answers | 0 (a top-1 retrieval baseline scores 3) |
| Avg questions asked | 1.2 |

"False answers" is the metric that matters: a system that confidently returns
the wrong bottle is worse than one that asks.

---

## Known limits (put these in the report, don't hide them)

- **Small VLMs miss small objects.** Mitigated by `max_image_side`; a
  crop-and-recheck pass is the obvious next step.
- **`query_parser.py` is a dictionary, not a parser.** It ignores grammar
  entirely and only looks for known nouns, which is why phrasing is irrelevant
  ("where my laptop", "yo where'd my macbook go" both work) — and why an unknown
  word is fatal ("labtop", Korean input). It fails *visibly* rather than
  hallucinating. Upgrading it to an LLM call is a drop-in swap: it only has to
  return the same `ParsedQuery`.
- **Relations are 2D only.** "Behind" is not recoverable from a single photo.
- **Merging can under-count.** An object visible in only one non-reference view
  is absorbed into the nearest reference object. We accept this: over-counting
  produces questions no one can answer, which is worse.

---

## Roadmap

**v1 (this repo) — static photos.** Offline indexing removes the query-time VLM
cost, so no embedding filter is needed and recall is exhaustive.

**v2 — live video.** Pre-indexing everything stops being possible: 30 frames a
second, and the scene keeps changing. Analysis has to move back to query time,
which means a cheap filter (CLIP) becomes necessary again. The design then
splits:

```
indexed history   →  dictionary lookup   (exhaustive, fast)
recent frames     →  CLIP → VLM          (not yet indexed, slower)
```

Two rules carry over from v1's argument. The filter must be generous — it ranks
candidates, it does not decide presence, and there is no similarity threshold
that means "the bottle is here". And when the system reports "not found" it must
say what it actually looked at ("nothing in the indexed history; the last 30
seconds are still being processed"), because a silent coverage gap is what makes
a user stop looking.
