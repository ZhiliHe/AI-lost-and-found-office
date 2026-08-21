"""How much of what is actually there did the indexer find?

    python eval/run_recall.py

run_eval.py measures the CONVERSATION: given what the index holds, does the
agent ask the right questions and land on the right object. This measures the
layer underneath - the index itself - against data/ground_truth.json, which was
written by looking at the photos, not by looking at the model's output.

Three numbers, and the split between the first two is the point:

  object recall    a thing in the photo has SOME object of its kind in the
                   index, in the right place
  attribute recall ...and the colour matches too
  false positives  objects in the index that answer to nothing in the photo

Recall high, attribute recall much lower means the model sees things and
describes them badly - a vocabulary or prompt problem. Both low means it is
not seeing them at all - a resolution problem, which is what tiling addresses.
Those two failures look identical from inside the agent and need opposite
fixes, so they are worth separating before anyone argues about what to fix.
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import truth                  # noqa: E402
from src.config import load_config      # noqa: E402
from src.retrieval import SceneIndex    # noqa: E402
from src.vocab import find_object_types  # noqa: E402


def indexed_objects(index):
    """(location, object) for everything in the index, all views."""
    return [(scene.get("location"), obj)
            for scene in index.scenes
            for obj in scene.get("objects", [])]


def searchable(object_type):
    """Can a user reach this type by typing its name?

    An object the indexer found but the vocabulary cannot name is invisible in
    practice. Counting it as recalled would report a system that works better
    than the one a person can actually use.
    """
    return bool(find_object_types(str(object_type)))


def main():
    cfg = load_config()
    index = SceneIndex.load(cfg["paths"]["index"])
    ground_truth = truth.load_ground_truth()

    found_here = defaultdict(list)
    for location, obj in indexed_objects(index):
        found_here[location].append(obj)

    rows = []
    claimed = set()               # id() of index objects some truth explains
    for location, wanted in truth.in_scope(ground_truth):
        by_type = [obj for obj in found_here.get(location, [])
                   if truth.same_type(wanted["type"], obj.get("type"))]
        by_colour = [obj for obj in by_type
                     if (obj.get("attributes") or {}).get("color") == wanted.get("color")]
        for obj in by_type:
            claimed.add(id(obj))
        rows.append({
            "location": location,
            "wanted": wanted,
            "seen": bool(by_type),
            "colour_ok": bool(by_colour),
            "searchable": searchable(wanted["type"]),
        })

    print(f"{'place':<20}{'object':<28}{'found':<8}{'colour':<8}{'searchable'}")
    print("-" * 78)
    for row in rows:
        wanted = row["wanted"]
        name = f"{wanted.get('color', '?')} {wanted['type']}"
        print(f"{row['location']:<20}{name:<28}"
              f"{'yes' if row['seen'] else 'NO':<8}"
              f"{('yes' if row['colour_ok'] else 'no') if row['seen'] else '-':<8}"
              f"{'yes' if row['searchable'] else 'NO WORD'}")

    total = len(rows)
    seen = sum(row["seen"] for row in rows)
    colour_ok = sum(row["colour_ok"] for row in rows)
    reachable = sum(row["seen"] and row["searchable"] for row in rows)
    unclaimed = [(location, obj) for location, obj in indexed_objects(index)
                 if id(obj) not in claimed]

    print("-" * 78)
    print(f"Object recall         {seen}/{total}  ({100 * seen / total:.0f}%)")
    print(f"Attribute recall      {colour_ok}/{total}  ({100 * colour_ok / total:.0f}%)"
          f"   (found AND the right colour)")
    print(f"Reachable by name     {reachable}/{total}  ({100 * reachable / total:.0f}%)"
          f"   (found AND the vocabulary has a word for it)")
    print(f"Unexplained objects   {len(unclaimed)} of {len(indexed_objects(index))} "
          f"index entries match nothing in the ground truth")

    misses = [row for row in rows if not row["seen"]]
    if misses:
        print("\nNot found at all:")
        for row in misses:
            wanted = row["wanted"]
            print(f"  {row['location']:<20}{wanted.get('color', '?')} {wanted['type']}"
                  f"  - {wanted.get('where', '')}")

    wrong_colour = [row for row in rows if row["seen"] and not row["colour_ok"]]
    if wrong_colour:
        print("\nFound, but described with a different colour:")
        for row in wrong_colour:
            wanted = row["wanted"]
            print(f"  {row['location']:<20}{wanted.get('color', '?')} {wanted['type']}"
                  f"  - {wanted.get('where', '')}")

    # Type-matching alone cannot see a SECOND copy of something real: six
    # chargers on a table that holds one are all "explained" by that one. So
    # count them. Per view, never summed across views - the same charger
    # photographed twice is one charger, and conflating the two would report
    # multi-view merging as a defect instead of measuring it.
    expected = defaultdict(int)
    for location, wanted in truth.in_scope(ground_truth):
        expected[(location, wanted["type"])] += 1
    per_view = defaultdict(int)
    for scene in index.scenes:
        seen_here = defaultdict(int)
        for obj in scene.get("objects", []):
            seen_here[(scene.get("location"), obj.get("type"))] += 1
        for slot, count in seen_here.items():
            per_view[slot] = max(per_view[slot], count)

    over = []
    for slot, count in sorted(per_view.items()):
        location, kind = slot
        # Compare against every ground-truth type the indexer is allowed to
        # confuse this one with, or a bottle counted as a mug reads as an
        # invented object AND a missing one.
        allowed = sum(number for (place, other), number in expected.items()
                      if place == location and truth.same_type(other, kind))
        if count > allowed:
            over.append((location, kind, allowed, count))
    if over:
        print("\nCounted more than are there (worst single view):")
        for location, kind, allowed, count in over:
            print(f"  {location:<20}{kind:<16}{allowed} real, {count} reported")

    if unclaimed:
        print("\nIn the index, not in the ground truth (false positives, or things "
              "we forgot to write down):")
        counted = defaultdict(int)
        for location, obj in unclaimed:
            counted[(location, obj.get("type"),
                     (obj.get("attributes") or {}).get("color"))] += 1
        for (location, kind, colour), count in sorted(counted.items()):
            print(f"  {location:<20}{colour or '?'} {kind}  x{count}")


if __name__ == "__main__":
    main()
