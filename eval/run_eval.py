"""Small evaluation harness - this is the "evaluation with several test cases"
deliverable from the project brief.

    python eval/run_eval.py

It replays scripted conversations against the agent and reports four numbers.
The important one is FALSE ANSWERS: cases where the agent confidently answered
a question it should have asked about. A plain retrieval baseline scores 100%
there, and that is exactly the weakness our system is supposed to fix - so put
this table in the report and the slides.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import truth                 # noqa: E402
from src.agent import PICK_KEY, SAFE_PREDICATES, Session, nearest_anchor_type  # noqa: E402
from src.config import load_config     # noqa: E402
from src.retrieval import SceneIndex   # noqa: E402
from src.query_parser import parse     # noqa: E402


def find_targets(index, descriptor):
    """Every indexed object that answers this case correctly.

    More than one, because a thing photographed from two angles is two objects
    in the index and either is the right answer. Empty means the indexer never
    found it - see eval/truth.py for why that is reported separately.
    """
    return truth.resolve(index, descriptor)


def neighbours_of(targets):
    """Everything the owner could truthfully say their object was next to.

    Across ALL views, not just one. A person knows what sat around their bottle;
    a single photo only shows the neighbours that angle happened to catch, and
    the two views of the same desk disagree about which one is "nearest". Asking
    the simulated owner from one view while the agent merged both is how you
    manufacture failures that no real user would ever produce.
    """
    anchors = []
    for scene, obj in targets:
        anchor, _ = nearest_anchor_type(scene, obj, predicate_order=SAFE_PREDICATES)
        if anchor and anchor not in anchors:
            anchors.append(anchor)
    return anchors


def answer_as_the_user(reply, scene, obj, targets=()):
    """Reply the way the real owner of `obj` would.

    Deriving answers from the target object instead of hardcoding strings means
    the suite survives renaming a room or dropping an attribute - which has
    already happened twice. If the agent asks something the owner could not
    know, we say so honestly rather than feeding it a lucky guess.
    """
    key = reply.asked_key

    # The agent gave up on words and showed photos. A real owner recognises
    # their own belonging instantly, so the simulated user does too - anything
    # else would measure our patience, not the agent.
    if key == PICK_KEY:
        for position, object_id in enumerate(reply.options, 1):
            if object_id == obj["id"]:
                return str(position)
        return "none"

    if key == "location":
        return scene.get("location", "i don't know")

    # "near" isn't stored in attributes (it's a computed spatial relation, see
    # agent.nearest_anchor_type), so without this the generic value = obj.get(
    # "attributes", {}).get(key) below always sees None and every simulated
    # user answers "i don't know" - which would fail any eval case whose
    # target can only be resolved by asking about a neighbour.
    if key == "near":
        anchors = neighbours_of(targets) or []
        if not anchors:
            anchor, _ = nearest_anchor_type(scene, obj,
                                            predicate_order=SAFE_PREDICATES)
            anchors = [anchor] if anchor else []
        options = reply.options or []
        if len(options) == 1:
            return "yes" if options[0] in anchors else "no"
        # Name one the agent offered if any of them is true. Volunteering a
        # neighbour it has never heard of is realistic, but it tells us nothing
        # about clarification - it only tests the not-found path, which
        # not-found-01 already covers.
        for option in options:
            if option in anchors:
                return option
        return anchors[0] if anchors else "i don't know"

    value = obj.get("attributes", {}).get(key)
    options = reply.options or []

    # yes/no shape: "Was it the one with X?"
    if len(options) == 1:
        return "yes" if value == options[0] else "no"
    if value is None:
        return "i don't know"
    return str(value)


def run_case(index, cfg, case):
    descriptor = case.get("expect")
    scripted = case.get("answers") or []

    targets = find_targets(index, descriptor) if descriptor else []
    missing = bool(descriptor) and not targets
    # The simulated user owns one specific object. Any view of it is a correct
    # answer, but the person answering questions is thinking of one thing.
    scene, obj = targets[0] if targets else (None, None)
    acceptable = {found_obj["id"] for _, found_obj in targets}

    session = Session(index, cfg)
    reply = session.start(case["query"])
    asked = 0

    # "choose" is a question too - the agent is asking which photo is yours.
    while reply.kind in ("question", "choose") and asked < 10:
        if asked < len(scripted):
            answer = scripted[asked]
        elif obj is not None:
            answer = answer_as_the_user(reply, scene, obj, targets)
        else:
            break
        asked += 1
        reply = session.reply(answer)

    found = None
    if reply.kind == "answer" and reply.candidates:
        found = reply.candidates[0]["object"]["id"]

    if descriptor is None:
        ok = reply.kind == "none_found"
    elif missing:
        ok = False               # cannot be found; counted as recall, not reasoning
    else:
        ok = found in acceptable

    # A "false answer" is answering with no questions when the query was
    # genuinely ambiguous. This is the metric that separates us from top-1 retrieval.
    false_answer = (case.get("difficulty") == "ambiguous"
                    and asked == 0 and reply.kind == "answer")
    parsed = parse(case["query"], known_locations=index.locations())
    parse_ok = parsed.target_type is not None
    transparent_not_found = reply.kind != "none_found" or "indexed data" in reply.text
    confidence_ok = reply.kind != "answer" or getattr(reply, "confidence", None) in (
        "high", "medium")
    comparison_ok = reply.kind != "question" or "Candidate " in reply.text

    return {"id": case["id"], "difficulty": case.get("difficulty", "-"),
            "ok": ok, "asked": asked, "kind": reply.kind, "missing": missing,
            "known_gap": bool(case.get("known_gap")),
            "found": found,
            "expect": truth.describe(descriptor) if descriptor else "nothing",
            "false_answer": false_answer,
            "parse_ok": parse_ok, "transparent_not_found": transparent_not_found,
            "confidence_ok": confidence_ok, "comparison_ok": comparison_ok,
            "text": reply.text}


def main():
    cfg = load_config()
    try:
        index = SceneIndex.load(cfg["paths"]["index"])
    except FileNotFoundError:
        print("No scene_index.json. Run: python scripts/make_dummy_index.py")
        sys.exit(1)

    cases_path = Path(__file__).parent / "test_queries.json"
    with open(cases_path, "r", encoding="utf-8") as fh:
        cases = [case for case in json.load(fh) if "id" in case]

    results = [run_case(index, cfg, case) for case in cases]

    print(f"{'id':<16}{'difficulty':<14}{'result':<10}{'asked':<7}{'outcome'}")
    print("-" * 78)
    for r in results:
        if r["known_gap"]:
            mark = "GAP" if not r["ok"] else "FIXED"
        elif r["missing"]:
            mark = "MISSING"
        else:
            mark = "PASS" if r["ok"] else "FAIL"
        flag = "  <- answered without asking!" if r["false_answer"] else ""
        print(f"{r['id']:<16}{r['difficulty']:<14}{mark:<10}{r['asked']:<7}{r['kind']}{flag}")

    # Known vocabulary gaps are excluded from the headline number and reported
    # on their own line. They are not reasoning failures - the object is in the
    # index, there is simply no word to ask for it with - and burying them in
    # the accuracy figure hides the one thing a reader could act on.
    scored = [r for r in results if not r["known_gap"]]
    gaps = [r for r in results if r["known_gap"]]
    total = len(scored)
    passed = sum(r["ok"] for r in scored)
    ambiguous = [r for r in scored if r["difficulty"] == "ambiguous"]
    clarified = sum(r["asked"] > 0 for r in ambiguous)
    turns = [r["asked"] for r in scored if r["asked"]]

    print("-" * 78)
    print(f"Resolution accuracy   {passed}/{total}  ({100 * passed / total:.0f}%)")
    absent = [r for r in scored if r["missing"]]
    if absent:
        print(f"Target not indexed    {len(absent)}/{total}  "
              f"(recall failure, not reasoning: {', '.join(r['id'] for r in absent)})")
    if gaps:
        fixed = sum(r["ok"] for r in gaps)
        print(f"Known vocab gaps      {len(gaps) - fixed} still unsearchable "
              f"({', '.join(r['expect'] for r in gaps if not r['ok'])})")
    if ambiguous:
        print(f"Clarification rate    {clarified}/{len(ambiguous)} of ambiguous queries "
              f"triggered a question")
    print(f"False answers         {sum(r['false_answer'] for r in scored)} "
          f"(lower is better; a top-1 retrieval baseline scores {len(ambiguous)})")
    print(f"Query parse accuracy  {sum(r['parse_ok'] for r in scored)}/{total}")
    print(f"Confidence handling   {sum(r['confidence_ok'] for r in scored)}/{total}")
    print(f"Transparent not-found {sum(r['transparent_not_found'] for r in scored)}/{total}")
    print(f"Candidate comparison  {sum(r['comparison_ok'] for r in scored)}/{total}")
    spatial = [r for r in scored if r["difficulty"] == "spatial"]
    if spatial:
        print(f"Spatial retrieval     {sum(r['ok'] for r in spatial)}/{len(spatial)}")
    small = [r for r in scored if r["difficulty"] == "small-object"]
    if small:
        print(f"Small-object verify   {sum(r['ok'] for r in small)}/{len(small)}")
    print(f"Avg questions asked   {sum(turns) / len(turns):.1f}" if turns else
          "Avg questions asked   0")

    failures = [r for r in scored if not r["ok"]]
    if failures:
        print("\nFailures:")
        for r in failures:
            print(f"  {r['id']}: expected {r['expect']}, got {r['found'] or r['kind']}")
            print(f"      \"{r['text']}\"")


if __name__ == "__main__":
    main()
