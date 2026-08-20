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

from src.agent import Session          # noqa: E402
from src.config import load_config     # noqa: E402
from src.retrieval import SceneIndex   # noqa: E402
from src.query_parser import parse     # noqa: E402


def find_target(index, object_id):
    """Look up the object a test case says is the right answer."""
    for scene in index.scenes:
        for obj in scene.get("objects", []):
            if obj["id"] == object_id:
                return scene, obj
    return None, None


def answer_as_the_user(reply, scene, obj):
    """Reply the way the real owner of `obj` would.

    Deriving answers from the target object instead of hardcoding strings means
    the suite survives renaming a room or dropping an attribute - which has
    already happened twice. If the agent asks something the owner could not
    know, we say so honestly rather than feeding it a lucky guess.
    """
    key = reply.asked_key
    if key == "location":
        return scene.get("location", "i don't know")

    value = obj.get("attributes", {}).get(key)
    options = reply.options or []

    # yes/no shape: "Was it the one with X?"
    if len(options) == 1:
        return "yes" if value == options[0] else "no"
    if value is None:
        return "i don't know"
    return str(value)


def run_case(index, cfg, case):
    session = Session(index, cfg)
    reply = session.start(case["query"])
    asked = 0

    expect = case.get("expect")
    scripted = case.get("answers") or []
    scene, obj = (None, None)
    if expect and expect != "any":
        scene, obj = find_target(index, expect)

    while reply.kind == "question" and asked < 10:
        if asked < len(scripted):
            answer = scripted[asked]
        elif obj is not None:
            answer = answer_as_the_user(reply, scene, obj)
        else:
            break
        asked += 1
        reply = session.reply(answer)

    found = None
    if reply.kind == "answer" and reply.candidates:
        found = reply.candidates[0]["object"]["id"]

    expect = case.get("expect")
    if expect is None:
        ok = reply.kind == "none_found"
    elif expect == "any":
        ok = reply.kind == "answer"
    else:
        ok = found == expect

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
            "ok": ok, "asked": asked, "kind": reply.kind,
            "found": found, "expect": expect, "false_answer": false_answer,
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
        cases = json.load(fh)

    results = [run_case(index, cfg, case) for case in cases]

    print(f"{'id':<14}{'difficulty':<12}{'result':<8}{'asked':<7}{'outcome'}")
    print("-" * 78)
    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        flag = "  <- answered without asking!" if r["false_answer"] else ""
        print(f"{r['id']:<14}{r['difficulty']:<12}{mark:<8}{r['asked']:<7}{r['kind']}{flag}")

    total = len(results)
    passed = sum(r["ok"] for r in results)
    ambiguous = [r for r in results if r["difficulty"] == "ambiguous"]
    clarified = sum(r["asked"] > 0 for r in ambiguous)
    turns = [r["asked"] for r in results if r["asked"]]

    print("-" * 78)
    print(f"Resolution accuracy   {passed}/{total}  ({100 * passed / total:.0f}%)")
    if ambiguous:
        print(f"Clarification rate    {clarified}/{len(ambiguous)} of ambiguous queries "
              f"triggered a question")
    print(f"False answers         {sum(r['false_answer'] for r in results)} "
          f"(lower is better; a top-1 retrieval baseline scores {len(ambiguous)})")
    print(f"Query parse accuracy  {sum(r['parse_ok'] for r in results)}/{total}")
    print(f"Confidence handling   {sum(r['confidence_ok'] for r in results)}/{total}")
    print(f"Transparent not-found {sum(r['transparent_not_found'] for r in results)}/{total}")
    print(f"Candidate comparison  {sum(r['comparison_ok'] for r in results)}/{total}")
    spatial = [r for r in results if r["difficulty"] == "spatial"]
    if spatial:
        print(f"Spatial retrieval     {sum(r['ok'] for r in spatial)}/{len(spatial)}")
    small = [r for r in results if r["difficulty"] == "small-object"]
    if small:
        print(f"Small-object verify   {sum(r['ok'] for r in small)}/{len(small)}")
    print(f"Avg questions asked   {sum(turns) / len(turns):.1f}" if turns else
          "Avg questions asked   0")

    failures = [r for r in results if not r["ok"]]
    if failures:
        print("\nFailures:")
        for r in failures:
            print(f"  {r['id']}: expected {r['expect']}, got {r['found'] or r['kind']}")
            print(f"      \"{r['text']}\"")


if __name__ == "__main__":
    main()
