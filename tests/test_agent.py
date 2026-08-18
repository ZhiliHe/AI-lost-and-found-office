"""The behaviour that actually distinguishes us from plain retrieval:
knowing when NOT to answer.
"""

import pytest

from src.agent import Session, choose_question, _split_quality
from src.query_parser import parse
from src.retrieval import find_candidates, rank_scenes


# --- retrieval ------------------------------------------------------------- #

def test_finds_all_bottles(index):
    candidates = find_candidates(index, parse("where is my bottle"))
    assert len(candidates) == 3


def test_color_narrows_candidates(index):
    candidates = find_candidates(index, parse("where is my black bottle"))
    assert len(candidates) == 2
    assert all(c["object"]["attributes"]["color"] == "black" for c in candidates)


def test_spatial_relation_narrows_to_one(index):
    candidates = find_candidates(index, parse("find the black bottle beside the laptop"))
    assert len(candidates) == 1
    assert candidates[0]["object"]["id"] == "office_01_o1"


def test_relation_is_actually_enforced(index):
    # the only umbrella lives in the lounge, and there is no bottle there
    candidates = find_candidates(index, parse("find the bottle beside the umbrella"))
    assert candidates == []


def test_rank_scenes_puts_relevant_scene_first(index):
    scenes = rank_scenes(index, parse("black bottle beside the laptop"), top_k=5)
    assert scenes[0]["scene_id"] == "office_01"


def test_location_filter(index):
    candidates = find_candidates(index, parse("my bottle in the classroom",
                                              known_locations=index.locations()))
    assert len(candidates) == 1
    assert candidates[0]["scene"]["location"] == "classroom"


# --- the clarification loop ------------------------------------------------ #

def test_unambiguous_query_answers_immediately(index, config):
    reply = Session(index, config).start("find the blue bottle")
    assert reply.kind == "answer"
    assert "blue bottle" in reply.text


def test_answer_includes_spatial_context(index, config):
    reply = Session(index, config).start("find the black bottle beside the laptop")
    assert reply.kind == "answer"
    assert "office" in reply.text
    assert "laptop" in reply.text


def test_ambiguous_query_asks_instead_of_guessing(index, config):
    reply = Session(index, config).start("where is my bottle")
    assert reply.kind == "question"
    assert reply.asked_key == "color"
    assert set(reply.options) == {"black", "blue"}


def test_full_clarification_converges(index, config):
    session = Session(index, config)

    first = session.start("where is my bottle")
    assert first.kind == "question"

    second = session.reply("black")
    # two black metal bottles remain, in different rooms - so it asks WHERE
    assert second.kind == "question"
    assert second.asked_key == "location"

    final = session.reply("office")
    assert final.kind == "answer"
    assert final.candidates[0]["object"]["id"] == "office_01_o1"


def test_agent_gives_up_rather_than_looping_forever(index, config):
    session = Session(index, {"agent": {"max_clarify_turns": 1, "top_k_scenes": 5}})
    assert session.start("where is my bottle").kind == "question"
    reply = session.reply("i don't know")
    assert reply.kind == "giveup"
    assert reply.candidates            # still shows a shortlist


def test_dont_know_does_not_add_a_constraint(index, config):
    session = Session(index, config)
    session.start("where is my bottle")
    session.reply("not sure")
    assert session.constraints == {}
    assert "color" in session.asked     # but we never ask it twice


def test_unknown_object_is_reported_not_guessed(index, config):
    reply = Session(index, config).start("where is my skateboard")
    assert reply.kind == "none_found"


def test_no_object_named_asks_for_the_object(index, config):
    reply = Session(index, config).start("where is it")
    assert reply.kind == "none_found"
    assert "looking for" in reply.text


def test_over_filtering_backs_off_instead_of_dead_ending(index, config):
    session = Session(index, config)
    session.start("where is my bottle")
    reply = session.reply("green")          # no green bottle exists
    # constraint was impossible, so it is dropped and we surface the shortlist
    assert reply.kind in ("question", "giveup")
    assert reply.candidates


# --- question selection ---------------------------------------------------- #

def test_split_quality_zero_when_attribute_is_uniform(index):
    candidates = find_candidates(index, parse("black bottle"))
    assert _split_quality(candidates, "color") == 0.0


def test_choose_question_skips_already_asked(index):
    candidates = find_candidates(index, parse("where is my bottle"))
    key, _ = choose_question(candidates, already_asked=["color"])
    assert key != "color"


def test_choose_question_falls_back_to_location(index):
    candidates = find_candidates(index, parse("where is my bottle"))
    key, options = choose_question(
        candidates, already_asked=["color", "size", "material"])
    assert key == "location"
    assert len(options) > 1


# --- multiple camera angles of the same place ------------------------------ #

def _view(scene_id, objects):
    from src.spatial import compute_relations
    built = [{"id": f"{scene_id}_o{i}", "type": t,
              "attributes": {"color": c, "material": None, "size": "medium"},
              "bbox": box, "confidence": conf}
             for i, (t, c, _unused, box, conf) in enumerate(objects)]
    return {"scene_id": scene_id, "location": "office",
            "image_path": f"data/images/office/{scene_id}.jpg",
            "width": 4032, "height": 3024, "caption": "a desk",
            "objects": built, "relations": compute_relations(built, (4032, 3024))}


@pytest.fixture
def two_views():
    """One desk, two camera angles. The bottle and laptop appear in both; the
    phone is only visible from the second angle."""
    from src.retrieval import SceneIndex
    return SceneIndex({"version": 1, "scenes": [
        _view("office_01", [
            ("bottle", "red", None, [1318, 1711, 1657, 2195], 0.96),
            ("laptop", "blue", "apple", [1895, 907, 2661, 1300], 0.95)]),
        _view("office_02", [
            ("bottle", "red", None, [900, 1500, 1300, 2100], 0.90),
            ("laptop", "blue", "apple", [2000, 800, 2900, 1400], 0.94),
            ("phone", "silver", None, [3000, 1200, 3300, 1500], 0.88)]),
    ]})


def test_same_object_in_two_views_counts_once(two_views):
    candidates = find_candidates(two_views, parse("where is my bottle"))
    assert len(candidates) == 1
    assert set(candidates[0]["seen_in"]) == {"office_01", "office_02"}


def test_without_merging_the_same_bottle_is_double_counted(two_views):
    candidates = find_candidates(two_views, parse("where is my bottle"),
                                 merge_across_views=False)
    assert len(candidates) == 2      # this is the behaviour we are fixing


def test_merging_keeps_the_clearest_view(two_views):
    # office_02 shows the bottle larger, so that view should win
    candidates = find_candidates(two_views, parse("where is my bottle"))
    assert candidates[0]["scene"]["scene_id"] == "office_02"


def test_extra_view_adds_objects_hidden_in_the_first(two_views, config):
    # the phone is only visible from the second angle
    reply = Session(two_views, config).start("find my phone")
    assert reply.kind == "answer"
    assert "phone" in reply.text


def test_two_of_the_same_thing_in_ONE_photo_are_not_merged():
    from src.retrieval import SceneIndex
    index = SceneIndex({"version": 1, "scenes": [_view("office_01", [
        ("bottle", "black", None, [100, 100, 300, 600], 0.9),
        ("bottle", "black", None, [900, 100, 1100, 600], 0.9)])]})
    assert len(find_candidates(index, parse("where is my bottle"))) == 2


def test_view_counts_are_capped_at_the_best_single_view():
    """The real reason merging matters: one photo saw 2 bottles, another saw 3.
    The answer is 3, not 5. Attribute noise across angles used to make them all
    look distinct, and the agent would announce five bottles that do not exist.
    """
    from src.retrieval import SceneIndex
    index = SceneIndex({"version": 1, "scenes": [
        _view("office_front", [
            ("bottle", "black", None, [100, 100, 300, 600], 0.9),
            ("bottle", "blue", None, [400, 100, 600, 600], 0.9)]),
        _view("office_left", [
            # same three bottles, and the model read the colours slightly
            # differently from this angle
            ("bottle", "black", None, [150, 120, 360, 640], 0.9),
            ("bottle", "blue", None, [450, 120, 660, 640], 0.9),
            ("bottle", "silver", None, [800, 120, 1000, 640], 0.9)]),
    ]})
    candidates = find_candidates(index, parse("where is my bottle"))
    assert len(candidates) == 3


def test_material_read_differently_from_two_angles_stays_one_object():
    """The model called the same red bottle "glass" from the front and
    "plastic" from the left. That must not become two bottles."""
    from src.retrieval import SceneIndex
    scenes = [
        {"scene_id": "office_front", "location": "office",
         "image_path": "data/images/office/office_front.jpg",
         "width": 4000, "height": 3000, "caption": "desk", "relations": [],
         "objects": [{"id": "office_front_o0", "type": "bottle",
                      "attributes": {"color": "red", "material": "glass", "size": "medium"},
                      "bbox": [1300, 1700, 1650, 2200], "confidence": 0.9}]},
        {"scene_id": "office_left", "location": "office",
         "image_path": "data/images/office/office_left.jpg",
         "width": 4000, "height": 3000, "caption": "desk", "relations": [],
         "objects": [{"id": "office_left_o0", "type": "bottle",
                      "attributes": {"color": "red", "material": "plastic", "size": "medium"},
                      "bbox": [900, 1500, 1300, 2100], "confidence": 0.9}]},
    ]
    index = SceneIndex({"version": 1, "scenes": scenes})

    candidates = find_candidates(index, parse("where is my bottle"))
    assert len(candidates) == 1
    assert sorted(candidates[0]["observed"]["material"]) == ["glass", "plastic"]

    # and the owner can answer with EITHER material and still find it
    for material in ("glass", "plastic"):
        found = find_candidates(index, parse("where is my red bottle"),
                                constraints={"material": material})
        assert len(found) == 1, material


def test_logo_differences_between_angles_do_not_split_one_object():
    from src.retrieval import SceneIndex
    index = SceneIndex({"version": 1, "scenes": [
        _view("office_front", [("bottle", "blue", "lululemon", [100, 100, 300, 600], 0.9)]),
        _view("office_left", [("bottle", "blue", None, [150, 120, 360, 640], 0.9)]),
    ]})
    # brand is not part of the merge key, so these are one bottle
    assert len(find_candidates(index, parse("where is my bottle"))) == 1


def test_view_dependent_description_is_marked_as_such(index, config):
    """If the only relation we have is a camera artefact, say "in this photo"
    rather than stating it as a fact about the room."""
    from src.agent import describe_place
    scene = {"location": "office", "objects": [
        {"id": "a", "type": "bottle"}, {"id": "b", "type": "laptop"}],
        "relations": [{"subject": "a", "predicate": "left_of", "object": "b"}]}
    text = describe_place(scene, {"id": "a", "type": "bottle"})
    assert "in this photo" in text


def test_invariant_relation_is_stated_plainly(index, config):
    from src.agent import describe_place
    scene = {"location": "office", "objects": [
        {"id": "a", "type": "bottle"}, {"id": "b", "type": "laptop"}],
        "relations": [{"subject": "a", "predicate": "beside", "object": "b"},
                      {"subject": "a", "predicate": "left_of", "object": "b"}]}
    text = describe_place(scene, {"id": "a", "type": "bottle"})
    assert "beside the laptop" in text
    assert "in this photo" not in text


# --- the attribute schema is now exactly three keys ------------------------- #

def test_indexer_emits_only_the_three_surviving_attributes():
    """brand_or_logo and state are GONE from the schema, not merely null.
    A null key invites someone to start filling it in again."""
    from src.indexer import clean_object
    obj = clean_object({"type": "bottle", "color": "black", "material": "metal",
                        "size": "medium", "brand_or_logo": "lululemon",
                        "state": "closed", "bbox": [100, 100, 200, 400],
                        "confidence": 0.9},
                       "office_01", 0, "norm1000", (1000, 1000), (1000, 1000))
    assert set(obj["attributes"]) == {"color", "material", "size"}


def test_dummy_index_has_no_dead_attribute_keys():
    from src.vlm import DummyBackend
    import json
    payload = json.loads(DummyBackend().describe(None, "x"))
    for obj in payload["objects"]:
        assert "brand_or_logo" not in obj
        assert "state" not in obj


def test_saying_no_rules_the_value_out(index, config):
    """Answering "no" to "is it metal?" means NOT metal - it used to be stored
    as "material unknown", which quietly matched everything."""
    from src.vocab import negate
    from src.retrieval import find_candidates as fc

    candidates = fc(index, parse("where is my bottle"),
                    constraints={"material": negate("metal")})
    for cand in candidates:
        observed = cand.get("observed", {}).get("material", [])
        assert observed != ["metal"], cand["object"]["id"]


# --- query-time VLM verification ------------------------------------------- #

def test_verifier_drops_candidates_the_model_rejects(index, config):
    """The indexer froze a mistake into the index; the query-time check catches
    it before the user is ever asked about it."""
    seen = {}

    def fake_verifier(candidates, description):
        seen["description"] = description
        # pretend the model rejected everything in the classroom
        return [c for c in candidates if c["scene"]["location"] != "classroom"]

    session = Session(index, config, verifier=fake_verifier)
    reply = session.start("where is my black bottle")

    assert "black bottle" == seen["description"]
    assert all(c["scene"]["location"] != "classroom" for c in reply.candidates)


def test_verifier_runs_once_per_object(index, config):
    calls = []

    def counting_verifier(candidates, description):
        calls.append(len(candidates))
        return candidates

    session = Session(index, config, verifier=counting_verifier)
    session.start("where is my bottle")
    first = len(calls)
    session.reply("black")           # narrowing must not re-verify the same objects
    assert len(calls) == first


def test_verification_never_empties_the_result(index, config):
    """A verifier that rejects everything must not turn a findable object into
    "not found" - that is the worst outcome for a lost-and-found."""
    from src.verify import verify_candidates

    class RejectEverything:
        def describe(self, *_, **__):
            return '{"match": false, "reason": "no", "confidence": 0.99}'

    candidates = find_candidates(index, parse("where is my bottle"))
    cfg = {"paths": {"index": "data/scene_index.json", "debug": "data/debug"}}
    survivors = verify_candidates(candidates, "bottle", RejectEverything(), cfg)
    assert survivors            # never empty


def test_index_paths_are_posix_style():
    """A Windows teammate must not write "data\\images\\x.jpg" into the shared
    index - no Mac or Linux machine could open it afterwards."""
    import subprocess
    import sys
    import json
    from pathlib import Path

    subprocess.run([sys.executable, "scripts/make_dummy_index.py"],
                   check=True, capture_output=True)
    payload = json.loads(Path("data/scene_index.json").read_text(encoding="utf-8"))
    for scene in payload["scenes"]:
        assert "\\" not in scene["image_path"], scene["image_path"]
