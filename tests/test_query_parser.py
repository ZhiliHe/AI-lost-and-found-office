from src.query_parser import parse
from src.vocab import find_colors, find_object_types, normalize_color, normalize_object_type


def test_parse_simple_query():
    q = parse("Where is my bottle?")
    assert q.target_type == "bottle"
    assert q.predicate is None
    assert q.anchor_type is None


def test_parse_color_and_relation():
    q = parse("Find the black bottle beside the laptop")
    assert q.target_type == "bottle"
    assert q.attributes["color"] == "black"
    assert q.predicate == "beside"
    assert q.anchor_type == "laptop"


def test_parse_next_to_is_beside():
    q = parse("Find the blue bottle next to a computer")
    assert q.target_type == "bottle"
    assert q.attributes["color"] == "blue"
    assert q.predicate == "beside"
    assert q.anchor_type == "laptop"        # 'computer' normalises to laptop


def test_parse_picks_up_location():
    q = parse("is my backpack in the lounge?", known_locations=["office", "lounge"])
    assert q.target_type == "backpack"
    assert q.location == "lounge"


def test_parse_unknown_object_returns_none_target():
    q = parse("where did I put that thing")
    assert q.target_type is None


def test_describe_is_human_readable():
    q = parse("Find the black bottle beside the laptop")
    assert q.describe() == "black bottle beside the laptop"


def test_color_aliases_normalise():
    assert normalize_color("Dark Grey") == "black"
    assert normalize_color("navy") == "blue"
    assert normalize_color(" CLEAR ") == "transparent"


def test_object_synonyms_normalise():
    assert normalize_object_type("water bottle") == "bottle"
    assert normalize_object_type("MacBook") == "laptop"
    assert normalize_object_type("earbuds") == "headphones"


def test_find_helpers_handle_plurals():
    assert "bottle" in find_object_types("I see two bottles here")
    assert "black" in find_colors("a black and blue bag")
