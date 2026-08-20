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


def test_parse_handles_typos_and_descriptive_sentences():
    q = parse("I think I left my dark grey botle nxt to the labtop")
    assert q.target_type == "bottle"
    assert q.attributes["color"] == "black"
    assert q.predicate == "beside"
    assert q.anchor_type == "laptop"


# --- real folder names, typed the way a person actually types them ---------- #

REAL_LOCATIONS = ["6f_meetingroom", "703", "711", "7f_tearoom",
                  "8f_lobbytable", "hotelroom"]


def test_locations_match_how_people_speak():
    """Folders are named for filing ("7f_tearoom"); people say "the tearoom"."""
    cases = {
        "bottle in the tearoom": "7f_tearoom",
        "laptop in the meeting room": "6f_meetingroom",
        "bottle in the hotel room": "hotelroom",
        "phone in the lobby table": "8f_lobbytable",
        "charger in 7f tearoom": "7f_tearoom",
        "my keys in room 711": "711",
        "keys in 703": "703",
    }
    for query, expected in cases.items():
        assert parse(query, known_locations=REAL_LOCATIONS).location == expected, query


def test_a_floor_number_alone_is_not_a_location():
    """"is it on the 7th floor" must not silently mean the tearoom - several
    rooms share a floor, and guessing one would hide the others."""
    parsed = parse("is my bottle on the 7f", known_locations=REAL_LOCATIONS)
    assert parsed.location is None


def test_no_location_mentioned_stays_none():
    assert parse("where is my bottle", known_locations=REAL_LOCATIONS).location is None


def test_keys_are_recognised():
    for query in ("where are my keys", "wheres my key", "i lost my keychain"):
        assert parse(query).target_type == "keys", query


def test_fixtures_are_not_indexed():
    """The prompt says to ignore furniture, but a 2B model still reported a
    sink, a faucet, a water cooler and an ice maker in the tea room. Nobody
    loses a sink - it only wastes a detection slot."""
    from src.vocab import is_portable

    for fixture in ("sink", "faucet", "water cooler", "ice maker", "plant",
                    "hair dryer", "towel", "desk", "chair", "monitor"):
        assert not is_portable(fixture), fixture

    for belonging in ("bottle", "keys", "glasses", "laptop", "wallet", "umbrella"):
        assert is_portable(belonging), belonging
