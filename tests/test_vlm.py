"""Parsing the model's answer.

A 2B model listing 15 objects WILL sometimes run out of tokens partway through.
Losing the whole image because entry 12 is half-written is unacceptable when a
single image costs 40 seconds, so these tests pin the salvage behaviour.
"""

import pytest

from src.vlm import VLMError, extract_json, salvage

COMPLETE = ('{"n_objects": 3, "objects": ['
            '{"type":"bottle","color":"black","bbox":[1,2,3,4],"confidence":0.9},'
            '{"type":"laptop","color":"blue","bbox":[5,6,7,8],"confidence":0.8},'
            '{"type":"mug","color":"white","bbox":[9,10,11,12],"confidence":0.7}'
            '], "caption": "a desk"}')


def test_parses_clean_json():
    parsed = extract_json(COMPLETE)
    assert len(parsed["objects"]) == 3
    assert parsed["caption"] == "a desk"
    assert not parsed.get("truncated")


def test_strips_markdown_fences():
    parsed = extract_json("```json\n" + COMPLETE + "\n```")
    assert len(parsed["objects"]) == 3


def test_ignores_chatter_around_the_json():
    parsed = extract_json("Sure! Here is the result:\n" + COMPLETE + "\nHope that helps.")
    assert len(parsed["objects"]) == 3


def test_salvages_complete_objects_when_output_is_cut_off():
    cut = COMPLETE[:COMPLETE.index('{"type":"mug"')] + '{"type":"mug","color":"whi'
    parsed = extract_json(cut)
    assert [o["type"] for o in parsed["objects"]] == ["bottle", "laptop"]
    assert parsed["truncated"] is True
    assert parsed["n_objects"] == 3          # what the model claimed it would list


def test_salvage_is_string_aware():
    # braces and escaped quotes inside a value must not break the depth count
    text = ('{"objects":[{"type":"card","color":"say {hi} \\"ok\\"",'
            '"bbox":[1,2,3,4]},{"type":"pen","bbox":[5,6,7,8]}')
    parsed = extract_json(text)
    assert [o["type"] for o in parsed["objects"]] == ["card", "pen"]


def test_salvage_skips_entries_without_a_box():
    text = '{"objects":[{"type":"ghost"},{"type":"pen","bbox":[5,6,7,8]}'
    parsed = extract_json(text)
    assert [o["type"] for o in parsed["objects"]] == ["pen"]


def test_salvage_returns_none_when_there_is_nothing_to_save():
    assert salvage("I could not see the image clearly.") is None
    assert salvage('{"objects":[') is None


def test_empty_response_raises():
    with pytest.raises(VLMError):
        extract_json("")


def test_unparseable_response_raises():
    with pytest.raises(VLMError):
        extract_json("I'm sorry, I cannot help with that.")
