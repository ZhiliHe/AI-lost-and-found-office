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


# --- the model copying the prompt's example ---------------------------------

def test_prompt_example_objects_are_dropped():
    """A 2B model under pressure hands back the prompt's example instead of
    looking. Caught on a real desk photo: it returned the example laptop and
    charger, to the pixel, as two of eight "found" objects."""
    from src.indexer import _is_prompt_echo, PROMPT_EXAMPLE_BOXES

    resized = (1000, 1000)
    for box in PROMPT_EXAMPLE_BOXES:
        echoed = {"bbox": [float(v) for v in box]}
        assert _is_prompt_echo(echoed, resized), box

    real = {"bbox": [100.0, 150.0, 260.0, 400.0]}
    assert not _is_prompt_echo(real, resized)


def test_prompt_example_uses_unmistakable_placeholders():
    """If the example ever goes back to plausible values, the echo filter stops
    working silently - so assert the two stay in sync."""
    from src.prompts import SCENE_EXTRACTION_PROMPT
    from src.indexer import PROMPT_EXAMPLE_BOXES

    for box in PROMPT_EXAMPLE_BOXES:
        rendered = "[" + ", ".join(str(v) for v in box) + "]"
        assert rendered in SCENE_EXTRACTION_PROMPT, rendered
    # and the example must not look like a real answer
    assert "<noun>" in SCENE_EXTRACTION_PROMPT
