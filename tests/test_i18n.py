"""The spoken demo answers in the language it was asked in."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import i18n, voice                             # noqa: E402
from src.jarvis import (order_candidates_left_to_right,  # noqa: E402
                        resolve_position)
from src.vocab import detect_language, find_colors, find_object_types  # noqa: E402


# --- understanding what was said -------------------------------------------

def test_object_names_in_three_languages():
    for query in ("where is my bag", "내 가방 어딨어?", "我的包在哪里"):
        assert find_object_types(query) == ["backpack"], query


def test_cjk_needs_no_spaces():
    """Chinese never spaces its words and Korean often does not. Matching on
    word boundaries - correct for English, where " pen " must not hit "open" -
    finds nothing at all in either."""
    assert find_object_types("내가방어딨어") == ["backpack"]
    assert find_object_types("我的包在哪里") == ["backpack"]
    assert find_object_types("手机和耳机") == ["phone", "headphones"]


def test_longest_match_wins_at_each_position():
    """粉红色 must read as PINK. A plain longest-key-first loop found 红色
    (red) sitting inside it and returned both."""
    assert find_colors("粉红色的") == ["pink"]
    assert find_object_types("我的笔记本电脑在哪") == ["laptop"]


def test_a_new_type_needs_all_three_languages_at_once():
    """An English-only vocabulary entry is not a missing feature, it is a wrong
    answer. With no Korean word for a fan, "선풍기" was matched by the longest
    CJK substring we did know - "선", a cord - so asking for a fan searched for
    cables and confidently answered about the wrong thing."""
    assert find_object_types("선풍기") == ["fan"]
    assert find_object_types("내 손선풍기 어딨어") == ["fan"]
    assert find_object_types("我的小风扇在哪") == ["fan"]
    assert find_object_types("곱창밴드") == ["hair tie"]
    assert find_object_types("scrunchie") == ["hair tie"]
    # and the word it used to be confused with still works
    assert find_object_types("충전선 어딨어") == ["cable"]


def test_every_type_the_index_can_return_has_a_name_in_each_language():
    """A type the search can find but i18n cannot name comes back as an English
    word inside a Korean sentence - "베이지색 fan이 711호에 있습니다" - which
    reads as broken for reasons unrelated to what actually went wrong."""
    import json
    from pathlib import Path
    index_path = Path(__file__).resolve().parent.parent / "data" / "scene_index.json"
    if not index_path.exists():
        return
    with open(index_path, "r", encoding="utf-8") as handle:
        scenes = json.load(handle).get("scenes", [])
    types = {obj["type"] for scene in scenes for obj in scene.get("objects", [])}
    unnamed = sorted(t for t in types if t not in i18n.TYPE_NAMES)
    assert not unnamed, f"no Korean/Chinese name for: {unnamed}"


def test_traditional_characters_reach_the_same_entries():
    """Whisper picks a script per utterance and often writes Mandarin in
    traditional characters whatever the speaker uses. Our tables are simplified,
    so without a substitution pass "我的雨傘在哪裡" matches nothing and the demo
    stops understanding Chinese mid-sentence, for a reason nothing on screen
    could explain."""
    assert find_object_types("我的雨傘在哪裡") == ["umbrella"]
    assert find_object_types("我的筆記本電腦在哪") == ["laptop"]
    assert find_object_types("小風扇") == ["fan"]
    assert find_colors("綠色的瓶子") == ["green"]
    assert find_colors("銀色") == ["silver"]
    assert voice.hears_wake_word("嘿羅帕")[0]
    assert i18n.normalize_answer("綠色", "color", ["green", "red"], "zh") == "green"
    assert resolve_position("最後一個", 3) == 2


def test_we_only_ever_write_simplified_chinese():
    """Answering a simplified question in traditional characters is the kind of
    detail a native speaker reads as carelessness. Everything we display is a
    template we wrote, so this is enforceable rather than hoped for."""
    from pathlib import Path
    from src.vocab import TRADITIONAL_TO_SIMPLE
    root = Path(__file__).resolve().parent.parent
    written = []
    for module in ("src/i18n.py", "src/jarvis.py", "src/app.py"):
        text = (root / module).read_text(encoding="utf-8")
        # The traditional->simplified table is the one place traditional
        # characters belong: it exists to recognise them on the way in.
        if "TRADITIONAL_TO_SIMPLE" in text:
            continue
        written += [(module, ch) for ch in text if ch in TRADITIONAL_TO_SIMPLE]
    assert not written, f"traditional characters in output text: {written}"


def test_colours_in_three_languages():
    assert find_colors("the black one") == ["black"]
    assert find_colors("검은색이야") == ["black"]
    assert find_colors("黑色") == ["black"]


def test_language_detection():
    assert detect_language("where is my bag") == "en"
    assert detect_language("내 가방 어딨어?") == "ko"
    assert detect_language("我的包在哪里") == "zh"


# --- answering --------------------------------------------------------------

def test_korean_particles_follow_the_word():
    """이/가 and 을/를 depend on whether the word ends in a consonant. Getting
    it wrong is the clearest possible sign a sentence was assembled by a
    machine - "마우스이" is not a thing anyone writes."""
    assert i18n.josa("가방", "이", "가") == "이"
    assert i18n.josa("마우스", "이", "가") == "가"
    assert i18n.josa("우산", "을", "를") == "을"
    assert i18n.josa("커피", "을", "를") == "를"


def test_answers_are_rendered_per_language():
    obj = {"type": "backpack", "attributes": {"color": "black"}}
    assert i18n.describe_object(obj, "en") == "black backpack"
    assert i18n.describe_object(obj, "ko") == "검은색 가방"
    assert i18n.describe_object(obj, "zh") == "黑色背包"


def test_a_missing_phrase_falls_back_to_english():
    """A wrong-language answer is recoverable in front of an audience. A raw
    {placeholder} on the projector is not."""
    text = i18n.phrase("found", "de", what="black bag", where="in the 703")
    assert "{" not in text
    assert "black bag" in text


def test_two_places_never_read_as_the_same_words():
    """The agent offers places as multiple choice. Two rooms that render
    identically produce "the lobby table or the lobby table" - a question with
    no answer, where whichever the person picks is a coin toss."""
    for lang in ("ko", "zh"):
        spoken = [i18n.location_name(place, lang) for place in i18n.LOCATION_NAMES]
        assert len(set(spoken)) == len(spoken), (lang, spoken)


def test_the_room_is_asked_before_the_neighbour():
    """"Which room?" is a memory people have. "What was it next to?" asks them
    to recall the furniture around something they have already lost - they
    answer "I don't know" and the turn is spent for nothing."""
    from src.agent import KEY_PRIOR
    assert KEY_PRIOR["location"] > KEY_PRIOR["near"]
    # ...but colour still comes first: they came to us because they do not know
    # where it is, so opening with "which room?" asks them their own question.
    assert KEY_PRIOR["color"] > KEY_PRIOR["location"]


def test_options_are_joined_in_the_asking_language():
    assert i18n.join_options("color", ["black", "blue"], "en") == "black or blue"
    assert i18n.join_options("color", ["black", "blue"], "ko") == "검은색 아니면 파란색"
    assert i18n.join_options("color", ["black", "blue"], "zh") == "黑色 还是 蓝色"


# --- understanding the ANSWER to a question --------------------------------

def test_localised_answers_map_back_to_english_options():
    """The question went out in Korean, so the answer comes back in Korean -
    but the agent only knows canonical English values. Without this the reply
    is silently discarded and the agent asks the same thing again, which looks
    exactly like a bug to an audience."""
    cases = [
        ("금속이야", "material", ["metal", "plastic"], "ko", "metal"),
        ("작은거", "size", ["small", "large"], "ko", "small"),
        ("703호", "location", ["703", "7f_tearoom"], "ko", "703"),
        ("노트북 옆에", "near", ["laptop", "umbrella"], "ko", "laptop"),
        ("金属", "material", ["metal", "plastic"], "zh", "metal"),
        ("雨伞旁边", "near", ["laptop", "umbrella"], "zh", "umbrella"),
    ]
    for text, key, options, lang, expected in cases:
        assert i18n.normalize_answer(text, key, options, lang) == expected, text


def test_an_answer_naming_nothing_is_left_alone():
    """"I don't know" must reach the agent as-is, so it can burn the turn
    honestly instead of being mapped to a random option."""
    assert i18n.normalize_answer("몰라", "material", ["metal", "plastic"], "ko") is None
    assert i18n.normalize_answer("不知道", "size", ["small", "large"], "zh") is None


# --- what actually gets said ------------------------------------------------

def test_the_comparison_table_is_shown_but_not_read_aloud():
    """The table is there so the audience can SEE the difference between
    candidates. Read out loud it is fifteen seconds of attributes that buries
    the question it was supposed to support."""
    reply = ("What colour is yours? (white or blue)\n"
             "Candidate A: white, metal, large, laptop, 703\n"
             "Candidate B: blue, metal, large, laptop, 703")
    assert voice.spoken_form(reply) == "What colour is yours? (white or blue)"


def test_one_very_long_sentence_is_cut_at_a_sentence_end():
    text = ("I found it. " * 40).strip()
    said = voice.spoken_form(text)
    assert len(said) <= voice.SPOKEN_LIMIT
    assert said.endswith(".")


# --- knowing when a sentence has ended --------------------------------------

def _chunks(collector, sample_rate, loud, count, amplitude=0.2):
    """Feed `count` chunks of speech or silence. Returns the phrase, if any."""
    import numpy as np
    rng = np.random.default_rng(0)
    size = int(sample_rate * 0.4)
    result = None
    for _ in range(count):
        data = (rng.normal(0, amplitude, size).astype("float32") if loud
                else np.zeros(size, dtype="float32"))
        result = collector.add(sample_rate, data) or result
    return result


def test_quiet_before_anything_is_said_is_not_a_phrase():
    """The microphone is on for the whole demo. An empty room must never turn
    into a search."""
    collector = voice.PhraseCollector()
    assert _chunks(collector, 48000, loud=False, count=10) is None


def test_speech_then_a_pause_ends_the_sentence():
    """This is the whole hands-free mechanism: you stop talking, it answers.
    No button in between."""
    collector = voice.PhraseCollector()
    assert _chunks(collector, 48000, loud=True, count=3) is None
    phrase = _chunks(collector, 48000, loud=False, count=4)
    assert phrase is not None
    sample_rate, samples = phrase
    assert sample_rate == 48000
    assert len(samples) > 48000        # over a second of audio


def test_a_cough_is_not_a_question():
    collector = voice.PhraseCollector()
    import numpy as np
    collector.add(48000, np.random.default_rng(1).normal(0, 0.3, 4800).astype("float32"))
    assert _chunks(collector, 48000, loud=False, count=4) is None


def test_int16_from_the_browser_is_understood():
    """Some browsers hand over int16 and some float32. Judging int16 samples
    against a 0-1 threshold makes silence look like shouting."""
    import numpy as np
    quiet16 = np.zeros(16000, dtype="int16")
    assert float(np.abs(voice.to_mono_float(quiet16)).mean()) == 0.0
    loud16 = (np.ones(16000) * 8000).astype("int16")
    assert 0.2 < float(np.abs(voice.to_mono_float(loud16)).mean()) < 0.3


def test_stereo_is_mixed_down():
    import numpy as np
    stereo = np.zeros((1000, 2), dtype="float32")
    assert voice.to_mono_float(stereo).shape == (1000,)


# --- "hey lopa" -------------------------------------------------------------

def test_the_name_is_heard_however_whisper_spells_it():
    """Whisper has never met this word and writes it differently every time.
    Korean has no separate l and r, so 로파 comes back as lopa or ropa about
    equally often, and English speakers' "hey lopa" lands somewhere nearby."""
    for said in ("헤이 로파", "헤이로파", "Hey Lopa!", "hey, ropa",
                 "헤이 로빠", "嘿罗帕", "헤이 로퍼"):
        heard, _ = voice.hears_wake_word(said)
        assert heard, said


def test_the_question_after_the_name_is_kept():
    """Waking up and then asking the person to repeat themselves is what makes
    voice assistants tiring. One breath has to be enough."""
    heard, rest = voice.hears_wake_word("헤이 로파, 내 가방 어딨어?")
    assert heard and rest == "내 가방 어딨어?"
    heard, rest = voice.hears_wake_word("Hey Lopa - where is my bottle?")
    assert heard and rest == "where is my bottle?"


def test_any_homophone_of_the_name_wakes_it():
    """The name is not a word in Chinese or Korean, so Whisper picks whichever
    characters sound right - and picks differently every time. Adding the exact
    spelling you just saw is a losing game: the next attempt produces another
    homophone. The syllables are listed and combined instead."""
    for said in ("嘿罗帕", "嘿洛趴", "嘿罗怕", "黑萝爬", "嗨逻巴", "罗帕",
                 "해이 로바", "하이 러퍼", "에이 노파"):
        assert voice.hears_wake_word(said)[0], said


def test_the_name_plus_a_stray_particle_still_just_wakes_it():
    """Whisper rounds a short utterance off with a particle. Searching for
    "呀" answers "I am not sure what you are looking for", which from the other
    side of the room is indistinguishable from never having woken up."""
    gate = voice.WakeGate()
    assert gate.check("嘿罗帕呀")[0] == voice.WakeGate.WOKE
    assert voice.WakeGate().check("헤이 로파아")[0] == voice.WakeGate.WOKE
    # ...but a real question after the name still goes straight through
    action, said = voice.WakeGate().check("嘿罗帕，我的雨伞在哪")
    assert action == voice.WakeGate.SPEAK and said == "我的雨伞在哪"


def test_only_the_three_languages_we_understand_are_chosen():
    """Whisper knows ninety-nine languages and will decide a short, quiet
    Korean sentence was Malay or Japanese. That is not a near miss our matching
    can recover from - it is a different script, so nothing matches, the name is
    never heard, and the room sees a system that ignored someone. Its own
    distribution is asked for and the best of OUR three is taken."""
    assert voice.best_understood(
        {"ms": 0.61, "ja": 0.20, "id": 0.09, "ko": 0.06, "zh": 0.02}) == "ko"
    assert voice.best_understood({"ja": 0.55, "zh": 0.30, "ko": 0.10}) == "zh"
    assert voice.best_understood({"vi": 0.9}) in voice.UNDERSTOOD_LANGUAGES


def test_ordinary_conversation_does_not_wake_it():
    for said in ("오늘 날씨 좋네", "where did I leave my keys", "我们去吃饭吧"):
        heard, _ = voice.hears_wake_word(said)
        assert not heard, said


def test_it_stays_awake_for_the_rest_of_the_exchange():
    """Having to say the name again to answer "blue" would be absurd."""
    gate = voice.WakeGate()
    assert gate.check("오늘 점심 뭐 먹지")[0] == voice.WakeGate.IGNORE
    assert gate.check("헤이 로파")[0] == voice.WakeGate.WOKE
    action, said = gate.check("내 노트북 어딨어?")
    assert action == voice.WakeGate.SPEAK and said == "내 노트북 어딨어?"
    assert gate.check("파란색")[0] == voice.WakeGate.SPEAK


def test_it_goes_back_to_sleep():
    gate = voice.WakeGate(timeout=0)
    gate.check("헤이 로파")
    assert gate.check("파란색")[0] == voice.WakeGate.IGNORE


def test_the_gate_can_be_turned_off_entirely():
    """Rehearsing, or a quiet room where the name is just friction."""
    gate = voice.WakeGate(enabled=False)
    assert gate.check("내 노트북 어딨어?")[0] == voice.WakeGate.SPEAK


# --- pointing at the photos -------------------------------------------------

def test_positional_picking_in_three_languages():
    for text, expected in [("the left one", 0), ("왼쪽거", 0), ("左边的", 0),
                           ("second", 1), ("가운데", 1), ("第二个", 1),
                           ("the last one", 2), ("오른쪽", 2), ("最右", 2)]:
        assert resolve_position(text, 3) == expected, text


def test_a_colour_is_not_a_position():
    assert resolve_position("black", 3) is None
    assert resolve_position("검은색", 3) is None


def test_left_means_left_on_the_desk_when_it_can():
    """With every candidate in one photo, "the left one" should mean the object
    further left on that desk - not just the first thumbnail we happened to
    render."""
    scene = {"scene_id": "703_front"}
    candidates = [
        {"scene": scene, "object": {"id": "right", "bbox": [3000, 100, 3200, 400]}},
        {"scene": scene, "object": {"id": "left", "bbox": [200, 100, 400, 400]}},
        {"scene": scene, "object": {"id": "middle", "bbox": [1500, 100, 1700, 400]}},
    ]
    order = [c["object"]["id"] for c in order_candidates_left_to_right(candidates)]
    assert order == ["left", "middle", "right"]


def test_across_rooms_left_means_the_first_photo():
    """Two different rooms share no left-to-right, so the only honest reading
    of "the left one" is the first thumbnail shown."""
    candidates = [
        {"scene": {"scene_id": "A"}, "object": {"id": "first", "bbox": [3000, 0, 3200, 400]}},
        {"scene": {"scene_id": "B"}, "object": {"id": "second", "bbox": [200, 0, 400, 400]}},
    ]
    order = [c["object"]["id"] for c in order_candidates_left_to_right(candidates)]
    assert order == ["first", "second"]


def test_green_means_the_wake_word_was_heard_and_nothing_else():
    """The light has to mean one thing. It was also green whenever the wake
    word was switched off - so typing a question lit it up, and "Lopa is
    listening" stopped being information."""
    from src import app
    from src.voice import WakeGate

    assert app.banner_state(None) == "asleep"

    gate = WakeGate()
    assert app.banner_state(gate) == "asleep"
    gate.check("헤이 로파")
    assert app.banner_state(gate) == "awake"

    # switched off is its own state, not the same green as being addressed
    off = WakeGate(enabled=False)
    assert app.banner_state(off) == "always"
    assert 'class="lopa-state awake"' not in app.banner_html("always")


def test_the_greeting_alone_wakes_it():
    """People trail off in a demo: the name once, then just "hey" after that."""
    for said in ("헤이", "헤이!", "hey", "Hey!", "嘿", "嗨",
                 "헤이 내 가방 어딨어", "hey where is my bottle"):
        assert voice.hears_wake_word(said)[0], said
    action, rest = voice.WakeGate().check("헤이 내 가방 어딨어")
    assert action == voice.WakeGate.SPEAK and rest == "내 가방 어딨어"


def test_a_greeting_buried_in_a_sentence_does_not():
    """Spacing is thrown away when matching the full name, which is right for
    Korean and Chinese - but "hey" sits inside "t-hey" and "hi" inside "hi-s".
    Matched the same way, the bare greeting would wake on "where did they go":
    exactly the presentation chatter the wake word exists to ignore."""
    for said in ("where did they go", "his bag is black", "the honey jar",
                 "they went to the tearoom", "highlight the laptop"):
        assert not voice.hears_wake_word(said)[0], said


def test_lists_can_be_answered_by_number_or_side_in_any_language():
    """Every question arrives as a numbered list on screen - rooms, colours,
    photos - so people answer it the way people answer lists. Accepting only
    "2", and only while photos were showing, meant "오른쪽 거" was read as a
    colour, matched nothing, and cost a turn."""
    cases = [("오른쪽 거", 2), ("왼쪽거", 0), ("두번째", 1), ("2", 1), ("2번", 1),
             ("2번째", 1), ("第二个", 1), ("最右", 2), ("the second one", 1),
             ("number 3", 2), ("last one", 2)]
    for text, expected in cases:
        assert resolve_position(text, 3) == expected, text


def test_a_colour_is_still_not_a_position():
    for text in ("black", "검은색", "黑色", "透明"):
        assert resolve_position(text, 3) is None, text
