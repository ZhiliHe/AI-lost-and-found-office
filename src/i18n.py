"""Answer in the language the question was asked in.

WHY THIS IS A SEPARATE FILE

The demo is spoken, in Korean, Chinese and English. Whisper transcribes but does
not translate, so a query arrives as Korean text and an English answer breaks
the illusion immediately - the whole point is that you talk to it and it talks
back.

Everything here is TEMPLATES, not machine translation. Our agent produces about
seven distinct sentences, all with the same slots, so a table covers them
completely. That matters more than it sounds:

  - it is instant, where an LLM round-trip costs 1-2s on this laptop
  - it says the same thing every time, so a live demo cannot surprise us
  - it never invents a room name or a colour that is not in the index

A model-based translator would be more flexible and strictly worse here.

If a phrase is missing for a language, we fall back to English rather than
showing a broken template - a wrong-language answer is recoverable, a
`{place}` on screen is not.
"""

from .vocab import detect_language, to_simplified  # noqa: F401  (re-exported)

# --- object types -----------------------------------------------------------
TYPE_NAMES = {
    "bottle":     {"ko": "물병",     "zh": "水瓶"},
    "backpack":   {"ko": "가방",     "zh": "背包"},
    "laptop":     {"ko": "노트북",   "zh": "笔记本电脑"},
    "notebook":   {"ko": "노트",     "zh": "笔记本"},
    "book":       {"ko": "책",       "zh": "书"},
    "phone":      {"ko": "휴대폰",   "zh": "手机"},
    "charger":    {"ko": "충전기",   "zh": "充电器"},
    "cable":      {"ko": "케이블",   "zh": "数据线"},
    "mug":        {"ko": "컵",       "zh": "杯子"},
    "keys":       {"ko": "열쇠",     "zh": "钥匙"},
    "wallet":     {"ko": "지갑",     "zh": "钱包"},
    "umbrella":   {"ko": "우산",     "zh": "雨伞"},
    "headphones": {"ko": "이어폰",   "zh": "耳机"},
    "glasses":    {"ko": "안경",     "zh": "眼镜"},
    "mouse":      {"ko": "마우스",   "zh": "鼠标"},
    "pen":        {"ko": "펜",       "zh": "笔"},
    "card":       {"ko": "카드",     "zh": "卡"},
    "tumbler":    {"ko": "텀블러",   "zh": "保温杯"},
    "power bank": {"ko": "보조배터리", "zh": "充电宝"},
    "tablet":     {"ko": "태블릿",   "zh": "平板"},
    "watch":      {"ko": "시계",     "zh": "手表"},
    # Added to the vocabulary after the ground truth was checked against the
    # photos. A type the search can find but i18n cannot name comes back as an
    # English word inside a Korean sentence - "베이지색 fan이 711호에 있습니다" -
    # which looks broken in a way that has nothing to do with what went wrong.
    "fan":        {"ko": "선풍기",   "zh": "风扇"},
    "tissue":     {"ko": "휴지",     "zh": "纸巾"},
    "hair tie":   {"ko": "머리끈",   "zh": "发圈"},
    "skateboard": {"ko": "스케이트보드", "zh": "滑板"},
    "case":       {"ko": "케이스",   "zh": "盒子"},
}

# --- colours ----------------------------------------------------------------
COLOR_NAMES = {
    "black":       {"ko": "검은색",   "zh": "黑色"},
    "white":       {"ko": "흰색",     "zh": "白色"},
    "gray":        {"ko": "회색",     "zh": "灰色"},
    "silver":      {"ko": "은색",     "zh": "银色"},
    "red":         {"ko": "빨간색",   "zh": "红色"},
    "orange":      {"ko": "주황색",   "zh": "橙色"},
    "yellow":      {"ko": "노란색",   "zh": "黄色"},
    "green":       {"ko": "초록색",   "zh": "绿色"},
    "blue":        {"ko": "파란색",   "zh": "蓝色"},
    "purple":      {"ko": "보라색",   "zh": "紫色"},
    "pink":        {"ko": "분홍색",   "zh": "粉色"},
    "brown":       {"ko": "갈색",     "zh": "棕色"},
    "beige":       {"ko": "베이지색", "zh": "米色"},
    "gold":        {"ko": "금색",     "zh": "金色"},
    "transparent": {"ko": "투명한",   "zh": "透明"},
    "multicolor":  {"ko": "여러 색",  "zh": "多色"},
}

# --- materials and sizes, for the clarifying questions ----------------------
MATERIAL_NAMES = {
    "metal":   {"ko": "금속",   "zh": "金属"},
    "plastic": {"ko": "플라스틱", "zh": "塑料"},
    "glass":   {"ko": "유리",   "zh": "玻璃"},
    "fabric":  {"ko": "천",     "zh": "布"},
    "leather": {"ko": "가죽",   "zh": "皮革"},
    "paper":   {"ko": "종이",   "zh": "纸"},
    "wood":    {"ko": "나무",   "zh": "木头"},
}

SIZE_NAMES = {
    "small":  {"ko": "작은",   "zh": "小"},
    "medium": {"ko": "중간",   "zh": "中"},
    "large":  {"ko": "큰",     "zh": "大"},
}

# --- where things are -------------------------------------------------------
# Folder names are for filing ("7f_tearoom"); these are for speaking.
LOCATION_NAMES = {
    "703":             {"ko": "703호",        "zh": "703房间"},
    "711":             {"ko": "711호",        "zh": "711房间"},
    "6f_meetingroom":  {"ko": "6층 회의실",    "zh": "6楼会议室"},
    "7f_tearoom":      {"ko": "7층 탕비실",    "zh": "7楼茶水间"},
    # Two different tables. They used to render identically, so the agent
    # offered "the 8th floor lobby table or the 8th floor lobby table" and the
    # person had no way to choose - the question looked like a bug, and
    # whichever they picked was a coin toss.
    "8f_lobbytable":   {"ko": "8층 로비 테이블 A", "zh": "8楼大堂桌A"},
    "8f_lobbytable_b": {"ko": "8층 로비 테이블 B", "zh": "8楼大堂桌B"},
    "hotelroom":       {"ko": "호텔 방",       "zh": "酒店房间"},
}

# --- spatial relations ------------------------------------------------------
RELATION_NAMES = {
    "on":           {"en": "on top of",     "ko": "위에",     "zh": "上面"},
    "beside":       {"en": "beside",        "ko": "옆에",     "zh": "旁边"},
    "near":         {"en": "near",          "ko": "근처에",   "zh": "附近"},
    "inside":       {"en": "inside",        "ko": "안에",     "zh": "里面"},
    "overlapping":  {"en": "touching",      "ko": "맞닿아",   "zh": "紧挨着"},
    "same_surface": {"en": "on the same surface as", "ko": "같은 곳에", "zh": "同一面上"},
    "left_of":      {"en": "left of",       "ko": "왼쪽에",   "zh": "左边"},
    "right_of":     {"en": "right of",      "ko": "오른쪽에", "zh": "右边"},
    "above":        {"en": "above",         "ko": "위쪽에",   "zh": "上方"},
    "below":        {"en": "below",         "ko": "아래쪽에", "zh": "下方"},
}

# --- sentences --------------------------------------------------------------
# What it says when it hears its name and nothing else.
#
# Spoken: one short line, in whichever language the name was called in. The
# person is mid-thought and about to ask the real question - reading them three
# translations out loud would be six seconds of nothing.
WAKE_REPLY = {
    "en": "Listening. What are you looking for?",
    "ko": "네, 듣고 있습니다. 무엇을 찾으시나요?",
    "zh": "在听，您在找什么？",
}

# Shown: all three at once. The name alone does not say which language the
# person wants, and on a projector this is the moment the audience learns they
# may answer in any of them - which is the point of the whole feature and is
# invisible if we only print the one we guessed.
WAKE_PROMPT = "\n".join((
    "한국어  네, 듣고 있습니다 — 찾으시는 물건을 말씀해 주세요",
    "English  Listening — say what you lost",
    "中文  在听 — 请说出您要找的东西",
))

PHRASES = {
    "found": {
        "en": "Found it - the {what} is {where}.",
        "ko": "찾았습니다. {what}{josa_i} {where} 있습니다.",
        "zh": "找到了。{what}在{where}。",
    },
    "found_unsure": {
        "en": "I think I found it - the {what} is {where}. "
              "Please check the photo.",
        "ko": "아마 이것 같습니다. {what}{josa_i} {where} 있습니다. 사진을 확인해 주세요.",
        "zh": "应该是这个。{what}在{where}。请核对照片。",
    },
    "picked": {
        "en": "That one is {where}.",
        "ko": "그것은 {where} 있습니다.",
        "zh": "那个在{where}。",
    },
    "choose": {
        "en": "I found {n} that I cannot tell apart. Which one is yours?",
        "ko": "{n}개를 찾았는데 설명만으로는 구분이 안 됩니다. 어느 것인가요?",
        "zh": "找到{n}个，光凭描述分不出来。是哪一个？",
    },
    "ask_color": {
        "en": "I found {n} possible matches. What colour is yours? ({options})",
        "ko": "{n}개를 찾았습니다. 무슨 색인가요? ({options})",
        "zh": "找到{n}个。是什么颜色？({options})",
    },
    "ask_size": {
        "en": "I found {n} possible matches. Roughly how big was it? ({options})",
        "ko": "{n}개를 찾았습니다. 크기가 어느 정도였나요? ({options})",
        "zh": "找到{n}个。大概多大？({options})",
    },
    "ask_material": {
        "en": "I found {n} possible matches. What was it made of? ({options})",
        "ko": "{n}개를 찾았습니다. 재질이 무엇이었나요? ({options})",
        "zh": "找到{n}个。是什么材质？({options})",
    },
    "ask_location": {
        "en": "I found {n} possible matches. Which place was it in? ({options})",
        "ko": "{n}개를 찾았습니다. 어디에 두셨나요? ({options})",
        "zh": "找到{n}个。放在哪里了？({options})",
    },
    "ask_near": {
        "en": "I found {n} possible matches. Was it near anything you remember? ({options})",
        "ko": "{n}개를 찾았습니다. 무엇 옆에 있었는지 기억하시나요? ({options})",
        "zh": "找到{n}个。记得旁边有什么吗？({options})",
    },
    "not_found": {
        "en": "I couldn't find a {what}. It may be outside the photographed area "
              "or hidden from view.",
        "ko": "{what}{josa_eul} 찾지 못했습니다. 촬영 범위 밖이거나 가려져 있을 수 있습니다.",
        "zh": "没有找到{what}。可能不在拍摄范围内，或者被挡住了。",
    },
    "not_here": {
        "en": "Then I don't have it. I checked every photo I have of {where}.",
        "ko": "그렇다면 제가 가진 사진에는 없습니다. {where}의 사진은 모두 확인했습니다.",
        "zh": "那我这里没有。{where}的照片我都看过了。",
    },
    # "list every bottle you have" - the opposite of narrowing down, so it gets
    # its own sentence rather than being squeezed into the found/choose ones.
    "found_all": {
        "en": "I found {n} {what}:",
        "ko": "{what} {n}개를 찾았습니다:",
        "zh": "找到{n}个{what}：",
    },
    "give_up": {
        "en": "I still see {n} possible matches. Here are the most likely ones:",
        "ko": "아직 {n}개가 남았습니다. 가장 가능성이 높은 것들입니다:",
        "zh": "还有{n}个。这些是最可能的：",
    },
    "no_target": {
        "en": "I'm not sure what you're looking for. Could you name the item?",
        "ko": "무엇을 찾으시는지 잘 모르겠습니다. 물건 이름을 말씀해 주시겠어요?",
        "zh": "我不确定您在找什么。可以说一下是什么物品吗？",
    },
}

# How choices are joined inside a question. "black or blue" reads fine in
# English and badly in Korean.
OPTION_JOINER = {"en": " or ", "ko": " 아니면 ", "zh": " 还是 "}


def join_options(key, values, lang):
    """"black or blue" / "검은색 아니면 파란색" / "黑色 还是 蓝色"."""
    names = [option_name(key, v, lang) for v in values]
    return OPTION_JOINER.get(lang, " or ").join(str(n) for n in names if n)


# Which phrase belongs to which clarification key.
QUESTION_PHRASE = {
    "color": "ask_color", "size": "ask_size", "material": "ask_material",
    "location": "ask_location", "near": "ask_near",
}


# --- Korean particles -------------------------------------------------------
def has_final_consonant(word):
    """Does the last Hangul syllable end in a consonant?

    Korean picks 이/가 and 을/를 by this, and getting it wrong is the single
    most obvious sign that a sentence was assembled by a machine. "마우스가"
    is right; "마우스이" is not.
    """
    for ch in reversed(str(word)):
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            return (code - 0xAC00) % 28 != 0
        if ch.isalnum():
            # a digit or Latin letter - guess by how it is read aloud
            return ch.lower() in "0136780lmnr"
    return False


def josa(word, with_final, without_final):
    return with_final if has_final_consonant(word) else without_final


# --- lookups ----------------------------------------------------------------
def _name(table, value, lang):
    if not value:
        return ""
    entry = table.get(str(value).lower())
    if not entry:
        return str(value)
    return entry.get(lang) or str(value)


def type_name(value, lang):
    return _name(TYPE_NAMES, value, lang) if lang != "en" else str(value or "")


def color_name(value, lang):
    return _name(COLOR_NAMES, value, lang) if lang != "en" else str(value or "")


def material_name(value, lang):
    return _name(MATERIAL_NAMES, value, lang) if lang != "en" else str(value or "")


def size_name(value, lang):
    return _name(SIZE_NAMES, value, lang) if lang != "en" else str(value or "")


def location_name(value, lang):
    if lang == "en":
        return str(value or "")
    return _name(LOCATION_NAMES, value, lang)


def relation_name(predicate, lang):
    entry = RELATION_NAMES.get(predicate, {})
    return entry.get(lang) or entry.get("en") or str(predicate).replace("_", " ")


def option_name(key, value, lang):
    """A clarification option, in the asking language."""
    if lang == "en":
        return str(value)
    if key == "color":
        return color_name(value, lang)
    if key == "material":
        return material_name(value, lang)
    if key == "size":
        return size_name(value, lang)
    if key == "location":
        return location_name(value, lang)
    if key == "near":
        return type_name(value, lang)
    return str(value)


def describe_object(obj, lang, wanted=None, observed=None):
    """"the blue laptop" / "파란색 노트북" / "蓝色笔记本电脑"."""
    attributes = dict(obj.get("attributes") or {})
    observed = observed or {}
    # honour the word the owner used, if any view agrees with it
    for key, value in (wanted or {}).items():
        if value and not isinstance(value, tuple) and value in (observed.get(key) or {}):
            attributes[key] = value

    colour = color_name(attributes.get("color"), lang)
    kind = type_name(obj.get("type"), lang)
    if lang == "zh":
        return f"{colour}{kind}"
    return " ".join(part for part in (colour, kind) if part)


def phrase(key, lang, **values):
    """Render a sentence. Falls back to English if a language is missing."""
    table = PHRASES.get(key, {})
    template = table.get(lang) or table.get("en") or ""

    if lang == "ko":
        subject = str(values.get("what", ""))
        values.setdefault("josa_i", josa(subject, "이", "가"))
        values.setdefault("josa_eul", josa(subject, "을", "를"))
    else:
        values.setdefault("josa_i", "")
        values.setdefault("josa_eul", "")

    try:
        return template.format(**values)
    except (KeyError, IndexError):
        # never show a raw {slot} to a user - fall back to the English wording
        return (table.get("en") or "").format(**values)


# --- rendering a whole reply ------------------------------------------------
#
# We rebuild the sentence from the Reply's STRUCTURE (kind, candidates,
# asked_key, options), never by translating its English text. Translating
# generated prose back out of English would be fragile and would drift every
# time someone edits a string in agent.py; the structure is stable.


def place_phrase(scene, obj, lang, prefer_anchor=None):
    """'in the 703, beside the laptop' / '703호에서 노트북 옆에'."""
    from .agent import FALLBACK_PREDICATES, SAFE_PREDICATES, nearest_anchor_type

    where = location_name(scene.get("location"), lang)
    anchor, predicate = nearest_anchor_type(
        scene, obj, anchor_filter=prefer_anchor,
        predicate_order=tuple(SAFE_PREDICATES) + tuple(FALLBACK_PREDICATES))

    if lang == "en":
        if not anchor:
            return f"in the {where}"
        relation = relation_name(predicate, "en")
        tail = " in this photo" if predicate in ("left_of", "right_of",
                                                 "above", "below") else ""
        return f"in the {where}, {relation} the {anchor}{tail}"

    if not anchor:
        return f"{where}에" if lang == "ko" else where

    anchor_word = type_name(anchor, lang)
    relation = relation_name(predicate, lang)
    if lang == "ko":
        # a view-dependent relation is only true of the photo we are showing,
        # and Korean has no neat way to slip that in mid-sentence, so we say it
        # outright rather than quietly overstating what we know.
        note = " (이 사진 기준)" if predicate in ("left_of", "right_of",
                                              "above", "below") else ""
        return f"{where}에서 {anchor_word} {relation}{note}"
    note = "（以这张照片为准）" if predicate in ("left_of", "right_of",
                                          "above", "below") else ""
    return f"{where}{anchor_word}{relation}{note}"


# Suffixes a person adds that are not part of the word itself.
_TRIM = ("색", "色", "的", "이야", "예요", "요", "인데", "거", "것", "이", "가", "은", "는")


def _stem(word):
    word = "".join(ch for ch in str(word).lower() if ch.isalnum())
    changed = True
    while changed:
        changed = False
        for suffix in _TRIM:
            if len(word) > len(suffix) + 1 and word.endswith(suffix):
                word = word[: -len(suffix)]
                changed = True
    return word


def normalize_answer(text, key, options, lang):
    """Map a spoken answer back to one of the ENGLISH options.

    The agent's options are canonical English ("metal", "703", "laptop"), but
    the question was asked in Korean or Chinese so the answer comes back that
    way too. Without this the agent silently discards a perfectly good reply
    and asks again - which in a live demo looks exactly like it is broken.

    Returns the matching option, or None if the answer names none of them.
    """
    if not options or lang == "en":
        return None
    # A traditional-character answer has to reach the same options a
    # simplified one does; Whisper chooses the script, the speaker does not.
    said = "".join(ch for ch in to_simplified(text).lower() if ch.isalnum())
    if not said:
        return None

    best = None
    for option in options:
        name = option_name(key, option, lang)
        for candidate in (name, _stem(name)):
            token = "".join(ch for ch in str(candidate).lower() if ch.isalnum())
            if len(token) >= 1 and token in said:
                if best is None or len(token) > best[0]:
                    best = (len(token), option)
    return best[1] if best else None


def describe_candidate(cand, lang, wanted=None):
    return describe_object(cand["object"], lang, wanted=wanted,
                           observed=cand.get("observed"))


def localize(reply, lang, parsed=None, wanted=None):
    """Re-render a Reply in `lang`. Returns the English text unchanged for 'en'.

    Anything we have no template for falls through to the original text, so a
    new reply kind added in agent.py degrades to English instead of vanishing.
    """
    if lang == "en" or not reply:
        return reply.text

    kind = reply.kind
    candidates = reply.candidates or []

    if kind == "answer" and candidates:
        cand = candidates[0]
        what = describe_candidate(cand, lang, wanted)
        where = place_phrase(cand["scene"], cand["object"], lang,
                             getattr(parsed, "anchor_type", None))
        key = "found" if reply.confidence != "medium" else "found_unsure"
        return phrase(key, lang, what=what, where=where)

    if kind == "choose":
        return phrase("choose", lang, n=len(candidates))

    if kind == "question" and reply.asked_key in QUESTION_PHRASE:
        return phrase(QUESTION_PHRASE[reply.asked_key], lang,
                      n=len(candidates),
                      options=join_options(reply.asked_key, reply.options, lang))

    if kind == "list":
        # The header plus one line per match. shortlist_lines already numbers
        # them and names the room in the right language.
        target = getattr(parsed, "target_type", None)
        header = phrase("found_all", lang, n=len(candidates),
                        what=type_name(target, lang) if target else "")
        return "\n".join([header.strip()]
                          + shortlist_lines(candidates, lang, wanted))

    if kind == "giveup":
        return phrase("give_up", lang, n=len(candidates))

    if kind == "none_found":
        if parsed is not None and getattr(parsed, "target_type", None):
            what = describe_object(
                {"type": parsed.target_type,
                 "attributes": {"color": (parsed.attributes or {}).get("color")}},
                lang)
            return phrase("not_found", lang, what=what)
        return phrase("no_target", lang)

    return reply.text


def shortlist_lines(candidates, lang, wanted=None):
    """Numbered one-liners for the pick-from-photos prompt."""
    lines = []
    for position, cand in enumerate(candidates, 1):
        what = describe_candidate(cand, lang, wanted)
        where = location_name(cand["scene"].get("location"), lang)
        lines.append(f"{position}. {what} - {where}")
    return lines
