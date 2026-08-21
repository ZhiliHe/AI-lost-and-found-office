"""Controlled vocabulary.

Why this file exists: a 2B model asked for free-text attributes will call the
same bottle "black", "dark", "charcoal" and "dark gray" across three runs, and
then no query ever matches. We pin the allowed values here, inject them into
the VLM prompt, and normalise anything the model returns anyway.

If you add an object type or colour, add it HERE and nowhere else.
"""

# --- colours the VLM is allowed to use --------------------------------------
COLORS = [
    "black", "white", "gray", "silver", "red", "orange", "yellow",
    "green", "blue", "purple", "pink", "brown", "beige", "gold",
    "transparent", "multicolor",
]

# map sloppy model output -> canonical colour. This is also used for queries,
# so it includes common non-English words where the mapping is unambiguous.
COLOR_ALIASES = {
    "dark": "black", "dark gray": "black", "dark grey": "black",
    "charcoal": "black", "grey": "gray", "light gray": "gray",
    "light grey": "gray", "navy": "blue", "sky blue": "blue",
    "light blue": "blue", "dark blue": "blue", "teal": "blue",
    "maroon": "red", "burgundy": "red", "crimson": "red",
    "clear": "transparent", "see-through": "transparent",
    "cream": "beige", "tan": "beige", "khaki": "beige",
    "golden": "gold", "yellowish": "yellow", "violet": "purple",
}

# --- object types -----------------------------------------------------------
# The key is canonical; the list is what a user might actually type.
OBJECT_SYNONYMS = {
    "bottle":     ["bottle", "water bottle", "flask", "thermos", "tumbler"],
    "backpack":   ["backpack", "bag", "rucksack", "knapsack", "schoolbag"],
    "laptop":     ["laptop", "macbook", "computer", "notebook computer"],
    "notebook":   ["notebook", "notepad", "note book", "writing pad"],
    "book":       ["book", "textbook"],
    "phone":      ["phone", "smartphone", "mobile", "cellphone", "iphone"],
    "charger":    ["charger", "adapter", "power adapter", "charging brick"],
    "cable":      ["cable", "cord", "wire", "usb cable", "charging cable"],
    "mug":        ["mug", "cup", "coffee cup", "tea cup"],
    "keys":       ["keys", "key", "keychain", "keyring"],
    "wallet":     ["wallet", "purse", "billfold"],
    "umbrella":   ["umbrella"],
    "headphones": ["headphones", "earphones", "earbuds", "headset", "airpods"],
    "glasses":    ["glasses", "eyeglasses", "spectacles", "sunglasses"],
    "mouse":      ["mouse", "computer mouse"],
    "pen":        ["pen", "pencil", "marker"],
    "card":       ["card", "id card", "student card", "badge", "lanyard"],
    "tumbler":    ["tumbler", "flask"],
    "power bank": ["power bank", "powerbank", "battery", "portable charger",
                   "external battery"],
    "tablet":     ["tablet", "ipad"],
    "watch":      ["watch", "smartwatch"],
    "skateboard": ["skateboard", "board"],
    "tissue":     ["tissue", "napkin", "serviette", "paper towel", "kitchen roll", "toilet paper", "wet wipe", "Kleenex", "loo roll", "baby wipe"],
    "fan":        ["fan", "hand fan", "handheld fan", "handled fan",
                   "mini fan", "portable fan", "usb fan", "electric fan",
                   "手持风扇", "小风扇", "手持小风扇"],
    # The lavender scrunchie on the 8f table. It is in the index and in the
    # ground truth; without a word for it, it was the one object a person
    # could see in the photo and still not search for.
    "hair tie":   ["hair tie", "hairtie", "scrunchie", "hair band", "hairband",
                   "hair elastic", "bobble", "ponytail holder"],
}

# --- Korean and Chinese, for spoken queries ---------------------------------
#
# The demo is voice-driven in three languages, so a query arrives as Korean or
# Chinese TEXT (Whisper transcribes, it does not translate). These entries are
# what let "내 가방 어딨어?" and "我的包在哪里" reach the same index as
# "where is my bag".
#
# They live in a separate dict from OBJECT_SYNONYMS on purpose: the VLM only
# ever writes English types into the index, so nothing here should be matched
# against model output - only against what a person says.
#
# NOTE on 笔记본 / notebook: Chinese 笔记本 means both "laptop" and "paper
# notebook", exactly like the English word. FUZZY_TYPE_GROUPS already handles
# that ambiguity, so we map it to laptop and let the agent ask.
OBJECT_SYNONYMS_CJK = {
    "bottle":     ["물병", "보틀", "텀블러", "병", "水瓶", "瓶子", "保温杯"],
    "backpack":   ["가방", "백팩", "책가방", "包", "背包", "书包", "袋子"],
    "laptop":     ["노트북", "랩탑", "맥북", "笔记本", "电脑", "笔记本电脑"],
    "phone":      ["폰", "핸드폰", "휴대폰", "스마트폰", "手机", "电话"],
    "charger":    ["충전기", "어댑터", "充电器", "适配器"],
    "cable":      ["케이블", "충전선", "선", "数据线", "充电线", "线"],
    "mug":        ["머그", "머그컵", "컵", "杯子", "马克杯", "水杯"],
    "keys":       ["열쇠", "키링", "钥匙", "钥匙扣"],
    "wallet":     ["지갑", "钱包"],
    "umbrella":   ["우산", "雨伞", "伞"],
    "headphones": ["이어폰", "에어팟", "헤드폰", "이어버즈", "耳机", "耳塞"],
    "glasses":    ["안경", "선글라스", "眼镜", "墨镜"],
    "mouse":      ["마우스", "鼠标"],
    "pen":        ["볼펜", "연필", "笔", "铅笔"],
    "watch":      ["시계", "스마트워치", "手表", "腕表"],
    "tablet":     ["태블릿", "아이패드", "平板"],
    "book":       ["책", "书", "书本"],
    "power bank": ["보조배터리", "充电宝", "移动电源"],
    "card":       ["카드", "명함", "卡", "名片"],
    # Types added to OBJECT_SYNONYMS after the ground truth was checked. An
    # English-only entry is not neutral here: with no Korean word for a fan,
    # "선풍기" was matched by the longest CJK substring we DO know, which is
    # "선" - a cord - so asking for a fan searched for cables and answered
    # about the wrong thing entirely. A missing translation is a wrong answer,
    # not a missing feature.
    "fan":        ["선풍기", "손선풍기", "미니선풍기", "휴대용선풍기", "부채",
                   "风扇", "手持风扇", "小风扇", "电风扇", "扇子"],
    "tissue":     ["휴지", "티슈", "냅킨", "물티슈",
                   "纸巾", "抽纸", "面巾纸", "湿巾"],
    "hair tie":   ["머리끈", "곱창밴드", "헤어끈", "머리띠",
                   "发圈", "发绳", "皮筋"],
    "case":       ["케이스", "파우치", "盒子", "收纳盒"],
    "skateboard": ["스케이트보드", "보드", "滑板"],
}

COLOR_ALIASES_CJK = {
    "검정": "black", "검은": "black", "까만": "black", "블랙": "black",
    "黑": "black", "黑色": "black",
    "흰": "white", "하얀": "white", "화이트": "white", "白": "white", "白色": "white",
    "회색": "gray", "灰": "gray", "灰色": "gray",
    "은색": "silver", "실버": "silver", "银": "silver", "银色": "silver",
    "빨간": "red", "빨강": "red", "레드": "red", "红": "red", "红色": "red",
    "주황": "orange", "오렌지": "orange", "橙": "orange", "橙色": "orange",
    "노란": "yellow", "노랑": "yellow", "黄": "yellow", "黄色": "yellow",
    "초록": "green", "녹색": "green", "그린": "green", "민트": "green",
    "绿": "green", "绿色": "green",
    "파란": "blue", "파랑": "blue", "블루": "blue", "남색": "blue",
    "蓝": "blue", "蓝色": "blue",
    "보라": "purple", "퍼플": "purple", "紫": "purple", "紫色": "purple",
    "분홍": "pink", "핑크": "pink", "粉": "pink", "粉色": "pink",
    "粉红": "pink", "粉红色": "pink",
    "갈색": "brown", "브라운": "brown", "棕": "brown", "棕色": "brown",
    "베이지": "beige", "米色": "beige",
    "금색": "gold", "골드": "gold", "金": "gold", "金色": "gold",
    "투명": "transparent", "透明": "transparent",
}


QUERY_OBJECT_ALIASES = {
    "botle": "bottle", "bottel": "bottle", "bttle": "bottle",
    "labtop": "laptop", "lap top": "laptop", "mac book": "laptop",
    "chargre": "charger", "chager": "charger", "adaptor": "charger",
    "usb": "cable", "cabel": "cable", "kyes": "keys", "key chain": "keys",
}

# Fixtures. The prompt already says to ignore furniture, but a 2B model still
# reported a sink, a faucet, a water cooler and an ice maker in the tea room.
# Nobody loses a sink, so these are dropped at index time - they only waste
# detection slots and clutter the candidate list.
NOT_PORTABLE = {
    # room fixtures the tea room and lobby shots turned up
    "sink", "faucet", "tap", "water cooler", "ice maker", "fridge",
    "refrigerator", "microwave", "kettle", "plant", "pot plant", "radiator",
    "air conditioner", "lamp", "light", "clock", "mirror", "painting",
    "picture", "curtain", "blind", "door", "window", "wall", "floor",
    "ceiling", "desk", "table", "chair", "sofa", "shelf", "cabinet",
    "drawer", "monitor", "screen", "printer", "television", "tv",
    # supplied by the building, not brought by a person
    "hair dryer", "hairdryer", "towel", "soap", "tissue", "bin", "trash can",
}


def is_portable(obj_type):
    """Could a person pick this up and walk off with it?"""
    return str(obj_type or "").strip().lower() not in NOT_PORTABLE
# Categories a small VLM genuinely mixes up. A query for one member matches any
# member of the group, and the agent then asks which one the user meant.
# Discovered from real output: Qwen3-VL-2B labelled a tall water bottle a "mug".
FUZZY_TYPE_GROUPS = [
    {"bottle", "tumbler", "mug", "cup"},
    {"charger", "cable", "power bank"},
]

# NOTE for the team: "notebook" is deliberately ambiguous (laptop vs paper
# notebook). Keep it that way - it is a free source of genuinely hard queries
# for the demo and the evaluation set.

OBJECT_TYPES = sorted(OBJECT_SYNONYMS.keys())

_SYNONYM_LOOKUP = {
    syn.lower(): canonical
    for canonical, syns in OBJECT_SYNONYMS.items()
    for syn in syns
}
_SYNONYM_LOOKUP.update({k.lower(): v for k, v in QUERY_OBJECT_ALIASES.items()})

# CJK synonyms are kept in their OWN lookup because they are matched
# differently - see _has_cjk below.
_CJK_SYNONYM_LOOKUP = {
    syn: canonical
    for canonical, syns in OBJECT_SYNONYMS_CJK.items()
    for syn in syns
}


def _has_cjk(text):
    """Does this string contain Korean, Chinese or Japanese characters?

    It matters because those scripts have no reliable word boundaries. English
    needs " pen " so that "open" does not match; Chinese writes 我的包在哪里
    with no spaces at all, so the same rule would find nothing. CJK terms are
    therefore matched as plain substrings.
    """
    for ch in str(text):
        code = ord(ch)
        if (0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF
                or 0x3130 <= code <= 0x318F          # Hangul
                or 0x4E00 <= code <= 0x9FFF          # CJK ideographs
                or 0x3040 <= code <= 0x30FF):        # kana
            return True
    return False


def _scan_cjk(text, table):
    """Left-to-right longest match. Returns canonical values, in order.

    A plain "longest key first" loop is not enough: 红色 is a key and so is
    粉红, so "粉红色" matched RED before anything looked at PINK. Scanning
    position by position and taking the longest key that starts there gets
    粉红 first and leaves nothing behind for 红 to grab.
    """
    raw = str(text)
    longest = max((len(k) for k in table), default=1)
    hits = []
    position = 0
    while position < len(raw):
        for size in range(min(longest, len(raw) - position), 0, -1):
            chunk = raw[position:position + size]
            if chunk in table:
                value = table[chunk]
                if value not in hits:
                    hits.append(value)
                position += size
                break
        else:
            position += 1
    return hits


def detect_language(text):
    """'ko' | 'zh' | 'en'. Used to answer in the language that was asked."""
    hangul = sum(1 for ch in str(text)
                 if 0xAC00 <= ord(ch) <= 0xD7A3 or 0x3130 <= ord(ch) <= 0x318F)
    han = sum(1 for ch in str(text) if 0x4E00 <= ord(ch) <= 0x9FFF)
    if hangul:
        return "ko"
    if han:
        return "zh"
    return "en"

# --- attribute keys the VLM should try to fill ------------------------------
ATTRIBUTE_KEYS = ["color", "material", "size"]

MATERIALS = ["metal", "plastic", "glass", "fabric", "leather", "paper", "wood", "unknown"]
SIZES = ["small", "medium", "large"]

# Constraint sentinel: the user ruled a value OUT. Stored as ("not", "metal")
# when someone answers "no" to "is it metal?" - which is real information, and
# used to be thrown away.
#
# There is deliberately no "this attribute is absent" sentinel any more. Every
# attribute we still index (colour, material, size) is a property every object
# HAS; a null means "the model could not tell", never "the object lacks it".
# The two attributes where null meant something - brand_or_logo and state -
# were both removed for being unreliable.
NOT = "not"


def negate(value):
    return (NOT, value)


def is_negated(value):
    return isinstance(value, tuple) and len(value) == 2 and value[0] == NOT


def normalize_color(value):
    """'Dark Grey ' -> 'black'.  Unknown values pass through lowercased."""
    if not value:
        return None
    v = str(value).strip().lower()
    v = COLOR_ALIASES.get(v, v)
    return v


def normalize_object_type(value):
    """'water bottle' -> 'bottle'.  Unknown values pass through lowercased."""
    if not value:
        return None
    v = str(value).strip().lower()
    if v in _SYNONYM_LOOKUP:
        return _SYNONYM_LOOKUP[v]
    # try the longest matching synonym contained in the string
    best = None
    for syn, canonical in _SYNONYM_LOOKUP.items():
        if syn in v and (best is None or len(syn) > len(best[0])):
            best = (syn, canonical)
    return best[1] if best else v


def _padded(text):
    """Lowercase, punctuation -> spaces, wrapped in spaces.

    Without this, "Where is my bottle?" never matches " bottle " and the parser
    silently finds no target. Learned the hard way; keep it.
    """
    cleaned = "".join(ch if ch.isalnum() else " " for ch in str(text).lower())
    return " " + " ".join(cleaned.split()) + " "


def _close_token_match(token, choices, max_distance=1):
    """Return a vocabulary word close to a user's typo, or None.

    Kept deliberately conservative: only single-token words of length >= 5 are
    eligible, so "pen" does not start matching arbitrary short words.
    """
    import difflib

    if len(token) < 5:
        return None
    matches = difflib.get_close_matches(token, choices, n=1, cutoff=0.84)
    if not matches:
        return None
    candidate = matches[0]
    if abs(len(candidate) - len(token)) > max_distance:
        return None
    return candidate


def find_object_types(text):
    """Return every canonical object type mentioned in a free-text string,
    longest match first so 'notebook computer' beats 'notebook'.

    Note what this does NOT do: it never parses grammar. "Where is my laptop?",
    "where my laptop" and "yo where'd my macbook go" all work for the same
    reason - everything except the known noun is thrown away. That is why
    phrasing is irrelevant and why an unknown word is fatal.
    """
    hits = []

    # CJK first, by substring - those scripts have no word boundaries to use.
    # Longest match wins so 笔记本电脑 beats 电脑, and 책가방 beats 가방.
    if _has_cjk(text):
        hits = _scan_cjk(text, _CJK_SYNONYM_LOOKUP)
        if hits:
            return hits

    t = _padded(text)
    for syn in sorted(_SYNONYM_LOOKUP, key=len, reverse=True):
        if f" {syn} " in t or f" {syn}s " in t:
            canonical = _SYNONYM_LOOKUP[syn]
            if canonical not in hits:
                hits.append(canonical)
            t = t.replace(f" {syn} ", " ").replace(f" {syn}s ", " ")

    if hits:
        return hits

    # Nothing matched on word boundaries. If the user typed the whole thing as
    # one run-on token ("Whereismylaptop"), fall back to substring search.
    # ONLY in that case - doing this generally would match "pen" inside "open"
    # and "key" inside "keyboard".
    single_token = t.strip()
    if single_token and " " not in single_token and len(single_token) > 6:
        for syn in sorted(_SYNONYM_LOOKUP, key=len, reverse=True):
            if len(syn) >= 5 and syn in single_token:
                return [_SYNONYM_LOOKUP[syn]]

    for token in t.strip().split():
        match = _close_token_match(token, [s for s in _SYNONYM_LOOKUP if " " not in s])
        if match:
            canonical = _SYNONYM_LOOKUP[match]
            if canonical not in hits:
                hits.append(canonical)
    return hits


def find_colors(text):
    """Return every colour mentioned in a free-text string."""
    hits = []

    # CJK by substring, for the same reason as find_object_types: "검은색" is
    # written without spaces around it, and Chinese has no spaces at all.
    # Longest first so 粉红 beats 红 and 파란색 beats 파란.
    if _has_cjk(text):
        hits = _scan_cjk(text, COLOR_ALIASES_CJK)
        if hits:
            return hits

    t = _padded(text)
    for name in sorted(list(COLOR_ALIASES) + COLORS, key=len, reverse=True):
        if f" {name} " in t:
            c = normalize_color(name)
            if c not in hits:
                hits.append(c)
            t = t.replace(f" {name} ", " ")
    for token in t.strip().split():
        match = _close_token_match(token, COLORS)
        if match and match not in hits:
            hits.append(match)
    return hits
