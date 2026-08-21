"""The spoken demo. Talk to it in Korean, Chinese or English.

    python -m src.jarvis              # speak; type if it mishears
    python -m src.jarvis --text       # keyboard only (quiet room not required)
    python -m src.jarvis --lang ko    # force a language instead of detecting

    You: 내 가방 어딨어?
    AI : 3개를 찾았는데 설명만으로는 구분이 안 됩니다. 어느 것인가요?
         1. 검은색 가방 - 703호
         2. 흰색 가방 - 호텔 방
         3. 주황색 가방 - 호텔 방
    You: 왼쪽거
    AI : 그것은 703호에서 노트북 옆에 있습니다.

WHAT THIS FILE IS AND IS NOT

It is a shell. Every decision - which candidates match, whether to ask, what to
ask - still happens in src/agent.py, unchanged and still deterministic. This
file only handles getting words in, putting words out, and choosing a language.

Keeping it separate matters: the demo can be as forgiving as it likes (mishears,
retries, "the left one") without any of that leaking into the logic we have to
defend in the report.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from . import i18n, voice
from .agent import PICK_KEY, Session
from .config import load_config
from .retrieval import SceneIndex
from .visualize import render_result

# "the left one" - people point with words when they can see the photos.
POSITION_WORDS = {
    "first": 0, "1st": 0, "left": 0, "leftmost": 0,
    "첫": 0, "첫번째": 0, "왼쪽": 0, "왼": 0, "제일왼쪽": 0,
    "第一": 0, "左边": 0, "左": 0, "最左": 0,
    "second": 1, "2nd": 1, "middle": 1, "centre": 1, "center": 1,
    "두번째": 1, "가운데": 1, "중간": 1,
    "第二": 1, "中间": 1,
    "third": 2, "3rd": 2,
    "세번째": 2, "第三": 2,
    "last": -1, "right": -1, "rightmost": -1,
    "마지막": -1, "오른쪽": -1, "오른": -1, "제일오른쪽": -1,
    "最后": -1, "右边": -1, "右": -1, "最右": -1,
}

GREETING = {
    "en": "Ready. What are you looking for?",
    "ko": "준비됐습니다. 무엇을 찾으시나요?",
    "zh": "准备好了。您在找什么？",
}

HEARD = {"en": "heard", "ko": "들은 말", "zh": "听到"}


def order_candidates_left_to_right(candidates):
    """Sort the shortlist the way the viewer sees it.

    If every candidate is in the SAME photo, "the left one" means the one
    further left on that desk, so we sort by box position. Across different
    rooms there is no shared left-to-right, so we keep the display order and
    "left" just means the first photo shown.
    """
    scenes = {c["scene"]["scene_id"] for c in candidates}
    if len(scenes) == 1:
        return sorted(candidates, key=lambda c: c["object"]["bbox"][0])
    return list(candidates)


def resolve_position(text, count):
    """"왼쪽거" -> 0, "the last one" -> count-1. None if it is not positional."""
    squashed = "".join(ch for ch in str(text).lower() if ch.isalnum())
    best = None
    for word, index in POSITION_WORDS.items():
        if word in squashed and (best is None or len(word) > len(best[0])):
            best = (word, index)
    if best is None:
        return None
    index = best[1]
    if index < 0:
        index = count + index
    return index if 0 <= index < count else None


def open_file(path):
    try:
        if sys.platform == "win32":
            import os
            os.startfile(str(path))                       # noqa: S606
            return
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.run([opener, str(path)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, AttributeError):
        pass


def show_photos(candidates, cfg, limit=4):
    project_root = Path(cfg["paths"]["index"]).parent.parent
    for position in range(min(limit, len(candidates))):
        out = render_result(candidates, cfg["paths"]["debug"], project_root,
                            focus=position)
        if out:
            open_file(out)


class Conversation:
    """One demo session: language, the agent, and the shortlist on screen."""

    def __init__(self, index, cfg, forced_lang=None, quiet=False):
        self.index = index
        self.cfg = cfg
        self.forced_lang = forced_lang
        self.quiet = quiet
        self.lang = forced_lang or "en"
        self.session = Session(index, cfg)
        self.shown = []            # the shortlist, in the order it is on screen

    # -- output ----------------------------------------------------------
    def respond(self, text, lang=None):
        lang = lang or self.lang
        print(f"AI : {text}")
        if not self.quiet:
            voice.say(text, lang)

    # -- input -----------------------------------------------------------
    def handle(self, text):
        if self.forced_lang is None:
            detected = i18n.detect_language(text)
            # Short replies like "2" or "yes" carry no script, so keep whatever
            # language the conversation is already in.
            if detected != "en" or not self.session.pending_key:
                self.lang = detected

        # "the left one" only means something while photos are on screen
        if self.session.pending_key == PICK_KEY and self.shown:
            position = resolve_position(text, len(self.shown))
            if position is not None:
                text = str(position + 1)

        # The question went out in Korean or Chinese, so the answer comes back
        # that way. The agent only knows canonical English options, so translate
        # the answer before handing it over - otherwise a perfectly good reply
        # is discarded and the agent asks the same thing again.
        elif self.session.pending_key and self.lang != "en":
            matched = i18n.normalize_answer(
                text, self.session.pending_key,
                self.session.pending_options, self.lang)
            if matched is not None:
                text = str(matched)

        reply = (self.session.reply(text) if self.session.pending_key
                 else self.session.start(text))
        return self.render(reply)

    def render(self, reply):
        wanted = self.session._wanted()
        spoken = i18n.localize(reply, self.lang, self.session.parsed, wanted)

        if reply.kind == "choose":
            self.shown = order_candidates_left_to_right(reply.candidates)
            reply.candidates[:] = self.shown          # keep numbering in sync
            self.respond(spoken)
            for line in i18n.shortlist_lines(self.shown, self.lang, wanted):
                print(f"     {line}")
            show_photos(self.shown, self.cfg)
            return reply

        self.shown = []
        self.respond(spoken)
        if reply.candidates and reply.kind != "question":
            show_photos(reply.candidates, self.cfg,
                        limit=1 if reply.kind == "answer" else 3)
        return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="store_true",
                        help="keyboard only - no microphone")
    parser.add_argument("--quiet", action="store_true",
                        help="do not speak the answers")
    parser.add_argument("--lang", choices=["en", "ko", "zh"], default=None,
                        help="force a language instead of detecting it")
    parser.add_argument("--wake", action="store_true",
                        help='stay quiet until it hears "Hey Lopa"')
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        index = SceneIndex.load(cfg["paths"]["index"])
    except FileNotFoundError:
        print("No scene_index.json yet. Run: python -m src.indexer")
        sys.exit(1)

    print(f"Loaded {len(index.scenes)} photos from "
          f"{len(index.locations())} places: {', '.join(index.locations())}")

    speaking = not args.quiet and voice.can_speak()
    if not args.quiet and not speaking:
        print("(no `say` on this machine - answers will be text only)")

    listening = False
    if not args.text:
        if voice.can_listen():
            listening = voice.warm_up(on_progress=print)
        else:
            missing = " ".join(voice.missing_packages())
            print(f"(speech input needs: pip install {missing} - typing instead)")

    conversation = Conversation(index, cfg, forced_lang=args.lang,
                                quiet=not speaking)
    greeting_lang = args.lang or "en"
    conversation.respond(GREETING[greeting_lang], greeting_lang)

    # One keyboard reader for the whole session, not one per turn. Starting a
    # fresh thread each time round the loop leaves every previous one still
    # blocked on input(): after a few turns several threads are competing for
    # stdin and a typed line lands in whichever one happens to win.
    typed = None
    waiting = False
    ear = voice.WakeGate(enabled=args.wake and listening)
    if ear.enabled:
        print('(asleep - say "Hey Lopa" to wake it)')

    while True:
        text = None
        if listening:
            voice.wait_until_quiet()      # do not record our own answer
            if typed is None:
                _, typed = voice.read_line_async()
            if not waiting:
                print("\n[speak - or type and press enter]")
                waiting = True

            heard = voice.listen(on_progress=print)
            if typed.get("line"):
                text = typed["line"]
                typed = None              # consumed; a new reader starts above
            elif heard:
                text, detected = heard
                print(f"You: {text}   ({HEARD.get(conversation.lang, 'heard')}: {detected})")
                action, said = ear.check(text)
                if action == voice.WakeGate.IGNORE:
                    print('     (asleep - say "Hey Lopa")')
                    continue
                if action == voice.WakeGate.WOKE:
                    lang = i18n.detect_language(text)
                    print(i18n.WAKE_PROMPT)
                    if not conversation.quiet:
                        voice.say(i18n.WAKE_REPLY.get(lang,
                                                      i18n.WAKE_REPLY["en"]), lang)
                    continue
                text = said
            if text is None:
                # Silence, or something Whisper could not make out. Go straight
                # back to listening - a demo that needs a keypress between
                # sentences is not a conversation.
                continue
            waiting = False
        else:
            try:
                text = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q", "종료", "退出"):
            break
        conversation.handle(text)


if __name__ == "__main__":
    main()
