"""Gradio demo UI.

    python -m src.app

Left: the conversation. Right: the scene image with the current candidates
highlighted, so the audience can see the candidate set shrink as the agent asks
questions. That shrinking is the story of the demo - make sure it is visible.
"""

import os
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from . import i18n, voice
from .agent import PICK_KEY, Session, describe_object, describe_place
from .config import load_config
from .jarvis import order_candidates_left_to_right, resolve_position
from .retrieval import SceneIndex

# LOPA_DEBUG=1 prints the live microphone level, so a mic that is simply too
# quiet can be told apart from one that is not connected at all.
DEBUG_AUDIO = bool(os.environ.get("LOPA_DEBUG"))
_LAST_MIC_MODE = None       # True streaming, False press-to-record, None none

CFG = load_config()
PROJECT_ROOT = Path(CFG["paths"]["index"]).parent.parent
IMAGE_ROOT = Path(CFG["paths"]["images"])
INDEX = None
HIGHLIGHT = (255, 87, 51)
DIMMED = (120, 120, 120)
APP_CSS = """
:root {
    --bg: #f5f1e7;
    --panel: #fffaf0;
    --ink: #2d2218;
    --muted: #6a5b4d;
    --brand: #e55d2d;
    --brand-soft: #f7c9b6;
    --edge: #d9c8b6;
}

#app-root {
    background:
        radial-gradient(circle at 12% -5%, #ffe2b8 0%, transparent 42%),
        radial-gradient(circle at 100% 0%, #ffd8ce 0%, transparent 36%),
        var(--bg);
    min-height: 100vh;
}

#hero {
    margin-bottom: 10px;
    padding: 16px 18px;
    border: 1px solid var(--edge);
    border-radius: 14px;
    background: linear-gradient(135deg, #fffefb 0%, #fff4e6 100%);
    box-shadow: 0 8px 24px rgba(120, 80, 40, 0.08);
}

#hero h1 {
    margin: 0;
    color: var(--ink);
    letter-spacing: 0.02em;
}

#hero p,
#hero li {
    color: var(--muted);
}

#left-panel,
#right-panel {
    border: 1px solid var(--edge);
    border-radius: 16px;
    padding: 10px;
    background: var(--panel);
    box-shadow: 0 10px 24px rgba(82, 59, 37, 0.08);
}

#send-btn {
    background: linear-gradient(120deg, #f17741 0%, #db5425 100%) !important;
    border: none !important;
    color: #fff !important;
}

#send-btn:hover {
    filter: brightness(1.03);
}

#reset-btn {
    border: 1px solid var(--edge) !important;
    color: var(--ink) !important;
}

#status {
    margin-top: 6px;
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px solid var(--edge);
    background: #fff;
}

/* The page is a light design. Chrome follows the operating system, so on a
   dark-mode laptop Gradio painted its own text white and left our cream
   panels underneath - white on cream, unreadable, and only on someone else's
   machine. Pinning the colours we depend on is the fix; inheriting a theme we
   did not design for is not something to discover on a projector. */
#app-root,
#app-root .prose,
#app-root label,
#app-root p,
#app-root span,
#app-root li,
#app-root h1, #app-root h2, #app-root h3, #app-root strong {
    color: var(--ink);
}

#app-root .message-wrap .message,
#app-root .bubble-wrap .message {
    color: var(--ink) !important;
}

#hero h1, #hero p, #hero strong, #hero em, #hero li {
    color: var(--ink);
}

#hero p, #hero li {
    color: var(--muted);
}

#status, #status p, #status strong {
    color: var(--ink);
    background: #fff;
}

#wake-banner {
    margin: 0 0 10px 0;
}

#wake-banner .lopa-state {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    border-radius: 12px;
    border: 1px solid var(--edge);
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}

#wake-banner .lopa-state .dot {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    flex: none;
}

#wake-banner .awake {
    background: linear-gradient(120deg, #eaf7ee 0%, #f4fbf6 100%);
    border-color: #9ad0ab;
    color: #1d6b34;
}

#wake-banner .awake .dot {
    background: #2f9e51;
    animation: lopa-pulse 1.4s ease-in-out infinite;
}

#wake-banner .asleep {
    background: #fffaf0;
    color: var(--muted);
}

#wake-banner .asleep .dot {
    background: #c3b4a4;
}

#wake-banner .always {
    background: #fff6e8;
    border-color: #e8c79a;
    color: #8a5a1c;
}

#wake-banner .always .dot {
    background: #d9932f;
}

@keyframes lopa-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.35; transform: scale(0.82); }
}

@media (max-width: 900px) {
    #hero {
        padding: 14px;
    }
}
"""


def _resolve(image_path):
    path = Path(image_path)
    if path.is_absolute():
        return path
    candidates = (
        PROJECT_ROOT / path,
        IMAGE_ROOT / path,
        IMAGE_ROOT.parent / path,
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def render_candidates(candidates, focus=0):
    """Draw the top candidates. The focused one is highlighted, the rest dimmed."""
    if not candidates:
        return None
    scene = candidates[focus]["scene"]
    path = _resolve(scene["image_path"])
    if not path.exists():
        return None

    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for position, cand in enumerate(candidates):
        if cand["scene"]["scene_id"] != scene["scene_id"]:
            continue
        x1, y1, x2, y2 = cand["object"]["bbox"]
        is_focus = position == focus
        draw.rectangle([x1, y1, x2, y2],
                       outline=HIGHLIGHT if is_focus else DIMMED,
                       width=6 if is_focus else 3)
    return image


def gallery_for(candidates, wanted=None, lang="en"):
    """One thumbnail per candidate, across scenes.

    Captions follow the language of the question: being asked "몇 번이요?" and
    then having to read English labels to answer is the one place where the
    demo would visibly stop being multilingual.
    """
    items = []
    for position, cand in enumerate(candidates[:6]):
        image = render_candidates(candidates, focus=position)
        if image is None:
            continue
        if lang == "en":
            caption = (f"{describe_object(cand['object'], cand, wanted)} - "
                       f"{describe_place(cand['scene'], cand['object'])}")
        else:
            caption = (f"{position + 1}. "
                       f"{i18n.describe_candidate(cand, lang, wanted)} - "
                       f"{i18n.location_name(cand['scene'].get('location'), lang)}")
        items.append((image, caption))
    return items


def _chatbot(**kwargs):
    """Gradio 4/5 need type="messages" explicitly; Gradio 6 removed the argument.
    Teammates will not all have the same version, so try both."""
    try:
        return gr.Chatbot(type="messages", **kwargs)
    except TypeError:
        return gr.Chatbot(**kwargs)


MIC_LABEL = 'Listening - say "Hey Lopa" (English / 한국어 / 中文)'


def _supports_streaming():
    """Can this Gradio stream the microphone? Asked without building anything,
    because the banner is laid out before the microphone is."""
    import inspect
    try:
        return "streaming" in inspect.signature(gr.Audio.__init__).parameters
    except (TypeError, ValueError):
        return False


def banner_html(state):
    """The one thing the room needs to know: is it listening to me right now?

    A voice demo has no cursor and no button being pressed, so without this the
    audience cannot tell "it is thinking" from "it never heard you" - and
    neither can the presenter, which is worse.

    Green means one specific thing: the wake word was heard and it is awake.
    Not "the microphone is on", and not "the wake word is switched off, so
    everything counts" - that state gets its own, quieter line. Once the light
    means two things it stops being worth looking at.
    """
    if state == "awake":
        return ('<div class="lopa-state awake"><span class="dot"></span>'
                'Lopa is listening</div>')
    if state == "always":
        return ('<div class="lopa-state always"><span class="dot"></span>'
                'Wake word off - Lopa answers anything it hears</div>')
    return ('<div class="lopa-state asleep"><span class="dot"></span>'
            'Asleep - say "Hey Lopa"</div>')


def banner_state(ear):
    """asleep | awake | always - from the gate itself, never from typing."""
    if ear is None:
        return "asleep"
    if not ear.enabled:
        return "always"
    return "awake" if ear.awake else "asleep"


def _checkbox(**kwargs):
    """A checkbox that explains itself, on a Gradio that supports it.

    `info` renders as a line of grey help text under the label. Older versions
    do not take the argument, and a switch nobody can interpret is still better
    than a page that will not build.
    """
    try:
        return gr.Checkbox(**kwargs)
    except TypeError:
        kwargs.pop("info", None)
        return gr.Checkbox(**kwargs)


def _microphone():
    """A microphone, if this Gradio and this machine can offer one.

    Streaming is what makes it hands-free: recording starts once and stays on,
    and src/voice.py decides where each sentence ends. The non-streaming form
    is kept as a fallback - the argument names moved between Gradio versions
    and not everyone on the team has the same one. A missing microphone must
    cost us the button, never the page.
    """
    attempts = ({"sources": ["microphone"], "type": "numpy", "streaming": True},
                {"source": "microphone", "type": "numpy", "streaming": True},
                {"sources": ["microphone"], "type": "filepath"},
                {"source": "microphone", "type": "filepath"})
    for kwargs in attempts:
        try:
            component = gr.Audio(label=MIC_LABEL, elem_id="mic", **kwargs)
        except TypeError:
            continue
        global _LAST_MIC_MODE
        _LAST_MIC_MODE = bool(kwargs.get("streaming"))
        return component, _LAST_MIC_MODE
    return None, False


# Press Record for the presenter, as soon as the page opens.
#
# Chrome will not let a page open a microphone silently - it shows its
# permission prompt, and answering that prompt IS the user's consent. What it
# does not require is a click on OUR button, so there is no reason to make
# someone walk to the laptop and press Record in front of an audience. After
# the first Allow, Chrome remembers the choice for this address and later
# openings start listening with nothing pressed at all.
#
# If any of this fails - a different Gradio version renaming the button, a
# denied permission - nothing breaks: the Record button is still sitting there
# to be pressed by hand, exactly as before.
AUTO_RECORD_JS = """
() => {
  const label = (b) =>
    ((b.getAttribute('aria-label') || '') + ' ' + (b.textContent || '')).trim();
  const press = () => {
    const panel = document.querySelector('#mic');
    if (!panel) return false;
    const buttons = Array.from(panel.querySelectorAll('button'));
    if (!buttons.length) return false;
    // Already running - a Stop control means the stream is open.
    if (buttons.some(b => /stop|정지/i.test(label(b)))) return true;
    // Gradio has renamed and re-iconed this button across versions, so match
    // by name first and fall back to "the one that is not any of the other
    // controls". Guessing wrong costs a harmless click; not finding it at all
    // costs the hands-free demo.
    let target = buttons.find(b => /record|녹음|录音/i.test(label(b)));
    if (!target) {
      target = buttons.find(b =>
        !/clear|remove|play|pause|download|undo|trim|edit|지우|재생/i.test(label(b)));
    }
    if (!target) return false;
    console.log('[lopa] starting the microphone:', label(target) || '(unnamed)');
    target.click();
    return true;
  };
  let tries = 0;
  const timer = setInterval(() => {
    if (press() || ++tries > 40) clearInterval(timer);
  }, 250);
}
"""


# Gradio follows the operating system's dark mode by adding a "dark" class to
# the body, and then paints its own panels near-black. This page is a light
# design - cream panels, dark ink - so on a dark-mode laptop the two fight:
# Gradio's white label text lands on our cream panel and our dark heading lands
# on Gradio's near-black block. Half the page becomes unreadable, and only on
# someone else's machine.
#
# Rather than write a second full theme the night before a demo, we hold the
# page in the light mode it was designed for. The observer is there because
# Gradio re-applies the class on some navigations.
FORCE_LIGHT_JS = """
() => {
  const strip = () => {
    document.body.classList.remove('dark');
    document.documentElement.classList.remove('dark');
    const app = document.querySelector('gradio-app');
    if (app) app.classList.remove('dark');
  };
  strip();
  new MutationObserver(strip).observe(document.body, {
    attributes: true, attributeFilter: ['class'],
  });
}
"""


def build():
    global INDEX
    INDEX = SceneIndex.load(CFG["paths"]["index"])
    locations = ", ".join(INDEX.locations())

    with gr.Blocks(title="AI Lost & Found", css=APP_CSS, elem_id="app-root") as demo:
        gr.Markdown(
            f"""
<div id="hero">
  <h1>AI Lost &amp; Found</h1>
  <p><strong>{len(INDEX.scenes)} scenes</strong> indexed across: {locations}</p>
  <p>Model used for indexing: <strong>{INDEX.payload.get('model')}</strong></p>
  <p>Try: <em>Where is my bottle?</em> | <em>Find the black bottle beside the laptop</em> | <em>I lost a black backpack</em></p>
  <p>Ask in <strong>English</strong>, <strong>한국어</strong> or <strong>中文</strong> - by keyboard or out loud. It answers in the language you asked.</p>
</div>
"""
        )

        session_state = gr.State(None)
        # Built before the microphone so it sits at the top of the page, which
        # is where a room full of people will look for it.
        wake_banner = gr.HTML(banner_html("asleep"), elem_id="wake-banner",
                              visible=_supports_streaming())

        with gr.Row():
            with gr.Column(scale=1, elem_id="left-panel"):
                chat = _chatbot(height=440, label="Conversation")
                box = gr.Textbox(
                    placeholder="Where is my bottle?",
                    label="Describe what you lost",
                    autofocus=True,
                )
                with gr.Row():
                    send = gr.Button("Send", variant="primary", elem_id="send-btn")
                    reset = gr.Button("New search", elem_id="reset-btn")
                mic, hands_free = _microphone()
                # A button that does what opening the page is supposed to do
                # on its own. Autostart depends on the browser having been
                # given permission already, and on Gradio not renaming its
                # record button - neither is something to discover in front of
                # an audience. One visible click always works.
                arm = gr.Button("🎤 Start listening", visible=hands_free)
                listening = gr.State(None)
                gate = gr.State(None)
                wake_only = _checkbox(
                    value=True,
                    label='Wake word: only answer after "Hey Lopa"',
                    info="The microphone stays open the whole time. With this "
                         "on, it acts only on sentences addressed to it - "
                         "everything else said in the room is ignored.",
                    visible=hands_free,
                )
                speak_back = _checkbox(
                    value=voice.can_speak(),
                    label="Speak answers out loud",
                    info="Reads the reply in the language it was asked in. "
                         "The photos and text appear either way.",
                    interactive=voice.can_speak(),
                )
            with gr.Column(scale=1, elem_id="right-panel"):
                view = gr.Gallery(
                    label="Candidate matches",
                    height=440,
                    columns=2,
                    object_fit="contain",
                    preview=True,
                    allow_preview=True,
                )
                status = gr.Markdown("", elem_id="status")

        def respond(message, history, session, speak):
            message = (message or "").strip()
            if not message:
                return history, session, gr.update(), gr.update()

            # A new question means the previous answer is stale - stop saying it
            # before we start saying this one.
            voice.stop()
            history = list(history or [])
            history.append({"role": "user", "content": message})

            if session is None:
                session = Session(INDEX, CFG)
                session.demo_lang = "en"

            # Answer in the language that was used to ask. Short replies like
            # "2" or "blue" carry no script of their own, so mid-conversation
            # we keep whatever language the conversation is already in.
            detected = i18n.detect_language(message)
            if detected != "en" or not session.pending_key:
                session.demo_lang = detected
            lang = getattr(session, "demo_lang", "en")

            # "the left one" / "왼쪽거" / "左边的" - people point with words
            # once the photos are on screen.
            asked = message
            if session.pending_key == PICK_KEY and session.candidates:
                position = resolve_position(message, len(session.candidates))
                if position is not None:
                    asked = str(position + 1)
            elif session.pending_key and session.pending_options:
                # Every question the agent asks arrives as a numbered list on
                # screen - rooms, colours, neighbours - so people answer it the
                # way people answer lists: "2", "두 번째", "the last one". Only
                # accepting that while photos were showing meant "오른쪽 거" was
                # read as a colour, matched nothing, and cost a turn.
                position = resolve_position(message, len(session.pending_options))
                if position is not None:
                    asked = str(session.pending_options[position])
                elif lang != "en":
                    matched = i18n.normalize_answer(
                        message, session.pending_key, session.pending_options, lang)
                    if matched is not None:
                        asked = str(matched)
            elif session.pending_key and lang != "en":
                # The question went out in Korean, so the answer comes back in
                # Korean - but the agent only knows canonical English values.
                matched = i18n.normalize_answer(
                    message, session.pending_key, session.pending_options, lang)
                if matched is not None:
                    asked = str(matched)

            # Always through reply(): it delegates to start() for a genuinely
            # new question, and otherwise keeps the search we are already in.
            # Calling start() directly whenever nothing was pending threw away
            # the object named three sentences ago, so "the green one" after a
            # shortlist answered "I'm not sure what you're looking for".
            reply = session.reply(asked)

            spoken = i18n.localize(reply, lang, session.parsed, session._wanted())
            history.append({"role": "assistant", "content": spoken})
            if speak:
                voice.say(spoken, lang)

            note = f"**{len(reply.candidates)}** candidate(s) · turn {session.turns}/{session.max_turns}"
            if session.constraints:
                known = ", ".join(f"{k}={v}" for k, v in session.constraints.items())
                note += f" · known: {known}"
            if reply.kind == "answer":
                note = "**Resolved.** " + note

            # Keep the final image visible even if a reply omits candidates.
            shown_candidates = reply.candidates or session.candidates
            if reply.kind == "choose" and shown_candidates:
                # Number the photos the way the viewer reads them, so "the left
                # one" means the object further left on the desk.
                shown_candidates = order_candidates_left_to_right(shown_candidates)
                reply.candidates[:] = shown_candidates
                session.candidates[:] = shown_candidates
            return history, session, \
                gallery_for(shown_candidates, session._wanted(), lang), note

        def clear():
            return [], None, [], ""

        def from_gallery(event: gr.SelectData, history, session, speak):
            """Clicking a photo means "that one is mine".

            It was only enlarging the picture. Pointing is the most direct
            answer available - shorter than any sentence, impossible to
            mishear, and the same gesture in every language - so it should
            settle the search, not zoom it.
            """
            if session is None or event is None:
                return gr.update(), session, gr.update(), gr.update()
            voice.stop()
            reply = session.pick_by_index(event.index)
            if reply is None:
                return gr.update(), session, gr.update(), gr.update()

            lang = getattr(session, "demo_lang", "en")
            spoken = i18n.localize(reply, lang, session.parsed, session._wanted())
            history = list(history or []) + [
                {"role": "user", "content": "(picked from the photos)"},
                {"role": "assistant", "content": spoken},
            ]
            if speak:
                voice.say(spoken, lang)
            shown = reply.candidates or session.candidates
            note = f"**Resolved.** **{len(reply.candidates)}** candidate(s)"
            return (history, session,
                    gallery_for(shown, session._wanted(), lang), note)

        def from_microphone(path):
            """Recording -> text in the box. The search itself runs in respond()."""
            if not path:
                return "", gr.update()
            if not voice.can_listen():
                missing = " ".join(voice.missing_packages())
                return "", f"Speech input needs: `pip install {missing}`"
            try:
                text, detected = voice.transcribe(path)
            except Exception as exc:                          # noqa: BLE001
                return "", f"Could not transcribe: {exc}"
            if not text:
                return "", "Heard nothing - try again, or just type it."
            return text, f"heard ({detected}): {text}"

        send.click(respond, [box, chat, session_state, speak_back],
                   [chat, session_state, view, status]) \
            .then(lambda: "", None, box)
        box.submit(respond, [box, chat, session_state, speak_back],
                   [chat, session_state, view, status]) \
           .then(lambda: "", None, box)
        reset.click(clear, None, [chat, session_state, view, status])
        try:
            view.select(from_gallery, [chat, session_state, speak_back],
                        [chat, session_state, view, status])
        except (AttributeError, TypeError):
            pass          # older Gradio: the photos stay click-to-enlarge only
        def from_stream(chunk, collector, ear, history, session, speak, wake):
            """One chunk of live microphone audio.

            Most calls do nothing: the person is not talking, or we are, or it
            has not heard its name yet. Only when voice.PhraseCollector says a
            sentence has ended AND the wake gate lets it through does this turn
            into a search - which is the point, since it means nobody has to
            press anything between questions.
            """
            if collector is None:
                collector = voice.PhraseCollector()
            if ear is None:
                ear = voice.WakeGate()
            ear.enabled = bool(wake)
            # Recomputed every chunk, so the banner goes back to "asleep" on
            # its own when the conversation stops - no extra timer, and the
            # room can see the state change as it happens.
            state = banner_html(banner_state(ear))
            idle = (gr.update(), session, gr.update(), gr.update(),
                    collector, ear, state)
            if chunk is None:
                return idle

            # Never listen to ourselves: the laptop speaker is louder to this
            # microphone than the person is, and Whisper will happily
            # transcribe our own answer back as the next question.
            if voice.is_talking():
                collector.reset()
                return idle

            sample_rate, samples = chunk
            phrase = collector.add(sample_rate, samples)
            if phrase is None:
                if DEBUG_AUDIO:
                    print(f"  mic level {collector.level:.4f}  "
                          f"threshold {collector.threshold():.4f}  "
                          f"{'SPEECH' if collector.heard_speech else 'quiet'}")
                return idle

            try:
                text, detected = voice.transcribe_samples(phrase[1], phrase[0])
            except Exception as exc:                      # noqa: BLE001
                return (gr.update(), session, gr.update(),
                        f"Could not transcribe: {exc}", collector, ear, state)
            if not text:
                return idle

            action, said = ear.check(text)
            # Always print it. When a wake word is not recognised, the only
            # thing that helps is seeing the exact string Whisper produced -
            # then it goes in voice.WAKE_WORDS and never misses again.
            print(f'heard [{detected}]: "{text}"   -> {action}')
            if action == voice.WakeGate.IGNORE:
                # Heard, understood, and deliberately not acted on. Say so in
                # the status line: during a demo, "it ignored me" and "it is
                # broken" look identical from the audience.
                return (gr.update(), session, gr.update(),
                        f'(asleep - say "Hey Lopa") heard: {text}',
                        collector, ear, banner_html("asleep"))
            if action == voice.WakeGate.WOKE:
                lang = i18n.detect_language(text)
                if speak:
                    voice.say(i18n.WAKE_REPLY.get(lang, i18n.WAKE_REPLY["en"]), lang)
                woken = list(history or []) + [
                    {"role": "assistant", "content": i18n.WAKE_PROMPT}]
                return (woken, session, gr.update(), "listening...",
                        collector, ear, banner_html(banner_state(ear)))

            history, session, gallery, note = respond(said, history, session, speak)
            return (history, session, gallery, note, collector, ear,
                    banner_html(banner_state(ear)))

        if mic is not None and hands_free:
            mic.stream(
                from_stream,
                [mic, listening, gate, chat, session_state, speak_back, wake_only],
                [chat, session_state, view, status, listening, gate, wake_banner],
                stream_every=0.4,
                show_progress="hidden",
            )
        elif mic is not None:
            try:
                # Talking over the person the moment they start speaking is the
                # single most annoying thing a voice demo can do.
                mic.start_recording(lambda: voice.stop(), None, None)
            except AttributeError:
                pass
            mic.stop_recording(from_microphone, mic, [box, status]) \
               .then(respond, [box, chat, session_state, speak_back],
                     [chat, session_state, view, status]) \
               .then(lambda: "", None, box) \
               .then(lambda: None, None, mic)
            # Clearing the recording is what makes the next turn one click.
            # Gradio keeps the finished clip in the component and offers "play"
            # until it is cleared, so without this the person has to press x
            # before they can record again - three clicks per sentence, in
            # front of an audience, every time.

        try:
            demo.load(None, None, None, js=FORCE_LIGHT_JS)
        except TypeError:
            pass

        if mic is not None and hands_free:
            try:
                demo.load(None, None, None, js=AUTO_RECORD_JS)
            except TypeError:
                pass          # older Gradio: the presenter presses Record once
            try:
                arm.click(None, None, None, js=AUTO_RECORD_JS)
            except TypeError:
                pass

    return demo


if __name__ == "__main__":
    page = build()
    print("microphone:", "streaming (hands-free)" if _LAST_MIC_MODE
          else "press-to-record" if _LAST_MIC_MODE is False else "none")
    print("speech model:", voice._WHISPER_REPO)
    print('say "Hey Lopa" to wake it. LOPA_DEBUG=1 shows the live mic level.')
    page.launch()
