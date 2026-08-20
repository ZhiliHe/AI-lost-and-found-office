"""Gradio demo UI.

    python -m src.app

Left: the conversation. Right: the scene image with the current candidates
highlighted, so the audience can see the candidate set shrink as the agent asks
questions. That shrinking is the story of the demo - make sure it is visible.
"""

from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from .agent import Session, describe_object, describe_place
from .config import load_config
from .retrieval import SceneIndex

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


def gallery_for(candidates, wanted=None):
    """One thumbnail per candidate, across scenes."""
    items = []
    for position, cand in enumerate(candidates[:6]):
        image = render_candidates(candidates, focus=position)
        if image is not None:
            caption = (f"{describe_object(cand['object'], cand, wanted)} - "
                       f"{describe_place(cand['scene'], cand['object'])}")
            items.append((image, caption))
    return items


def _chatbot(**kwargs):
    """Gradio 4/5 need type="messages" explicitly; Gradio 6 removed the argument.
    Teammates will not all have the same version, so try both."""
    try:
        return gr.Chatbot(type="messages", **kwargs)
    except TypeError:
        return gr.Chatbot(**kwargs)


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
</div>
"""
        )

        session_state = gr.State(None)

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

        def respond(message, history, session):
            message = (message or "").strip()
            if not message:
                return history, session, [], ""

            history = list(history or [])
            history.append({"role": "user", "content": message})

            if session is None:
                session = Session(INDEX, CFG)
            reply = (session.reply(message) if session.pending_key
                     else session.start(message))

            history.append({"role": "assistant", "content": reply.text})

            note = f"**{len(reply.candidates)}** candidate(s) · turn {session.turns}/{session.max_turns}"
            if session.constraints:
                known = ", ".join(f"{k}={v}" for k, v in session.constraints.items())
                note += f" · known: {known}"
            if reply.kind == "answer":
                note = "**Resolved.** " + note

            # Keep the final image visible even if a reply omits candidates.
            shown_candidates = reply.candidates or session.candidates
            return history, session, \
                gallery_for(shown_candidates, session._wanted()), note

        def clear():
            return [], None, [], ""

        send.click(respond, [box, chat, session_state], [chat, session_state, view, status]) \
            .then(lambda: "", None, box)
        box.submit(respond, [box, chat, session_state], [chat, session_state, view, status]) \
           .then(lambda: "", None, box)
        reset.click(clear, None, [chat, session_state, view, status])

    return demo


if __name__ == "__main__":
    build().launch()
