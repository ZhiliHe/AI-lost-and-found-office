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
INDEX = None
HIGHLIGHT = (255, 87, 51)
DIMMED = (120, 120, 120)


def _resolve(image_path):
    path = Path(image_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


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

    with gr.Blocks(title="AI Lost & Found") as demo:
        gr.Markdown(
            f"# AI Lost & Found\n"
            f"**{len(INDEX.scenes)} scenes** indexed across: {locations}  \n"
            f"Model used for indexing: `{INDEX.payload.get('model')}`\n\n"
            "Try: *Where is my bottle?* &nbsp;|&nbsp; *Find the black bottle beside the laptop* "
            "&nbsp;|&nbsp; *I lost a black backpack*"
        )

        session_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                chat = _chatbot(height=430, label="Conversation")
                box = gr.Textbox(placeholder="Where is my bottle?", label="You", autofocus=True)
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    reset = gr.Button("New search")
            with gr.Column(scale=1):
                view = gr.Gallery(label="Candidates", height=430, columns=2, object_fit="contain")
                status = gr.Markdown("")

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

            return history, session, gallery_for(reply.candidates, session._wanted()), note

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
