"""Terminal chat with the agent.

    python -m src.cli                      # interactive
    python -m src.cli "Where is my bottle?"

Use this rather than the Gradio app while developing - it starts instantly and
prints the internal state, so you can see WHY it asked a question.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .agent import Session, describe_object, describe_place
from .config import load_config
from .retrieval import SceneIndex
from .verify import make_verifier
from .visualize import render_result


def open_file(path):
    """Pop the image with whatever the OS uses. Best-effort only.

    Windows needs os.startfile, NOT subprocess.run(["start", ...]) - "start" is
    a cmd.exe builtin, not an executable, so spawning it raises FileNotFoundError.
    """
    try:
        if sys.platform == "win32":
            os.startfile(str(path))                      # noqa: S606
            return
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.run([opener, str(path)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, AttributeError):
        pass


def show(reply, verbose=False, wanted=None):
    print(f"\nAgent: {reply.text}")
    if reply.kind == "choose":
        for position, cand in enumerate(reply.candidates, 1):
            print(f"   {position}. {describe_object(cand['object'], cand, wanted)} "
                  f"- {describe_place(cand['scene'], cand['object'])}")
    elif reply.kind in ("giveup", "none_found") and reply.candidates:
        for position, cand in enumerate(reply.candidates, 1):
            print(f"   {position}. {describe_object(cand['object'], cand, wanted)} "
                  f"- {describe_place(cand['scene'], cand['object'])} "
                  f"[{cand['scene']['image_path']}]")
    elif reply.kind == "answer" and reply.candidates:
        cand = reply.candidates[0]
        print(f"   image: {cand['scene']['image_path']}  bbox: {cand['object']['bbox']}")
    if verbose:
        print(f"   (kind={reply.kind}, candidates={len(reply.candidates)}, "
              f"asked={reply.asked_key})")


def main():
    parser = argparse.ArgumentParser(description="Chat with the Lost & Found agent")
    parser.add_argument("query", nargs="*", help="initial query")
    parser.add_argument("--config", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-images", action="store_true",
                        help="do not draw or open result images")
    parser.add_argument("--verify", action="store_true",
                        help="re-check each candidate with the VLM at query "
                             "time (slower, catches indexing mistakes)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        index = SceneIndex.load(cfg["paths"]["index"])
    except FileNotFoundError:
        print("No scene_index.json yet. Run one of:")
        print("   python scripts/make_dummy_index.py     (no model needed)")
        print("   python -m src.indexer                  (real photos + VLM)")
        sys.exit(1)

    print(f"Loaded {len(index.scenes)} scenes from {len(index.locations())} locations: "
          f"{', '.join(index.locations())}")

    shown_already = set()
    verifier = None
    if args.verify:
        cfg.setdefault("agent", {})["verify_with_vlm"] = True
        from .vlm import load_backend
        print("Loading the model for query-time verification...")
        verifier = make_verifier(cfg, load_backend(cfg["vlm"]), on_progress=print)
        if verifier is None:
            print("   (could not load a backend - continuing without verification)")

    session = Session(index, cfg, verifier=verifier)
    first = " ".join(args.query).strip()

    while True:
        if first:
            user_text, first = first, None
            print(f"\nYou: {user_text}")
        else:
            try:
                user_text = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if not user_text:
            continue
        if user_text.lower() in ("quit", "exit", "q"):
            break

        was_fresh = not session.pending_key
        reply = (session.reply(user_text) if session.pending_key
                 else session.start(user_text))
        if was_fresh or reply.kind == "none_found":
            shown_already.clear()
        show(reply, args.verbose, session._wanted())

        # Draw the result and pop it open - the visual half of the deliverable,
        # without needing a web UI.
        # "choose" needs EVERY photo on screen at once - that is the whole point.
        # Only open photos we have not already put on screen. Re-opening the
        # same shortlist every time an answer fails to parse buries the user in
        # windows - which is exactly what happened.
        signature = (reply.kind, tuple(c["object"]["id"] for c in reply.candidates))
        if not args.no_images and reply.candidates and reply.kind != "question" \
                and signature not in shown_already:
            shown_already.add(signature)
            project_root = Path(cfg["paths"]["index"]).parent.parent
            how_many = len(reply.candidates) if reply.kind == "choose" \
                else min(3, len(reply.candidates))
            for position in range(how_many):
                out = render_result(reply.candidates, cfg["paths"]["debug"],
                                    project_root, focus=position)
                if out:
                    print(f"   -> {out}")
                    open_file(out)
                if reply.kind == "answer":
                    break

        if reply.kind in ("answer", "none_found", "giveup"):
            print("   (ask a new question, or type 'quit')")


if __name__ == "__main__":
    main()
