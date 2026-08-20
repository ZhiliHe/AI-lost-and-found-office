"""Query-time verification: ask the VLM to look again at a candidate.

WHY THIS EXISTS

The index is built once, offline. That makes queries instant, but it also
freezes any mistake the model made that day: a tall water bottle labelled
"mug", a cable read as a charger. Nothing downstream can ever notice.

This module is the second look. When the agent has a handful of candidates, it
crops each one out of its photo and asks the VLM a single yes/no question:
"is this the thing the user described?" Wrong candidates get dropped before we
ever ask the user about them.

It is OFF by default (`agent.verify_with_vlm` in config.yaml), because each
check costs a real model call - 2-5s on an 8GB Mac - and the demo should feel
instant. Turn it on to show the full pipeline, or when accuracy matters more
than speed.

    fast path (default)   index lookup                      ~10ms
    verified path         index lookup + VLM re-check       ~2-5s per candidate
"""

from pathlib import Path

from .prompts import VERIFY_CANDIDATE_PROMPT
from .vlm import extract_json

# How much context to include around the object. A bare crop of a black
# rectangle is unidentifiable; the surrounding desk is what makes it a bottle.
CROP_MARGIN = 0.35

# Below this, we do not trust the model's own "no" enough to act on it.
REJECT_BELOW_CONFIDENCE = 0.6
SMALL_OBJECT_AREA_FRACTION = 0.018
LOW_CANDIDATE_CONFIDENCE = 0.70


def _crop(cand, project_root, work_dir):
    """Cut the candidate out of its photo, with margin. Returns a path or None."""
    from PIL import Image

    scene = cand["scene"]
    image_path = Path(scene["image_path"])
    if not image_path.is_absolute():
        image_path = project_root / image_path
    if not image_path.exists():
        return None

    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = cand["object"]["bbox"]
    margin_x = (x2 - x1) * CROP_MARGIN
    margin_y = (y2 - y1) * CROP_MARGIN
    box = (max(0, int(x1 - margin_x)), max(0, int(y1 - margin_y)),
           min(image.width, int(x2 + margin_x)), min(image.height, int(y2 + margin_y)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"verify_{cand['object']['id']}.jpg"
    image.crop(box).save(out, quality=90)
    return out


def needs_verification(cand):
    """Only spend VLM calls on candidates where a second look is worth it."""
    obj = cand.get("object", {})
    scene = cand.get("scene", {})
    score = float(cand.get("score", obj.get("confidence", 0.5)) or 0.5)
    if score < LOW_CANDIDATE_CONFIDENCE:
        return True

    bbox = obj.get("bbox") or []
    if len(bbox) != 4:
        return False
    width = float(scene.get("width") or 0)
    height = float(scene.get("height") or 0)
    if width <= 0 or height <= 0:
        return False
    area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    return area / (width * height) < SMALL_OBJECT_AREA_FRACTION


def verify_candidates(candidates, description, backend, cfg, on_progress=None):
    """Drop candidates the VLM says are not what the user described.

    Deliberately conservative. A candidate survives unless the model says NO
    with real confidence, because a false rejection is invisible to the user -
    their object simply never turns up - while a false acceptance merely costs
    one clarification question.
    """
    if not candidates or backend is None:
        return candidates
    selected = [c for c in candidates if needs_verification(c)]
    if not selected:
        return candidates

    project_root = Path(cfg["paths"]["index"]).parent.parent
    work_dir = Path(cfg["paths"]["debug"]) / "verify"

    kept = [c for c in candidates if c not in selected]
    for cand in selected:
        crop_path = _crop(cand, project_root, work_dir)
        if crop_path is None:
            kept.append(cand)          # cannot check it, so do not punish it
            continue

        prompt = VERIFY_CANDIDATE_PROMPT.format(description=description)
        try:
            raw = backend.describe(str(crop_path), prompt,
                                   max_tokens=128, temperature=0.0)
            verdict = extract_json(raw) or {}
        except Exception as exc:                       # noqa: BLE001
            if on_progress:
                on_progress(f"   verify failed for {cand['object']['id']}: {exc}")
            kept.append(cand)
            continue

        match = verdict.get("match")
        confidence = float(verdict.get("confidence") or 0.0)
        rejected = match is False and confidence >= REJECT_BELOW_CONFIDENCE

        if on_progress:
            mark = "reject" if rejected else "keep  "
            on_progress(f"   verify {mark} {cand['object']['id']} "
                        f"(match={match}, conf={confidence:.2f}) "
                        f"{verdict.get('reason', '')}")

        if not rejected:
            cand = dict(cand)
            cand["verified"] = verdict
            kept.append(cand)

    # Never verify away every option - that would turn a findable object into
    # "not found", which is the worst possible outcome for a lost-and-found.
    return kept or candidates


def make_verifier(cfg, backend, on_progress=None):
    """Build the callable Session expects, or None if verification is off."""
    if not cfg.get("agent", {}).get("verify_with_vlm"):
        return None
    if backend is None:
        return None

    def verifier(candidates, description):
        return verify_candidates(candidates, description, backend, cfg,
                                 on_progress=on_progress)
    return verifier
