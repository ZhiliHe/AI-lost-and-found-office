import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import SceneIndex      # noqa: E402
from src.spatial import compute_relations  # noqa: E402


def _obj(oid, kind, color, bbox, material="metal", size="medium"):
    return {
        "id": oid, "type": kind,
        "attributes": {"color": color, "material": material, "size": size},
        "bbox": bbox, "confidence": 0.9,
    }


def _scene(scene_id, location, caption, objects):
    return {
        "scene_id": scene_id, "location": location,
        "image_path": f"data/images/{location}/{scene_id}.jpg",
        "width": 1200, "height": 800, "caption": caption,
        "objects": objects, "relations": compute_relations(objects, (1200, 800)),
    }


@pytest.fixture
def index():
    """Three black bottles across two rooms - one has a logo, one is beside a
    laptop. This is the ambiguity the agent is supposed to resolve."""
    scenes = [
        _scene("office_01", "office", "A desk with a laptop and two bottles.", [
            _obj("office_01_o0", "laptop", "silver", [430, 300, 780, 560]),
            _obj("office_01_o1", "bottle", "black", [800, 300, 870, 520]),
            _obj("office_01_o2", "bottle", "blue", [330, 330, 395, 530]),
        ]),
        _scene("classroom_01", "classroom", "Desks with a black bottle and a backpack.", [
            _obj("classroom_01_o0", "bottle", "black", [250, 320, 320, 520]),
            _obj("classroom_01_o1", "backpack", "blue", [480, 300, 760, 640]),
        ]),
        _scene("lounge_01", "lounge", "A sofa with a black backpack.", [
            _obj("lounge_01_o0", "backpack", "black", [200, 280, 500, 620]),
            _obj("lounge_01_o1", "umbrella", "red", [820, 250, 890, 620]),
        ]),
    ]
    return SceneIndex({"version": 1, "model": "test", "coord_mode": "norm1000",
                       "scenes": scenes})


@pytest.fixture
def config():
    return {"agent": {"max_clarify_turns": 3, "top_k_scenes": 5}}
