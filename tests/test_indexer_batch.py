"""Indexer batching: the CUDA path sends several images through ONE
describe_batch() call, and falls back to one-at-a-time if the batch fails.

Uses a fake backend - no torch, no GPU, no model download needed. Temp files
live under the repo (system temp is sandboxed on some setups) and are removed
by the fixture teardown.
"""

import json
import shutil
from pathlib import Path

import pytest

from src import indexer
from src.vlm import DummyBackend

TEST_TMP = Path(__file__).resolve().parent.parent / ".tmp_test_indexer"


@pytest.fixture()
def work_dir():
    shutil.rmtree(TEST_TMP, ignore_errors=True)
    TEST_TMP.mkdir()
    yield TEST_TMP
    shutil.rmtree(TEST_TMP, ignore_errors=True)


@pytest.fixture()
def three_photos(work_dir):
    images = work_dir / "images" / "office"
    images.mkdir(parents=True)
    from PIL import Image

    for name in ("a.jpg", "b.jpg", "c.jpg"):
        Image.new("RGB", (64, 48), (120, 120, 120)).save(images / name)
    return images


def _cfg(work_dir):
    return {
        "vlm": {"max_image_side": 1280, "coord_mode": "norm1000",
                "max_tokens": 1024, "temperature": 0.0, "batch_size": 2},
        "paths": {
            "images": str(work_dir / "images"),
            "index": str(work_dir / "scene_index.json"),
            "debug": str(work_dir / "debug"),
        },
    }


class _BatchingBackend(DummyBackend):
    """DummyBackend output, but with the CUDA backend's batch interface."""

    name = "cuda"
    supports_batch = True

    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.batch_calls = 0

    def describe_batch(self, items, **_kwargs):
        self.batch_calls += 1
        return [self.describe(path, prompt) for path, prompt in items]


def _indexed_scenes(work_dir):
    payload = json.loads((work_dir / "scene_index.json").read_text(encoding="utf-8"))
    return {s["scene_id"] for s in payload["scenes"]}


def test_batching_backend_processes_images_in_batches(three_photos, work_dir):
    cfg = _cfg(work_dir)
    backend = _BatchingBackend()

    indexer.build_index(cfg, backend=backend)

    assert _indexed_scenes(work_dir) == {"office_a", "office_b", "office_c"}
    # 3 images at batch_size 2 -> 2 describe_batch calls, not 3 describes
    assert backend.batch_calls == 2


def test_indexer_falls_back_to_one_by_one_when_batch_fails(three_photos, work_dir, monkeypatch):
    cfg = _cfg(work_dir)
    backend = _BatchingBackend()
    state = {"fail": True}

    real_describe_batch = backend.describe_batch

    def flaky_batch(items, **_kwargs):
        if state["fail"]:
            state["fail"] = False
            raise RuntimeError("simulated CUDA OOM")
        return real_describe_batch(items)

    monkeypatch.setattr(backend, "describe_batch", flaky_batch)

    indexer.build_index(cfg, backend=backend)

    # every image still got indexed despite the failed first batch
    assert _indexed_scenes(work_dir) == {"office_a", "office_b", "office_c"}
