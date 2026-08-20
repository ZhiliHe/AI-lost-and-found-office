"""CUDA backend: routing, device/dtype resolution, and the no-GPU guard.

These tests must run on machines WITHOUT CUDA too (any laptop), so nothing here
instantiates a real model. The availability check in CUDABackend.__init__ runs
before any download, so the no-CUDA test is real on every machine.
"""

import pytest

from src import vlm
from src.vlm import VLMError, load_backend, resolve_device, resolve_dtype


class _StubCuda:
    is_available = staticmethod(lambda: False)
    device_count = staticmethod(lambda: 0)
    current_device = staticmethod(lambda: 0)


class _StubTorchNoCuda:
    cuda = _StubCuda()


class _StubCudaOne:
    is_available = staticmethod(lambda: True)
    device_count = staticmethod(lambda: 1)
    current_device = staticmethod(lambda: 0)


class _StubTorchOneGpu:
    cuda = _StubCudaOne()


# --- registration and config routing --------------------------------------


def test_cuda_backend_is_registered():
    assert "cuda" in vlm.BACKENDS
    assert vlm.BACKENDS["cuda"] is vlm.CUDABackend
    assert vlm.CUDABackend.supports_batch is True


def test_load_backend_routes_cuda_and_passes_cuda_kwargs(monkeypatch):
    seen = {}

    class FakeCuda:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setitem(vlm.BACKENDS, "cuda", FakeCuda)

    # minimal config: the cuda defaults stay loose here, the backend itself
    # turns "auto" into concrete CUDA settings
    load_backend({"backend": "cuda", "model": "m"})
    assert seen["device"] == "auto"
    assert seen["dtype"] == "auto"
    assert seen["attn_implementation"] is None

    # explicit config is passed straight through
    load_backend({"backend": "cuda", "model": "m", "dtype": "float16",
                  "attn_implementation": "flash_attention_2",
                  "compile": True, "device": "cuda:1"})
    assert seen["dtype"] == "float16"
    assert seen["attn_implementation"] == "flash_attention_2"
    assert seen["compile"] is True
    assert seen["device"] == "cuda:1"


def test_load_backend_transformers_defaults_stay_auto(monkeypatch):
    seen = {}

    class FakeTf:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setitem(vlm.BACKENDS, "transformers", FakeTf)
    load_backend({"backend": "transformers", "model": "m"})
    # exactly the old behaviour: torch/accelerate decide everything
    assert seen["device"] == "auto"
    assert seen["dtype"] == "auto"
    assert seen["attn_implementation"] is None


def test_load_backend_rejects_unknown_backend():
    with pytest.raises(VLMError, match="unknown backend"):
        load_backend({"backend": "nope"})


# --- device resolution -----------------------------------------------------


def test_resolve_device_auto_and_cpu():
    assert resolve_device(None, None) == "auto"
    assert resolve_device("auto", None) == "auto"
    assert resolve_device("cpu", None) == "cpu"


def test_resolve_device_cuda_requires_cuda():
    with pytest.raises(VLMError, match="CUDA"):
        resolve_device("cuda", _StubTorchNoCuda)
    # "cuda:0" with no GPU at all hits the count check instead
    with pytest.raises(VLMError, match="0 GPU"):
        resolve_device("cuda:0", _StubTorchNoCuda)


def test_resolve_device_cuda_picks_first_gpu():
    assert resolve_device("cuda", _StubTorchOneGpu) == "cuda:0"
    assert resolve_device("cuda:0", _StubTorchOneGpu) == "cuda:0"


def test_resolve_device_cuda_index_out_of_range():
    with pytest.raises(VLMError, match="cuda:1"):
        resolve_device("cuda:1", _StubTorchOneGpu)


# --- dtype resolution ------------------------------------------------------


def test_resolve_dtype():
    torch = pytest.importorskip("torch")
    assert resolve_dtype("auto", torch) == "auto"
    assert resolve_dtype(None, torch) == "auto"
    assert resolve_dtype("float16", torch) is torch.float16
    assert resolve_dtype("bfloat16", torch) is torch.bfloat16
    with pytest.raises(VLMError, match="unknown dtype"):
        resolve_dtype("int8", torch)


# --- the no-GPU guard ------------------------------------------------------


def test_cuda_backend_raises_without_gpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if torch.cuda.is_available():
        pytest.skip("this machine HAS CUDA; the guard is trivially satisfied")

    # model is never touched - the CUDA check fires before any download
    with pytest.raises(VLMError, match="CUDA"):
        vlm.CUDABackend(model="irrelevant-no-download-happens")


# --- batched describe() -----------------------------------------------------
# Regression guard for the padding fix: Qwen2-VL images resize to different
# grids, so a batch needs padding=True or the processor raises "Unable to
# create tensor", silently downgrading the indexer to one-by-one.


def test_describe_batch_passes_padding_and_trims_per_row():
    import shutil
    from pathlib import Path

    from PIL import Image

    torch = pytest.importorskip("torch")
    tmp = Path(__file__).resolve().parent.parent / ".tmp_test_vlm"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        a, b = tmp / "a.jpg", tmp / "b.jpg"
        Image.new("RGB", (64, 48), (100, 100, 100)).save(a)
        Image.new("RGB", (48, 64), (120, 120, 120)).save(b)

        seen = {}

        class FakeProcessor:
            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True):
                return "t"

            def __call__(self, text, images, return_tensors, padding):
                seen["padding"] = padding
                # row0 prompt length 5, row1 length 10 -> padded to 10
                ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
                return {"input_ids": ids, "attention_mask": (ids > 0).long(),
                        "pixel_values": torch.zeros(2, 3),
                        "image_grid_thw": torch.zeros(2, 3)}

            def batch_decode(self, ids, skip_special_tokens=True):
                return ["decoded_row_%d" % i for i in range(ids.shape[0])]

        class FakeModel:
            device = "cpu"

            def generate(self, **kwargs):
                rows, length = kwargs["input_ids"].shape
                return torch.cat([kwargs["input_ids"],
                                  torch.full((rows, 3), 99)], dim=1)

        backend = object.__new__(vlm.CUDABackend)
        backend.torch = torch
        backend.max_tokens, backend.temperature, backend.device = 8, 0.0, "cpu"
        backend.model, backend.processor = FakeModel(), FakeProcessor()

        results = backend.describe_batch([(str(a), "p1"), (str(b), "p2")])

        assert seen["padding"] is True          # the regression fix
        assert len(results) == 2                # per-row trim, not one blob
        assert all(r == "decoded_row_0" for r in results)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
