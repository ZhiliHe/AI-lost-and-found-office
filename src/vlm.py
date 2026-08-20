"""VLM backend adapter.

The whole point of this file: nothing else in the codebase knows which model we
use. Swapping Qwen3-VL-2B for Qwen2.5-VL-3B, or for an 8B model on a bigger
machine, is one line in config.yaml and a re-index. No code changes.

Backends
--------
mlx          Apple Silicon.  pip install -U mlx-vlm
transformers CPU or auto-detected CUDA.  pip install -U "transformers>=4.57"
             torch accelerate
cuda         NVIDIA GPU, explicitly accelerated: forced CUDA device (never a
             silent CPU fallback), half precision, Flash Attention with a safe
             SDPA fallback, optional torch.compile, and batched indexing
             (several images per generate() call in src/indexer.py).
dummy        No model at all. Returns plausible canned JSON so that people
             working on retrieval / agent / UI are never blocked on a download.
"""

import json
import os
import re


class VLMError(RuntimeError):
    pass


def _iter_balanced_objects(text, start):
    """Yield every complete {...} literal after `start`, stopping at the ] that
    closes the enclosing array. String-aware, so braces inside a value do not
    confuse the depth count."""
    depth = 0
    begin = None
    in_string = False
    escaped = False

    for position in range(start, len(text)):
        char = text[position]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                begin = position
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and begin is not None:
                yield text[begin:position + 1]
                begin = None
        elif char == "]" and depth == 0:
            return


def salvage(text):
    """Recover whatever survived when the model's JSON is cut off mid-generation.

    A 2B model listing 15 objects can hit the token limit partway through
    object 12. Strict parsing then throws away all 11 good ones. Here we walk
    the "objects" array and keep every entry that is individually valid.

    Returns None if there is nothing to salvage.
    """
    key = text.find('"objects"')
    if key == -1:
        return None
    bracket = text.find("[", key)
    if bracket == -1:
        return None

    objects = []
    for chunk in _iter_balanced_objects(text, bracket + 1):
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "bbox" in parsed:
            objects.append(parsed)

    if not objects:
        return None

    result = {"objects": objects, "truncated": True}
    caption = re.search(r'"caption"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if caption:
        result["caption"] = caption.group(1)
    counted = re.search(r'"n_objects"\s*:\s*(\d+)', text)
    if counted:
        result["n_objects"] = int(counted.group(1))
    return result


def extract_json(text):
    """Models wrap JSON in ```json fences, prose, or both. Dig it out.

    Raises VLMError only when nothing at all is recoverable, so the indexer can
    skip that image and keep going instead of dying on image 2 of 40.
    """
    if not text:
        raise VLMError("empty response from model")

    # strip markdown fences
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # the outermost {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # last resort: keep the objects that are individually complete
    rescued = salvage(text)
    if rescued:
        return rescued

    raise VLMError("no parseable JSON in response")


def resolve_device(device, torch):
    """Turn a config `device` value into something torch can use.

    Returns "auto" unchanged (torch/accelerate decides), turns "cuda" into a
    concrete "cuda:N", and passes explicit values like "cpu" / "cuda:1"
    through. Raises VLMError when the requested device does not exist.
    """
    if device in (None, "auto"):
        return "auto"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise VLMError(
                f"config says device: {device!r} but torch was built without "
                "CUDA. Install the CUDA wheel "
                "(pip install torch --index-url "
                "https://download.pytorch.org/whl/cu128) or set "
                "vlm.device: cpu / vlm.device: auto.")
        return f"cuda:{torch.cuda.current_device()}"
    if isinstance(device, str) and device.startswith("cuda:"):
        index = int(device.split(":", 1)[1])
        if index >= torch.cuda.device_count():
            raise VLMError(
                f"config says device: {device!r} but torch only sees "
                f"{torch.cuda.device_count()} GPU(s)")
        return device
    return device


def resolve_dtype(dtype, torch):
    """Turn a config `dtype` value ("float16", "bfloat16", ...) into a dtype."""
    if dtype in (None, "auto"):
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype in mapping:
        return mapping[dtype]
    raise VLMError(
        f"unknown dtype {dtype!r}; choose from float16, bfloat16, float32, auto")


def _load_dtype_kwargs(dtype):
    """from_pretrained kwarg for the inference dtype.

    transformers>=5 renamed `torch_dtype` to `dtype` (same values, including
    "auto"); older versions only know `torch_dtype`. Return the right kwarg
    dict so both 4.x and 5.x work without deprecation warnings.
    """
    try:
        import transformers
        major = transformers.__version__.split(".")[0]
    except ImportError:
        major = "4"
    return {"dtype" if major >= "5" else "torch_dtype": dtype}


def cuda_device_info(torch):
    """Human-readable summary of the CUDA device torch will use, or None."""
    if not torch.cuda.is_available():
        return None
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    free, _ = torch.cuda.mem_get_info(index)
    return {
        "name": props.name,
        "compute": f"{props.major}.{props.minor}",
        "vram_gb": props.total_memory / (1024 ** 3),
        "free_gb": free / (1024 ** 3),
        "device": f"cuda:{index}",
    }


class DummyBackend:
    """Returns a fixed scene so the pipeline runs with zero dependencies."""

    name = "dummy"

    def __init__(self, **_):
        pass

    def describe(self, image_path, prompt, **_):
        return json.dumps({
            "caption": "A desk with a laptop and two bottles.",
            "objects": [
                {"type": "laptop", "color": "silver", "material": "metal",
                 "size": "large", "bbox": [300, 350, 700, 650], "confidence": 0.95},
                {"type": "bottle", "color": "black", "material": "metal",
                 "size": "medium", "bbox": [720, 380, 800, 620], "confidence": 0.88},
                {"type": "bottle", "color": "blue", "material": "plastic",
                 "size": "medium", "bbox": [180, 400, 260, 640], "confidence": 0.81},
            ],
        })


class MLXBackend:
    """mlx-vlm on Apple Silicon."""

    name = "mlx"

    def __init__(self, model, max_tokens=1024, temperature=0.0, verbose=False, **_):
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError as exc:
            raise VLMError(
                "mlx-vlm is not installed. Run:  pip install -U mlx-vlm\n"
                "(Or set vlm.backend: dummy in config.yaml to work without a model.)"
            ) from exc

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        self.model, self.processor = load(model)
        self.config = load_config(model)

    def describe(self, image_path, prompt, **_):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=1)
        result = generate(
            self.model, self.processor, formatted, [str(image_path)],
            max_tokens=self.max_tokens, temperature=self.temperature,
            verbose=self.verbose,
        )
        # mlx-vlm has returned both a bare string and a result object across
        # versions; handle both so a pip upgrade doesn't break the team.
        return result if isinstance(result, str) else getattr(result, "text", str(result))


class TransformersBackend:
    """HuggingFace transformers on CPU or an auto-detected GPU.

    With the default `device: auto` this behaves exactly as the original
    backend did (device_map="auto", torch_dtype="auto"). Explicit `device` /
    `dtype` values let you force e.g. `device: cuda:0` + `dtype: float16`
    without the CUDA-only backend's extras.
    """

    name = "transformers"
    supports_batch = False

    def __init__(self, model, max_tokens=1024, temperature=0.0, device="auto",
                 dtype="auto", attn_implementation=None, compile=False,
                 verbose=False, **_):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise VLMError(
                'transformers backend needs:  pip install -U "transformers>=4.57" torch accelerate'
            ) from exc

        self.torch = torch
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose

        self.device = resolve_device(device, torch)
        self.dtype = resolve_dtype(dtype, torch)
        self.attn_implementation = attn_implementation

        self.processor = AutoProcessor.from_pretrained(model)
        if self.device == "auto":
            # exactly the original behaviour: let accelerate/auto decide
            self.model = AutoModelForImageTextToText.from_pretrained(
                model, device_map="auto",
                attn_implementation=attn_implementation,
                **_load_dtype_kwargs("auto"),
            )
        else:
            load_kwargs = _load_dtype_kwargs(self.dtype) if self.dtype != "auto" else {}
            if attn_implementation:
                load_kwargs["attn_implementation"] = attn_implementation
            self.model = AutoModelForImageTextToText.from_pretrained(
                model, **load_kwargs)
            self.model.to(self.device)
        self.model.eval()

        self._compiled = False
        if compile and self.device != "auto":
            try:
                self.model = torch.compile(self.model)
                self._compiled = True
            except Exception as exc:                 # noqa: BLE001
                if self.verbose:
                    print(f"(torch.compile failed, continuing uncompiled: {exc})")

    def _infer_device(self):
        """Where the inputs must go: the forced device, or the model's own."""
        if self.device != "auto":
            return self.device
        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device
        return device

    def describe(self, image_path, prompt, max_tokens=None, temperature=None, **_):
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(self._infer_device())

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_tokens,
                do_sample=(temperature if temperature is not None
                           else self.temperature) > 0,
                temperature=(temperature if temperature is not None
                             else self.temperature) or None,
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


class CUDABackend(TransformersBackend):
    """The CUDA-accelerated transformers backend.

    This is the backend to pick on an NVIDIA machine. It forces the model onto
    the GPU (it never silently falls back to CPU), runs it in half precision,
    tries Flash Attention with an automatic SDPA fallback, can torch.compile
    the model, and batches several images into ONE generate() call while
    indexing - the single biggest CUDA throughput win, because generation is
    GPU-bound and the GPU is never left idle between prompts.

    config.yaml:
        backend: cuda
        dtype: float16        # auto here means float16; bfloat16 on Ampere+
        attn_implementation: flash_attention_2   # auto-falls back to sdpa
        compile: false        # true = torch.compile (slow first call, faster rest)
        batch_size: 2         # images per generation call in src/indexer.py
    """

    name = "cuda"
    supports_batch = True

    def __init__(self, model, max_tokens=1024, temperature=0.0, device="auto",
                 dtype="auto", attn_implementation="auto", compile=False,
                 verbose=False, **_):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise VLMError(
                'cuda backend needs:  pip install -U "transformers>=4.57" torch accelerate\n'
                "and a CUDA-enabled torch: pip install torch --index-url "
                "https://download.pytorch.org/whl/cu128"
            ) from exc

        # The whole point of this backend is the GPU - fail loudly and
        # helpfully here rather than silently grinding away on the CPU.
        if not torch.cuda.is_available():
            raise VLMError(
                "backend 'cuda' needs a CUDA-enabled torch build, but "
                f"torch {torch.__version__} reports no CUDA device.\n"
                "  Fix: pip install torch --index-url "
                "https://download.pytorch.org/whl/cu128\n"
                "  (or set vlm.backend: transformers to let torch pick the "
                "device itself)")

        self.torch = torch
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose

        # auto defaults that only make sense for this backend
        if device in (None, "", "auto"):
            device = "cuda"
        if dtype in (None, "", "auto"):
            dtype = "float16"              # half precision is the point
        if attn_implementation in (None, "", "auto"):
            attn_implementation = "flash_attention_2"

        self.device = resolve_device(device, torch)
        self.dtype = resolve_dtype(dtype, torch)

        # Our workload is a fixed image size and a fixed prompt: tell cuDNN to
        # benchmark once instead of on every call, and let TF32/FP16 matmuls
        # use the fast paths.
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        self.processor = AutoProcessor.from_pretrained(model)

        # Load with the requested attention implementation, falling back
        # sdpa -> model default. flash_attention_2 needs the flash-attn
        # package AND model support; a failed attempt just reloads from cache.
        attempts = [attn_implementation, "sdpa", None]
        self.attn_implementation = None
        self.model = None
        last_error = None
        for impl in attempts:
            load_kwargs = _load_dtype_kwargs(self.dtype)
            if impl:
                load_kwargs["attn_implementation"] = impl
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model, **load_kwargs)
                self.attn_implementation = impl
                break
            except Exception as exc:                 # noqa: BLE001 - probing
                last_error = exc
        if self.model is None:
            raise VLMError(f"could not load {model} on CUDA: {last_error}")

        self.model.to(self.device)
        self.model.eval()

        self._compiled = False
        if compile:
            try:
                self.model = torch.compile(self.model)
                self._compiled = True
            except Exception as exc:                 # noqa: BLE001
                if self.verbose:
                    print(f"(torch.compile failed, continuing uncompiled: {exc})")

        info = cuda_device_info(torch)
        if info:
            print(f"CUDA backend ready: {info['name']} "
                  f"(compute {info['compute']}, {info['vram_gb']:.1f} GB VRAM, "
                  f"{info['free_gb']:.1f} GB free)")
        print(f"  device={self.device} dtype={self.dtype} "
              f"attention={self.attn_implementation or 'default'} "
              f"compile={'on' if self._compiled else 'off'}")

    def describe(self, image_path, prompt, max_tokens=None, temperature=None, **_):
        return self.describe_batch([(image_path, prompt)],
                                   max_tokens=max_tokens,
                                   temperature=temperature)[0]

    def describe_batch(self, items, max_tokens=None, temperature=None):
        """Run several (image_path, prompt) pairs in ONE generate() call.

        Generation is GPU-bound, so batching is the biggest CUDA win: N images
        in one call cost barely more than one. The indexer uses this when
        vlm.batch_size > 1. Returns raw texts in the same order as items.
        """
        from PIL import Image

        paths, prompts = zip(*items)
        images = [Image.open(path).convert("RGB") for path in paths]
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))

        # padding=True is REQUIRED for batching: each image resizes to its own
        # grid (different WxH -> different number of vision tokens), so the
        # input_ids rows differ in length and must be padded to the longest.
        # Without it the processor raises "Unable to create tensor ... use
        # padding=True", which used to silently downgrade us to one-by-one.
        inputs = self.processor(text=texts, images=images, return_tensors="pt",
                                padding=True)
        inputs = {key: (value.to(self.device) if hasattr(value, "to") else value)
                  for key, value in inputs.items()}

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_tokens,
                do_sample=(temperature if temperature is not None
                           else self.temperature) > 0,
                temperature=(temperature if temperature is not None
                             else self.temperature) or None,
            )

        # Trim each row by ITS OWN prompt length: scene-extraction and
        # verification prompts differ, so the padded max would be wrong for
        # every row except the longest one.
        lengths = inputs["attention_mask"].sum(dim=1)
        results = []
        for row, length in zip(out, lengths.tolist()):
            trimmed = row[length:]
            results.append(self.processor.batch_decode(
                trimmed.unsqueeze(0), skip_special_tokens=True)[0])
        return results


BACKENDS = {"mlx": MLXBackend, "transformers": TransformersBackend,
            "cuda": CUDABackend, "dummy": DummyBackend}


def load_backend(vlm_cfg):
    """vlm_cfg is the `vlm:` block from config.yaml."""
    # On networks where huggingface.co is unreachable, transformers hangs
    # forever on the online HEAD check even when the model is cached. When
    # `vlm.hf_offline: true`, force pure-cache loading (the model must already
    # be downloaded, e.g. via scripts/download_model.py or a mirror).
    if vlm_cfg.get("hf_offline"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    backend = vlm_cfg.get("backend", "dummy")
    if backend not in BACKENDS:
        raise VLMError(f"unknown backend {backend!r}; choose from {list(BACKENDS)}")

    kwargs = {
        "model": vlm_cfg.get("model"),
        "max_tokens": vlm_cfg.get("max_tokens", 1024),
        "temperature": vlm_cfg.get("temperature", 0.0),
        "verbose": vlm_cfg.get("verbose", False),
    }
    if backend in ("transformers", "cuda"):
        attn = vlm_cfg.get("attn_implementation", "auto")
        kwargs.update(
            device=vlm_cfg.get("device", "auto"),
            dtype=vlm_cfg.get("dtype", "auto"),
            attn_implementation=None if attn == "auto" else attn,
            compile=vlm_cfg.get("compile", False),
        )
    return BACKENDS[backend](**kwargs)
