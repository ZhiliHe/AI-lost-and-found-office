"""VLM backend adapter.

The whole point of this file: nothing else in the codebase knows which model we
use. Swapping Qwen3-VL-2B for Qwen2.5-VL-3B, or for an 8B model on a bigger
machine, is one line in config.yaml and a re-index. No code changes.

Backends
--------
mlx          Apple Silicon.  pip install -U mlx-vlm
transformers CUDA or CPU.    pip install -U "transformers>=4.57" torch accelerate
dummy        No model at all. Returns plausible canned JSON so that people
             working on retrieval / agent / UI are never blocked on a download.
"""

import json
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
    """HuggingFace transformers, for whoever has an NVIDIA machine."""

    name = "transformers"

    def __init__(self, model, max_tokens=1024, temperature=0.0, **_):
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
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model, torch_dtype="auto", device_map="auto",
        )

    def describe(self, image_path, prompt, **_):
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(self.model.device)

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0, temperature=self.temperature or None,
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


BACKENDS = {"mlx": MLXBackend, "transformers": TransformersBackend, "dummy": DummyBackend}


def load_backend(vlm_cfg):
    """vlm_cfg is the `vlm:` block from config.yaml."""
    backend = vlm_cfg.get("backend", "dummy")
    if backend not in BACKENDS:
        raise VLMError(f"unknown backend {backend!r}; choose from {list(BACKENDS)}")
    return BACKENDS[backend](
        model=vlm_cfg.get("model"),
        max_tokens=vlm_cfg.get("max_tokens", 1024),
        temperature=vlm_cfg.get("temperature", 0.0),
        verbose=vlm_cfg.get("verbose", False),
    )
