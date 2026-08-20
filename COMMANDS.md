# Command reference

Everything you can run, with the flags that actually exist.

> **zsh warning:** do NOT paste a command with a trailing `# comment`. macOS zsh
> does not treat `#` as a comment in interactive shells, so it gets passed to the
> program as an argument. Run one line at a time.

---

## 0. Setup

```bash
cd ~/Desktop/summercamp팀플/ai-lost-and-found
source .venv/bin/activate
pip install -r requirements.txt
```

Only three packages are required: PyYAML, Pillow, pytest. Everything below works
with just those, except the real indexer.

---

## 1. Get something running with no model

```bash
python scripts/make_dummy_index.py
python -m src.cli "Where is my bottle?"
```

`make_dummy_index.py` writes fake scenes AND placeholder images. Same schema as
the real index, so every other command works against it.

**It overwrites `data/scene_index.json`.** If you have a real index you care
about, back it up first:

```bash
cp data/scene_index.json data/scene_index.real.json
```

---

## 2. Indexing real photos

Photos go in `data/images/<location>/<name>.jpg`. One folder = one place; every
photo in a folder is treated as a different view of that same place.

```bash
python -m src.indexer --limit 1 -v
python -m src.visualize
python -m src.indexer
```

Always do `--limit 1` then `visualize` first. If the boxes are not sitting on the
objects, every relation computed from them is wrong.

| flag | meaning |
| --- | --- |
| `--limit N` | stop after N images — smoke test |
| `--only office` | index one location folder only |
| `--force` | re-index scenes already in the index |
| `-v` / `--verbose` | print raw model output and timing |
| `--config path` | use a different config file |

Common combinations:

```bash
python -m src.indexer --only office --force -v
python -m src.indexer --force
```

`--force` is what you need after changing the prompt or swapping models.
Without it, already-indexed scenes are skipped.

---

## 3. Checking the boxes are right

```bash
python -m src.visualize
python -m src.visualize --scene office_front
open data/debug/
```

Writes annotated copies to `data/debug/`. Every indexed object gets a box and a
label. This is the fastest way to see what the model actually saw.

If boxes are in the wrong place, fix `vlm.coord_mode` in `config.yaml`:

| value | meaning |
| --- | --- |
| `norm1000` | coordinates normalised 0–1000 — what our prompt asks for |
| `abs_resized` | absolute pixels of the resized image the model was shown |

---

## 4. Asking questions

```bash
python -m src.cli
python -m src.cli "Where is my bottle?"
python -m src.cli "Find the black bottle next to the laptop"
```

With no argument it starts an interactive session. With an argument it answers
that first, then keeps the session open. Type `quit` to leave.

| flag | meaning |
| --- | --- |
| `-v` / `--verbose` | show candidate counts, parsed query, which attribute it chose |
| `--no-images` | do not draw or open result photos |
| `--verify` | re-check each candidate with the VLM at query time (slow) |
| `--config path` | use a different config file |

`-v` is what you want while developing — it shows *why* it asked what it asked:

```bash
python -m src.cli -v "where is my bottle"
```

Result photos are written to `data/debug/result_*.jpg` and opened automatically.

### Query forms that work

```bash
python -m src.cli "where is my laptop"
python -m src.cli "wheres my black bottle"
python -m src.cli "yo where'd my macbook go"
python -m src.cli "find the bottle next to the laptop"
python -m src.cli "is there a charger in the office"
python -m src.cli "bottle"
```

Grammar is ignored entirely — only known nouns, colours and relation words are
read. Unknown words fail visibly rather than guessing.

---

## 5. Web UI (optional)

```bash
pip install gradio
python -m src.app
```

Opens `http://127.0.0.1:7860`. Not required — the CLI already shows photos.
Gradio pulls in ~40 packages, so skip it on a slow network.

---

## 6. Tests

```bash
python -m pytest -q
python -m pytest -v
python -m pytest tests/test_agent.py -v
python -m pytest -k merge -v
python -m pytest -k "not slow" -x
```

| flag | meaning |
| --- | --- |
| `-q` | one line of output |
| `-v` | list every test name |
| `-k pattern` | only tests whose name matches |
| `-x` | stop at the first failure |
| `--tb=short` | shorter tracebacks |

---

## 7. Evaluation

```bash
python eval/run_eval.py
```

No flags. Reads `eval/test_queries.json` and prints the table for the report.

Answers are derived from the ground-truth target object, so adding a case only
means adding a query and the object id it should resolve to.

---

## 8. Downloading a model

```bash
python scripts/download_model.py --check
python scripts/download_model.py
python scripts/download_model.py --endpoint https://hf-mirror.com
```

Needs `pip install requests`. Downloads one file at a time and resumes, which is
what makes it work on a network where `huggingface_hub` dies.

| flag | meaning |
| --- | --- |
| `--check` | list what is reachable and what would be downloaded |
| `--repo NAME` | a different model repo |
| `--endpoint URL` | force one mirror |
| `--out DIR` | where to put it — default `models/<repo name>` |
| `--only PATTERN` | only files matching |
| `--retries N` | retry count per file |

Then point `config.yaml` at the folder:

```yaml
vlm:
  model: models/Qwen3-VL-2B-Instruct-4bit
```

---

## 9. Inspecting the index by hand

```bash
python -c "import json;d=json.load(open('data/scene_index.json'));print(d['backend'],d['model'],len(d['scenes']))"
```

Object counts per scene:

```bash
python -c "
import json
d = json.load(open('data/scene_index.json'))
for s in d['scenes']:
    print(s['scene_id'], len(s['objects']), [o['type'] for o in s['objects']])
"
```

What the agent actually sees after merging:

```bash
python -c "
from src.config import load_config
from src.retrieval import SceneIndex, find_candidates
from src.query_parser import parse
cfg = load_config(); idx = SceneIndex.load(cfg['paths']['index'])
for c in find_candidates(idx, parse('where is my bottle')):
    print(c['object']['id'], c['object']['attributes'], c.get('seen_in'))
"
```

That last one is the most useful debugging command in the project — it shows
exactly which objects survived merging and which views each came from.

---

## 10. Git

```bash
git status
git add -A
git commit -m "message"
git push
```

If an editor opens and you are stuck in vim: `Esc`, then `:wq`, then Enter.
To skip the editor entirely:

```bash
GIT_EDITOR=true git rebase --continue
```

Never commit `models/` — it is gitignored, keep it that way.

---

## Config knobs worth knowing

All in `config.yaml`.

| key | effect |
| --- | --- |
| `vlm.backend` | `mlx` (Apple) / `transformers` (CUDA, CPU) / `dummy` |
| `vlm.model` | model id or local folder |
| `vlm.max_image_side` | 1280. Lower = faster, misses small objects |
| `vlm.max_tokens` | 3072. Raising this makes a 2B model ramble, not improve |
| `vlm.coord_mode` | fix this first if boxes land in the wrong place |
| `agent.max_clarify_turns` | 3. How many questions before showing a shortlist |
| `agent.top_k_scenes` | 5. Scenes passed to the candidate stage |
| `agent.verify_with_vlm` | false. Query-time VLM re-check |

After changing anything under `vlm:`, re-run with `--force`:

```bash
python -m src.indexer --force
```

Changes under `agent:` take effect immediately — no re-indexing needed.
