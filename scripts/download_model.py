"""Download a model WITHOUT huggingface_hub.

Why this exists: on the campus network in China, `huggingface_hub` fails even
with HF_ENDPOINT pointed at a mirror. It lists the repo fine, then dies on the
per-file download because it follows a redirect to a CDN host that is blocked,
and it does all 14 files in parallel so one failure kills the whole thing.

This script talks plain HTTP to the mirror, downloads files ONE AT A TIME, and
resumes where it left off. It also tells you exactly which URL failed and why,
instead of a 60-line traceback.

    # 1. see what is reachable and what would be downloaded
    python scripts/download_model.py --check

    # 2. actually download
    python scripts/download_model.py

    # 3. try a different mirror if the default is down
    python scripts/download_model.py --endpoint https://hf-mirror.com

Then point config.yaml at the folder it created:

    vlm:
      model: models/Qwen3-VL-2B-Instruct-4bit

Downloading to a local folder is worth doing even when the Hub works: the demo
then needs no network at all, so it cannot fail in front of the audience.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import requests

DEFAULT_REPO = "mlx-community/Qwen3-VL-2B-Instruct-4bit"

# Tried in order. The first one that answers the repo listing wins.
ENDPOINTS = [
    os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com",
    "https://hf-mirror.com",
    "https://huggingface.co",
]

# Files we never need; skipping them saves time and avoids pointless failures.
SKIP_SUFFIXES = (".gitattributes", ".md")


def list_files(endpoint, repo, revision, timeout):
    url = f"{endpoint}/api/models/{repo}/revision/{revision}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    files = [s["rfilename"] for s in payload.get("siblings", [])]
    return [f for f in files if not f.endswith(SKIP_SUFFIXES)]


def human(num):
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.1f}{unit}"
        num /= 1024


def download_one(endpoint, repo, revision, filename, out_dir, timeout, retries=100):
    url = f"{endpoint}/{repo}/resolve/{revision}/{filename}"
    target = Path(out_dir) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True,
                              timeout=timeout, allow_redirects=True) as response:
                if response.status_code == 416:      # already complete
                    break
                if response.status_code not in (200, 206):
                    raise requests.HTTPError(
                        f"HTTP {response.status_code} from {response.url}")

                total = response.headers.get("Content-Length")
                total = (int(total) + have) if total else None
                if target.exists() and total and target.stat().st_size == total:
                    print(f"    already have {filename}")
                    return

                mode = "ab" if have and response.status_code == 206 else "wb"
                if mode == "wb":
                    have = 0
                written = have
                last_print = time.time()
                with open(part, mode) as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
                        if time.time() - last_print > 1:
                            pct = f"{100 * written / total:.0f}%" if total else ""
                            print(f"\r    {filename}  {human(written)} {pct}   ",
                                  end="", flush=True)
                            last_print = time.time()
                print(f"\r    {filename}  {human(written)}  done            ")
            part.replace(target)
            return
        except (requests.RequestException, OSError) as exc:
            # The mirror drops the connection every 100-150MB. That is normal
            # here, not a real failure - so retry a lot, and back off gently.
            wait = min(2 + attempt, 5)
            got = part.stat().st_size if part.exists() else 0
            print(f"\n    attempt {attempt}/{retries} dropped after {human(got)}: "
                  f"{type(exc).__name__}")
            if attempt == retries:
                raise
            print(f"    resuming from {human(got)} in {wait}s")
            time.sleep(wait)

    if part.exists():
        part.replace(target)


def main():
    parser = argparse.ArgumentParser(description="Mirror-friendly model downloader")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", default=None, help="default: models/<repo name>")
    parser.add_argument("--endpoint", default=None, help="force one endpoint")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=100,
                        help="per-file retry attempts (the mirror drops often)")
    parser.add_argument("--only", default=None,
                        help="download just this filename, e.g. model.safetensors")
    parser.add_argument("--check", action="store_true",
                        help="only test connectivity and list files")
    args = parser.parse_args()

    endpoints = [args.endpoint] if args.endpoint else ENDPOINTS
    seen, ordered = set(), []
    for endpoint in endpoints:
        if endpoint and endpoint not in seen:
            seen.add(endpoint)
            ordered.append(endpoint.rstrip("/"))

    files, working = None, None
    for endpoint in ordered:
        print(f"trying {endpoint} ...", end=" ", flush=True)
        try:
            files = list_files(endpoint, args.repo, args.revision, args.timeout)
            working = endpoint
            print(f"OK, {len(files)} files")
            break
        except Exception as exc:
            print(f"FAILED ({type(exc).__name__}: {str(exc)[:100]})")

    if not working:
        print("\nNo endpoint reachable. Things to try:")
        print("  - turn a VPN on or off (whichever you are not doing now)")
        print("  - switch to phone hotspot; campus networks often block this")
        print("  - use ModelScope instead (see README)")
        sys.exit(1)

    out_dir = Path(args.out or Path("models") / args.repo.split("/")[-1])
    print(f"\nrepo     {args.repo}")
    print(f"endpoint {working}")
    print(f"target   {out_dir}")
    for filename in files:
        print(f"  - {filename}")

    if args.check:
        print("\n--check only, nothing downloaded.")
        print("If the listing worked but downloads fail, the mirror proxies the API")
        print("but not the file CDN. Try --endpoint https://huggingface.co, or ModelScope.")
        return

    if args.only:
        files = [f for f in files if f == args.only]
        if not files:
            print(f"\nNo file named {args.only!r} in this repo.")
            sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for position, filename in enumerate(files, 1):
        print(f"\n[{position}/{len(files)}] {filename}")
        try:
            download_one(working, args.repo, args.revision, filename,
                         out_dir, args.timeout, retries=args.retries)
        except Exception as exc:
            print(f"    GAVE UP: {type(exc).__name__}: {str(exc)[:200]}")
            failed.append(filename)

    print()
    if failed:
        print(f"{len(failed)} file(s) failed: {failed}")
        print("Re-run this script - completed files are skipped and partial ones resume.")
        sys.exit(1)

    print(f"All files in {out_dir}")
    print("\nNow set this in config.yaml:")
    print(f"  vlm:\n    backend: mlx\n    model: {out_dir}")
    print("\nThen:  python -m src.indexer --limit 1")


if __name__ == "__main__":
    main()
