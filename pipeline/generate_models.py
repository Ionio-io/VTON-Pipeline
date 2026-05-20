"""
Batch Base Model Generator  --  fal.ai + GPT Image 2
====================================================
Generates all 27 body-type x skin-tone base model images defined in
prompts.json.  These images are used as the identity source throughout
the VTON pipeline.

Features:
    - Resume-safe: skips models whose output PNG already exists.
    - Configurable delay between requests to respect rate limits.
    - Saves a generation_log.json with latency and metadata per image.
    - Filters by gender with --gender flag.

Usage:
    python generate_models.py                 # generate all 27
    python generate_models.py --resume        # skip already-generated images
    python generate_models.py --gender male   # only male body types (15 images)
    python generate_models.py --gender female # only female body types (12 images)
    python generate_models.py --delay 3       # 3 s between requests

Environment:
    FAL_KEY  --  fal.ai API key  (https://fal.ai/dashboard/keys)
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import fal_client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAL_KEY = os.environ.get("FAL_KEY", "")
if not FAL_KEY:
    raise EnvironmentError(
        "FAL_KEY is not set.\n"
        "Get your key at https://fal.ai/dashboard/keys and run:\n"
        "  export FAL_KEY=your_key_here"
    )
os.environ["FAL_KEY"] = FAL_KEY

FAL_ENDPOINT = "openai/gpt-image-2"
IMAGE_SIZE   = {"width": 1024, "height": 1536}   # portrait -- good for full-body shots
QUALITY      = "high"
OUTPUT_DIR   = Path("generated_images")
LOG_PATH     = Path("generation_log.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts(path: str = "prompts.json") -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_log() -> dict:
    """Load the generation log; return empty dict if it does not exist yet."""
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_log(log: dict) -> None:
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def save_image(image_url: str, prompt_entry: dict) -> Path:
    """Download an image from a URL and save it to OUTPUT_DIR/<id>.png."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{prompt_entry['id']}.png"
    urllib.request.urlretrieve(image_url, out_path)
    return out_path


def generate_image(prompt_entry: dict) -> dict:
    """
    Call GPT Image 2 to generate one base model image.

    Args:
        prompt_entry: Single entry from prompts.json.

    Returns:
        Log dict with output path, URL, latency, and metadata.
    """
    t0 = time.time()

    result = fal_client.subscribe(
        FAL_ENDPOINT,
        arguments={
            "prompt":     prompt_entry["prompt"],
            "image_size": IMAGE_SIZE,
            "quality":    QUALITY,
        },
    )

    elapsed     = round(time.time() - t0, 2)
    image_url   = result["images"][0]["url"]
    output_path = save_image(image_url, prompt_entry)

    return {
        "id":              prompt_entry["id"],
        "gender":          prompt_entry["gender"],
        "body_type_key":   prompt_entry["body_type_key"],
        "body_type_name":  prompt_entry["body_type_name"],
        "skin_tone_key":   prompt_entry["skin_tone_key"],
        "skin_tone_name":  prompt_entry["skin_tone_name"],
        "hex_midpoint":    prompt_entry["hex_midpoint"],
        "output_path":     str(output_path),
        "image_url":       image_url,
        "latency_seconds": elapsed,
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def print_summary(log: dict, total: int, skipped: int, failed: list) -> None:
    sep = "-" * 55
    print(f"\n{sep}")
    print("  GENERATION SUMMARY")
    print(sep)
    print(f"  Total prompts   : {total}")
    print(f"  Generated       : {len(log) - skipped}")
    print(f"  Skipped (exist) : {skipped}")
    print(f"  Failed          : {len(failed)}")

    latencies = [v["latency_seconds"] for v in log.values() if "latency_seconds" in v]
    if latencies:
        avg = round(sum(latencies) / len(latencies), 2)
        print(f"  Avg latency     : {avg}s")
        print(f"  Total API time  : {round(sum(latencies), 1)}s")

    if failed:
        print(f"  Failed IDs      : {', '.join(failed)}")

    print(sep)
    print(f"  Images saved to : {OUTPUT_DIR}/")
    print(f"  Log saved to    : {LOG_PATH}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch generate all 27 VTON base model images via GPT Image 2"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds to wait between API requests  (default: 2)"
    )
    parser.add_argument(
        "--gender", choices=["male", "female"],
        help="Generate only one gender  (default: both)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip models whose output PNG already exists"
    )
    args = parser.parse_args()

    prompts = load_prompts()
    log     = load_log()

    if args.gender:
        gender_word = "man" if args.gender == "male" else "woman"
        prompts     = [p for p in prompts if p["gender"] == gender_word]
        print(f"Filtered to {args.gender}: {len(prompts)} prompts")

    OUTPUT_DIR.mkdir(exist_ok=True)

    failed  = []
    skipped = 0
    total   = len(prompts)

    print(f"\nStarting batch generation: {total} images")
    print(f"  Endpoint : {FAL_ENDPOINT}")
    print(f"  Quality  : {QUALITY}")
    print(f"  Size     : {IMAGE_SIZE['width']}x{IMAGE_SIZE['height']}")
    print(f"  Delay    : {args.delay}s between requests")
    print(f"  Resume   : {args.resume}\n")

    for i, prompt_entry in enumerate(prompts, 1):
        pid        = prompt_entry["id"]
        out_path   = OUTPUT_DIR / f"{pid}.png"

        if args.resume and out_path.exists():
            print(f"[{i:02d}/{total}] Skipping {pid} (already exists)")
            skipped += 1
            continue

        print(f"[{i:02d}/{total}] Generating {pid}  "
              f"({prompt_entry['body_type_name']} / {prompt_entry['skin_tone_name']})...")

        try:
            result  = generate_image(prompt_entry)
            log[pid] = result
            save_log(log)
            print(f"         Done in {result['latency_seconds']}s -> {result['output_path']}")

        except Exception as exc:
            print(f"         FAILED: {exc}")
            failed.append(pid)
            log[pid] = {
                "id":        pid,
                "error":     str(exc),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            save_log(log)

        if i < total:
            time.sleep(args.delay)

    print_summary(log, total, skipped, failed)


if __name__ == "__main__":
    main()
