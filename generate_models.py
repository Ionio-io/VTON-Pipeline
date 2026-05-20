"""
Batch Base Model Generator — fal.ai + GPT Image 2
==================================================
Generates all 27 body_type × skin_tone base model images from prompts.json.
These images are used as the identity source in the VTON pipeline.

Features:
  - Resume-safe: skips already-generated images
  - Rate limiting with configurable delay between requests
  - Saves generation_log.json with latency + metadata per image
  - Prints a summary at the end

Usage:
    python generate_models.py                    # generate all 27
    python generate_models.py --delay 3          # 3s delay between requests
    python generate_models.py --gender male      # only male body types
    python generate_models.py --gender female    # only female body types
    python generate_models.py --resume           # skip already-generated images

Requirements:
    pip install fal-client

Environment:
    FAL_KEY — your fal.ai API key (https://fal.ai/dashboard/keys)
"""

import os
import json
import argparse
import time
from pathlib import Path

import fal_client

# ── Config ────────────────────────────────────────────────────────────────────

FAL_KEY = os.environ.get("FAL_KEY", "")
if not FAL_KEY:
    raise EnvironmentError(
        "FAL_KEY environment variable is not set.\n"
        "Get your key at https://fal.ai/dashboard/keys and run:\n"
        "  export FAL_KEY=your_key_here"
    )
os.environ["FAL_KEY"] = FAL_KEY

MODEL_ID   = "openai/gpt-image-2"
IMAGE_SIZE = {"width": 1024, "height": 1536}   # Portrait — good for full-body shots
QUALITY    = "high"
OUTPUT_DIR = Path("generated_images")
LOG_PATH   = Path("generation_log.json")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_prompts(path: str = "prompts.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_log() -> dict:
    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_log(log: dict):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def save_image(image_url: str, prompt_entry: dict) -> Path:
    import urllib.request
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename    = f"{prompt_entry['id']}.png"
    output_path = OUTPUT_DIR / filename
    urllib.request.urlretrieve(image_url, output_path)
    return output_path


def generate_image(prompt_entry: dict) -> dict:
    start = time.time()

    result = fal_client.subscribe(
        MODEL_ID,
        arguments={
            "prompt":     prompt_entry["prompt"],
            "image_size": IMAGE_SIZE,
            "quality":    QUALITY,
        },
    )

    elapsed   = round(time.time() - start, 2)
    image_url = result["images"][0]["url"]
    output_path = save_image(image_url, prompt_entry)

    return {
        "id":             prompt_entry["id"],
        "gender":         prompt_entry["gender"],
        "body_type_key":  prompt_entry["body_type_key"],
        "body_type_name": prompt_entry["body_type_name"],
        "skin_tone_key":  prompt_entry["skin_tone_key"],
        "skin_tone_name": prompt_entry["skin_tone_name"],
        "hex_midpoint":   prompt_entry["hex_midpoint"],
        "output_path":    str(output_path),
        "image_url":      image_url,
        "latency_seconds": elapsed,
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def print_summary(log: dict, total: int, skipped: int, failed: list):
    print("\n" + "═" * 55)
    print("  GENERATION SUMMARY")
    print("═" * 55)
    print(f"  Total prompts   : {total}")
    print(f"  Generated       : {len(log) - skipped}")
    print(f"  Skipped (exist) : {skipped}")
    print(f"  Failed          : {len(failed)}")
    if log:
        latencies = [v["latency_seconds"] for v in log.values() if "latency_seconds" in v]
        if latencies:
            print(f"  Avg latency     : {round(sum(latencies)/len(latencies), 2)}s")
            print(f"  Total time      : {round(sum(latencies), 1)}s")
    if failed:
        print(f"\n  ❌  Failed IDs: {', '.join(failed)}")
    print("═" * 55)
    print(f"  Images saved to : {OUTPUT_DIR}/")
    print(f"  Log saved to    : {LOG_PATH}")
    print("═" * 55 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch generate all VTON base model images")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds to wait between requests (default: 2)")
    parser.add_argument("--gender", choices=["male", "female"],
                        help="Filter by gender (default: both)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-generated images")
    args = parser.parse_args()

    prompts = load_prompts()
    log     = load_log()

    if args.gender:
        gender_word = "man" if args.gender == "male" else "woman"
        prompts = [p for p in prompts if p["gender"] == gender_word]
        print(f"🔍  Filtered to {args.gender} only: {len(prompts)} prompts")

    OUTPUT_DIR.mkdir(exist_ok=True)

    failed  = []
    skipped = 0
    total   = len(prompts)

    print(f"\n🚀  Starting batch generation: {total} images")
    print(f"    Model    : {MODEL_ID}")
    print(f"    Quality  : {QUALITY}")
    print(f"    Size     : {IMAGE_SIZE['width']}x{IMAGE_SIZE['height']}")
    print(f"    Delay    : {args.delay}s between requests")
    print(f"    Resume   : {args.resume}\n")

    for i, prompt_entry in enumerate(prompts, 1):
        pid         = prompt_entry["id"]
        output_path = OUTPUT_DIR / f"{pid}.png"

        if args.resume and output_path.exists():
            print(f"[{i:02d}/{total}] ⏭️   Skipping {pid} (already exists)")
            skipped += 1
            continue

        print(f"[{i:02d}/{total}] 🎨  Generating {pid} "
              f"({prompt_entry['body_type_name']} / {prompt_entry['skin_tone_name']})...")

        try:
            result = generate_image(prompt_entry)
            log[pid] = result
            save_log(log)
            print(f"         ✅  Done in {result['latency_seconds']}s → {result['output_path']}")

        except Exception as e:
            print(f"         ❌  Failed: {e}")
            failed.append(pid)
            log[pid] = {"id": pid, "error": str(e), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            save_log(log)

        if i < total:
            time.sleep(args.delay)

    print_summary(log, total, skipped, failed)


if __name__ == "__main__":
    main()
