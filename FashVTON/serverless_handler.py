#!/usr/bin/env python3
"""
FASHN VTON v1.5 — RunPod Serverless Handler + Test Client
==========================================================

AS SERVER (inside Docker / RunPod worker):
    python serverless_handler.py

AS TEST CLIENT (from your laptop):
    export RUNPOD_ENDPOINT_ID="your-endpoint-id"
    export RUNPOD_API_KEY="your-api-key"
    python serverless_handler.py --test --person person.jpg --garment garment.jpg
"""

import sys
import os
import argparse

# ============================================================
# MODE: TEST CLIENT (runs on your laptop)
# ============================================================
if "--test" in sys.argv:
    import base64
    import time
    import json

    try:
        import requests
    except ImportError:
        print("pip install requests")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--person", required=True, help="Path to person image")
    parser.add_argument("--garment", required=True, help="Path to garment image")
    parser.add_argument("--category", default="tops", choices=["tops", "bottoms", "one-pieces"])
    parser.add_argument("--garment-type", default="flat-lay", choices=["model", "flat-lay"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="result.png")
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID", ""))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY", ""))
    args = parser.parse_args()

    if not args.endpoint_id:
        print("ERROR: Set RUNPOD_ENDPOINT_ID env var or pass --endpoint-id")
        sys.exit(1)
    if not args.api_key:
        print("ERROR: Set RUNPOD_API_KEY env var or pass --api-key")
        sys.exit(1)
    if not os.path.isfile(args.person):
        print(f"ERROR: File not found: {args.person}")
        sys.exit(1)
    if not os.path.isfile(args.garment):
        print(f"ERROR: File not found: {args.garment}")
        sys.exit(1)

    with open(args.person, "rb") as f:
        person_b64 = base64.b64encode(f.read()).decode()
    with open(args.garment, "rb") as f:
        garment_b64 = base64.b64encode(f.read()).decode()

    url = f"https://api.runpod.ai/v2/{args.endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "person_image": person_b64,
            "garment_image": garment_b64,
            "category": args.category,
            "garment_type": args.garment_type,
            "steps": args.steps,
            "seed": args.seed,
        }
    }

    print(f"Submitting: {args.person} + {args.garment} → {args.endpoint_id}")
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code != 200:
        print(f"Submit failed ({r.status_code}): {r.text}")
        sys.exit(1)

    job_id = r.json()["id"]
    print(f"Job: {job_id}")
    print("Waiting for result (first request may take ~60s for cold start)...")

    status_url = f"https://api.runpod.ai/v2/{args.endpoint_id}/status/{job_id}"
    for attempt in range(60):
        time.sleep(3)
        r = requests.get(status_url, headers=headers, timeout=15)
        state = r.json().get("status", "UNKNOWN")
        sys.stdout.write(f"\r  Status: {state} ({(attempt+1)*3}s)")
        sys.stdout.flush()

        if state == "COMPLETED":
            img_b64 = r.json()["output"]["image"]
            img_bytes = base64.b64decode(img_b64)
            with open(args.output, "wb") as f:
                f.write(img_bytes)
            print(f"\n\nSaved: {args.output} ({len(img_bytes)/1024:.0f}KB)")
            sys.exit(0)

        if state in ("FAILED", "CANCELLED", "TIMED_OUT"):
            print(f"\n\nJob {state}: {r.json()}")
            sys.exit(1)

    print("\n\nTimeout — job did not complete in 3 minutes.")
    sys.exit(1)


# ============================================================
# MODE: SERVERLESS WORKER (runs inside Docker on RunPod)
# ============================================================
import io
import base64

sys.path.insert(0, "/fashn-vton-1.5")
os.chdir("/fashn-vton-1.5")

import runpod
from PIL import Image
from fashn_vton import TryOnPipeline

print("Loading FASHN VTON v1.5...")
PIPELINE = TryOnPipeline(weights_dir="/fashn-vton-1.5/weights", device="cuda")
print("Model ready. Waiting for requests.\n")


def handler(job):
    """
    Input JSON:
        person_image:  base64 string (required)
        garment_image: base64 string (required)
        category:      "tops" | "bottoms" | "one-pieces" (default: "tops")
        garment_type:  "model" | "flat-lay" (default: "flat-lay")
        steps:         int (default: 20)
        seed:          int (default: 42)

    Output JSON:
        image: base64 PNG string
    """
    try:
        inp = job["input"]

        person = Image.open(io.BytesIO(base64.b64decode(inp["person_image"]))).convert("RGB")
        garment = Image.open(io.BytesIO(base64.b64decode(inp["garment_image"]))).convert("RGB")

        result = PIPELINE(
            person_image=person,
            garment_image=garment,
            category=inp.get("category", "tops"),
            garment_photo_type=inp.get("garment_type", "flat-lay"),
            num_samples=1,
            num_timesteps=inp.get("steps", 20),
            guidance_scale=1.5,
            seed=inp.get("seed", 42),
            segmentation_free=True,
        )

        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode("utf-8")}

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
