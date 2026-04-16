#!/usr/bin/env python3
"""
Client for RunPod segmentation endpoint.

Usage:
    export RUNPOD_API_KEY=your_key
    export RUNPOD_ENDPOINT_ID=your_endpoint_id

    python client.py person.jpg
    python client.py person.jpg --type lower --output-dir ./results
"""

import os
import sys
import time
import base64
import argparse

import runpod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to person image")
    parser.add_argument("--type", default="upper", choices=["upper", "lower", "dress", "all"])
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    args = parser.parse_args()

    if not args.api_key or not args.endpoint_id:
        print("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID")
        sys.exit(1)

    runpod.api_key = args.api_key
    endpoint = runpod.Endpoint(args.endpoint_id)

    with open(args.image, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    print(f"Submitting: {args.image} (type={args.type})")
    run = endpoint.run({"input": {"image": image_b64, "cloth_type": args.type}})
    print(f"Job: {run.job_id}")

    start = time.time()
    while time.time() - start < 120:
        status = run.status()
        print(f"\r  {status} ({time.time() - start:.0f}s)", end="", flush=True)
        if status == "COMPLETED":
            break
        elif status == "FAILED":
            print(f"\nFailed: {run.output()}")
            sys.exit(1)
        time.sleep(2)
    print()

    result = run.output()
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"Processing time: {result['processing_time']}s")
    print("\nDetected:")
    for label, pct in sorted(result["detected_labels"].items(), key=lambda x: -x[1]):
        print(f"  {label:20s} {pct:5.1f}%")

    os.makedirs(args.output_dir, exist_ok=True)
    for key in ["segmentation_map", "overlay", "clothing_mask",
                 "agnostic_mask", "extracted_full", "extracted_cropped"]:
        if key in result:
            path = os.path.join(args.output_dir, f"{key}.png")
            with open(path, "wb") as f:
                f.write(base64.b64decode(result[key]))
            print(f"Saved: {path}")

    print(f"\nDone! Check {args.output_dir}/")


if __name__ == "__main__":
    main()
