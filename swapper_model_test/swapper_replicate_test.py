"""
Replicate InSwapper Test
========================
Runs a real face-swapper model against the existing VTON outputs.

Default flow:
    source face/base model image: generated_images/<model_id>.png
    target image:                tryon_output/<precomputed try-on>.png
    output:                      tryon_output/swapper_replicate_test/*.png

This uses ddvinh1/inswapper on Replicate, which exposes source_img and
target_img inputs for an InSwapper-style face swap. It is a swapper model,
not an image generation/editing model.

Usage:
    python swapper_replicate_test.py --dry-run
    python swapper_replicate_test.py
    python swapper_replicate_test.py --sample-index 2
    python swapper_replicate_test.py --all-samples
    python swapper_replicate_test.py --source generated_images/M-REC_S3.png --target tryon_output/sample1.png

Required:
    $env:REPLICATE_API_TOKEN="your-token"
"""

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


VERSION_ID = "25bdae46f2713138640b6e8c04dc4ca18625ce95b1863936b053eee42d9ba6db"
PREDICTIONS_URL = "https://api.replicate.com/v1/predictions"
REPO_ROOT = Path(__file__).resolve().parents[1]
TRYON_LOG = REPO_ROOT / "tryon_output" / "tryon_log.json"
OUTPUT_DIR = REPO_ROOT / "tryon_output" / "swapper_replicate_test"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_token() -> str:
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise EnvironmentError(
            "REPLICATE_API_TOKEN is not set. In PowerShell run: "
            "$env:REPLICATE_API_TOKEN='your-token'"
        )
    return token


def path_or_data_uri(value: str) -> str:
    if value.startswith(("http://", "https://", "data:")):
        return value

    path = Path(value)
    if not path.exists() and not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def request_json(url: str, token: str, payload: dict | None = None) -> dict:
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Replicate HTTP {exc.code}: {body}") from exc


def create_prediction(source_img: str, target_img: str, token: str) -> dict:
    payload = {
        "version": VERSION_ID,
        "input": {
            "source_img": source_img,
            "target_img": target_img,
            "source_indexes": "-1",
            "target_indexes": "-1",
            "face_restore": True,
            "background_enhance": False,
            "face_upsample": True,
            "upscale": 1,
            "codeformer_fidelity": 0.5,
        },
    }
    return request_json(PREDICTIONS_URL, token, payload)


def poll_prediction(prediction: dict, token: str, poll_seconds: float = 2.0) -> dict:
    get_url = prediction["urls"]["get"]
    while prediction["status"] not in ("succeeded", "failed", "canceled"):
        print(f"      status={prediction['status']}")
        time.sleep(poll_seconds)
        prediction = request_json(get_url, token)
    return prediction


def download_output(output_url: str, output_path: Path) -> None:
    req = urllib.request.Request(output_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        output_path.write_bytes(resp.read())


def infer_sample_pairs() -> list[dict]:
    if not TRYON_LOG.exists():
        return []

    pairs = []
    for row in load_json(TRYON_LOG):
        model_id = row.get("model_id")
        output_path = row.get("output_path")
        if not model_id or not output_path:
            continue
        pairs.append(
            {
                "model_id": model_id,
                "source": str(REPO_ROOT / "generated_images" / f"{model_id}.png"),
                "target": str(REPO_ROOT / output_path),
                "product_title": row.get("product_title", ""),
                "product_type": row.get("product_type", ""),
            }
        )
    return pairs


def build_jobs(args: argparse.Namespace) -> list[dict]:
    if args.source or args.target:
        if not args.source or not args.target:
            raise ValueError("--source and --target must be passed together.")
        return [
            {
                "label": args.label or "custom",
                "source": args.source,
                "target": args.target,
                "metadata": {},
            }
        ]

    pairs = infer_sample_pairs()
    if not pairs:
        raise FileNotFoundError(
            "Could not infer sample pairs. Pass --source and --target, or run tryon_samples.py first."
        )

    selected = pairs if args.all_samples else [pairs[args.sample_index - 1]]
    jobs = []
    for i, pair in enumerate(selected, 1):
        jobs.append(
            {
                "label": args.label or f"sample{i}_{pair['model_id']}",
                "source": pair["source"],
                "target": pair["target"],
                "metadata": {
                    "model_id": pair["model_id"],
                    "product_title": pair["product_title"],
                    "product_type": pair["product_type"],
                },
            }
        )
    return jobs


def run_job(job: dict, token: str | None, dry_run: bool) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{job['label']}_inswapper.png"

    print("\n" + "=" * 68)
    print(f"  InSwapper job : {job['label']}")
    print(f"  Source face   : {job['source']}")
    print(f"  Target try-on : {job['target']}")
    if job["metadata"].get("product_title"):
        print(f"  Product       : {job['metadata']['product_title']}")
    print("=" * 68)

    if dry_run:
        return {
            "label": job["label"],
            "source": job["source"],
            "target": job["target"],
            "output_path": str(output_path),
            "dry_run": True,
            **job["metadata"],
        }

    source_img = path_or_data_uri(job["source"])
    target_img = path_or_data_uri(job["target"])

    started = time.time()
    print("\n  Creating Replicate prediction...")
    prediction = create_prediction(source_img, target_img, token)
    prediction = poll_prediction(prediction, token)
    latency = round(time.time() - started, 2)

    if prediction["status"] != "succeeded":
        raise RuntimeError(f"Prediction ended with status={prediction['status']}: {prediction.get('error')}")

    output = prediction["output"]
    output_url = output[0] if isinstance(output, list) else output
    print(f"\n  Downloading output -> {output_path}")
    download_output(output_url, output_path)

    return {
        "label": job["label"],
        "source": job["source"],
        "target": job["target"],
        "output_path": str(output_path),
        "replicate_url": output_url,
        "prediction_id": prediction.get("id"),
        "latency_seconds": latency,
        **job["metadata"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real InSwapper face swap via Replicate.")
    parser.add_argument("--source", help="Source identity image path, URL, or data URI.")
    parser.add_argument("--target", help="Target try-on image path, URL, or data URI.")
    parser.add_argument("--label", help="Output file label.")
    parser.add_argument("--sample-index", type=int, default=1, help="1-based row from tryon_output/tryon_log.json.")
    parser.add_argument("--all-samples", action="store_true", help="Run all rows from tryon_output/tryon_log.json.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and write log without calling API.")
    args = parser.parse_args()

    jobs = build_jobs(args)
    token = None if args.dry_run else require_token()

    print(f"\nReplicate InSwapper Test - {len(jobs)} job(s)")
    print(f"Version : {VERSION_ID}")
    print(f"Output  : {OUTPUT_DIR}")

    results = []
    for job in jobs:
        results.append(run_job(job, token, args.dry_run))

    log_path = OUTPUT_DIR / "swapper_replicate_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 68)
    print(f"  Finished {len(results)} job(s)")
    print(f"  Log -> {log_path}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
