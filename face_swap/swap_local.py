"""
Local InsightFace InSwapper Test
================================
Runs a local face swap using InsightFace + inswapper_128.onnx.

Default flow:
    source face/base model image: generated_images/<model_id>.png
    target image:                tryon_output/<precomputed try-on>.png
    output:                      tryon_output/swapper_local_test/*.png

This is a real local swapper path. It does not call GPT/image-edit models.

Recommended environment:
    Python 3.10 or 3.11

Install:
    python -m venv .venv-swapper
    .venv-swapper\\Scripts\\activate
    python -m pip install -U pip setuptools wheel
    python -m pip install insightface==0.7.3 onnxruntime opencv-python numpy

Model:
    Put inswapper_128.onnx in one of:
      - models/inswapper_128.onnx
      - C:\\Users\\<you>\\.insightface\\models\\inswapper_128.onnx
      - C:\\Users\\<you>\\.insightface\\models\\inswapper_128\\inswapper_128.onnx

Usage:
    python swapper_local_insightface.py --dry-run
    python swapper_local_insightface.py
    python swapper_local_insightface.py --sample-index 2
    python swapper_local_insightface.py --all-samples
    python swapper_local_insightface.py --source generated_images/M-REC_S3.png --target tryon_output/sample1.png
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRYON_LOG = REPO_ROOT / "tryon_output" / "tryon_log.json"
OUTPUT_DIR = REPO_ROOT / "tryon_output" / "swapper_local_test"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def import_insightface():
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "Missing insightface. Use Python 3.10/3.11 and install:\n"
            "python -m pip install insightface==0.7.3 onnxruntime opencv-python numpy"
        ) from exc
    return insightface, FaceAnalysis


def import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Missing opencv-python. Install swapper dependencies with:\n"
            "python -m pip install insightface==0.7.3 onnxruntime opencv-python numpy"
        ) from exc
    return cv2


def model_candidates(custom_model: str | None) -> list[Path]:
    candidates = []
    if custom_model:
        candidates.append(Path(custom_model))

    home = Path.home()
    candidates.extend(
        [
            Path(__file__).resolve().parent / "models" / "inswapper_128.onnx",
            Path("models/inswapper_128.onnx"),
            home / ".insightface" / "models" / "inswapper_128.onnx",
            home / ".insightface" / "models" / "inswapper_128" / "inswapper_128.onnx",
        ]
    )
    return candidates


def resolve_swapper_model(custom_model: str | None) -> Path:
    for candidate in model_candidates(custom_model):
        paths = [candidate]
        if not candidate.is_absolute():
            paths.append(REPO_ROOT / candidate)
        for path in paths:
            if path.exists():
                return path

    searched = "\n".join(f"  - {p}" for p in model_candidates(custom_model))
    raise FileNotFoundError(
        "Could not find inswapper_128.onnx. Put it in one of these locations:\n"
        f"{searched}"
    )


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

    if args.sample_index < 1 or args.sample_index > len(pairs):
        raise IndexError(f"--sample-index must be between 1 and {len(pairs)}")

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


def resolve_image_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def read_image(path: str):
    cv2 = import_cv2()
    image_path = resolve_image_path(path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def pick_face(faces, role: str):
    if not faces:
        raise RuntimeError(f"No face detected in {role} image.")

    # Use the largest detected face, which is the most stable default for
    # clean e-commerce model photos.
    return max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
    )


def init_models(args: argparse.Namespace):
    insightface, FaceAnalysis = import_insightface()
    providers = ["CPUExecutionProvider"]

    app = FaceAnalysis(name=args.detector_pack, providers=providers)
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    model_path = resolve_swapper_model(args.model)
    swapper = insightface.model_zoo.get_model(str(model_path), providers=providers)
    return app, swapper, model_path


def run_job(job: dict, app, swapper, dry_run: bool) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{job['label']}_inswapper.png"

    print("\n" + "=" * 68)
    print(f"  Local InSwapper job : {job['label']}")
    print(f"  Source face         : {job['source']}")
    print(f"  Target try-on       : {job['target']}")
    if job["metadata"].get("product_title"):
        print(f"  Product             : {job['metadata']['product_title']}")
    print("=" * 68)

    if dry_run:
        for key in ("source", "target"):
            path = resolve_image_path(job[key])
            if not path.exists():
                raise FileNotFoundError(f"{key} image not found: {path}")
        return {
            "label": job["label"],
            "source": job["source"],
            "target": job["target"],
            "output_path": str(output_path),
            "dry_run": True,
            **job["metadata"],
        }

    started = time.time()
    cv2 = import_cv2()
    source_img = read_image(job["source"])
    target_img = read_image(job["target"])

    print("  Detecting faces...")
    source_face = pick_face(app.get(source_img), "source")
    target_face = pick_face(app.get(target_img), "target")

    print("  Running inswapper_128...")
    swapped = swapper.get(target_img, target_face, source_face, paste_back=True)

    cv2.imwrite(str(output_path), swapped)
    latency = round(time.time() - started, 2)

    print(f"  Saved -> {output_path}")
    print(f"  Done in {latency}s")

    return {
        "label": job["label"],
        "source": job["source"],
        "target": job["target"],
        "output_path": str(output_path),
        "latency_seconds": latency,
        **job["metadata"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local InsightFace inswapper_128 face swaps.")
    parser.add_argument("--source", help="Source identity image path.")
    parser.add_argument("--target", help="Target try-on image path.")
    parser.add_argument("--label", help="Output file label.")
    parser.add_argument("--sample-index", type=int, default=1, help="1-based row from tryon_output/tryon_log.json.")
    parser.add_argument("--all-samples", action="store_true", help="Run all rows from tryon_output/tryon_log.json.")
    parser.add_argument("--model", help="Path to inswapper_128.onnx.")
    parser.add_argument("--detector-pack", default="buffalo_l", help="InsightFace detector pack name.")
    parser.add_argument("--det-size", type=int, default=640, help="Face detector input size.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and write log without loading models.")
    args = parser.parse_args()

    print(f"\nPython: {sys.version.split()[0]}")
    if sys.version_info >= (3, 12):
        print("Warning: InsightFace is usually easiest on Python 3.10/3.11, not 3.12+.")

    jobs = build_jobs(args)
    app = swapper = None
    model_path = None

    if not args.dry_run:
        app, swapper, model_path = init_models(args)

    print(f"\nLocal InSwapper Test - {len(jobs)} job(s)")
    print(f"Output : {OUTPUT_DIR}")
    if model_path:
        print(f"Model  : {model_path}")

    results = []
    for job in jobs:
        results.append(run_job(job, app, swapper, args.dry_run))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "swapper_local_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 68)
    print(f"  Finished {len(results)} job(s)")
    print(f"  Log -> {log_path}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
