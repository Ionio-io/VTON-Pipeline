"""
Z-Image Turbo Virtual Try-On  --  fal.ai API
============================================
Generates virtual try-on images by editing a base model photo via
the fal.ai hosted Z-Image Turbo image-to-image endpoint.

Workflow:
    1. Upload the base model PNG (generated_images/<id>.png) to fal storage.
    2. Build a structured garment-description prompt from the product JSON.
    3. Call fal-ai/z-image/turbo/image-to-image with turbo settings.
    4. Download and save the result to tryon_output/zimage_fal/.

No local GPU required -- all inference runs on fal.ai infrastructure.
Cost: ~$0.004 per 768x1024 image.

Usage:
    python vton_zimage_fal.py --model M-REC_S3 --product 0 --gender male
    python vton_zimage_fal.py --batch --gender male --max-products 5
    python vton_zimage_fal.py --batch --gender both --max-products 10
    python vton_zimage_fal.py --model F-HG_S4 --product 2 --gender female --strength 0.50

Environment:
    FAL_KEY  --  fal.ai API key  (https://fal.ai/dashboard/keys)
"""

import argparse
import json
import os
import re
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
        "Get your key at https://fal.ai/dashboard/keys then run:\n"
        "  Windows : set FAL_KEY=your_key\n"
        "  Mac/Linux: export FAL_KEY=your_key"
    )
os.environ["FAL_KEY"] = FAL_KEY

# fal.ai endpoint for Z-Image Turbo image-to-image
FAL_ENDPOINT = "fal-ai/z-image/turbo/image-to-image"

OUTPUT_DIR = Path("tryon_output/zimage_fal")
IMG_SIZE   = {"width": 768, "height": 1024}   # portrait aspect for fashion

# Turbo inference settings -- guidance_scale must stay 0.0
NUM_STEPS       = 9      # 8 actual DiT forward passes
GUIDANCE_SCALE  = 0.0
DEFAULT_STRENGTH = 0.55  # how aggressively to edit; lower preserves face/body more

NEGATIVE_PROMPT = (
    "deformed, extra limbs, blurry, low quality, watermark, text, logo, "
    "artifacts, bad anatomy, distorted clothing, ugly, duplicate"
)

# Default model subsets used in batch mode
BATCH_MALE_MODELS   = ["M-REC_S3", "M-REC_S4", "M-INV_S3"]
BATCH_FEMALE_MODELS = ["F-HG_S3",  "F-HG_S4",  "F-REC_S3"]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_tryon_prompt(product: dict, model_meta: dict) -> str:
    """
    Build a structured text prompt that describes the desired try-on output.

    The prompt combines model identity attributes (gender, body type, skin tone)
    with garment details extracted from the product JSON.  A single descriptive
    sentence is pulled from the product description to add fabric/fit context.
    """
    gender    = model_meta["gender"]            # "man" or "woman"
    body_type = model_meta["body_type_name"]
    skin_tone = model_meta["skin_tone_name"]
    title     = product["title"]
    category  = product.get("product_type") or "clothing"

    # Strip HTML tags and normalise whitespace from the product description
    desc_raw   = product.get("description", "")
    desc_clean = re.sub(r"<[^>]+>", " ", desc_raw).strip()
    desc_clean = re.sub(r"\s+", " ", desc_clean)

    # Extract the first sentence that mentions garment construction details
    garment_detail = ""
    for sentence in re.split(r"(?<=[.!?])\s+", desc_clean):
        sentence = sentence.strip()
        if len(sentence) > 15 and any(
            kw in sentence.lower()
            for kw in ["fabric", "fit", "cut", "style", "crafted", "feature",
                       "design", "wear", "silhouette", "waist", "sleeve"]
        ):
            garment_detail = sentence
            break

    # Fall back to the first sentence if no construction-detail sentence is found
    if not garment_detail and desc_clean:
        garment_detail = desc_clean.split(".")[0].strip()

    detail_clause = f" {garment_detail}." if garment_detail else ""

    return (
        f"Full-body studio photograph of an Indian {gender} "
        f"with {skin_tone} skin tone and {body_type} body type, "
        f"wearing {title} ({category}).{detail_clause} "
        "White seamless studio background, soft diffused overhead lighting, "
        "sharp full-body focus, professional e-commerce fashion photography, 4K."
    )


# ---------------------------------------------------------------------------
# fal.ai helpers
# ---------------------------------------------------------------------------

def upload_image(image_path: Path) -> str:
    """Upload a local PNG to fal storage and return the CDN URL."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    data = image_path.read_bytes()
    print(f"  Uploading {image_path.name} ...")
    url = fal_client.upload(data, "image/png")
    print(f"  -> {url[:72]}")
    return url


def call_tryon(image_url: str, prompt: str, strength: float, seed: int) -> str:
    """
    Call the Z-Image Turbo img2img endpoint and return the output image URL.

    Args:
        image_url: fal CDN URL of the base model image.
        prompt:    Garment + identity description prompt.
        strength:  Edit strength (0.0 = no change, 1.0 = ignore source image).
        seed:      Random seed for reproducibility.

    Returns:
        CDN URL of the generated try-on image.
    """
    result = fal_client.subscribe(
        FAL_ENDPOINT,
        arguments={
            "prompt":              prompt,
            "image_url":           image_url,
            "image_size":          IMG_SIZE,
            "strength":            strength,
            "num_inference_steps": NUM_STEPS,
            "guidance_scale":      GUIDANCE_SCALE,
            "negative_prompt":     NEGATIVE_PROMPT,
            "seed":                seed,
            "num_images":          1,
            "output_format":       "png",
        },
        with_logs=False,
    )
    return result["images"][0]["url"]


def download_image(url: str, dest: Path) -> None:
    """Download an image from a URL and write it to disk."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    print(f"  Saved -> {dest}")


# ---------------------------------------------------------------------------
# Try-on runner
# ---------------------------------------------------------------------------

def run_tryon(
    product:    dict,
    model_meta: dict,
    strength:   float,
    seed:       int,
    out_path:   Path,
) -> None:
    """
    Run a single try-on: upload base model image, call the API, save result.

    Args:
        product:    Product dict from the westside dataset JSON.
        model_meta: Model entry from prompts.json.
        strength:   img2img edit strength.
        seed:       Random seed.
        out_path:   Destination path for the output image.
    """
    model_id     = model_meta["id"]
    base_img     = Path("generated_images") / f"{model_id}.png"
    image_url    = upload_image(base_img)
    prompt       = build_tryon_prompt(product, model_meta)

    print(f"  Prompt: {prompt[:120]}...")

    t0      = time.time()
    out_url = call_tryon(image_url, prompt, strength, seed)
    elapsed = time.time() - t0

    download_image(out_url, out_path)
    print(f"  Done in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_prompts() -> dict:
    """Return prompts.json as a dict keyed by model ID."""
    with open("prompts.json", encoding="utf-8") as f:
        raw = json.load(f)
    return {entry["id"]: entry for entry in raw}


def load_products(gender: str) -> list:
    """Load the product list for the given gender."""
    path = (
        "westside_dataset/products_women.json"
        if gender == "female"
        else "westside_dataset/products_men.json"
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Single and batch runners
# ---------------------------------------------------------------------------

def run_single(args, prompts: dict) -> None:
    products   = load_products(args.gender)
    product    = products[args.product]
    model_meta = prompts.get(args.model)

    if model_meta is None:
        raise ValueError(f"Model ID '{args.model}' not found in prompts.json")

    handle   = product["handle"][:40]
    out_path = OUTPUT_DIR / f"{args.model}_{handle}.png"

    print(f"\nProduct  : {product['title']}")
    print(f"Model    : {args.model}  "
          f"({model_meta['body_type_name']} / {model_meta['skin_tone_name']})")
    print(f"Strength : {args.strength}  |  Steps : {NUM_STEPS}  |  Seed : {args.seed}\n")

    run_tryon(product, model_meta, args.strength, args.seed, out_path)

    print(f"\n[OK] Result saved -> {out_path}")


def run_batch(args, prompts: dict) -> None:
    genders = ["male", "female"] if args.gender == "both" else [args.gender]

    dataset_map = {
        "male":   (BATCH_MALE_MODELS,),
        "female": (BATCH_FEMALE_MODELS,),
    }

    total_done = 0

    for gender in genders:
        model_ids = dataset_map[gender][0]
        products  = load_products(gender)[: args.max_products]

        for product in products:
            for mid in model_ids:
                if mid not in prompts:
                    continue

                handle   = product["handle"][:40]
                out_path = OUTPUT_DIR / f"{mid}_{handle}.png"

                if out_path.exists():
                    print(f"  [skip] {out_path.name}")
                    total_done += 1
                    continue

                print(f"\n[{total_done + 1}] {mid} x {product['title'][:50]}")

                try:
                    run_tryon(product, prompts[mid], args.strength, args.seed, out_path)
                except Exception as exc:
                    print(f"  ERROR: {exc}")

                total_done += 1

    print(f"\nBatch complete: {total_done} images -> {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Z-Image Turbo virtual try-on via fal.ai API"
    )
    parser.add_argument(
        "--model", default="M-REC_S3",
        help="Model ID from prompts.json  (default: M-REC_S3)"
    )
    parser.add_argument(
        "--product", type=int, default=0,
        help="0-based product index in the dataset JSON  (default: 0)"
    )
    parser.add_argument(
        "--gender", choices=["male", "female", "both"], default="male",
        help="Product dataset to use  (default: male)"
    )
    parser.add_argument(
        "--strength", type=float, default=DEFAULT_STRENGTH,
        help="Edit strength 0.35-0.80; lower preserves face/body more  (default: 0.55)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility  (default: 42)"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Batch mode: run multiple products x models"
    )
    parser.add_argument(
        "--max-products", type=int, default=5,
        help="Max products per gender in batch mode  (default: 5)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts()

    if args.batch:
        run_batch(args, prompts)
    else:
        run_single(args, prompts)
