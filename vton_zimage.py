"""
Z-Image Turbo Virtual Try-On  --  Local GPU Runner
==================================================
Runs virtual try-on inference locally using ZImageImg2ImgPipeline
from diffusers.  The base model PNG is partially noised and then
denoised guided by a garment-description text prompt.

Requirements:
    pip install diffusers>=0.38.0 transformers accelerate torch pillow
    GPU with at least 16 GB VRAM recommended (runs on CPU with --cpu flag,
    but expect ~5 minutes per image).

Usage:
    python vton_zimage.py --model M-REC_S3 --product 0 --gender male
    python vton_zimage.py --batch --gender male --max-products 5
    python vton_zimage.py --batch --gender both
    python vton_zimage.py --model F-HG_S4 --product 2 --gender female --strength 0.50
    python vton_zimage.py --model M-TRI_S3 --product 1 --gender male --cpu

No API keys required -- model weights are downloaded from Hugging Face on
first run (~12 GB download).
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from diffusers import ZImageImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_MODEL_REPO = "Tongyi-MAI/Z-Image-Turbo"
OUTPUT_DIR    = Path("tryon_output/zimage")
IMG_SIZE      = (768, 1024)   # (width, height) -- portrait for fashion shots

# Turbo inference settings
NUM_STEPS      = 9     # 8 actual DiT forward passes (turbo mode)
GUIDANCE_SCALE = 0.0   # must be 0.0 for distilled turbo models
DEFAULT_STRENGTH = 0.55  # 0.0 = no change, 1.0 = ignore source image entirely

NEGATIVE_PROMPT = (
    "deformed, extra limbs, blurry, low quality, watermark, text, logo, "
    "artifacts, bad anatomy, distorted clothing, ugly, duplicate, "
    "out of frame, mutation, disfigured"
)

# Model subsets for batch mode (covers a range of body types and skin tones)
BATCH_MALE_MODELS   = ["M-REC_S3", "M-REC_S4", "M-INV_S3", "M-TRI_S4", "M-OVL_S3"]
BATCH_FEMALE_MODELS = ["F-HG_S3",  "F-HG_S4",  "F-REC_S3", "F-SPO_S4", "F-TRAP_S3"]


# ---------------------------------------------------------------------------
# Pipeline loader (singleton -- model is loaded once and reused)
# ---------------------------------------------------------------------------

_pipeline = None


def get_pipeline(device: str) -> ZImageImg2ImgPipeline:
    """
    Load ZImageImg2ImgPipeline onto the target device.

    Uses bfloat16 on GPU for speed and memory efficiency.
    Caches the loaded pipeline in a module-level variable so repeated
    calls within the same process reuse the already-loaded weights.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print(f"Loading {HF_MODEL_REPO} on {device} ...")
    t0    = time.time()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    pipe  = ZImageImg2ImgPipeline.from_pretrained(HF_MODEL_REPO, torch_dtype=dtype)
    pipe  = pipe.to(device)

    # Uncomment to compile the transformer for ~20% throughput gain on repeated calls:
    # pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")

    print(f"Model ready in {time.time() - t0:.1f}s")
    _pipeline = pipe
    return pipe


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_tryon_prompt(product: dict, model_meta: dict) -> str:
    """
    Build a structured text prompt that describes the desired output image.

    Combines model identity (gender, body type, skin tone) with garment
    details extracted from the product JSON.  A descriptive sentence about
    fabric or fit is pulled from the product description when available.
    """
    gender    = model_meta["gender"]          # "man" or "woman"
    body_type = model_meta["body_type_name"]
    skin_tone = model_meta["skin_tone_name"]
    title     = product["title"]
    category  = product.get("product_type") or "clothing"

    # Strip HTML and normalise whitespace
    desc_raw   = product.get("description", "")
    desc_clean = re.sub(r"<[^>]+>", " ", desc_raw).strip()
    desc_clean = re.sub(r"\s+", " ", desc_clean)

    # Find a sentence that mentions construction/fit details
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
# Try-on runner
# ---------------------------------------------------------------------------

def run_tryon(
    product:    dict,
    model_meta: dict,
    pipe:       ZImageImg2ImgPipeline,
    strength:   float = DEFAULT_STRENGTH,
    seed:       int   = 42,
) -> Image.Image:
    """
    Generate a try-on image for one product x model combination.

    Loads the base model PNG from generated_images/, resizes it to IMG_SIZE,
    then runs img2img inference guided by the garment description prompt.

    Args:
        product:    Product entry from westside dataset JSON.
        model_meta: Model entry from prompts.json.
        pipe:       Loaded ZImageImg2ImgPipeline instance.
        strength:   Edit strength (lower = more of the original is preserved).
        seed:       Random seed for reproducibility.

    Returns:
        PIL Image of the generated try-on.
    """
    model_id   = model_meta["id"]
    base_path  = Path("generated_images") / f"{model_id}.png"

    if not base_path.exists():
        raise FileNotFoundError(f"Base model image not found: {base_path}")

    init_image = load_image(str(base_path)).resize(IMG_SIZE, Image.LANCZOS)
    prompt     = build_tryon_prompt(product, model_meta)
    generator  = torch.Generator(device=pipe.device.type).manual_seed(seed)

    print(f"  Prompt ({len(prompt)} chars): {prompt[:120]}...")

    result = pipe(
        prompt              = prompt,
        image               = init_image,
        strength            = strength,
        num_inference_steps = NUM_STEPS,
        guidance_scale      = GUIDANCE_SCALE,
        negative_prompt     = NEGATIVE_PROMPT,
        generator           = generator,
    ).images[0]

    return result


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_prompts() -> dict:
    """Return prompts.json as a dict keyed by model ID."""
    with open("prompts.json", encoding="utf-8") as f:
        raw = json.load(f)
    return {entry["id"]: entry for entry in raw}


def load_products(gender: str) -> list:
    """Load the product list JSON for the given gender."""
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

def run_single(args, prompts: dict, pipe: ZImageImg2ImgPipeline) -> None:
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

    t0  = time.time()
    img = run_tryon(product, model_meta, pipe, args.strength, args.seed)
    img.save(out_path)

    print(f"\n[OK] Saved -> {out_path}  ({time.time() - t0:.1f}s)")


def run_batch(args, prompts: dict, pipe: ZImageImg2ImgPipeline) -> None:
    genders = ["male", "female"] if args.gender == "both" else [args.gender]

    dataset_map = {
        "male":   BATCH_MALE_MODELS,
        "female": BATCH_FEMALE_MODELS,
    }

    total_done = 0

    for gender in genders:
        model_ids = dataset_map[gender]
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

                t0  = time.time()
                img = run_tryon(product, prompts[mid], pipe, args.strength, args.seed)
                img.save(out_path)
                print(f"  Saved -> {out_path}  ({time.time() - t0:.1f}s)")

                total_done += 1

    print(f"\nBatch complete: {total_done} images -> {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Z-Image Turbo virtual try-on -- local GPU runner"
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
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference (very slow, ~5 min/image)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.cpu:
        device = "cpu"
        print("WARNING: Running on CPU -- expect ~5 minutes per image.")
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        print("WARNING: No CUDA device found, falling back to CPU.")

    prompts = load_prompts()
    pipe    = get_pipeline(device)

    if args.batch:
        run_batch(args, prompts, pipe)
    else:
        run_single(args, prompts, pipe)
