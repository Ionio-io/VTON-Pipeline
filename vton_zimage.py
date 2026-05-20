"""
Z-Image Turbo Virtual Try-On — Local Runner
============================================
Dresses base model images (generated_images/*.png) in Westside products
using ZImageImg2ImgPipeline (diffusers ≥ 0.38.0).

How it works:
  1. Load the pre-generated base model PNG as the init image.
  2. Build a garment description prompt from the product JSON.
  3. Run ZImageImg2ImgPipeline with strength=0.55 so the face/body
     is preserved while only the clothing area changes.
  4. Save to tryon_output/zimage/.

Speed: ~1-3 s/image on A100, ~5-8 s on T4 (16 GB VRAM).

Requirements:
    pip install diffusers>=0.38.0 transformers accelerate torch pillow

Usage:
    # Single try-on
    python vton_zimage.py --model M-REC_S3 --product 0 --gender male

    # Batch: first 5 men's products × 3 male models
    python vton_zimage.py --batch --gender male --max-products 5

    # All products, all models (long!)
    python vton_zimage.py --batch --gender both

Environment:
    No API keys required — runs fully locally.
    Needs ~16 GB VRAM for bfloat16. Use --cpu for CPU (very slow, ~5 min/image).
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

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_REPO      = "Tongyi-MAI/Z-Image-Turbo"
OUTPUT_DIR      = Path("tryon_output/zimage")
IMG_SIZE        = (768, 1024)    # width × height — portrait fashion shot

# Turbo settings (do not change guidance_scale — must be 0.0 for turbo)
NUM_STEPS       = 9              # 8 actual DiT forwards
GUIDANCE_SCALE  = 0.0
STRENGTH        = 0.55           # lower → more of original preserved

NEGATIVE_PROMPT = (
    "deformed, extra limbs, blurry, low quality, watermark, text, logo, "
    "artifacts, bad anatomy, distorted clothing, ugly, duplicate, "
    "out of frame, mutation, disfigured"
)

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_tryon_prompt(product: dict, model_meta: dict) -> str:
    gender     = model_meta["gender"]          # "man" / "woman"
    body_type  = model_meta["body_type_name"]
    skin_tone  = model_meta["skin_tone_name"]
    title      = product["title"]
    category   = product.get("product_type") or "clothing"

    desc_raw   = product.get("description", "")
    desc_clean = re.sub(r"<[^>]+>", " ", desc_raw).strip()
    desc_clean = re.sub(r"\s+", " ", desc_clean)

    # Pull one useful sentence that mentions fabric/fit/style
    garment_detail = ""
    for sent in re.split(r"(?<=[.!?])\s+", desc_clean):
        sent = sent.strip()
        if len(sent) > 15 and any(
            w in sent.lower()
            for w in ["fabric", "fit", "cut", "style", "crafted", "feature",
                      "design", "wear", "silhouette", "waist", "sleeve"]
        ):
            garment_detail = sent
            break
    if not garment_detail and desc_clean:
        garment_detail = desc_clean.split(".")[0].strip()

    detail_clause = f" {garment_detail}." if garment_detail else ""

    return (
        f"Full-body studio photograph of an Indian {gender} with {skin_tone} "
        f"skin tone and {body_type} body type, wearing {title} ({category})."
        f"{detail_clause} "
        "White seamless studio background, soft diffused overhead lighting, "
        "sharp full-body focus, professional e-commerce fashion photography, 4K."
    )


# ── Model loader (singleton) ──────────────────────────────────────────────────

_pipe = None

def get_pipeline(device: str = "cuda") -> ZImageImg2ImgPipeline:
    global _pipe
    if _pipe is not None:
        return _pipe

    print(f"Loading {MODEL_REPO} on {device} …")
    t0 = time.time()

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    pipe  = ZImageImg2ImgPipeline.from_pretrained(MODEL_REPO, torch_dtype=dtype)
    pipe  = pipe.to(device)

    # Optional: compile transformer for ~20% speed boost on repeated calls
    # pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")

    print(f"Model loaded in {time.time() - t0:.1f}s")
    _pipe = pipe
    return pipe


# ── Try-on runner ─────────────────────────────────────────────────────────────

def run_tryon(
    product:    dict,
    model_meta: dict,
    pipe:       ZImageImg2ImgPipeline,
    seed:       int = 42,
    strength:   float = STRENGTH,
) -> Image.Image:
    model_id   = model_meta["id"]
    model_path = Path("generated_images") / f"{model_id}.png"
    if not model_path.exists():
        raise FileNotFoundError(f"Base model image not found: {model_path}")

    init_img = load_image(str(model_path)).resize(IMG_SIZE, Image.LANCZOS)
    prompt   = build_tryon_prompt(product, model_meta)

    print(f"  Prompt ({len(prompt)} chars): {prompt[:120]}…")

    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)

    result = pipe(
        prompt            = prompt,
        image             = init_img,
        strength          = strength,
        num_inference_steps = NUM_STEPS,
        guidance_scale    = GUIDANCE_SCALE,
        negative_prompt   = NEGATIVE_PROMPT,
        generator         = generator,
    ).images[0]

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Z-Image Turbo VTON runner")
    p.add_argument("--model",        default="M-REC_S3",
                   help="Model ID from prompts.json (e.g. M-REC_S3)")
    p.add_argument("--product",      type=int, default=0,
                   help="Product index in the JSON file (0-based)")
    p.add_argument("--gender",       choices=["male", "female", "both"],
                   default="male", help="Which product dataset to use")
    p.add_argument("--strength",     type=float, default=STRENGTH,
                   help="img2img strength (0.4–0.75). Lower = more original preserved")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--batch",        action="store_true",
                   help="Batch mode: run multiple products × models")
    p.add_argument("--max-products", type=int, default=5,
                   help="Max products per gender in batch mode")
    p.add_argument("--cpu",          action="store_true",
                   help="Force CPU inference (slow — ~5 min/image)")
    return p.parse_args()


def load_prompts() -> dict:
    with open("prompts.json") as f:
        raw = json.load(f)
    return {p["id"]: p for p in raw}


MALE_MODELS   = ["M-REC_S3", "M-REC_S4", "M-INV_S3", "M-TRI_S4", "M-OVL_S3"]
FEMALE_MODELS = ["F-HG_S3",  "F-HG_S4",  "F-REC_S3", "F-SPO_S4", "F-TRAP_S3"]


def run_batch(args, prompts: dict, pipe):
    genders = (["male", "female"] if args.gender == "both"
               else [args.gender])

    dataset_map = {
        "male":   ("westside_dataset/products_men.json",   MALE_MODELS),
        "female": ("westside_dataset/products_women.json", FEMALE_MODELS),
    }

    total, done = 0, 0
    for g in genders:
        ds_path, model_ids = dataset_map[g]
        products = json.loads(Path(ds_path).read_text())
        products = products[: args.max_products]

        for prod in products:
            for mid in model_ids:
                if mid not in prompts:
                    continue
                total += 1
                handle  = prod["handle"][:40]
                out_path = OUTPUT_DIR / f"{mid}_{handle}.png"
                if out_path.exists():
                    print(f"  [skip] {out_path.name}")
                    done += 1
                    continue

                print(f"\n[{done+1}/{total}] {mid} × {prod['title'][:50]}")
                t0  = time.time()
                img = run_tryon(prod, prompts[mid], pipe,
                                seed=args.seed, strength=args.strength)
                img.save(out_path)
                print(f"  Saved → {out_path}  ({time.time()-t0:.1f}s)")
                done += 1

    print(f"\n✅  Batch done: {done} images → {OUTPUT_DIR}/")


def run_single(args, prompts: dict, pipe):
    dataset = ("westside_dataset/products_women.json"
               if args.gender == "female"
               else "westside_dataset/products_men.json")
    products = json.loads(Path(dataset).read_text())
    product  = products[args.product]

    if args.model not in prompts:
        raise ValueError(f"Model ID '{args.model}' not found in prompts.json")
    model_meta = prompts[args.model]

    handle   = product["handle"][:40]
    out_path = OUTPUT_DIR / f"{args.model}_{handle}.png"

    print(f"\nProduct : {product['title']}")
    print(f"Model   : {args.model}  ({model_meta['body_type_name']} / {model_meta['skin_tone_name']})")
    print(f"Strength: {args.strength}  |  Steps: {NUM_STEPS}  |  Seed: {args.seed}\n")

    t0  = time.time()
    img = run_tryon(product, model_meta, pipe,
                    seed=args.seed, strength=args.strength)
    img.save(out_path)
    print(f"\n✅  Saved → {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("⚠  Running on CPU — expect ~5 minutes per image.")

    prompts = load_prompts()
    pipe    = get_pipeline(device)

    if args.batch:
        run_batch(args, prompts, pipe)
    else:
        run_single(args, prompts, pipe)
