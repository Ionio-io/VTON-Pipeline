"""
Virtual Try-On  --  Batch Sample Generator
==========================================
Composites garments onto base model images using the GPT Image 2
edit endpoint via fal.ai.

Workflow for each sample:
    1. Upload the base model PNG and up to MAX_GARMENT_IMGS garment
       reference photos to fal storage.
    2. Build a structured prompt that instructs the model to dress
       the person in the referenced garment while preserving identity.
    3. Call openai/gpt-image-2/edit and save the result.

Output images are saved to tryon_output/ with a log file.

Usage:
    python tryon_samples.py

    Edit the SAMPLES list below to choose which products and base
    models to composite.

Quality:
    Change QUALITY to control speed vs detail:
        "low"    ~30 s   (good for rapid iteration)
        "medium" ~2-4 m  (better print reproduction)
        "high"   ~5-8 m  (best quality, use before final delivery)

Environment:
    FAL_KEY  --  fal.ai API key  (https://fal.ai/dashboard/keys)
"""

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

FAL_ENDPOINT     = "openai/gpt-image-2/edit"
IMAGE_SIZE       = "portrait_4_3"   # 768x1024 -- portrait for fashion
QUALITY          = "low"            # "low" | "medium" | "high"
MAX_GARMENT_IMGS = 1                # number of garment reference images to upload
OUTPUT_DIR       = Path("tryon_output")


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
# Each entry: (products_json_path, product_index, model_id_from_prompts_json)
# Add or remove rows to change what gets generated.

SAMPLES = [
    # Sample 1 -- Men's Jeans on Rectangle / Medium-Olive model
    ("westside_dataset/products_men.json",   0,  "M-REC_S3"),

    # Sample 2 -- Men's Shirt on Inverted Trapezoid / Light-Brown model
    ("westside_dataset/products_men.json",   8,  "M-INV_S4"),

    # Sample 3 -- Women's Dress on Hourglass / Light-Brown model
    ("westside_dataset/products_women.json", 7,  "F-HG_S4"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def upload_image(local_path: str) -> str:
    """Upload a local image file to fal storage and return the CDN URL."""
    name = Path(local_path).name
    print(f"      Uploading {name} ...")
    with open(local_path, "rb") as f:
        data = f.read()
    ext  = Path(local_path).suffix.lower().lstrip(".")
    mime = {
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")
    url = fal_client.upload(data, mime)
    print(f"         -> {url[:60]}...")
    return url


def build_prompt(product: dict, model_meta: dict) -> str:
    """
    Build the try-on instruction prompt for GPT Image 2 edit.

    The prompt treats Image 1 as the person reference and Images 2+
    as garment reference photos, asking the model to dress the person
    in the garment while preserving all identity attributes.
    """
    gender       = "man" if model_meta["gender"] == "man" else "woman"
    body_type    = model_meta["body_type_name"]
    skin_tone    = model_meta["skin_tone_name"]
    hex_color    = model_meta["hex_midpoint"]
    product_type = product["product_type"]
    title        = product["title"]

    return (
        f"Image 1 is the BASE MODEL -- a full-body studio photo of an Indian {gender}.\n"
        "Images 2 onwards are GARMENT REFERENCE PHOTOS showing the product from multiple angles.\n\n"
        f"Generate a professional full-body studio photograph of the person from Image 1 "
        f"wearing the {product_type} shown in Images 2+.\n\n"
        "PERSON -- keep EXACTLY as in Image 1:\n"
        "- Same face, hair, and expression\n"
        f"- {body_type} body build\n"
        f"- {skin_tone} skin tone (hex {hex_color})\n"
        "- Same relaxed neutral front-facing pose, arms slightly away from body\n\n"
        f'GARMENT -- "{title}":\n'
        "- Preserve the EXACT color, fabric texture, print, and pattern from the reference images\n"
        "- Preserve every design detail -- collar, buttons, cut, fit, embroidery, zippers\n"
        "- Garment drapes and fits naturally on the model's body shape\n\n"
        "SHOT REQUIREMENTS:\n"
        "- Full body visible from head to toe\n"
        "- White seamless studio background\n"
        "- Soft diffused overhead lighting, no harsh shadows\n"
        "- Professional e-commerce fashion photography, sharp full-body focus, 4K"
    )


def build_negative_prompt() -> str:
    return (
        "extra limbs, deformed hands, blurry, low quality, cropped, "
        "watermark, text overlay, logo, wrong garment color, changed pattern, "
        "different person, multiple people, shadows, gradient background, "
        "accessories, jewelry, sunglasses, hat"
    )


def save_image(image_url: str, output_path: Path) -> None:
    """Download a URL and write the bytes to output_path."""
    req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        output_path.write_bytes(resp.read())


# ---------------------------------------------------------------------------
# Sample runner
# ---------------------------------------------------------------------------

def run_sample(
    idx:           int,
    product_file:  str,
    product_idx:   int,
    model_id:      str,
    model_prompts: list,
) -> dict:
    """
    Run a single try-on: upload images, call the API, save the result.

    Args:
        idx:           1-based sample index for naming output files.
        product_file:  Path to the products JSON dataset.
        product_idx:   0-based index of the target product.
        model_id:      Model ID string from prompts.json (e.g. "M-REC_S3").
        model_prompts: Full list of model prompt dicts from prompts.json.

    Returns:
        Log dict with output path, URL, and timing information.
    """
    products   = load_json(product_file)
    product    = products[product_idx]
    model_meta = next(m for m in model_prompts if m["id"] == model_id)

    garment_paths = product["local_images"][:MAX_GARMENT_IMGS]

    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  Sample {idx} -- {model_id}")
    print(f"  Product : {product['title'][:55]}")
    print(f"  Category: {product['product_type']}")
    print(f"  Model   : {model_meta['body_type_name']} / {model_meta['skin_tone_name']}")
    print(f"  Garment images : {min(len(garment_paths), MAX_GARMENT_IMGS)}")
    print(sep)

    print("\n  Step 1 -- Uploading images to fal storage...")
    all_urls = [upload_image(f"generated_images/{model_id}.png")]
    for gp in garment_paths:
        if Path(gp).exists():
            all_urls.append(upload_image(gp))
    print(f"  Uploaded {len(all_urls)} image(s)")

    prompt = build_prompt(product, model_meta)
    neg    = build_negative_prompt()
    print(f"\n  Step 2 -- Prompt ready ({len(prompt)} chars)")

    print(f"\n  Step 3 -- Calling {FAL_ENDPOINT} [quality={QUALITY}]...")
    t0 = time.time()

    result = fal_client.subscribe(
        FAL_ENDPOINT,
        arguments={
            "prompt":        prompt,
            "image_urls":    all_urls,
            "image_size":    IMAGE_SIZE,
            "quality":       QUALITY,
            "num_images":    1,
            "output_format": "png",
        },
        with_logs=True,
    )

    elapsed = round(time.time() - t0, 2)

    OUTPUT_DIR.mkdir(exist_ok=True)
    gen_url     = result["images"][0]["url"]
    handle      = product["handle"][:40]
    output_path = OUTPUT_DIR / f"sample{idx}_{model_id}_{handle}.png"

    save_image(gen_url, output_path)

    print(f"\n  Done in {elapsed}s")
    print(f"  Saved -> {output_path}")

    return {
        "sample":          idx,
        "model_id":        model_id,
        "product_title":   product["title"],
        "product_type":    product["product_type"],
        "output_path":     str(output_path),
        "fal_url":         gen_url,
        "latency_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_prompts = load_json("prompts.json")
    log = []

    print(f"\nVirtual Try-On -- {len(SAMPLES)} sample(s)")
    print(f"  Endpoint : {FAL_ENDPOINT}")
    print(f"  Quality  : {QUALITY}")
    print(f"  Size     : {IMAGE_SIZE}")
    print(f"  Output   : {OUTPUT_DIR}/\n")

    for i, (product_file, product_idx, model_id) in enumerate(SAMPLES, 1):
        try:
            result = run_sample(i, product_file, product_idx, model_id, model_prompts)
            log.append(result)
        except Exception as exc:
            print(f"\n  ERROR  Sample {i} failed: {exc}")
            log.append({"sample": i, "error": str(exc)})

        if i < len(SAMPLES):
            print("\n  Waiting 3s before next sample...")
            time.sleep(3)

    log_path = OUTPUT_DIR / "tryon_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    sep = "=" * 62
    success = sum(1 for r in log if "output_path" in r)
    print(f"\n{sep}")
    print(f"  DONE -- {success}/{len(SAMPLES)} samples generated")
    print(f"  Results -> {OUTPUT_DIR}/")
    print(f"  Log     -> {log_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
