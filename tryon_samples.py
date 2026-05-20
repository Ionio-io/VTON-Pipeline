"""
Virtual Try-On — Batch Sample Generator
========================================
Sends garment images + base model image to GPT Image 2 via fal.ai
and generates try-on results.

Uploads local images to fal storage, then calls the image-to-image endpoint.

Output: tryon_output/ folder with result images.

Usage:
    python tryon_samples.py

Environment:
    FAL_KEY — your fal.ai API key (https://fal.ai/dashboard/keys)
"""

import os
import json
import time
import urllib.request
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

MODEL_ID         = "openai/gpt-image-2/edit"
IMAGE_SIZE       = "portrait_4_3"   # 768x1024 — portrait for fashion
QUALITY          = "low"            # fastest; upgrade to medium/high after validating
MAX_GARMENT_IMGS = 1                # 1 garment ref keeps processing fast
OUTPUT_DIR       = Path("tryon_output")

# ── Samples to generate ────────────────────────────────────────────────────────
# Format: (product_json_file, product_index, model_id_from_prompts_json)

SAMPLES = [
    # Sample 1 — Men's Jeans on Rectangle / Medium-Olive model
    ("westside_dataset/products_men.json",   0,  "M-REC_S3"),

    # Sample 2 — Men's Shirt on Inverted Trapezoid / Light-Brown model
    ("westside_dataset/products_men.json",   8,  "M-INV_S4"),

    # Sample 3 — Women's Dress on Hourglass / Light-Brown model
    ("westside_dataset/products_women.json", 7,  "F-HG_S4"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def upload_image(local_path: str) -> str:
    print(f"      ⬆  Uploading {Path(local_path).name} ...")
    with open(local_path, "rb") as f:
        data = f.read()
    ext  = Path(local_path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp"}.get(ext, "image/jpeg")
    url = fal_client.upload(data, mime)
    print(f"         → {url[:60]}...")
    return url


def build_prompt(product: dict, model_meta: dict) -> str:
    gender       = "man" if model_meta["gender"] == "man" else "woman"
    body_type    = model_meta["body_type_name"]
    skin_tone    = model_meta["skin_tone_name"]
    hex_color    = model_meta["hex_midpoint"]
    product_type = product["product_type"]
    title        = product["title"]

    return f"""Image 1 is the BASE MODEL — a full-body studio photo of an Indian {gender}.
Images 2 onwards are GARMENT REFERENCE PHOTOS showing the product from multiple angles.

Generate a professional full-body studio photograph of the person from Image 1 wearing the {product_type} shown in Images 2+.

PERSON — keep EXACTLY as in Image 1:
- Same face, hair, and expression
- {body_type} body build
- {skin_tone} skin tone (hex {hex_color})
- Same relaxed neutral front-facing pose, arms slightly away from body

GARMENT — "{title}":
- Preserve the EXACT color, fabric texture, print, and pattern from the reference images
- Preserve every design detail — collar, buttons, cut, fit, embroidery, zippers
- Garment drapes and fits naturally on the model's body shape

SHOT REQUIREMENTS:
- Full body visible from head to toe
- White seamless studio background
- Soft diffused overhead lighting, no harsh shadows
- Professional e-commerce fashion photography, sharp full-body focus, 4K"""


def build_negative_prompt() -> str:
    return (
        "extra limbs, deformed hands, blurry, low quality, cropped, "
        "watermark, text overlay, logo, wrong garment color, changed pattern, "
        "different person, multiple people, shadows, gradient background, "
        "accessories, jewelry, sunglasses, hat"
    )


def save_result_image(image_url: str, output_path: Path):
    req = urllib.request.Request(
        image_url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        output_path.write_bytes(resp.read())


# ── Main ──────────────────────────────────────────────────────────────────────

def run_sample(idx: int, product_file: str, product_idx: int, model_id: str,
               model_prompts: list[dict]) -> dict:

    products   = load_json(product_file)
    product    = products[product_idx]
    model_meta = next(m for m in model_prompts if m["id"] == model_id)

    model_img_path = f"generated_images/{model_id}.png"
    garment_paths  = product["local_images"][:MAX_GARMENT_IMGS]

    print(f"\n{'─'*62}")
    print(f"  Sample {idx} — {model_id}")
    print(f"  Product : {product['title'][:55]}")
    print(f"  Category: {product['product_type']}")
    print(f"  Model   : {model_meta['body_type_name']} / {model_meta['skin_tone_name']}")
    print(f"  Garment images: {min(len(garment_paths), MAX_GARMENT_IMGS)}")
    print(f"{'─'*62}")

    print("\n  Step 1 — Uploading images to fal storage...")
    all_urls = []
    all_urls.append(upload_image(model_img_path))
    for gp in garment_paths:
        if Path(gp).exists():
            all_urls.append(upload_image(gp))

    print(f"  Uploaded {len(all_urls)} images total")

    prompt = build_prompt(product, model_meta)
    neg    = build_negative_prompt()
    print(f"\n  Step 2 — Prompt ready ({len(prompt)} chars)")

    print(f"\n  Step 3 — Calling {MODEL_ID} ...")
    start = time.time()

    result = fal_client.subscribe(
        MODEL_ID,
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

    elapsed = round(time.time() - start, 2)

    OUTPUT_DIR.mkdir(exist_ok=True)
    gen_url     = result["images"][0]["url"]
    handle      = product["handle"][:40]
    output_name = f"sample{idx}_{model_id}_{handle}.png"
    output_path = OUTPUT_DIR / output_name

    save_result_image(gen_url, output_path)

    print(f"\n  ✅  Done in {elapsed}s")
    print(f"  Saved → {output_path}")

    return {
        "sample":          idx,
        "model_id":        model_id,
        "product_title":   product["title"],
        "product_type":    product["product_type"],
        "output_path":     str(output_path),
        "fal_url":         gen_url,
        "latency_seconds": elapsed,
    }


def main():
    model_prompts = load_json("prompts.json")
    log = []

    print(f"\n🎽  Virtual Try-On — {len(SAMPLES)} Samples")
    print(f"    Model     : {MODEL_ID}")
    print(f"    Quality   : {QUALITY}")
    print(f"    Size      : {IMAGE_SIZE}")
    print(f"    Output    : {OUTPUT_DIR}/\n")

    for i, (product_file, product_idx, model_id) in enumerate(SAMPLES, 1):
        try:
            result = run_sample(i, product_file, product_idx, model_id, model_prompts)
            log.append(result)
        except Exception as e:
            print(f"\n  ❌  Sample {i} failed: {e}")
            log.append({"sample": i, "error": str(e)})

        if i < len(SAMPLES):
            print("\n  Waiting 3s before next sample...")
            time.sleep(3)

    log_path = OUTPUT_DIR / "tryon_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'═'*62}")
    print(f"  DONE — {sum(1 for r in log if 'output_path' in r)}/{len(SAMPLES)} samples generated")
    print(f"  Results → {OUTPUT_DIR}/")
    print(f"  Log     → {log_path}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
