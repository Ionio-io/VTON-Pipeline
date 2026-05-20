"""
Print Fidelity Test
===================
Tests how accurately openai/gpt-image-2/edit preserves complex garment prints.

Two test cases chosen deliberately:
  A) Bombay Paisley Red Floral Cotton Top  (women's #16)
     Dense micro-print (Ajrakh-style block-print medallions).
     Hardest fidelity test — hallucination is immediately obvious.

  B) Nuon Teal Checkered Relaxed-Fit Shirt  (men's #16)
     Large-scale oversized checkered geometry (teal + cream squares).
     Tests whether large geometric patterns fare better than micro-prints.

Usage:
    python print_fidelity_test.py

Environment:
    FAL_KEY — your fal.ai API key (https://fal.ai/dashboard/keys)
"""

import os
import re
import time
import json
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
IMAGE_SIZE       = "portrait_4_3"
QUALITY          = "low"
MAX_GARMENT_IMGS = 2
OUTPUT_DIR       = Path("tryon_output/print_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Test cases ────────────────────────────────────────────────────────────────

TESTS = [
    (
        "westside_dataset/products_women.json",
        16,
        "F-HG_S3",
        "MICRO_PRINT",
        "Dense micro-print: repeating Ajrakh-style medallions (~1mm scale per motif). "
        "Hardest fidelity test — model must reproduce red+black+gold geometric medallions "
        "exactly, not hallucinate a 'similar' pattern."
    ),
    (
        "westside_dataset/products_men.json",
        16,
        "M-TRI_S4",
        "GEO_CHECKERED",
        "Large-scale checkered geometry: ~10 oversized teal+cream squares on torso. "
        "Tests whether large geometric repeats are easier to preserve than micro-prints."
    ),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_prompts() -> dict:
    with open("prompts.json") as f:
        raw = json.load(f)
    return {p["id"]: p for p in raw}


def upload_image(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    with open(p, "rb") as f:
        data = f.read()
    print(f"      ⬆  Uploading {p.name} ...")
    url = fal_client.upload(data, mime)
    print(f"         → {url[:60]}...")
    return url


def build_print_prompt(product: dict, model_meta: dict) -> str:
    title       = product["title"]
    category    = (product.get("product_type") or product.get("category") or "clothing")
    gender      = "female" if model_meta["gender"] == "F" else "male"
    body_type   = model_meta["body_type_name"]
    skin_tone   = model_meta["skin_tone_name"]

    desc = product.get("description", "")
    pattern_phrase = ""
    for sent in re.split(r'(?<=[.!?])\s+', re.sub(r'<[^>]+>', '', desc)):
        low = sent.lower()
        if any(w in low for w in ["print", "pattern", "check", "floral", "motif",
                                    "stripe", "graphic", "geometric", "design"]):
            pattern_phrase = sent.strip()
            break

    return f"""VIRTUAL TRY-ON — PRINT FIDELITY TEST

Image 1 = BASE MODEL (full body {gender} model, {body_type} body type, {skin_tone} skin tone).
Image 2+ = GARMENT REFERENCE IMAGES (multiple angles of the actual cloth).

TASK: Place the exact garment from the reference images onto the model.

CRITICAL — PRINT FIDELITY RULES:
1. COPY the garment's print EXACTLY as seen in the reference. Do NOT simplify, blur, or generalise the pattern.
2. Every individual motif, color, and repeat unit of the pattern must appear in the output exactly as in the reference photos.
3. The pattern "{pattern_phrase}" must be faithfully reproduced — not replaced with a vaguely similar texture.
4. Preserve exact colors: if the reference shows red+black+gold medallions, the output must have red+black+gold medallions at the same density.
5. Pattern scale must match the reference — do NOT enlarge or shrink the repeat units.

GARMENT: {title} ({category})
BODY: Preserve the model's face, skin tone, body proportions EXACTLY — only change the clothing.
BACKGROUND: Clean white studio background, soft diffused lighting.
FRAMING: Full body, head to toe, 4K fashion photography quality.

DO NOT: guess the pattern, invent a new pattern, or use a placeholder texture.
DO: faithfully render every detail visible in the garment reference images."""


def save_result_image(url: str, out_path: Path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        out_path.write_bytes(resp.read())
    print(f"  Saved → {out_path}")


def run_test(idx: int, dataset: str, product_idx: int, model_id: str,
             label: str, reason: str, prompts: dict) -> dict:
    product_list = json.loads(Path(dataset).read_text())
    product      = product_list[product_idx]
    model_meta   = prompts[model_id]
    handle       = product["handle"][:50]

    print(f"\n{'═'*62}")
    print(f"  Test {idx+1} — {label}")
    print(f"  Product : {product['title']}")
    print(f"  Why     : {reason[:80]}...")
    print(f"  Model   : {model_meta['body_type_name']} / {model_meta['skin_tone_name']}")
    print(f"  Garment images: {MAX_GARMENT_IMGS}")
    print(f"{'═'*62}\n")

    print("  Step 1 — Uploading images to fal storage...")
    model_img_path = f"generated_images/{model_id}.png"
    garment_paths  = product["local_images"][:MAX_GARMENT_IMGS]

    urls = []
    urls.append(upload_image(model_img_path))
    for gp in garment_paths:
        urls.append(upload_image(gp))
    print(f"  Uploaded {len(urls)} images total\n")

    prompt = build_print_prompt(product, model_meta)
    print(f"  Step 2 — Prompt ready ({len(prompt)} chars)\n")

    print(f"  Step 3 — Calling {MODEL_ID} ...")
    t0 = time.time()
    result = fal_client.subscribe(
        MODEL_ID,
        arguments={
            "prompt":        prompt,
            "image_urls":    urls,
            "image_size":    IMAGE_SIZE,
            "quality":       QUALITY,
            "num_images":    1,
            "output_format": "png",
        },
        with_logs=True,
    )
    elapsed = time.time() - t0

    out_url  = result["images"][0]["url"]
    out_path = OUTPUT_DIR / f"test{idx+1}_{label}_{model_id}_{handle}.png"
    print(f"\n  ✅  Done in {elapsed:.1f}s")
    save_result_image(out_url, out_path)

    return {
        "test":      label,
        "product":   product["title"],
        "model":     model_id,
        "elapsed_s": round(elapsed, 1),
        "out_file":  str(out_path),
        "reason":    reason,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n🔬  Print Fidelity Test — {len(TESTS)} garments")
    print(f"    Model   : {MODEL_ID}")
    print(f"    Quality : {QUALITY}")
    print(f"    Garment refs per test: {MAX_GARMENT_IMGS}")
    print(f"    Output  : {OUTPUT_DIR}/\n")

    prompts = load_prompts()
    log = []

    for i, (dataset, prod_idx, model_id, label, reason) in enumerate(TESTS):
        entry = run_test(i, dataset, prod_idx, model_id, label, reason, prompts)
        log.append(entry)
        if i < len(TESTS) - 1:
            print("\n  Waiting 3s before next test...")
            time.sleep(3)

    log_path = OUTPUT_DIR / "print_fidelity_log.json"
    log_path.write_text(json.dumps(log, indent=2))

    print(f"\n{'═'*62}")
    print(f"  DONE — {len(log)}/{len(TESTS)} tests completed")
    print(f"  Results → {OUTPUT_DIR}/")
    print(f"  Log     → {log_path}")
    print(f"{'═'*62}\n")
    print("  WHAT TO LOOK FOR:")
    print("  MICRO_PRINT  — Do the tiny red+black medallions appear? Or are they")
    print("                 blurred into a generic 'red texture'?")
    print("  GEO_CHECKERED — Are the teal+cream squares correctly sized/spaced?")
    print("                  Or does the AI 'invent' a different grid?")
