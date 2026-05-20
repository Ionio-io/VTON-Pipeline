"""
Sophisticated Print Fidelity Test — quality=medium
====================================================
Tests the two most artistically complex garment prints in the catalog
at quality=medium (sharper detail than the low-quality run).

Test A — ETA Off-White Graphic Printed Relaxed-Fit Cotton Shirt (men's #49)
  Non-repeating scenic artwork: atmospheric misty mountains, blue lotus
  flowers, lily-pad leaves. Every element is compositionally placed once.
  Hardest possible print test: the model must reconstruct a painted scene.

Test B — Vark Pink Floral Printed A-Line Ethnic Set (women's #6)
  Dense large-scale tropical floral across a full 3-piece ethnic set
  (kurta + palazzo + dupatta). Multi-colour with metallic accents.
  Tests consistent print rendering across different silhouette shapes.

Usage:
    python sophisticated_print_test.py

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
QUALITY          = "medium"         # upgraded from low; sharper print detail
MAX_GARMENT_IMGS = 3                # 3 reference angles → max pattern info
OUTPUT_DIR       = Path("tryon_output/sophisticated_print_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Test cases ────────────────────────────────────────────────────────────────

TESTS = [
    (
        "westside_dataset/products_men.json",
        49,
        "M-OVL_S3",
        "SCENIC_LANDSCAPE",
        (
            "Non-repeating scenic artwork print: atmospheric misty mountains, "
            "blue lotus flowers, lily-pad leaves, layered botanical composition. "
            "Every element is unique — no tiling. The hardest generative test: "
            "model must reconstruct a painted scene, not a pattern."
        ),
    ),
    (
        "westside_dataset/products_women.json",
        6,
        "F-SPO_S4",
        "TROPICAL_FLORAL_ETHNIC",
        (
            "Dense large-scale tropical floral across a full 3-piece ethnic set "
            "(kurta + palazzo + dupatta). Multi-colour: hot pink base + cream/teal/"
            "gold florals + metallic accents. Tests consistent print rendering "
            "across different silhouette shapes and a sheer dupatta overlay."
        ),
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
    data = p.read_bytes()
    print(f"      ⬆  Uploading {p.name} ...")
    url = fal_client.upload(data, mime)
    print(f"         → {url[:60]}...")
    return url


def build_sophisticated_prompt(product: dict, model_meta: dict) -> str:
    title      = product["title"]
    category   = product.get("product_type") or product.get("category") or "clothing"
    gender     = "female" if model_meta["gender"] == "F" else "male"
    body_type  = model_meta["body_type_name"]
    skin_tone  = model_meta["skin_tone_name"]

    desc_clean = re.sub(r'<[^>]+>', '', product.get("description", ""))
    print_sentence = ""
    for sent in re.split(r'(?<=[.!?])\s+', desc_clean):
        if any(w in sent.lower() for w in
               ["print", "pattern", "floral", "motif", "graphic",
                "illustration", "design", "painted", "landscape", "botanical"]):
            print_sentence = sent.strip()
            break

    return f"""VIRTUAL TRY-ON — SOPHISTICATED PRINT FIDELITY TEST (quality=medium)

IMAGES SUPPLIED:
  • Image 1 = BASE MODEL — {gender}, {body_type} body type, {skin_tone} skin tone.
  • Images 2-4 = GARMENT REFERENCE — multiple angles of the actual garment.

TASK: Dress the model in this exact garment: {title} ({category}).

═══ CRITICAL — PRINT ARTWORK RULES ═══════════════════════════════════
This garment carries a SOPHISTICATED DESIGNED PRINT — NOT a simple repeating tile.
Treat the garment print as a unique piece of artwork that must be reproduced faithfully:

1. SCENE / COMPOSITION: Reproduce the exact visual elements of the print — their
   shapes, positions relative to each other, and spatial arrangement.
   "{print_sentence}"

2. COLOUR ACCURACY: Every colour in the print must match the reference exactly.
   Do not substitute colours. If the reference shows teal + gold + cream on pink,
   the output must show teal + gold + cream on pink.

3. DETAIL DENSITY: Maintain the density of illustrated elements. Do not simplify
   complex areas into blobs or solid fills. Every petal, leaf vein, and fine line
   visible in the reference must appear in the output.

4. NO HALLUCINATION: Do NOT invent new motifs or generalize the print into a
   "similar-looking" pattern. Copy what is in the reference, nothing else.

5. SCALE: The print elements must appear at the correct scale — not enlarged or
   shrunk relative to the garment area.
═══════════════════════════════════════════════════════════════════════

MODEL IDENTITY (DO NOT CHANGE):
  Preserve the model's face, skin tone ({skin_tone}), and body proportions exactly.
  Only the clothing changes.

OUTPUT REQUIREMENTS:
  • Clean white studio background, soft even lighting
  • Full body head-to-toe frame
  • 4K fashion editorial quality
  • Garment falls naturally on the {body_type} body type"""


def save_result(url: str, out_path: Path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        out_path.write_bytes(resp.read())
    print(f"  Saved → {out_path}")


def run_test(idx: int, dataset: str, prod_idx: int,
             model_id: str, label: str, reason: str, prompts: dict) -> dict:
    products  = json.loads(Path(dataset).read_text())
    product   = products[prod_idx]
    model     = prompts[model_id]
    handle    = product["handle"][:45]

    print(f"\n{'═'*62}")
    print(f"  Test {idx+1} — {label}")
    print(f"  Product  : {product['title']}")
    print(f"  Why      : {reason[:90]}...")
    print(f"  Model    : {model['body_type_name']} / {model['skin_tone_name']}")
    print(f"  Garment refs: {MAX_GARMENT_IMGS}  |  Quality: {QUALITY}")
    print(f"{'═'*62}\n")

    print("  Step 1 — Uploading images...")
    garment_images = product["local_images"][:MAX_GARMENT_IMGS]
    urls = [upload_image(f"generated_images/{model_id}.png")]
    for gp in garment_images:
        urls.append(upload_image(gp))
    print(f"  Uploaded {len(urls)} images total\n")

    prompt = build_sophisticated_prompt(product, model)
    print(f"  Step 2 — Prompt ready ({len(prompt)} chars)\n")

    print(f"  Step 3 — Calling {MODEL_ID} [quality={QUALITY}] ...")
    print(f"           (medium quality typically takes 2-4 minutes)\n")
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
    print(f"\n  ✅  Done in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    save_result(out_url, out_path)

    return {
        "test":      label,
        "product":   product["title"],
        "model":     model_id,
        "quality":   QUALITY,
        "elapsed_s": round(elapsed, 1),
        "out_file":  str(out_path),
        "reason":    reason,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n🎨  Sophisticated Print Fidelity Test — {len(TESTS)} garments")
    print(f"    Model   : {MODEL_ID}")
    print(f"    Quality : {QUALITY}  ← upgraded for sharper detail")
    print(f"    Garment refs per test: {MAX_GARMENT_IMGS}")
    print(f"    Output  : {OUTPUT_DIR}/\n")
    print(f"    NOTE: medium quality is ~2-4× slower than low.")
    print(f"    Expected runtime: ~4-8 minutes total.\n")

    prompts = load_prompts()
    log = []

    for i, (dataset, prod_idx, model_id, label, reason) in enumerate(TESTS):
        entry = run_test(i, dataset, prod_idx, model_id, label, reason, prompts)
        log.append(entry)
        if i < len(TESTS) - 1:
            print("\n  Waiting 3s before next test...")
            time.sleep(3)

    log_path = OUTPUT_DIR / "sophisticated_print_log.json"
    log_path.write_text(json.dumps(log, indent=2))

    print(f"\n{'═'*62}")
    print(f"  ALL DONE — {len(log)}/{len(TESTS)} tests at quality=medium")
    print(f"  Results → {OUTPUT_DIR}/")
    print(f"  Log     → {log_path}")
    print(f"{'═'*62}\n")
    print("  WHAT TO LOOK FOR:")
    print("  SCENIC_LANDSCAPE ——  Can the AI reconstruct the lotus flowers,")
    print("    misty mountains, lily pads as a coherent scene? Or does it")
    print("    collapse to a vague 'floral shirt' texture?")
    print("  TROPICAL_FLORAL_ETHNIC — Are the cream/teal/gold florals on")
    print("    hot pink correctly rendered across the kurta AND palazzo?")
    print("    Does the dupatta maintain its sheer quality over the print?")
