"""
Sophisticated Print Fidelity Test  --  quality=medium
=====================================================
Tests GPT Image 2 edit on the two most artistically complex garment prints
in the Westside catalogue, at quality=medium for sharper detail.

    Test A -- ETA Off-White Graphic Printed Relaxed-Fit Cotton Shirt  (men's #49)
    Non-repeating scenic artwork: atmospheric misty mountains, blue lotus
    flowers, lily-pad leaves.  Every element is compositionally placed once.
    Hardest possible print test: the model must reconstruct a painted scene.

    Test B -- Vark Pink Floral Printed A-Line Ethnic Set  (women's #6)
    Dense large-scale tropical floral across a full 3-piece ethnic set
    (kurta + palazzo + dupatta).  Multi-colour with metallic accents.
    Tests consistent print rendering across different silhouette shapes.

Usage:
    python sophisticated_print_test.py

Environment:
    FAL_KEY  --  fal.ai API key  (https://fal.ai/dashboard/keys)
"""

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
        "Get your key at https://fal.ai/dashboard/keys and run:\n"
        "  export FAL_KEY=your_key_here"
    )
os.environ["FAL_KEY"] = FAL_KEY

FAL_ENDPOINT     = "openai/gpt-image-2/edit"
IMAGE_SIZE       = "portrait_4_3"
QUALITY          = "medium"         # upgraded from low for sharper print detail
MAX_GARMENT_IMGS = 3                # 3 reference angles -> maximum pattern information
OUTPUT_DIR       = Path("tryon_output/sophisticated_print_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
# Each entry: (dataset_path, product_index, model_id, label, reason)

TESTS = [
    (
        "westside_dataset/products_men.json",
        49,
        "M-OVL_S3",
        "SCENIC_LANDSCAPE",
        (
            "Non-repeating scenic artwork print: atmospheric misty mountains, "
            "blue lotus flowers, lily-pad leaves, layered botanical composition. "
            "Every element is unique - no tiling. The hardest generative test: "
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts() -> dict:
    with open("prompts.json", encoding="utf-8") as f:
        raw = json.load(f)
    return {entry["id"]: entry for entry in raw}


def upload_image(path: str) -> str:
    """Upload a local image to fal storage and return the CDN URL."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    data = p.read_bytes()
    print(f"      Uploading {p.name} ...")
    url = fal_client.upload(data, mime)
    print(f"         -> {url[:60]}...")
    return url


def build_sophisticated_prompt(product: dict, model_meta: dict) -> str:
    """
    Build a prompt that enforces faithful reproduction of sophisticated
    single-composition artwork prints (not tiled repeating patterns).
    """
    title     = product["title"]
    category  = product.get("product_type") or product.get("category") or "clothing"
    gender    = model_meta["gender"]          # "man" or "woman"
    body_type = model_meta["body_type_name"]
    skin_tone = model_meta["skin_tone_name"]

    # Extract one sentence that describes the print composition
    desc_clean = re.sub(r"<[^>]+>", " ", product.get("description", "")).strip()
    print_sentence = ""
    for sent in re.split(r"(?<=[.!?])\s+", desc_clean):
        if any(w in sent.lower() for w in
               ["print", "pattern", "floral", "motif", "graphic",
                "illustration", "design", "painted", "landscape", "botanical"]):
            print_sentence = sent.strip()
            break

    return (
        "VIRTUAL TRY-ON - SOPHISTICATED PRINT FIDELITY TEST (quality=medium)\n\n"
        "IMAGES SUPPLIED:\n"
        f"  Image 1     = BASE MODEL - {gender}, {body_type} body type, {skin_tone} skin tone.\n"
        "  Images 2-4  = GARMENT REFERENCE - multiple angles of the actual garment.\n\n"
        f"TASK: Dress the model in this exact garment: {title} ({category}).\n\n"
        "CRITICAL - PRINT ARTWORK RULES\n"
        "This garment carries a SOPHISTICATED DESIGNED PRINT - NOT a simple repeating tile.\n"
        "Treat the garment print as a unique piece of artwork that must be reproduced faithfully:\n\n"
        "1. SCENE / COMPOSITION: Reproduce the exact visual elements of the print - their\n"
        "   shapes, positions relative to each other, and spatial arrangement.\n"
        f'   "{print_sentence}"\n\n'
        "2. COLOUR ACCURACY: Every colour in the print must match the reference exactly.\n"
        "   Do not substitute colours. If the reference shows teal + gold + cream on pink,\n"
        "   the output must show teal + gold + cream on pink.\n\n"
        "3. DETAIL DENSITY: Maintain the density of illustrated elements. Do not simplify\n"
        "   complex areas into blobs or solid fills. Every petal, leaf vein, and fine line\n"
        "   visible in the reference must appear in the output.\n\n"
        "4. NO HALLUCINATION: Do NOT invent new motifs or generalise the print into a\n"
        '   "similar-looking" pattern. Copy what is in the reference, nothing else.\n\n'
        "5. SCALE: The print elements must appear at the correct scale - not enlarged or\n"
        "   shrunk relative to the garment area.\n\n"
        "MODEL IDENTITY (DO NOT CHANGE):\n"
        f"  Preserve the model's face, skin tone ({skin_tone}), and body proportions exactly.\n"
        "  Only the clothing changes.\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "  - Clean white studio background, soft even lighting\n"
        "  - Full body head-to-toe frame\n"
        "  - 4K fashion editorial quality\n"
        f"  - Garment falls naturally on the {body_type} body type"
    )


def save_image(url: str, out_path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        out_path.write_bytes(resp.read())
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(
    idx:        int,
    dataset:    str,
    prod_idx:   int,
    model_id:   str,
    label:      str,
    reason:     str,
    prompts:    dict,
) -> dict:
    """Run a single sophisticated print test and return a log entry."""
    products = json.loads(Path(dataset).read_text(encoding="utf-8"))
    product  = products[prod_idx]
    model    = prompts[model_id]
    handle   = product["handle"][:45]

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Test {idx + 1} - {label}")
    print(f"  Product  : {product['title']}")
    print(f"  Reason   : {reason[:90]}...")
    print(f"  Model    : {model['body_type_name']} / {model['skin_tone_name']}")
    print(f"  Garment refs : {MAX_GARMENT_IMGS}  |  Quality : {QUALITY}")
    print(f"{sep}\n")

    print("  Step 1 - Uploading images...")
    garment_images = product["local_images"][:MAX_GARMENT_IMGS]
    urls = [upload_image(f"generated_images/{model_id}.png")]
    for gp in garment_images:
        urls.append(upload_image(gp))
    print(f"  Uploaded {len(urls)} image(s)\n")

    prompt = build_sophisticated_prompt(product, model)
    print(f"  Step 2 - Prompt ready ({len(prompt)} chars)\n")

    print(f"  Step 3 - Calling {FAL_ENDPOINT} [quality={QUALITY}]...")
    print("           (medium quality typically takes 2-4 minutes)\n")
    t0 = time.time()
    result = fal_client.subscribe(
        FAL_ENDPOINT,
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
    out_path = OUTPUT_DIR / f"test{idx + 1}_{label}_{model_id}_{handle}.png"

    print(f"\n  Done in {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    save_image(out_url, out_path)

    return {
        "test":      label,
        "product":   product["title"],
        "model":     model_id,
        "quality":   QUALITY,
        "elapsed_s": round(elapsed, 1),
        "out_file":  str(out_path),
        "reason":    reason,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\nSophisticated Print Fidelity Test - {len(TESTS)} garment(s)")
    print(f"  Endpoint     : {FAL_ENDPOINT}")
    print(f"  Quality      : {QUALITY}  (upgraded for sharper detail)")
    print(f"  Garment refs : {MAX_GARMENT_IMGS} per test")
    print(f"  Output       : {OUTPUT_DIR}/")
    print(f"  NOTE: medium quality is ~2-4x slower than low.")
    print(f"  Expected runtime: ~4-8 minutes total.\n")

    prompts = load_prompts()
    log     = []

    for i, (dataset, prod_idx, model_id, label, reason) in enumerate(TESTS):
        entry = run_test(i, dataset, prod_idx, model_id, label, reason, prompts)
        log.append(entry)
        if i < len(TESTS) - 1:
            print("\n  Waiting 3s before next test...")
            time.sleep(3)

    log_path = OUTPUT_DIR / "sophisticated_print_log.json"
    log_path.write_text(json.dumps(log, indent=2))

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  ALL DONE - {len(log)}/{len(TESTS)} tests at quality=medium")
    print(f"  Results -> {OUTPUT_DIR}/")
    print(f"  Log     -> {log_path}")
    print(f"{sep}")
    print()
    print("  WHAT TO LOOK FOR:")
    print("  SCENIC_LANDSCAPE       - Can the AI reconstruct the lotus flowers,")
    print("    misty mountains, lily pads as a coherent scene? Or does it")
    print("    collapse to a vague 'floral shirt' texture?")
    print("  TROPICAL_FLORAL_ETHNIC - Are the cream/teal/gold florals on")
    print("    hot pink correctly rendered across the kurta AND palazzo?")
    print("    Does the dupatta maintain its sheer quality over the print?")
