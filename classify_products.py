"""
LLM Product Classifier — GPT-4o Vision via OpenRouter
======================================================
Classifies each Westside product using local images sent as base64 to GPT-4o.

For each product the LLM returns:
  can_use        : true/false — usable for virtual try-on?
  gender         : male | female | unisex
  image_type     : product | tryon | mixed
  category       : normalized category (t-shirt, jeans, dress, kurta, etc.)
  best_image_idx : 1-based index of the best image for the try-on step
  reasoning      : one-line explanation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CONFIG — set OPENROUTER_API_KEY in your environment
  or paste it directly below before running
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os

# ── ⚙️  CONFIGURE HERE ───────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")

# GPT-4o is the best vision model — handles Indian fashion, garment types,
# flat-lay vs tryon distinction extremely well.
MODEL = "openai/gpt-4o"

# Max images to send per product (GPT-4o handles up to ~10 well; all are local)
MAX_IMAGES = 5

# ─────────────────────────────────────────────────────────────────────────────

import json
import time
import base64
import argparse
import urllib.request
import urllib.error
from pathlib import Path

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

INPUT_FILES = {
    "male":   Path("westside_dataset/products_men.json"),
    "female": Path("westside_dataset/products_women.json"),
}
OUTPUT_FILES = {
    "male":   Path("westside_dataset/classified_men.json"),
    "female": Path("westside_dataset/classified_women.json"),
}

VALID_CATEGORIES = {
    "t-shirt", "shirt", "polo", "top", "blouse", "dress",
    "kurta", "ethnic-suit", "pants", "jeans", "shorts", "skirt",
    "jacket", "blazer", "hoodie", "sweatshirt", "joggers",
    "loungewear", "co-ord-set", "saree", "other",
}

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert fashion analyst for a virtual try-on (VTON) pipeline \
specialized in Indian e-commerce clothing. You receive product images and \
return structured JSON. You MUST respond with ONLY valid JSON — \
no markdown fences, no extra text.\
"""

def build_user_prompt(product: dict, n_images: int) -> str:
    return f"""\
Analyze this clothing product:
  Title         : {product['title']}
  Store category: {product['product_type']}
  Tagged gender : {product['gender']}
  Images sent   : {n_images} (examine ALL of them)

Return ONLY this JSON object:

{{
  "can_use": true,
  "gender": "male",
  "image_type": "tryon",
  "category": "jeans",
  "best_image_idx": 1,
  "reasoning": "one sentence"
}}

RULES for each field:

can_use → false ONLY if: not a wearable garment, image is corrupt/unusable,
          or it is purely underwear/intimate wear with no outer clothing.
          Ethnic wear (kurtas, sarees, suits) = true.

gender  → "male" | "female" | "unisex"
          Override the tagged gender only if the images clearly contradict it.

image_type → "product" : garment shown flat, on a ghost/invisible mannequin,
                          on a hanger, or isolated on clean/white background
                          — NO real human body visible
             "tryon"   : a real human model is WEARING the garment;
                          you can see a person's face or body in the clothing
             "mixed"   : some images are product-type, some are tryon-type

category → pick EXACTLY ONE from:
           t-shirt | shirt | polo | top | blouse | dress | kurta | ethnic-suit |
           pants | jeans | shorts | skirt | jacket | blazer | hoodie |
           sweatshirt | joggers | loungewear | co-ord-set | saree | other

best_image_idx → 1-based index of the SINGLE BEST image for try-on processing.
   Priority:
   1st choice: a PRODUCT image (flat-lay / mannequin) showing the FULL garment
   2nd choice: a TRYON image with full garment visible and clean background
   Never pick: close-ups, back-only shots, or images where garment is cut off

reasoning → one sentence covering: image_type decision + category + best image choice\
"""

# ── Image helpers ─────────────────────────────────────────────────────────────

def image_to_base64(path: str) -> tuple[str, str]:
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return mime, b64


def build_image_content(local_paths: list[str], max_images: int) -> list[dict]:
    content = []
    for path in local_paths[:max_images]:
        if not Path(path).exists():
            continue
        try:
            mime, b64 = image_to_base64(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url":    f"data:{mime};base64,{b64}",
                    "detail": "high",
                }
            })
        except Exception as e:
            print(f"      ⚠️  Could not encode {path}: {e}")
    return content

# ── API call ──────────────────────────────────────────────────────────────────

def call_llm(product: dict) -> dict:
    local_paths = product.get("local_images", [])
    if not local_paths:
        raise ValueError("No local images found for this product")

    image_blocks = build_image_content(local_paths, MAX_IMAGES)
    if not image_blocks:
        raise ValueError("Could not encode any images")

    n_images = len(image_blocks)

    content = [
        {"type": "text", "text": build_user_prompt(product, n_images)},
        *image_blocks,
    ]

    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens":  300,
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/Ionio-io/VTON-Pipeline",
            "X-Title":       "VTON Fashion Classifier",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    raw_text = result["choices"][0]["message"]["content"].strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    return json.loads(raw_text)


def validate_classification(cls: dict, product: dict) -> dict:
    cls.setdefault("can_use",        True)
    cls.setdefault("gender",         product.get("gender", "unisex"))
    cls.setdefault("image_type",     "tryon")
    cls.setdefault("category",       "other")
    cls.setdefault("best_image_idx", 1)
    cls.setdefault("reasoning",      "")

    cat = cls["category"].lower().strip()
    cls["category"] = cat if cat in VALID_CATEGORIES else "other"

    n_imgs = len(product.get("local_images", []))
    idx = int(cls.get("best_image_idx", 1))
    cls["best_image_idx"] = max(1, min(idx, n_imgs if n_imgs else 1))

    return cls

# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_json(path: Path) -> list[dict]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_json(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ── Main classification loop ──────────────────────────────────────────────────

def classify_gender(gender_key: str, delay: float):
    input_path  = INPUT_FILES[gender_key]
    output_path = OUTPUT_FILES[gender_key]

    products = load_json(input_path)
    existing = load_json(output_path)
    done_ids = {p["id"] for p in existing}
    results  = list(existing)

    pending = [p for p in products if p["id"] not in done_ids]

    print(f"\n{'─'*62}")
    print(f"  {gender_key.upper()} — {len(products)} total | "
          f"{len(existing)} done | {len(pending)} remaining")
    print(f"{'─'*62}")

    for i, product in enumerate(pending, 1):
        n_local = len(product.get("local_images", []))
        print(f"\n  [{i:02d}/{len(pending)}] {product['title'][:58]}")
        print(f"           type={product['product_type']}  "
              f"local_images={n_local}")

        try:
            cls = call_llm(product)
            cls = validate_classification(cls, product)

            best_idx  = cls["best_image_idx"] - 1
            imgs      = product.get("local_images", [])
            best_path = imgs[best_idx] if imgs else ""

            entry = {
                **product,
                "llm_can_use":         cls["can_use"],
                "llm_gender":          cls["gender"],
                "llm_image_type":      cls["image_type"],
                "llm_category":        cls["category"],
                "llm_best_image_idx":  cls["best_image_idx"],
                "llm_best_image_path": best_path,
                "llm_reasoning":       cls["reasoning"],
            }
            results.append(entry)
            save_json(results, output_path)

            flag = "✅" if cls["can_use"] else "🚫"
            print(f"           {flag} can_use={cls['can_use']}  "
                  f"type={cls['image_type']}  "
                  f"cat={cls['category']}  "
                  f"best=img#{cls['best_image_idx']}")
            print(f"           💬 {cls['reasoning'][:90]}")

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"           ❌ HTTP {e.code}: {body[:150]}")
            results.append({**product, "llm_error": f"HTTP {e.code}: {body[:150]}"})
            save_json(results, output_path)

        except Exception as e:
            print(f"           ❌ {type(e).__name__}: {e}")
            results.append({**product, "llm_error": str(e)})
            save_json(results, output_path)

        if i < len(pending):
            time.sleep(delay)

    usable = sum(1 for r in results if r.get("llm_can_use"))
    errors = sum(1 for r in results if "llm_error" in r)
    print(f"\n  {gender_key.upper()} complete — "
          f"{usable} usable, {len(results)-usable-errors} not usable, {errors} errors")
    print(f"  Saved → {output_path}")


def print_final_summary():
    from collections import Counter
    print(f"\n{'═'*62}")
    print("  CLASSIFICATION SUMMARY")
    print(f"{'═'*62}")
    for gender_key, path in OUTPUT_FILES.items():
        data = load_json(path)
        if not data:
            continue
        usable = [p for p in data if p.get("llm_can_use")]
        cats   = Counter(p.get("llm_category", "?") for p in usable)
        types  = Counter(p.get("llm_image_type", "?") for p in usable)
        print(f"\n  {gender_key.upper()} ({len(usable)}/{len(data)} usable):")
        print(f"    Image types : {dict(types)}")
        print(f"    Categories  :", dict(cats.most_common()))
    print(f"\n{'═'*62}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Classify Westside products with GPT-4o Vision via OpenRouter"
    )
    parser.add_argument(
        "--gender", choices=["male", "female"], default=None,
        help="Only classify one gender (default: both)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Seconds between API calls (default: 1.5)"
    )
    args = parser.parse_args()

    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        print("❌  Set OPENROUTER_API_KEY environment variable before running.")
        print("    export OPENROUTER_API_KEY=sk-or-...")
        return

    print(f"\n🤖  GPT-4o Vision Classifier")
    print(f"    Model    : {MODEL}")
    print(f"    Images   : local base64 (up to {MAX_IMAGES} per product)")
    print(f"    Delay    : {args.delay}s between calls")

    genders = ["male", "female"] if args.gender is None else [args.gender]
    for g in genders:
        classify_gender(g, delay=args.delay)

    print_final_summary()


if __name__ == "__main__":
    main()
