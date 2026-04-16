#!/usr/bin/env python3
"""
Clothing Extraction + Tagging — End-to-End
============================================

Takes a person photo → extracts each clothing item → tags it with:
  1. CATEGORY    — "Upper-clothes", "Pants", "Dress", etc. (from segmentation)
  2. DETAIL      — "t-shirt", "jeans", "hoodie", etc. (from CLIP zero-shot)
  3. COLOR       — "navy blue", "black", "red", etc. (from pixel analysis)
  4. CONFIDENCE  — how sure the model is

Setup (one time):
    pip install transformers torch torchvision Pillow opencv-python-headless numpy

Usage:
    python tag_clothes.py --image person.jpg
    python tag_clothes.py --image person.jpg --output-dir ./results

Outputs:
    output/
    ├── 1_all_items_tagged.png           ← Overview: all items labeled on the photo
    ├── 2_extracted_Upper-clothes.png    ← Shirt cut out + tag overlay
    ├── 2_extracted_Pants.png            ← Pants cut out + tag overlay
    ├── ...                              ← One per detected clothing item
    └── tags.json                        ← Machine-readable tags for every item
"""

import os
import sys
import json
import argparse
import time
from collections import Counter

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont


# ─── ATR Labels ──────────────────────────────────────────────────────────────

LABELS = [
    "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt",
    "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face",
    "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf",
]

COLORS = [
    [0, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 170],
    [0, 255, 0], [85, 0, 255], [0, 0, 255], [0, 255, 170],
    [170, 170, 0], [255, 255, 0], [255, 200, 0], [255, 0, 0],
    [0, 170, 255], [0, 85, 255], [255, 85, 170], [255, 170, 170],
    [170, 0, 255], [0, 255, 255],
]

# Which label IDs are actual clothing items (not body parts or background)
CLOTHING_LABEL_IDS = {
    1: "Hat",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "Dress",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    16: "Bag",
    17: "Scarf",
}

# CLIP will try to classify each item into these detailed categories
DETAIL_LABELS = {
    "Upper-clothes": [
        "t-shirt", "dress shirt", "polo shirt", "blouse", "tank top",
        "sweater", "hoodie", "jacket", "blazer", "cardigan",
        "crop top", "vest", "coat", "sweatshirt", "tunic",
    ],
    "Pants": [
        "jeans", "dress pants", "chinos", "sweatpants", "cargo pants",
        "shorts", "leggings", "trousers", "joggers", "corduroy pants",
    ],
    "Skirt": [
        "mini skirt", "midi skirt", "maxi skirt", "pencil skirt",
        "pleated skirt", "denim skirt", "a-line skirt",
    ],
    "Dress": [
        "casual dress", "formal dress", "maxi dress", "mini dress",
        "cocktail dress", "sundress", "shirt dress", "wrap dress",
        "bodycon dress", "evening gown",
    ],
    "Hat": [
        "baseball cap", "beanie", "sun hat", "fedora", "bucket hat",
        "beret", "visor", "cowboy hat", "trucker hat",
    ],
    "Bag": [
        "handbag", "backpack", "tote bag", "crossbody bag", "clutch",
        "shoulder bag", "messenger bag", "fanny pack",
    ],
    "Scarf": [
        "winter scarf", "silk scarf", "bandana", "shawl", "wrap",
    ],
    "Belt": [
        "leather belt", "fabric belt", "chain belt", "dress belt",
    ],
    "Sunglasses": [
        "aviator sunglasses", "round sunglasses", "square sunglasses",
        "cat-eye sunglasses", "sports sunglasses",
    ],
}

# Shoes get merged into one item
SHOE_DETAILS = [
    "sneakers", "dress shoes", "boots", "sandals", "heels",
    "loafers", "running shoes", "slip-ons", "flats", "oxford shoes",
]

# Named colors for detection
COLOR_MAP = {
    "black":       (0, 0, 0),
    "white":       (255, 255, 255),
    "red":         (200, 30, 30),
    "dark red":    (139, 0, 0),
    "blue":        (30, 30, 200),
    "navy blue":   (0, 0, 128),
    "light blue":  (135, 180, 230),
    "green":       (30, 150, 30),
    "dark green":  (0, 100, 0),
    "yellow":      (230, 220, 40),
    "orange":      (230, 140, 30),
    "purple":      (128, 0, 128),
    "pink":        (230, 130, 160),
    "brown":       (139, 90, 43),
    "beige":       (210, 190, 160),
    "gray":        (140, 140, 140),
    "dark gray":   (70, 70, 70),
    "light gray":  (200, 200, 200),
    "olive":       (128, 128, 0),
    "maroon":      (128, 0, 0),
    "teal":        (0, 128, 128),
    "cream":       (255, 253, 208),
    "khaki":       (195, 176, 145),
}


# ─── Load Models ─────────────────────────────────────────────────────────────

def load_segmentation_model(device: str):
    """Load SegFormer-B2 for clothing segmentation."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    model_id = "mattmdjaga/segformer_b2_clothes"
    print(f"  Loading segmentation: {model_id}")
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    model.to(device).eval()
    return processor, model


def load_clip_model(device: str):
    """Load CLIP for detailed clothing classification."""
    from transformers import CLIPProcessor, CLIPModel

    model_id = "openai/clip-vit-base-patch32"
    print(f"  Loading CLIP: {model_id}")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.to(device).eval()
    return processor, model


# ─── Segmentation ───────────────────────────────────────────────────────────

@torch.inference_mode()
def run_segmentation(image: Image.Image, processor, model, device: str) -> np.ndarray:
    """Returns (H, W) array with label ID per pixel."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    up = torch.nn.functional.interpolate(
        logits, size=(image.size[1], image.size[0]),
        mode="bilinear", align_corners=False,
    )
    return up.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)


# ─── CLIP Classification ────────────────────────────────────────────────────

@torch.inference_mode()
def classify_with_clip(
    image: Image.Image,
    candidate_labels: list,
    clip_processor,
    clip_model,
    device: str,
) -> list:
    """
    Zero-shot classify an image against candidate text labels.
    Returns list of (label, confidence) sorted by confidence descending.
    """
    text_inputs = [f"a photo of {label}" for label in candidate_labels]
    inputs = clip_processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = clip_model(**inputs)
    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=0).cpu().numpy()

    results = list(zip(candidate_labels, probs.tolist()))
    results.sort(key=lambda x: -x[1])
    return results


# ─── Color Detection ────────────────────────────────────────────────────────

def detect_dominant_color(image: Image.Image, mask: np.ndarray, top_k: int = 2) -> list:
    """
    Find the dominant color(s) of the clothing pixels.
    Returns list of (color_name, hex_code, percentage).
    """
    img_array = np.array(image)
    # Only consider pixels where mask is white
    clothing_pixels = img_array[mask > 0]

    if len(clothing_pixels) == 0:
        return [("unknown", "#000000", 100.0)]

    # Use cv2 kmeans to find dominant colors
    pixels = clothing_pixels.astype(np.float32)
    k = min(top_k + 1, len(pixels))  # +1 because one might be near-black edges
    if k < 1:
        k = 1

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels_km, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    # Count pixels per cluster
    label_counts = Counter(labels_km.flatten())
    total = sum(label_counts.values())

    results = []
    for cluster_id, count in label_counts.most_common(top_k):
        center = centers[cluster_id].astype(int)
        pct = (count / total) * 100

        # Find closest named color
        color_name = _closest_color_name(center)
        hex_code = "#{:02x}{:02x}{:02x}".format(center[0], center[1], center[2])

        results.append((color_name, hex_code, round(pct, 1)))

    return results


def _closest_color_name(rgb: np.ndarray) -> str:
    """Find the closest named color to an RGB value."""
    min_dist = float("inf")
    closest = "unknown"
    for name, ref_rgb in COLOR_MAP.items():
        dist = np.sqrt(np.sum((rgb - np.array(ref_rgb)) ** 2))
        if dist < min_dist:
            min_dist = dist
            closest = name
    return closest


# ─── Extraction ──────────────────────────────────────────────────────────────

def extract_item(image: Image.Image, parse_map: np.ndarray, label_ids: list) -> dict:
    """Extract a clothing item and compute its bounding box."""
    mask = np.zeros_like(parse_map, dtype=np.uint8)
    for lid in label_ids:
        mask[parse_map == lid] = 255

    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None

    # Bounding box
    y_min, y_max = int(coords[0].min()), int(coords[0].max())
    x_min, x_max = int(coords[1].min()), int(coords[1].max())

    # Full RGBA extraction
    img_array = np.array(image)
    # Smooth edges slightly
    mask_smooth = cv2.GaussianBlur(mask, (5, 5), 0)
    alpha = np.where(mask > 0, np.maximum(mask, mask_smooth), mask_smooth)

    rgba = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = img_array
    rgba[:, :, 3] = alpha
    full_img = Image.fromarray(rgba, "RGBA")

    # Cropped version
    pad = 20
    h, w = mask.shape
    crop_box = (
        max(0, x_min - pad),
        max(0, y_min - pad),
        min(w, x_max + pad),
        min(h, y_max + pad),
    )
    cropped_img = full_img.crop(crop_box)

    # Pixel count
    pixel_count = int(np.sum(mask > 0))
    pixel_pct = round((pixel_count / parse_map.size) * 100, 1)

    return {
        "mask": mask,
        "full_image": full_img,
        "cropped_image": cropped_img,
        "bbox": [x_min, y_min, x_max, y_max],
        "pixel_count": pixel_count,
        "pixel_percentage": pixel_pct,
    }


# ─── Tagged Image Generation ────────────────────────────────────────────────

def draw_tag_on_image(image: Image.Image, tag_text: str, position: str = "top") -> Image.Image:
    """Draw a tag label on an extracted clothing image."""
    img = image.copy().convert("RGBA")
    w, h = img.size

    # Create overlay for the tag
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Calculate text size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), tag_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Tag background
    pad = 8
    tag_w = text_w + pad * 2
    tag_h = text_h + pad * 2

    if position == "top":
        tx = (w - tag_w) // 2
        ty = 5
    else:
        tx = (w - tag_w) // 2
        ty = h - tag_h - 5

    # Draw rounded-ish tag background
    draw.rectangle([tx, ty, tx + tag_w, ty + tag_h], fill=(0, 0, 0, 200))
    draw.text((tx + pad, ty + pad), tag_text, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(img, overlay)


def draw_tags_on_original(
    image: Image.Image,
    parse_map: np.ndarray,
    items: list,
) -> Image.Image:
    """Draw bounding boxes and tags on the original image."""
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        color = tuple(COLORS[item["label_ids"][0]]) + (200,)

        # Bounding box
        for offset in range(2):  # thicker line
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        # Tag background
        tag_line1 = item["tag_text"]
        tag_line2 = item.get("detail_text", "")

        bbox1 = draw.textbbox((0, 0), tag_line1, font=font)
        text_w = bbox1[2] - bbox1[0]
        text_h = bbox1[3] - bbox1[1]

        pad = 5
        tag_bg_h = text_h + pad * 2
        if tag_line2:
            bbox2 = draw.textbbox((0, 0), tag_line2, font=font_small)
            text_w = max(text_w, bbox2[2] - bbox2[0])
            tag_bg_h += bbox2[3] - bbox2[1] + 3

        tag_x = x1
        tag_y = max(0, y1 - tag_bg_h - 2)

        draw.rectangle(
            [tag_x, tag_y, tag_x + text_w + pad * 2, tag_y + tag_bg_h],
            fill=(0, 0, 0, 200),
        )
        draw.text((tag_x + pad, tag_y + pad), tag_line1, fill=(255, 255, 255, 255), font=font)
        if tag_line2:
            draw.text(
                (tag_x + pad, tag_y + pad + text_h + 3),
                tag_line2, fill=(180, 180, 180, 255), font=font_small,
            )

    return Image.alpha_composite(img, overlay).convert("RGB")


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract and tag clothing items from a person photo",
    )
    parser.add_argument("--image", required=True, help="Path to person image")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--device", default=None, help="Force cuda/cpu")
    parser.add_argument("--no-clip", action="store_true", help="Skip CLIP detailed classification")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print(" Clothing Extraction + Tagging")
    print("=" * 55)
    print(f"  Image:  {args.image}")
    print(f"  Device: {device}")
    print()

    # ── Load models ──
    t0 = time.time()
    print("Loading models...")
    seg_processor, seg_model = load_segmentation_model(device)

    clip_processor, clip_model = None, None
    if not args.no_clip:
        clip_processor, clip_model = load_clip_model(device)

    print(f"Models loaded in {time.time() - t0:.1f}s\n")

    # ── Load image ──
    image = Image.open(args.image).convert("RGB")
    print(f"Image: {image.size[0]} x {image.size[1]}")

    # ── Segmentation ──
    print("Running segmentation...", end=" ", flush=True)
    t1 = time.time()
    parse_map = run_segmentation(image, seg_processor, seg_model, device)
    print(f"done ({time.time() - t1:.2f}s)")

    # ── Find all clothing items ──
    detected_ids = np.unique(parse_map)
    clothing_found = {lid: CLOTHING_LABEL_IDS[lid] for lid in detected_ids if lid in CLOTHING_LABEL_IDS}

    # Merge left-shoe and right-shoe into "Shoes"
    has_left_shoe = 9 in clothing_found
    has_right_shoe = 10 in clothing_found
    if has_left_shoe or has_right_shoe:
        shoe_ids = []
        if has_left_shoe:
            shoe_ids.append(9)
            del clothing_found[9]
        if has_right_shoe:
            shoe_ids.append(10)
            del clothing_found[10]
        clothing_found[tuple(shoe_ids)] = "Shoes"

    if not clothing_found:
        print("\nNo clothing items detected in this image!")
        sys.exit(0)

    print(f"\nFound {len(clothing_found)} clothing item(s):")
    for lid, name in clothing_found.items():
        print(f"  - {name}")

    # ── Process each item ──
    print("\nProcessing each item...")
    all_items = []

    for lid, category_name in clothing_found.items():
        label_ids = list(lid) if isinstance(lid, tuple) else [lid]

        print(f"\n  [{category_name}]")

        # Extract
        extraction = extract_item(image, parse_map, label_ids)
        if extraction is None:
            print(f"    Skipped (no pixels found)")
            continue

        print(f"    Area: {extraction['pixel_percentage']}% of image")
        print(f"    Bbox: {extraction['bbox']}")

        # Detect color
        colors = detect_dominant_color(image, extraction["mask"], top_k=2)
        primary_color = colors[0][0]
        color_hex = colors[0][1]
        print(f"    Color: {primary_color} ({color_hex})")

        # CLIP detailed classification
        detail_type = None
        detail_confidence = None
        if clip_processor is not None:
            # Get candidate labels for this category
            if category_name == "Shoes":
                candidates = SHOE_DETAILS
            else:
                candidates = DETAIL_LABELS.get(category_name, [category_name.lower()])

            if candidates:
                clip_results = classify_with_clip(
                    extraction["cropped_image"].convert("RGB"),
                    candidates, clip_processor, clip_model, device,
                )
                detail_type = clip_results[0][0]
                detail_confidence = round(clip_results[0][1], 3)
                print(f"    Type: {detail_type} (confidence: {detail_confidence:.1%})")

                # Show top 3
                for label, conf in clip_results[:3]:
                    bar = "█" * int(conf * 20)
                    print(f"      {label:20s} {conf:5.1%} {bar}")

        # Build tag text
        if detail_type:
            tag_text = f"{primary_color} {detail_type}"
        else:
            tag_text = f"{primary_color} {category_name.lower()}"

        item_data = {
            "category": category_name,
            "detailed_type": detail_type,
            "tag": tag_text,
            "color": {
                "name": primary_color,
                "hex": color_hex,
                "all_colors": [(c[0], c[1], c[2]) for c in colors],
            },
            "confidence": detail_confidence,
            "bbox": extraction["bbox"],
            "pixel_percentage": extraction["pixel_percentage"],
            # For image generation
            "label_ids": label_ids,
            "tag_text": tag_text,
            "detail_text": f"{category_name} | {extraction['pixel_percentage']}%",
            "extraction": extraction,
        }
        all_items.append(item_data)

    # ── Generate outputs ──
    print("\n\nGenerating output images...")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Tagged overview on original image
    tagged_overview = draw_tags_on_original(image, parse_map, all_items)
    tagged_overview.save(os.path.join(args.output_dir, "1_all_items_tagged.png"))
    print(f"  1_all_items_tagged.png")

    # 2. Individual extracted items with tags
    for i, item in enumerate(all_items):
        ext = item["extraction"]
        tagged_img = draw_tag_on_image(ext["cropped_image"], item["tag_text"])
        filename = f"2_extracted_{item['category']}.png"
        tagged_img.save(os.path.join(args.output_dir, filename))
        print(f"  {filename}")

    # 3. Save tags.json
    json_items = []
    for item in all_items:
        json_item = {
            "category": item["category"],
            "detailed_type": item["detailed_type"],
            "tag": item["tag"],
            "color": item["color"],
            "confidence": item["confidence"],
            "bbox": item["bbox"],
            "pixel_percentage": item["pixel_percentage"],
        }
        json_items.append(json_item)

    tags_json = {
        "source_image": os.path.basename(args.image),
        "image_size": {"width": image.size[0], "height": image.size[1]},
        "total_items": len(json_items),
        "items": json_items,
    }

    json_path = os.path.join(args.output_dir, "tags.json")
    with open(json_path, "w") as f:
        json.dump(tags_json, f, indent=2)
    print(f"  tags.json")

    # ── Final summary ──
    total_time = time.time() - t0
    print()
    print("=" * 55)
    print(" RESULTS")
    print("=" * 55)
    print()
    for item in all_items:
        tag = item["tag"]
        cat = item["category"]
        conf = item["confidence"]
        conf_str = f" ({conf:.0%})" if conf else ""
        print(f"  {cat:16s} →  \"{tag}\"{conf_str}")
    print()
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Saved to: {os.path.abspath(args.output_dir)}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
