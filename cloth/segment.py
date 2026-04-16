#!/usr/bin/env python3
"""
Clothing Segmentation + Extraction — Complete End-to-End
=========================================================

Does TWO things:
  1. SEGMENTATION — Labels every pixel (face, shirt, pants, hair, etc.)
  2. EXTRACTION   — Cuts out the clothing as separate images with transparent background

Setup (one time):
    pip install transformers torch torchvision Pillow opencv-python-headless numpy

Usage:
    python segment.py --image person.jpg
    python segment.py --image person.jpg --type upper
    python segment.py --image person.jpg --type lower
    python segment.py --image person.jpg --type all
"""

import os
import sys
import argparse
import time

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw


# ─── ATR Label Schema (18 classes) ──────────────────────────────────────────

LABELS = [
    "Background",     # 0
    "Hat",            # 1
    "Hair",           # 2
    "Sunglasses",     # 3
    "Upper-clothes",  # 4
    "Skirt",          # 5
    "Pants",          # 6
    "Dress",          # 7
    "Belt",           # 8
    "Left-shoe",      # 9
    "Right-shoe",     # 10
    "Face",           # 11
    "Left-leg",       # 12
    "Right-leg",      # 13
    "Left-arm",       # 14
    "Right-arm",      # 15
    "Bag",            # 16
    "Scarf",          # 17
]

COLORS = [
    [0, 0, 0],        # 0  Background
    [255, 85, 0],     # 1  Hat
    [255, 170, 0],    # 2  Hair
    [255, 0, 170],    # 3  Sunglasses
    [0, 255, 0],      # 4  Upper-clothes
    [85, 0, 255],     # 5  Skirt
    [0, 0, 255],      # 6  Pants
    [0, 255, 170],    # 7  Dress
    [170, 170, 0],    # 8  Belt
    [255, 255, 0],    # 9  Left-shoe
    [255, 200, 0],    # 10 Right-shoe
    [255, 0, 0],      # 11 Face
    [0, 170, 255],    # 12 Left-leg
    [0, 85, 255],     # 13 Right-leg
    [255, 85, 170],   # 14 Left-arm
    [255, 170, 170],  # 15 Right-arm
    [170, 0, 255],    # 16 Bag
    [0, 255, 255],    # 17 Scarf
]

# Clothing label IDs grouped by type
CLOTHING_GROUPS = {
    "upper":   [4],              # Upper-clothes
    "lower":   [5, 6],           # Skirt, Pants
    "dress":   [7],              # Dress
    "all":     [4, 5, 6, 7, 8, 17],  # All clothing + belt + scarf
}

# For VTON agnostic mask: clothing + body parts underneath
AGNOSTIC_GROUPS = {
    "upper":   [4, 7, 14, 15],           # shirt + dress + arms
    "lower":   [5, 6, 7, 12, 13],        # pants/skirt + dress + legs
    "dress":   [4, 5, 6, 7, 14, 15],     # everything clothing + arms
    "all":     [4, 5, 6, 7, 8, 14, 15],  # all clothing + arms
}


# ─── Model ───────────────────────────────────────────────────────────────────

def load_model(device: str):
    """Load SegFormer-B2 fine-tuned on ATR clothing dataset."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    model_id = "mattmdjaga/segformer_b2_clothes"
    print(f"Loading model: {model_id}")
    print("(Downloads ~450 MB on first run, then cached)")

    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    model.to(device)
    model.eval()

    print(f"Loaded on: {device}")
    if device == "cuda":
        print(f"VRAM used: {torch.cuda.memory_allocated() / 1e6:.0f} MB")

    return processor, model


@torch.inference_mode()
def run_segmentation(image: Image.Image, processor, model, device: str) -> np.ndarray:
    """Returns parse_map: np.ndarray (H, W) with label ID per pixel."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logits = model(**inputs).logits

    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(image.size[1], image.size[0]),
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)


# ─── Output Generators ──────────────────────────────────────────────────────

def make_colored_map(parse_map: np.ndarray) -> Image.Image:
    """Segmentation: color every pixel by its label."""
    h, w = parse_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(COLORS):
        colored[parse_map == i] = color
    return Image.fromarray(colored)


def make_overlay(original: Image.Image, colored: Image.Image, alpha: float = 0.55) -> Image.Image:
    """Segmentation blended on top of original photo."""
    orig = np.array(original).astype(np.float32)
    col = np.array(colored).astype(np.float32)
    # Only blend where there's a non-background label
    has_label = (col.sum(axis=2) > 0).astype(np.float32)[..., np.newaxis]
    blended = orig * (1 - has_label * alpha) + col * (has_label * alpha)
    return Image.fromarray(blended.astype(np.uint8))


def make_mask(parse_map: np.ndarray, label_ids: list) -> np.ndarray:
    """Binary mask: 255 where pixel matches any of the label IDs, 0 elsewhere."""
    mask = np.zeros_like(parse_map, dtype=np.uint8)
    for lid in label_ids:
        mask[parse_map == lid] = 255
    return mask


def extract_clothing(
    original: Image.Image,
    parse_map: np.ndarray,
    label_ids: list,
    smooth_edges: bool = True,
) -> Image.Image:
    """
    EXTRACTION: Cut out clothing pixels from the original image.
    Returns RGBA image — clothing on transparent background.
    """
    orig_array = np.array(original)
    mask = make_mask(parse_map, label_ids)

    if smooth_edges:
        # Slight blur on mask edges for cleaner cutout
        mask_smooth = cv2.GaussianBlur(mask, (5, 5), 0)
        # But keep the core solid
        mask = np.where(mask > 0, np.maximum(mask, mask_smooth), mask_smooth)

    # Create RGBA image
    rgba = np.zeros((orig_array.shape[0], orig_array.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = orig_array       # RGB from original
    rgba[:, :, 3] = mask              # Alpha from mask

    return Image.fromarray(rgba, mode="RGBA")


def extract_clothing_cropped(
    original: Image.Image,
    parse_map: np.ndarray,
    label_ids: list,
    padding: int = 20,
) -> Image.Image:
    """
    EXTRACTION + CROP: Cut out clothing and crop to bounding box.
    No wasted transparent space — just the garment, tightly cropped.
    """
    full_rgba = extract_clothing(original, parse_map, label_ids)
    mask = make_mask(parse_map, label_ids)

    # Find bounding box of the clothing pixels
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return full_rgba  # No clothing found, return full image

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Add padding
    h, w = mask.shape
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)

    return full_rgba.crop((x_min, y_min, x_max, y_max))


def make_agnostic_mask(parse_map: np.ndarray, cloth_type: str) -> Image.Image:
    """VTON agnostic mask: region that would be replaced during try-on."""
    label_ids = AGNOSTIC_GROUPS.get(cloth_type, AGNOSTIC_GROUPS["upper"])
    mask = make_mask(parse_map, label_ids)

    # Dilate to catch edges
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Smooth
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    _, mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)

    return Image.fromarray(mask, mode="L")


def make_legend() -> Image.Image:
    """Color legend: which color = which label."""
    row_h = 25
    swatch_size = 18
    padding = 10
    text_w = 160
    total_w = swatch_size + text_w + padding * 3
    total_h = len(LABELS) * row_h + padding * 2

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        y = padding + i * row_h
        x1 = padding
        # Color swatch
        draw.rectangle(
            [x1, y + 3, x1 + swatch_size, y + 3 + swatch_size],
            fill=tuple(color),
            outline=(80, 80, 80),
        )
        # Label text
        text_color = (255, 255, 255) if i != 0 else (100, 100, 100)
        draw.text((x1 + swatch_size + padding, y + 4), f"{i:2d}  {label}", fill=text_color)

    return canvas


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clothing Segmentation + Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python segment.py --image person.jpg
    python segment.py --image person.jpg --type lower
    python segment.py --image person.jpg --type all --output-dir ./results
        """,
    )
    parser.add_argument("--image", required=True, help="Path to person image")
    parser.add_argument(
        "--type", default="upper",
        choices=["upper", "lower", "dress", "all"],
        help="Which clothing to extract (default: upper)",
    )
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--device", default=None, help="Force cuda or cpu")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print(" Clothing Segmentation + Extraction")
    print("=" * 55)
    print(f"  Image:   {args.image}")
    print(f"  Type:    {args.type}")
    print(f"  Device:  {device}")
    print(f"  Output:  {args.output_dir}/")
    print()

    # ── Load model ──
    t0 = time.time()
    processor, model = load_model(device)
    print(f"Model load: {time.time() - t0:.1f}s\n")

    # ── Load image ──
    image = Image.open(args.image).convert("RGB")
    w, h = image.size
    print(f"Image: {w} x {h}")

    # ── Run segmentation ──
    print("Running segmentation...", end=" ", flush=True)
    t1 = time.time()
    parse_map = run_segmentation(image, processor, model, device)
    print(f"done ({time.time() - t1:.2f}s)")

    # ── Print detected regions ──
    print("\nDetected regions:")
    print("-" * 50)
    for lid in np.unique(parse_map):
        count = np.sum(parse_map == lid)
        pct = (count / parse_map.size) * 100
        name = LABELS[lid] if lid < len(LABELS) else f"Unknown({lid})"
        bar = "█" * int(pct / 2)
        print(f"  {lid:2d}  {name:16s}  {pct:5.1f}%  {bar}")
    print("-" * 50)

    # ── Check if requested clothing exists ──
    target_ids = CLOTHING_GROUPS[args.type]
    found_ids = [lid for lid in target_ids if lid in np.unique(parse_map)]
    found_names = [LABELS[lid] for lid in found_ids]

    if not found_ids:
        print(f"\nWarning: No '{args.type}' clothing detected in this image!")
        print(f"  Looked for labels: {[LABELS[lid] for lid in target_ids]}")
        print(f"  Found clothing: {[LABELS[lid] for lid in np.unique(parse_map) if lid in CLOTHING_GROUPS['all']]}")
        print("  Outputs will be generated but extraction will be empty.")
    else:
        print(f"\nFound clothing: {found_names}")

    # ── Generate all outputs ──
    print("\nGenerating outputs...")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- SEGMENTATION outputs ---

    # 1. Colored segmentation map
    colored = make_colored_map(parse_map)
    colored.save(os.path.join(args.output_dir, "1_segmentation_map.png"))

    # 2. Overlay on original
    overlay = make_overlay(image, colored)
    overlay.save(os.path.join(args.output_dir, "2_overlay.png"))

    # 3. Clothing mask (binary)
    cloth_mask = make_mask(parse_map, target_ids)
    Image.fromarray(cloth_mask, mode="L").save(
        os.path.join(args.output_dir, f"3_clothing_mask_{args.type}.png")
    )

    # 4. VTON agnostic mask
    agnostic = make_agnostic_mask(parse_map, args.type)
    agnostic.save(os.path.join(args.output_dir, f"4_agnostic_mask_{args.type}.png"))

    # --- EXTRACTION outputs ---

    # 5. Extracted clothing (full image size, transparent background)
    extracted_full = extract_clothing(image, parse_map, target_ids)
    extracted_full.save(os.path.join(args.output_dir, f"5_extracted_{args.type}_full.png"))

    # 6. Extracted clothing (cropped tight to the garment)
    extracted_cropped = extract_clothing_cropped(image, parse_map, target_ids)
    extracted_cropped.save(os.path.join(args.output_dir, f"6_extracted_{args.type}_cropped.png"))

    # 7. If type is "all", also extract each piece individually
    if args.type == "all":
        for group_name, group_ids in [("upper", [4]), ("lower", [5, 6]), ("dress", [7])]:
            group_found = [lid for lid in group_ids if lid in np.unique(parse_map)]
            if group_found:
                ext = extract_clothing_cropped(image, parse_map, group_ids)
                ext.save(os.path.join(args.output_dir, f"6_extracted_{group_name}_cropped.png"))

    # 8. Legend
    legend = make_legend()
    legend.save(os.path.join(args.output_dir, "7_legend.png"))

    # 9. Save raw parse map for programmatic use
    np.save(os.path.join(args.output_dir, "parse_map.npy"), parse_map)

    # ── Summary ──
    total_time = time.time() - t0
    print()
    print("=" * 55)
    print(" OUTPUTS")
    print("=" * 55)
    print()
    print(" SEGMENTATION (label every pixel):")
    print(f"   1_segmentation_map.png       — colored label map")
    print(f"   2_overlay.png                — labels blended on photo")
    print(f"   3_clothing_mask_{args.type}.png     — binary: white = clothing")
    print(f"   4_agnostic_mask_{args.type}.png     — binary: white = VTON replace zone")
    print()
    print(" EXTRACTION (cut out the clothes):")
    print(f"   5_extracted_{args.type}_full.png    — clothing on transparent bg (full size)")
    print(f"   6_extracted_{args.type}_cropped.png — clothing on transparent bg (cropped tight)")
    print()
    print(" REFERENCE:")
    print(f"   7_legend.png                 — color-to-label guide")
    print(f"   parse_map.npy                — raw numpy array for code")
    print()
    print(f" Total time: {total_time:.1f}s")
    print(f" Saved to: {os.path.abspath(args.output_dir)}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
