#!/usr/bin/env python3
"""
RunPod Serverless Handler — Segmentation + Extraction
======================================================

Input:
{
    "input": {
        "image": "<base64>",
        "cloth_type": "upper"   // "upper", "lower", "dress", "all"
    }
}

Output:
{
    "segmentation_map": "<base64 png>",
    "overlay": "<base64 png>",
    "clothing_mask": "<base64 png>",
    "agnostic_mask": "<base64 png>",
    "extracted_full": "<base64 png>",        // clothing on transparent bg
    "extracted_cropped": "<base64 png>",     // clothing cropped tight
    "detected_labels": {"Upper-clothes": 15.3, "Face": 4.1, ...},
    "processing_time": 0.45
}
"""

import time
import base64
import io
import traceback

import numpy as np
import cv2
import torch
from PIL import Image

import runpod


# ─── Labels ──────────────────────────────────────────────────────────────────

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

CLOTHING_GROUPS = {
    "upper": [4], "lower": [5, 6], "dress": [7],
    "all": [4, 5, 6, 7, 8, 17],
}

AGNOSTIC_GROUPS = {
    "upper": [4, 7, 14, 15], "lower": [5, 6, 7, 12, 13],
    "dress": [4, 5, 6, 7, 14, 15], "all": [4, 5, 6, 7, 8, 14, 15],
}


# ─── Model ───────────────────────────────────────────────────────────────────

MODEL = {}


def load_model():
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "mattmdjaga/segformer_b2_clothes"

    print(f"[INIT] Loading {model_id} on {device}...")
    t = time.time()

    MODEL["processor"] = SegformerImageProcessor.from_pretrained(model_id)
    MODEL["model"] = SegformerForSemanticSegmentation.from_pretrained(model_id)
    MODEL["model"].to(device)
    MODEL["model"].eval()
    MODEL["device"] = device

    print(f"[INIT] Ready in {time.time() - t:.1f}s")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def b64_decode(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


@torch.inference_mode()
def segment(image: Image.Image) -> np.ndarray:
    inputs = MODEL["processor"](images=image, return_tensors="pt")
    inputs = {k: v.to(MODEL["device"]) for k, v in inputs.items()}
    logits = MODEL["model"](**inputs).logits
    up = torch.nn.functional.interpolate(
        logits, size=(image.size[1], image.size[0]),
        mode="bilinear", align_corners=False,
    )
    return up.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)


def colorize(pm: np.ndarray) -> Image.Image:
    h, w = pm.shape
    c = np.zeros((h, w, 3), dtype=np.uint8)
    for i, col in enumerate(COLORS):
        c[pm == i] = col
    return Image.fromarray(c)


def make_overlay(orig: Image.Image, colored: Image.Image, alpha=0.55) -> Image.Image:
    o = np.array(orig).astype(np.float32)
    c = np.array(colored).astype(np.float32)
    m = (c.sum(axis=2) > 0).astype(np.float32)[..., np.newaxis]
    return Image.fromarray((o * (1 - m * alpha) + c * (m * alpha)).astype(np.uint8))


def make_mask(pm: np.ndarray, ids: list) -> np.ndarray:
    mask = np.zeros_like(pm, dtype=np.uint8)
    for i in ids:
        mask[pm == i] = 255
    return mask


def extract_full(orig: Image.Image, pm: np.ndarray, ids: list) -> Image.Image:
    arr = np.array(orig)
    mask = make_mask(pm, ids)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = np.where(make_mask(pm, ids) > 0, np.maximum(make_mask(pm, ids), mask), mask)
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = arr
    rgba[:, :, 3] = mask
    return Image.fromarray(rgba, "RGBA")


def extract_cropped(orig: Image.Image, pm: np.ndarray, ids: list, pad=20) -> Image.Image:
    full = extract_full(orig, pm, ids)
    mask = make_mask(pm, ids)
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return full
    h, w = mask.shape
    y1 = max(0, coords[0].min() - pad)
    y2 = min(h, coords[0].max() + pad)
    x1 = max(0, coords[1].min() - pad)
    x2 = min(w, coords[1].max() + pad)
    return full.crop((x1, y1, x2, y2))


def make_agnostic(pm: np.ndarray, ct: str) -> Image.Image:
    ids = AGNOSTIC_GROUPS.get(ct, AGNOSTIC_GROUPS["upper"])
    mask = make_mask(pm, ids)
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    _, mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
    return Image.fromarray(mask, "L")


def get_labels(pm: np.ndarray) -> dict:
    total = pm.size
    result = {}
    for lid in np.unique(pm):
        name = LABELS[lid] if lid < len(LABELS) else f"Unknown({lid})"
        pct = round((np.sum(pm == lid) / total) * 100, 1)
        if pct > 0.1:
            result[name] = pct
    return result


# ─── Handler ─────────────────────────────────────────────────────────────────

def handler(job):
    try:
        inp = job["input"]
        cloth_type = inp.get("cloth_type", "upper")
        image = b64_decode(inp["image"])

        t = time.time()
        pm = segment(image)
        seg_time = time.time() - t

        target_ids = CLOTHING_GROUPS.get(cloth_type, CLOTHING_GROUPS["upper"])

        return {
            "segmentation_map": pil_to_b64(colorize(pm)),
            "overlay": pil_to_b64(make_overlay(image, colorize(pm))),
            "clothing_mask": pil_to_b64(Image.fromarray(make_mask(pm, target_ids), "L")),
            "agnostic_mask": pil_to_b64(make_agnostic(pm, cloth_type)),
            "extracted_full": pil_to_b64(extract_full(image, pm, target_ids)),
            "extracted_cropped": pil_to_b64(extract_cropped(image, pm, target_ids)),
            "detected_labels": get_labels(pm),
            "cloth_type": cloth_type,
            "processing_time": round(seg_time, 3),
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


print("[WORKER] Starting...")
load_model()
print("[WORKER] Ready.")
runpod.serverless.start({"handler": handler})
