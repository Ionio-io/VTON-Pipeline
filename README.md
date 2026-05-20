# VTON Pipeline

A complete **Virtual Try-On (VTON)** pipeline for Indian fashion e-commerce. Given a product catalogue, it:

1. Scrapes product data and images from a fashion store
2. Classifies each garment using GPT-4o Vision
3. Generates diverse base model images (body type × skin tone matrix)
4. Composites garments onto models using GPT Image 2 (fal.ai)
5. Swaps the model's face with a target identity using InsightFace

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VTON PIPELINE                               │
│                                                                     │
│  Step 1             Step 2              Step 3                      │
│  scrape_            classify_           generate_                   │
│  westside.py  ───▶  products.py  ───▶  models.py                   │
│  (scrape 100        (GPT-4o Vision      (GPT Image 2               │
│   products +         classifies          generates 27               │
│   images)            each garment)       base model PNGs)           │
│                                               │                     │
│                                               ▼                     │
│  Step 4                                  Step 5                     │
│  tryon_samples.py  ──────────────────▶  swapper_*/                 │
│  (GPT Image 2 edit                      (InsightFace face           │
│   composites garment                     swap for identity          │
│   onto base model)                       personalisation)           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/Ionio-io/VTON-Pipeline.git
cd VTON-Pipeline

# Create and activate a virtual environment (Python 3.11 recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Or export directly:

```bash
export FAL_KEY=your_fal_key_here
export OPENROUTER_API_KEY=your_openrouter_key_here
```

### 3. Run the pipeline

```bash
# Step 1 — Scrape product data
python scrape_westside.py

# Step 2 — Classify products (requires local images from Step 1)
python classify_products.py

# Step 3 — Generate base model images (skip if using pre-generated images/)
python generate_models.py --resume

# Step 4 — Generate try-on images
python tryon_samples.py

# Step 5 — Face swap (see swapper_model_test/ or swapper_handoff/)
```

---

## Repository Structure

```
VTON-Pipeline/
├── README.md
├── .env.example                  ← copy to .env, fill in API keys
├── requirements.txt
├── prompts.json                  ← 27 body-type × skin-tone model definitions
│
├── scrape_westside.py            ← Step 1: scrape products from westside.com
├── classify_products.py          ← Step 2: GPT-4o Vision product classifier
├── fix_missing_images.py         ← Step 2b: backfill missing product images
├── generate_models.py            ← Step 3: batch-generate base model images
├── tryon_samples.py              ← Step 4: VTON composite (GPT Image 2 edit)
├── vton_zimage.py                ← Step 4 alt: Z-Image Turbo VTON (local, fast)
├── print_fidelity_test.py        ← quality test: micro-print & geometric prints
├── sophisticated_print_test.py   ← quality test: scenic & ethnic prints (medium)
│
├── generated_images/             ← 27 pre-generated base model PNGs (Step 3 output)
│   ├── M-REC_S3.png              ← Male, Rectangle, Medium-Olive
│   ├── M-REC_S4.png
│   └── ...                       ← see prompts.json for full list
│
├── westside_dataset/
│   ├── products_men.json         ← 50 men's clothing items (metadata + image URLs)
│   └── products_women.json       ← 50 women's clothing items
│   └── images/                   ← scraped product images (run scrape_westside.py)
│
├── zimage_space/                 ← Z-Image Turbo VTON as a Hugging Face Space
│   ├── app.py                    ← Gradio app + /predict API endpoint
│   └── requirements.txt
│
├── swapper_model_test/           ← Step 5: face swap model testing
│   ├── README.md                 ← detailed setup + usage guide
│   ├── swapper_local_insightface.py   ← local InsightFace runner
│   ├── swapper_replicate_test.py      ← Replicate API runner
│   ├── call_hf_space.py               ← call a deployed HF Space
│   ├── requirements-local.txt
│   ├── requirements-replicate.txt
│   ├── models/                   ← place inswapper_128.onnx here
│   └── huggingface_space/        ← deploy this folder as a Gradio Space
│       ├── app.py
│       ├── requirements.txt
│       └── packages.txt
│
└── swapper_handoff/              ← ready-to-run end-to-end face-swap scripts
    ├── README.md
    ├── requirements.txt
    ├── run_hf_swap.py             ← call HF Space with custom source/target
    └── run_westside_sample.py     ← full pipeline: crop face → call HF Space
```

---

## Step-by-Step Guide

### Step 1 — Scrape Product Data

```bash
python scrape_westside.py
# Options:
#   --per-gender 50   (default: 50 products per gender)
#   --resume          (skip already-downloaded images)
#   --no-images       (metadata only, very fast)
```

Outputs:
- `westside_dataset/products_men.json`
- `westside_dataset/products_women.json`
- `westside_dataset/images/{men|women}/{product-handle}/1.jpg, 2.jpg, ...`

---

### Step 2 — Classify Products

Uses GPT-4o Vision (via OpenRouter) to classify each product:

```bash
export OPENROUTER_API_KEY=your_key
python classify_products.py
# Options:
#   --gender male|female   (only classify one gender)
#   --delay 1.5            (seconds between API calls, default: 1.5)
```

Each product gets: `can_use`, `gender`, `image_type`, `category`, `best_image_idx`, `reasoning`

Outputs:
- `westside_dataset/classified_men.json`
- `westside_dataset/classified_women.json`

---

### Step 3 — Generate Base Model Images

Generates 27 diverse Indian model images (5 male + 4 female body types × 3 skin tones):

```bash
export FAL_KEY=your_key
python generate_models.py --resume
# Options:
#   --gender male|female   (only generate one gender)
#   --delay 2.0            (seconds between requests)
#   --resume               (skip already-generated images)
```

Output: `generated_images/{model_id}.png` for each of the 27 model IDs in `prompts.json`

> **Pre-generated images are included** in this repository under `generated_images/` — skip this step if you want to use them directly.

---

### Step 4A — Generate Try-On Images (GPT Image 2)

Composites garments onto base models using GPT Image 2 edit endpoint:

```bash
export FAL_KEY=your_key
python tryon_samples.py
```

Edit the `SAMPLES` list at the top of `tryon_samples.py` to choose which products and models to use.

Output: `tryon_output/sample{n}_{model_id}_{product_handle}.png`

**Quality settings** (`QUALITY` variable in the script):
| Setting | Speed | Detail |
|---------|-------|--------|
| `low`   | ~30s  | Good for prototyping |
| `medium`| ~2-4m | Better print detail |
| `high`  | ~5-8m | Best quality |

---

### Step 4B — Z-Image Turbo Try-On (fast, local, text-guided)

An alternative try-on approach using [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) — a 6B-parameter distilled model that edits the base model image via a garment text prompt in **~2-8 seconds** (GPU).

**How it works:** `ZImageImg2ImgPipeline` partially noises the base model PNG and denoises it guided by the garment description. `strength=0.55` keeps the face and body shape intact while changing only the clothing.

#### Local runner

```bash
pip install diffusers>=0.38.0 transformers accelerate torch pillow

# Single try-on: model M-REC_S3 wearing men's product #0
python vton_zimage.py --model M-REC_S3 --product 0 --gender male

# Batch: first 5 men's products × 5 male base models
python vton_zimage.py --batch --gender male --max-products 5

# Tweak strength (lower = more of original preserved, higher = bigger edit)
python vton_zimage.py --model F-HG_S3 --product 2 --gender female --strength 0.50
```

Output: `tryon_output/zimage/{model_id}_{product_handle}.png`

**Speed on common hardware:**
| Hardware | Speed |
|----------|-------|
| A100 (40 GB) | ~1-2 s/image |
| A10G (24 GB) | ~2-3 s/image |
| T4 (16 GB)   | ~5-8 s/image |
| CPU          | ~5 min/image |

> Requires ~16 GB VRAM (`torch.bfloat16`). Use `--cpu` to run without a GPU (very slow).

#### Hugging Face Space deployment

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Gradio**
   - Hardware: **T4-small** (16 GB, ~$0.40/hr) or **A10G** for faster inference
2. Upload `zimage_space/app.py` and `zimage_space/requirements.txt` to the Space
3. The Space exposes a `/predict` API endpoint

**Programmatic API call:**
```python
from gradio_client import Client
client = Client("YOUR_USERNAME/YOUR_SPACE_NAME")
result = client.predict(
    model_image,                    # PIL Image or file path
    "Navy Blue Slim-Fit Shirt",     # garment title
    "Cotton, button-down collar",   # garment detail (optional)
    "man",                          # gender
    "Rectangle",                    # body type
    "Medium / Olive",               # skin tone
    0.55,                           # strength
    42,                             # seed
    api_name="/predict",
)
```

---

### Step 5 — Face Swap

Replaces the generated model's face with a target identity (personalised try-on).

Three options — see **[`swapper_model_test/README.md`](swapper_model_test/README.md)** for full setup:

#### Option A: Replicate API (easiest)

```bash
export REPLICATE_API_TOKEN=your_token
python swapper_model_test/swapper_replicate_test.py
# Or with explicit paths:
python swapper_model_test/swapper_replicate_test.py \
  --source generated_images/M-REC_S3.png \
  --target tryon_output/sample1_M-REC_S3_*.png
```

#### Option B: Hugging Face Space

1. Deploy `swapper_model_test/huggingface_space/` as a Gradio Space
2. Call it:

```bash
pip install gradio_client
python swapper_model_test/call_hf_space.py \
  --space YOUR_USERNAME/YOUR_SPACE_NAME \
  --source generated_images/M-REC_S3.png \
  --target tryon_output/sample1_M-REC_S3_*.png
```

#### Option C: Local InsightFace (Python 3.10/3.11)

```bash
py -3.10 -m venv .venv-swapper
.venv-swapper/Scripts/activate
pip install -r swapper_model_test/requirements-local.txt

# Place inswapper_128.onnx in swapper_model_test/models/
python swapper_model_test/swapper_local_insightface.py \
  --model swapper_model_test/models/inswapper_128.onnx
```

---

## Model Matrix (`prompts.json`)

27 base model images covering Indian body types and skin tones:

| ID | Gender | Body Type | Skin Tone |
|----|--------|-----------|-----------|
| M-REC_S3 | Male | Rectangle | Medium / Olive |
| M-REC_S4 | Male | Rectangle | Light Brown |
| M-REC_S5 | Male | Rectangle | Dark Brown |
| M-INV_S3 | Male | Inverted Trapezoid | Medium / Olive |
| M-INV_S4 | Male | Inverted Trapezoid | Light Brown |
| M-INV_S5 | Male | Inverted Trapezoid | Dark Brown |
| M-TRI_S3 | Male | Triangle | Medium / Olive |
| M-TRI_S4 | Male | Triangle | Light Brown |
| M-TRI_S5 | Male | Triangle | Dark Brown |
| M-OVL_S3 | Male | Oval | Medium / Olive |
| M-OVL_S4 | Male | Oval | Light Brown |
| M-OVL_S5 | Male | Oval | Dark Brown |
| M-TRP_S3 | Male | Trapezoid | Medium / Olive |
| M-TRP_S4 | Male | Trapezoid | Light Brown |
| M-TRP_S5 | Male | Trapezoid | Dark Brown |
| F-HG_S3 | Female | Hourglass | Medium / Olive |
| F-HG_S4 | Female | Hourglass | Light Brown |
| F-HG_S5 | Female | Hourglass | Dark Brown |
| F-REC_S3 | Female | Rectangle | Medium / Olive |
| F-REC_S4 | Female | Rectangle | Light Brown |
| F-REC_S5 | Female | Rectangle | Dark Brown |
| F-SPO_S3 | Female | Sporty | Medium / Olive |
| F-SPO_S4 | Female | Sporty | Light Brown |
| F-SPO_S5 | Female | Sporty | Dark Brown |
| F-TRAP_S3 | Female | Trapezoid | Medium / Olive |
| F-TRAP_S4 | Female | Trapezoid | Light Brown |
| F-TRAP_S5 | Female | Trapezoid | Dark Brown |

---

## API Keys Required

| Script | Service | Where to get |
|--------|---------|--------------|
| `generate_models.py` | [fal.ai](https://fal.ai/dashboard/keys) | `FAL_KEY` |
| `tryon_samples.py` | [fal.ai](https://fal.ai/dashboard/keys) | `FAL_KEY` |
| `print_fidelity_test.py` | [fal.ai](https://fal.ai/dashboard/keys) | `FAL_KEY` |
| `sophisticated_print_test.py` | [fal.ai](https://fal.ai/dashboard/keys) | `FAL_KEY` |
| `classify_products.py` | [OpenRouter](https://openrouter.ai/settings/keys) | `OPENROUTER_API_KEY` |
| `swapper_replicate_test.py` | [Replicate](https://replicate.com/account/api-tokens) | `REPLICATE_API_TOKEN` |
| `run_hf_swap.py` / `call_hf_space.py` | Hugging Face (optional for private spaces) | `HF_TOKEN` |

---

## Hugging Face Space Deployment

The `swapper_model_test/huggingface_space/` folder is a ready-to-deploy Gradio app:

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) (SDK: Gradio, Hardware: CPU Basic)
2. Upload `app.py`, `requirements.txt`, `packages.txt` to the Space
3. The Space exposes the `/swap` API endpoint

Full step-by-step instructions in [`swapper_model_test/README.md`](swapper_model_test/README.md).

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Base model generation | GPT Image 2 via [fal.ai](https://fal.ai) |
| VTON composition (Step 4A) | GPT Image 2 Edit via [fal.ai](https://fal.ai) |
| VTON composition (Step 4B) | [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) via diffusers |
| Product classification | GPT-4o Vision via [OpenRouter](https://openrouter.ai) |
| Face swap (cloud) | InsightFace inswapper_128 via [Replicate](https://replicate.com) |
| Face swap (local) | InsightFace + ONNX Runtime |
| Face swap (API) | Gradio Space on Hugging Face |
| Data scraping | Westside Shopify JSON API |

---

## Notes

- The scraper uses the public Shopify `products.json` endpoint — no authentication needed
- `classify_products.py` sends product images as **local base64** (not remote URLs) for better accuracy and to avoid CDN auth issues
- The face swap step uses only the face region and should preserve garment details intact
- For production use, verify licensing for `inswapper_128.onnx` — see `swapper_model_test/README.md`
