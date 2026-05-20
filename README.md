# VTON Pipeline

An end-to-end **Virtual Try-On pipeline** built for Indian fashion e-commerce.
Scrapes a product catalogue, classifies garments, generates diverse model images,
composites garments onto models, and optionally swaps the model's face with a
target identity for personalised try-on.

---

## How it works

```
Scrape          Classify         Generate Models       Try-On
products   -->  with GPT-4o  --> (body type x      --> composite
from            Vision           skin tone matrix)     garment onto
westside.com                     27 base PNGs          base model
                                                           |
                                                    Face Swap (optional)
                                                    replace model face
                                                    with target identity
```

---

## Repository Structure

```
VTON-Pipeline/
|
|-- pipeline/                    Core pipeline scripts (run in order)
|   |-- scrape.py                Step 1: scrape products from westside.com
|   |-- classify.py              Step 2: classify garments with GPT-4o Vision
|   |-- generate_models.py       Step 3: generate 27 base model images
|   |-- tryon_gpt.py             Step 4A: try-on using GPT Image 2 edit
|   `-- tryon_zimage.py          Step 4B: try-on using Z-Image Turbo (faster)
|
|-- face_swap/                   Step 5: identity personalisation
|   |-- swap_replicate.py        Run face swap via Replicate API (easiest)
|   |-- swap_local.py            Run face swap locally with InsightFace
|   |-- swap_hf_space.py         Call a deployed HF Space for face swap
|   |-- run_hf_swap.py           Swap faces on a batch of try-on images
|   |-- run_pipeline.py          Full end-to-end: try-on + face swap
|   |-- models/                  Place inswapper_128.onnx here
|   `-- requirements.txt
|
|-- spaces/                      Hugging Face Space deployments
|   |-- tryon/                   Z-Image Turbo try-on Space
|   |   |-- app.py
|   |   `-- requirements.txt
|   `-- face_swap/               InsightFace face-swap Space
|       |-- app.py
|       |-- requirements.txt
|       `-- packages.txt
|
|-- tests/                       Print quality and fidelity tests
|   |-- print_fidelity.py        GPT Image 2 print accuracy at quality=low
|   `-- sophisticated_print.py   Scenic and ethnic prints at quality=medium
|
|-- utils/                       Optional / advanced tools
|   |-- fix_missing_images.py    Backfill missing product images after scraping
|   `-- tryon_zimage_local.py    Z-Image Turbo local GPU runner (needs 16 GB VRAM)
|
|-- generated_images/            27 pre-generated base model PNGs (Step 3 output)
|-- westside_dataset/            Scraped product JSON and images
|-- prompts.json                 27 model definitions (body type x skin tone)
|-- requirements.txt             Core dependencies
`-- .env.example                 API key template
```

> All scripts are run from the **repository root**, e.g. `python pipeline/scrape.py`.
> Paths inside every script resolve relative to the current working directory.

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/Ionio-io/VTON-Pipeline.git
cd VTON-Pipeline

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

Or export directly:

```bash
export FAL_KEY=your_fal_key
export OPENROUTER_API_KEY=your_openrouter_key
```

### 3. Run the pipeline

```bash
python pipeline/scrape.py
python pipeline/classify.py
python pipeline/generate_models.py --resume
python pipeline/tryon_gpt.py
```

---

## Pipeline Steps

### Step 1 — Scrape Products

Scrapes 50 men's and 50 women's products from westside.com using the
public Shopify JSON API. Downloads product images and saves metadata.

```bash
python pipeline/scrape.py
python pipeline/scrape.py --per-gender 100   # more products
python pipeline/scrape.py --no-images        # metadata only, very fast
```

**Output:**
- `westside_dataset/products_men.json`
- `westside_dataset/products_women.json`
- `westside_dataset/images/{men|women}/{handle}/1.jpg, 2.jpg, ...`

---

### Step 2 — Classify Products

Sends each product's images (as base64) to GPT-4o Vision via OpenRouter.
Tags each item with category, usability flag, best image index, and reasoning.

```bash
export OPENROUTER_API_KEY=your_key
python pipeline/classify.py
python pipeline/classify.py --gender male    # one gender only
python pipeline/classify.py --delay 2        # seconds between API calls
```

**Output:**
- `westside_dataset/classified_men.json`
- `westside_dataset/classified_women.json`

If any product images are missing after scraping, run:

```bash
python utils/fix_missing_images.py
```

---

### Step 3 — Generate Base Models

Generates 27 diverse Indian model images covering 5 male and 4 female body
types across 3 skin tones, using GPT Image 2 via fal.ai.

```bash
export FAL_KEY=your_key
python pipeline/generate_models.py --resume
python pipeline/generate_models.py --gender male   # one gender only
```

**Output:** `generated_images/{model_id}.png` for each of the 27 IDs in `prompts.json`

> **Pre-generated images are included** in `generated_images/` — skip this step
> if you want to use them directly.

---

### Step 4A — Try-On with GPT Image 2

Uploads the base model PNG and garment reference photos to fal.ai, then
calls GPT Image 2 edit to composite the garment onto the model.
Best print fidelity; slower and more expensive than Z-Image Turbo.

```bash
export FAL_KEY=your_key
python pipeline/tryon_gpt.py
```

Edit the `SAMPLES` list at the top of `tryon_gpt.py` to choose products and models.

**Quality settings** (set `QUALITY` in the script):

| Setting  | Speed  | Notes                        |
|----------|--------|------------------------------|
| `low`    | ~30 s  | Good for rapid iteration     |
| `medium` | ~2-4 m | Better print reproduction    |
| `high`   | ~5-8 m | Best quality for delivery    |

**Output:** `tryon_output/sample{n}_{model_id}_{handle}.png`

---

### Step 4B — Try-On with Z-Image Turbo

Faster and cheaper alternative using Z-Image Turbo image-to-image via fal.ai.
Garment is described via text prompt — no reference image upload required.
Best for high-volume generation where speed matters more than print fidelity.

```bash
export FAL_KEY=your_key

# Single try-on
python pipeline/tryon_zimage.py --model M-REC_S3 --product 0 --gender male

# Batch: first 5 men's products across 3 models
python pipeline/tryon_zimage.py --batch --gender male --max-products 5

# Adjust edit strength (lower = model identity better preserved)
python pipeline/tryon_zimage.py --model F-HG_S4 --product 2 --gender female --strength 0.50
```

| Flag            | Default   | Description                                  |
|-----------------|-----------|----------------------------------------------|
| `--model`       | M-REC_S3  | Model ID from `prompts.json`                 |
| `--product`     | 0         | 0-based product index in dataset JSON        |
| `--gender`      | male      | `male`, `female`, or `both`                  |
| `--strength`    | 0.55      | Edit strength (lower = more original kept)   |
| `--batch`       | off       | Run multiple products x models               |
| `--max-products`| 5         | Max products per gender in batch mode        |

**Output:** `tryon_output/zimage_fal/{model_id}_{handle}.png`

**Cost:** ~$0.004 per 768x1024 image on fal.ai

> For local GPU inference (16 GB VRAM required), see `utils/tryon_zimage_local.py`.

---

### Step 5 — Face Swap

Replaces the model's face with a target identity for personalised try-on.
Three options depending on your setup:

#### Option A: Replicate API (recommended, no local GPU needed)

```bash
export REPLICATE_API_TOKEN=your_token
python face_swap/swap_replicate.py \
  --source generated_images/M-REC_S3.png \
  --target tryon_output/sample1_M-REC_S3_*.png
```

#### Option B: Hugging Face Space

Deploy `spaces/face_swap/` as a Gradio Space, then call it:

```bash
pip install gradio_client
python face_swap/swap_hf_space.py \
  --space YOUR_USERNAME/YOUR_SPACE \
  --source generated_images/M-REC_S3.png \
  --target tryon_output/sample1_M-REC_S3_*.png
```

#### Option C: Local InsightFace (Python 3.10 or 3.11)

```bash
pip install -r face_swap/requirements.txt
# Place inswapper_128.onnx in face_swap/models/
python face_swap/swap_local.py \
  --model face_swap/models/inswapper_128.onnx \
  --source generated_images/M-REC_S3.png \
  --target tryon_output/sample1_M-REC_S3_*.png
```

#### Full end-to-end pipeline

```bash
python face_swap/run_pipeline.py \
  --face your_photo.jpg \
  --product 0 \
  --gender male \
  --model M-REC_S3
```

---

## HF Space Deployments

### Try-On Space (`spaces/tryon/`)

Gradio app that runs Z-Image Turbo image-to-image inference.
Upload a base model image, describe the garment, get a try-on result.

**Deploy:**
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: Gradio
   - Hardware: T4-small (16 GB, ~$0.40/hr) or A10G
2. Upload `spaces/tryon/app.py` and `spaces/tryon/requirements.txt`

**API:**
```python
from gradio_client import Client

client = Client("YOUR_USERNAME/YOUR_SPACE")
result = client.predict(
    base_model_image,
    "Navy Blue Slim-Fit Shirt",
    "Cotton, button-down collar",
    "man",
    "Rectangle",
    "Medium / Olive",
    0.55,   # strength
    42,     # seed
    api_name="/predict",
)
```

### Face Swap Space (`spaces/face_swap/`)

Gradio app powered by InsightFace inswapper_128.
Upload source (face to take) and target (face to replace), get swapped result.

**Deploy:**
1. Create a new Space — SDK: Gradio, Hardware: CPU Basic
2. Upload `spaces/face_swap/app.py`, `requirements.txt`, `packages.txt`

---

## Model Matrix

27 base model images covering Indian body types and skin tones:

| ID         | Gender | Body Type          | Skin Tone      |
|------------|--------|--------------------|----------------|
| M-REC_S3   | Male   | Rectangle          | Medium / Olive |
| M-REC_S4   | Male   | Rectangle          | Light Brown    |
| M-REC_S5   | Male   | Rectangle          | Dark Brown     |
| M-INV_S3   | Male   | Inverted Trapezoid | Medium / Olive |
| M-INV_S4   | Male   | Inverted Trapezoid | Light Brown    |
| M-INV_S5   | Male   | Inverted Trapezoid | Dark Brown     |
| M-TRI_S3   | Male   | Triangle           | Medium / Olive |
| M-TRI_S4   | Male   | Triangle           | Light Brown    |
| M-TRI_S5   | Male   | Triangle           | Dark Brown     |
| M-OVL_S3   | Male   | Oval               | Medium / Olive |
| M-OVL_S4   | Male   | Oval               | Light Brown    |
| M-OVL_S5   | Male   | Oval               | Dark Brown     |
| M-TRP_S3   | Male   | Trapezoid          | Medium / Olive |
| M-TRP_S4   | Male   | Trapezoid          | Light Brown    |
| M-TRP_S5   | Male   | Trapezoid          | Dark Brown     |
| F-HG_S3    | Female | Hourglass          | Medium / Olive |
| F-HG_S4    | Female | Hourglass          | Light Brown    |
| F-HG_S5    | Female | Hourglass          | Dark Brown     |
| F-REC_S3   | Female | Rectangle          | Medium / Olive |
| F-REC_S4   | Female | Rectangle          | Light Brown    |
| F-REC_S5   | Female | Rectangle          | Dark Brown     |
| F-SPO_S3   | Female | Sporty             | Medium / Olive |
| F-SPO_S4   | Female | Sporty             | Light Brown    |
| F-SPO_S5   | Female | Sporty             | Dark Brown     |
| F-TRAP_S3  | Female | Trapezoid          | Medium / Olive |
| F-TRAP_S4  | Female | Trapezoid          | Light Brown    |
| F-TRAP_S5  | Female | Trapezoid          | Dark Brown     |

---

## API Keys

| Script                  | Service      | Environment Variable    | Get Key                                      |
|-------------------------|--------------|-------------------------|----------------------------------------------|
| `pipeline/generate_models.py` | fal.ai  | `FAL_KEY`          | [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) |
| `pipeline/tryon_gpt.py`       | fal.ai  | `FAL_KEY`          | [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) |
| `pipeline/tryon_zimage.py`    | fal.ai  | `FAL_KEY`          | [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) |
| `pipeline/classify.py`        | OpenRouter | `OPENROUTER_API_KEY` | [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) |
| `face_swap/swap_replicate.py` | Replicate | `REPLICATE_API_TOKEN` | [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens) |
| `face_swap/swap_hf_space.py`  | Hugging Face (private spaces only) | `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## Technology Stack

| Component               | Technology                                                  |
|-------------------------|-------------------------------------------------------------|
| Base model generation   | GPT Image 2 via [fal.ai](https://fal.ai)                   |
| Try-on (GPT)            | GPT Image 2 Edit via [fal.ai](https://fal.ai)              |
| Try-on (Z-Image)        | [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) via fal.ai |
| Product classification  | GPT-4o Vision via [OpenRouter](https://openrouter.ai)       |
| Face swap (cloud)       | InsightFace inswapper_128 via [Replicate](https://replicate.com) |
| Face swap (local)       | InsightFace + ONNX Runtime                                  |
| Data scraping           | Westside Shopify public JSON API                            |
