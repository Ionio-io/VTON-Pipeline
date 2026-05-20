"""
Z-Image Turbo Virtual Try-On — Hugging Face Space
==================================================
Upload a base model image + describe the garment → Z-Image Turbo edits
the model to wear it in under ~3 seconds on GPU.

Deploy:
  1. Create a new Space at https://huggingface.co/new-space
     - SDK: Gradio
     - Hardware: T4-small (16 GB, $0.40/hr) or A10G (24 GB, faster)
  2. Upload this app.py + requirements.txt to the Space

API (programmatic):
  from gradio_client import Client
  client = Client("YOUR_USERNAME/YOUR_SPACE_NAME")
  result = client.predict(
      model_image_path,     # path to base model PNG
      "Navy Blue Slim-Fit Shirt",  # garment title
      "Cotton, button-down collar, slim fit", # garment detail
      "male",               # gender
      "Rectangle",          # body type
      "Medium / Olive",     # skin tone
      0.55,                 # strength
      42,                   # seed
      api_name="/predict",
  )
"""

import re
import time

import gradio as gr
import torch
from diffusers import ZImageImg2ImgPipeline
from PIL import Image

# ── Model loading (once at startup) ──────────────────────────────────────────

MODEL_REPO = "Tongyi-MAI/Z-Image-Turbo"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE      = torch.bfloat16 if DEVICE != "cpu" else torch.float32

print(f"Loading {MODEL_REPO} on {DEVICE} …")
t0   = time.time()
pipe = ZImageImg2ImgPipeline.from_pretrained(MODEL_REPO, torch_dtype=DTYPE)
pipe = pipe.to(DEVICE)
print(f"Ready in {time.time()-t0:.1f}s")

# ── Prompt builder ────────────────────────────────────────────────────────────

NEGATIVE_PROMPT = (
    "deformed, extra limbs, blurry, low quality, watermark, text, logo, "
    "artifacts, bad anatomy, distorted clothing, ugly, duplicate"
)

IMG_SIZE = (768, 1024)   # width × height

BODY_TYPES = [
    "Rectangle", "Inverted Trapezoid", "Triangle", "Oval", "Trapezoid",
    "Hourglass", "Sporty",
]
SKIN_TONES = ["Medium / Olive", "Light Brown", "Dark Brown"]


def build_prompt(
    title:   str,
    detail:  str,
    gender:  str,
    body:    str,
    skin:    str,
) -> str:
    detail = detail.strip()
    detail_clause = f" {detail}." if detail else ""
    return (
        f"Full-body studio photograph of an Indian {gender} with {skin} "
        f"skin tone and {body} body type, wearing {title}."
        f"{detail_clause} "
        "White seamless studio background, soft diffused overhead lighting, "
        "sharp full-body focus, professional e-commerce fashion photography, 4K."
    )


# ── Inference ─────────────────────────────────────────────────────────────────

def run_tryon(
    model_image: Image.Image,
    garment_title: str,
    garment_detail: str,
    gender: str,
    body_type: str,
    skin_tone: str,
    strength: float,
    seed: int,
) -> tuple[Image.Image, str]:
    if model_image is None:
        raise gr.Error("Please upload a base model image.")
    if not garment_title.strip():
        raise gr.Error("Please enter a garment title.")

    init = model_image.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    prompt = build_prompt(garment_title, garment_detail, gender, body_type, skin_tone)

    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    t0 = time.time()
    result = pipe(
        prompt              = prompt,
        image               = init,
        strength            = float(strength),
        num_inference_steps = 9,
        guidance_scale      = 0.0,
        negative_prompt     = NEGATIVE_PROMPT,
        generator           = generator,
    ).images[0]
    elapsed = time.time() - t0

    info = f"Done in {elapsed:.1f}s | strength={strength} | seed={seed}\nPrompt: {prompt}"
    return result, info


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Z-Image Turbo VTON", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Z-Image Turbo — Virtual Try-On
        Upload a **base model image** (or use a pre-generated one from `generated_images/`),
        describe the garment, and get a try-on result in ~2-3 seconds.

        > **Note:** Z-Image Turbo edits via text — no garment reference image needed.
        > Use **strength 0.45–0.60** to keep the model's face/body intact.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_image = gr.Image(
                label="Base Model Image",
                type="pil",
                height=400,
            )
            gr.Markdown("*Upload one of the `generated_images/*.png` files*")

        with gr.Column(scale=1):
            garment_title = gr.Textbox(
                label="Garment Title",
                placeholder="e.g. Navy Blue Relaxed-Fit Mid-Rise Jeans",
            )
            garment_detail = gr.Textbox(
                label="Garment Detail (optional)",
                placeholder="e.g. Cotton fabric, slim fit, button closure",
                lines=2,
            )
            with gr.Row():
                gender = gr.Radio(
                    choices=["man", "woman"],
                    value="man",
                    label="Gender",
                )
            body_type = gr.Dropdown(
                choices=BODY_TYPES,
                value="Rectangle",
                label="Body Type",
            )
            skin_tone = gr.Dropdown(
                choices=SKIN_TONES,
                value="Medium / Olive",
                label="Skin Tone",
            )
            with gr.Row():
                strength = gr.Slider(
                    minimum=0.35, maximum=0.80, value=0.55, step=0.05,
                    label="Strength (lower = more original preserved)",
                )
                seed = gr.Number(value=42, label="Seed", precision=0)

            run_btn = gr.Button("Generate Try-On", variant="primary")

    with gr.Row():
        output_image = gr.Image(label="Try-On Result", height=500)
        info_box     = gr.Textbox(label="Info", lines=4, interactive=False)

    run_btn.click(
        fn=run_tryon,
        inputs=[model_image, garment_title, garment_detail,
                gender, body_type, skin_tone, strength, seed],
        outputs=[output_image, info_box],
        api_name="predict",
    )

    gr.Examples(
        examples=[
            [None, "Dark Brown Relaxed-Fit Mid-Rise Jeans",
             "Smooth cotton fabric, skater fit, mid-rise waist, multiple pockets",
             "man", "Rectangle", "Medium / Olive", 0.55, 42],
            [None, "Pink Floral Printed A-Line Ethnic Set",
             "Kurta with palazzo and dupatta, dense floral print, festive wear",
             "woman", "Hourglass", "Light Brown", 0.55, 7],
        ],
        inputs=[model_image, garment_title, garment_detail,
                gender, body_type, skin_tone, strength, seed],
        label="Example Inputs",
    )


if __name__ == "__main__":
    demo.launch()
