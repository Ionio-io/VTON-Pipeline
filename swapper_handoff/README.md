# VTON Face Swapper Handoff

This folder is the minimal package for running the Hugging Face Space face-swapper
from another laptop.

It uses the already deployed Space:

```text
manideep-e/vton-inswapper-test
https://huggingface.co/spaces/manideep-e/vton-inswapper-test
```

The Space runs `inswapper_128.onnx` behind a Gradio API endpoint:

```text
/swap
```

## What This Does

```text
base model face image + Westside model wearing garment image
-> Hugging Face Space /swap
-> output image with base model face on the Westside garment photo
```

Meaning:

```text
source = base model face / identity
target = Westside model already wearing garment
output = target garment image with source face swapped in
```

## Files

```text
swapper_handoff/
  README.md
  requirements.txt
  run_hf_swap.py
  run_westside_sample.py
```

## What To Copy To Another Laptop

Copy this folder:

```text
swapper_handoff/
```

And copy these repo data folders/files if you want the included sample runner:

```text
generated_images/
westside_dataset/
```

For the exact prepared sample, these two files are the minimum data needed:

```text
generated_images/M-OVL_S5.png
westside_dataset/images/men/nuon-black-text-design-oversized-fit-cotton-t-shirt-301066288/1.jpg
```

The sample runner expects this structure:

```text
your_project/
  generated_images/
  westside_dataset/
  swapper_handoff/
```

## Install

From the project root:

```powershell
python -m venv .venv-swapper-client
.\.venv-swapper-client\Scripts\activate
python -m pip install -U pip
python -m pip install -r swapper_handoff\requirements.txt
```

## Token

If the Space is public, you may not need a token.

If the Space is private, set a Hugging Face token in the terminal:

```powershell
$env:HF_TOKEN="PASTE_YOUR_HF_TOKEN_HERE"
```

Do not hardcode the token into scripts. If a token was pasted into chat or committed
anywhere, revoke/rotate it from:

```text
https://huggingface.co/settings/tokens
```

## Run The Prepared Westside Sample

This uses:

```text
source: generated_images/M-OVL_S5.png
target: westside_dataset/images/men/nuon-black-text-design-oversized-fit-cotton-t-shirt-301066288/1.jpg
```

Run:

```powershell
python swapper_handoff\run_westside_sample.py
```

Output:

```text
tryon_output/westside_model_faceswap_handoff/sample_m_ovl_s5_nuon_black_tshirt/
  01_source_base_model.png
  01b_source_face_crop.png
  02_target_westside_model_wearing_garment.jpg
  03_faceswapped_output.webp
  metadata.json
```

## Run Custom Source + Target

```powershell
python swapper_handoff\run_hf_swap.py `
  --source generated_images\M-OVL_S5.png `
  --target westside_dataset\images\men\nuon-black-text-design-oversized-fit-cotton-t-shirt-301066288\1.jpg `
  --output tryon_output\custom_faceswap.webp
```

If you already have a clean face crop, use it as `--source`.

## Private Space Example

```powershell
$env:HF_TOKEN="PASTE_YOUR_HF_TOKEN_HERE"

python swapper_handoff\run_hf_swap.py `
  --space manideep-e/vton-inswapper-test `
  --source generated_images\M-OVL_S5.png `
  --target westside_dataset\images\men\nuon-black-text-design-oversized-fit-cotton-t-shirt-301066288\1.jpg `
  --output tryon_output\custom_faceswap.webp
```

## Hugging Face Cost Question

As of the current Hugging Face docs, CPU Basic Spaces are free. Paid hardware is
billed while the Space is `Starting` or `Running`; paused time is not billed.

Your current Space was created on CPU Basic, so keeping it up on CPU Basic should
not charge you. If you switch to GPU or upgraded CPU hardware, then you can be
charged while it is starting/running. To avoid surprise costs, keep hardware on
CPU Basic or pause the Space when not testing.

Official pricing/docs:

```text
https://huggingface.co/pricing
https://huggingface.co/docs/hub/spaces-overview
https://huggingface.co/docs/hub/spaces-gpus
```

## Notes

- The output can look subtle when source and target faces are similar.
- Use a clean face crop as source for stronger identity transfer.
- InSwapper changes mainly the inner face. It does not replace the full head,
  hair, body, or garment.
- Only use this with consent for identity/face swapping.
