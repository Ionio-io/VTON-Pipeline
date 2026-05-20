# Swapper Model Test

This folder contains the real face-swapper test paths for the VTON flow.

It does **not** use GPT/image-generation edit models.

Flow:

```text
source identity image  +  target model-wearing VTON image
        -> face swapper model
        -> personalized try-on image
```

For the current repo:

```text
source identity image = generated_images/<model_id>.png
target try-on image   = tryon_output/<existing VTON output>.png
```

## Folder Contents

```text
swapper_model_test/
  README.md
  requirements-local.txt
  requirements-replicate.txt
  swapper_local_insightface.py
  swapper_replicate_test.py
  call_hf_space.py
  huggingface_space/
    app.py
    requirements.txt
    packages.txt
    README.md
  models/
    .gitkeep
```

## Option 1: Local InSwapper

Use this when you want everything running locally.

Recommended Python: **3.10 or 3.11**.

Do not use Python 3.14 for this path. `insightface` is likely to fail installing there.

### Setup

From the repo root:

```powershell
cd path\to\rag-web-ui-main

py -3.10 -m venv .venv-swapper
.\.venv-swapper\Scripts\activate

python -m pip install -U pip setuptools wheel
python -m pip install -r swapper_model_test\requirements-local.txt
```

### Add The Model

Download/provide `inswapper_128.onnx` and place it here:

```text
swapper_model_test/models/inswapper_128.onnx
```

The script also checks:

```text
models/inswapper_128.onnx
C:\Users\<you>\.insightface\models\inswapper_128.onnx
C:\Users\<you>\.insightface\models\inswapper_128\inswapper_128.onnx
```

### Where To Download The Model

Best/safest production route:

```text
Use InsightFace commercial licensing/API if this is for a client or production e-commerce use.
```

InsightFace's own docs say the old `inswapper_128.onnx` demo is no longer maintained and point commercial/production users toward their face-swapping product/licensing.

Fast local prototype route:

```text
Download a community mirror of inswapper_128.onnx from Hugging Face, then put it at:
swapper_model_test/models/inswapper_128.onnx
```

Known community mirror:

```text
https://huggingface.co/fofr/comfyui/blob/main/insightface/inswapper_128.onnx
```

Important:

```text
Treat community mirrors as prototype-only unless you have confirmed redistribution and commercial usage rights.
For production, do a license check or use a licensed/commercial face-swap API.
```

### Run

Dry run:

```powershell
python swapper_model_test\swapper_local_insightface.py --dry-run
```

Default one-sample run:

```powershell
python swapper_model_test\swapper_local_insightface.py `
  --model swapper_model_test\models\inswapper_128.onnx
```

Custom run:

```powershell
python swapper_model_test\swapper_local_insightface.py `
  --source generated_images\M-REC_S3.png `
  --target tryon_output\sample1_M-REC_S3_nuon-dark-brown-relaxed-fit-mid-rise-jea.png `
  --model swapper_model_test\models\inswapper_128.onnx `
  --label jeans_test
```

Run all rows from `tryon_output/tryon_log.json`:

```powershell
python swapper_model_test\swapper_local_insightface.py `
  --model swapper_model_test\models\inswapper_128.onnx `
  --all-samples
```

Output:

```text
tryon_output/swapper_local_test/
```

## Option 2: Replicate InSwapper API

Use this when local dependencies/model setup are annoying and you want a real swapper model through an API.

This path uses `ddvinh1/inswapper` on Replicate.

### Setup

From the repo root:

```powershell
cd path\to\rag-web-ui-main
$env:REPLICATE_API_TOKEN="your_replicate_token"
```

No pip install is required for the script itself.

### Run

Dry run:

```powershell
python swapper_model_test\swapper_replicate_test.py --dry-run
```

Default one-sample run:

```powershell
python swapper_model_test\swapper_replicate_test.py
```

Custom run:

```powershell
python swapper_model_test\swapper_replicate_test.py `
  --source generated_images\M-REC_S3.png `
  --target tryon_output\sample1_M-REC_S3_nuon-dark-brown-relaxed-fit-mid-rise-jea.png `
  --label jeans_test
```

Run all rows from `tryon_output/tryon_log.json`:

```powershell
python swapper_model_test\swapper_replicate_test.py --all-samples
```

Output:

```text
tryon_output/swapper_replicate_test/
```

## Which One To Use First?

Use this order:

1. Run Replicate first to validate the flow quickly.
2. If you specifically want Hugging Face API, deploy `huggingface_space/` as a Gradio Space and call it with `call_hf_space.py`.
3. Run local InSwapper after you have Python 3.10/3.11 and `inswapper_128.onnx`.
4. Compare outputs from:

```text
tryon_output/swapper_replicate_test/
tryon_output/swapper_local_test/
```

## Option 3: Hugging Face Space API

The plain Hugging Face model repo is only a raw ONNX file. It is not directly callable as:

```text
source image + target image -> swapped image
```

So the Hugging Face API path is:

```text
Deploy a tiny Gradio Space wrapper -> call the Space API.
```

Upload this folder to a new Hugging Face Space:

```text
swapper_model_test/huggingface_space/
```

Then call it locally:

```powershell
python -m pip install gradio_client

python swapper_model_test\call_hf_space.py `
  --space YOUR_USERNAME/YOUR_SPACE_NAME `
  --source generated_images\M-REC_S3.png `
  --target tryon_output\sample1_M-REC_S3_nuon-dark-brown-relaxed-fit-mid-rise-jea.png
```

The Space exposes:

```text
api_name="/swap"
```

### Detailed Hugging Face Space Creation

Useful official links:

```text
Create new Space:
https://huggingface.co/new-space

Spaces docs:
https://huggingface.co/docs/hub/spaces

Gradio Spaces docs:
https://huggingface.co/docs/hub/spaces-sdks-gradio

Spaces API endpoint docs:
https://huggingface.co/docs/hub/spaces-api-endpoints

Access token docs:
https://huggingface.co/docs/hub/security-tokens

Upload files docs:
https://huggingface.co/docs/huggingface_hub/guides/upload
```

#### A. Create The Space In Browser

1. Open:

```text
https://huggingface.co/new-space
```

2. Fill:

```text
Owner: your account/org
Space name: vton-inswapper-test
License: choose what matches your prototype
SDK: Gradio
Hardware: CPU Basic
Visibility: Private for client/prototype work
```

3. Click `Create Space`.

#### B. Upload In Browser

Inside the new Space page:

1. Go to `Files`.
2. Click `Add file`.
3. Upload these files from `swapper_model_test/huggingface_space/`:

```text
app.py
requirements.txt
packages.txt
README.md
```

4. Commit the files.
5. Watch the `Logs` tab until build finishes.

#### C. Push With Git Instead

Create a Hugging Face write token:

```text
https://huggingface.co/settings/tokens
```

Then:

```powershell
git lfs install
git clone https://huggingface.co/spaces/YOUR_USERNAME/vton-inswapper-test

Copy-Item swapper_model_test\huggingface_space\app.py vton-inswapper-test\app.py -Force
Copy-Item swapper_model_test\huggingface_space\requirements.txt vton-inswapper-test\requirements.txt -Force
Copy-Item swapper_model_test\huggingface_space\packages.txt vton-inswapper-test\packages.txt -Force
Copy-Item swapper_model_test\huggingface_space\README.md vton-inswapper-test\README.md -Force

cd vton-inswapper-test
git add app.py requirements.txt packages.txt README.md
git commit -m "Add InSwapper VTON API"
git push
```

When Git asks for credentials:

```text
Username: your Hugging Face username
Password: your Hugging Face write token
```

#### D. Test The Space UI

After build succeeds, open:

```text
https://huggingface.co/spaces/YOUR_USERNAME/vton-inswapper-test
```

Upload:

```text
Source Identity Image: generated_images/M-REC_S3.png
Target Try-On Image: tryon_output/sample1_M-REC_S3_nuon-dark-brown-relaxed-fit-mid-rise-jea.png
```

Click `Swap Face`.

#### E. Find The API Docs

On the running Space page:

```text
Click the three dots / menu or "Use via API" / "View API"
```

The endpoint should show:

```text
/swap
```

#### F. Call From Local Repo

```powershell
python -m pip install gradio_client

python swapper_model_test\call_hf_space.py `
  --space YOUR_USERNAME/vton-inswapper-test `
  --source generated_images\M-REC_S3.png `
  --target tryon_output\sample1_M-REC_S3_nuon-dark-brown-relaxed-fit-mid-rise-jea.png
```

For a private Space, update `call_hf_space.py` to pass your token:

```python
client = Client("YOUR_USERNAME/vton-inswapper-test", hf_token="hf_...")
```

Or set token from env in your own wrapper instead of hardcoding it.

## Expected Data

The default commands expect these files from the existing repo:

```text
generated_images/
tryon_output/tryon_log.json
tryon_output/sample*.png
```

If those are missing, pass explicit `--source` and `--target` image paths.

## Notes

- `source` means the face/identity to transfer.
- `target` means the already-generated model-wearing image.
- The swapper should preserve the clothes because it only replaces the face region.
- Use only with consent from the person whose face/identity is being swapped.
