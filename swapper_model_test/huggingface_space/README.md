---
title: InSwapper VTON Face Swap
sdk: gradio
app_file: app.py
pinned: false
---

# Hugging Face Space: InSwapper API

This is a Hugging Face Space wrapper around:

```text
ezioruan/inswapper_128.onnx
```

It exposes a Gradio API:

```text
/swap
```

Input:

```text
source_image = identity/face image
target_image = model-wearing VTON image
```

Output:

```text
swapped image
```

## Deploy To Hugging Face Spaces

1. Create a new Hugging Face Space.
2. SDK: `Gradio`
3. Hardware: CPU Basic is okay for prototype. GPU is better if you later switch to GPU ONNX Runtime.
4. Upload these files:

```text
app.py
requirements.txt
packages.txt
```

The Space downloads `inswapper_128.onnx` from Hugging Face on startup/cache.

## Call The Space API

Install the client locally:

```powershell
python -m pip install gradio_client
```

Then:

```python
from gradio_client import Client, handle_file

client = Client("YOUR_USERNAME/YOUR_SPACE_NAME")

result = client.predict(
    source_image=handle_file("generated_images/M-REC_S3.png"),
    target_image=handle_file("tryon_output/sample1_M-REC_S3_nuon-dark-brown-relaxed-fit-mid-rise-jea.png"),
    api_name="/swap",
)

print(result)
```

The `result` is usually a temporary file path or URL returned by Gradio.

## Notes

- This is still using a real swapper model, not an image-generation model.
- First request can be slow because the Space downloads/caches model files.
- For production, confirm commercial rights or use a licensed face-swap API.
