FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod>=1.7.0 \
    transformers>=4.40.0 \
    torch>=2.4.0 \
    torchvision>=0.19.0 \
    Pillow>=10.0.0 \
    opencv-python-headless>=4.9.0 \
    numpy>=1.26.0

# Pre-download model so cold start doesn't need internet
RUN python -c "\
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor; \
SegformerImageProcessor.from_pretrained('mattmdjaga/segformer_b2_clothes'); \
SegformerForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes'); \
print('Model cached.')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
