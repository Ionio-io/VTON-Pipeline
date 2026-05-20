import time

import cv2
import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download
import insightface
from insightface.app import FaceAnalysis


MODEL_REPO = "ezioruan/inswapper_128.onnx"
MODEL_FILE = "inswapper_128.onnx"

app = None
swapper = None


def load_models():
    global app, swapper

    if app is not None and swapper is not None:
        return app, swapper

    providers = ["CPUExecutionProvider"]

    detector = FaceAnalysis(name="buffalo_l", providers=providers)
    detector.prepare(ctx_id=0, det_size=(640, 640))

    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    face_swapper = insightface.model_zoo.get_model(model_path, providers=providers)

    app = detector
    swapper = face_swapper
    return app, swapper


def pick_largest_face(faces, label):
    if not faces:
        raise gr.Error(f"No face detected in {label} image.")

    return max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
    )


def rgb_to_bgr(image):
    if image is None:
        return None
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def swap_faces(source_image, target_image):
    started = time.time()
    detector, face_swapper = load_models()

    source_bgr = rgb_to_bgr(source_image)
    target_bgr = rgb_to_bgr(target_image)

    source_faces = detector.get(source_bgr)
    target_faces = detector.get(target_bgr)

    source_face = pick_largest_face(source_faces, "source")
    target_face = pick_largest_face(target_faces, "target")

    swapped = face_swapper.get(target_bgr, target_face, source_face, paste_back=True)
    latency = round(time.time() - started, 2)
    print(f"swap latency: {latency}s")

    return bgr_to_rgb(swapped)


with gr.Blocks(title="InSwapper VTON Face Swap") as demo:
    gr.Markdown("# InSwapper VTON Face Swap")
    gr.Markdown("Source = identity face. Target = precomputed model-wearing VTON image.")

    with gr.Row():
        source = gr.Image(label="Source Identity Image", type="numpy")
        target = gr.Image(label="Target Try-On Image", type="numpy")

    output = gr.Image(label="Swapped Output", type="numpy")
    run = gr.Button("Swap Face")

    run.click(
        fn=swap_faces,
        inputs=[source, target],
        outputs=output,
        api_name="swap",
    )


if __name__ == "__main__":
    demo.launch()
