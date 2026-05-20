"""
Run one prepared Westside model-wearing garment face swap sample.

This script:
  1. Copies the source base model image.
  2. Crops the source face for stronger identity transfer.
  3. Copies the Westside target model-wearing garment image.
  4. Calls the deployed HF Space.
  5. Saves a clean triplet folder with metadata.
"""

import json
import os
import shutil
from pathlib import Path

import cv2
from gradio_client import Client, handle_file


SPACE = "manideep-e/vton-inswapper-test"
API_NAME = "/swap"

SOURCE_MODEL_ID = "M-OVL_S5"
SOURCE_IMAGE = Path("generated_images/M-OVL_S5.png")
TARGET_IMAGE = Path(
    "westside_dataset/images/men/"
    "nuon-black-text-design-oversized-fit-cotton-t-shirt-301066288/1.jpg"
)

OUTPUT_DIR = Path(
    "tryon_output/westside_model_faceswap_handoff/"
    "sample_m_ovl_s5_nuon_black_tshirt"
)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def crop_largest_face(source_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {source_path}")

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.05, 4, minSize=(30, 30))
    if len(faces) == 0:
        raise RuntimeError(f"No face detected in source image: {source_path}")

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    margin = int(w * 0.9)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)

    crop = image[y1:y2, x1:x2]
    cv2.imwrite(str(output_path), crop)


def call_space(source: Path, target: Path, output: Path) -> None:
    hf_token = os.environ.get("HF_TOKEN") or None
    client = Client(SPACE, hf_token=hf_token)
    result = client.predict(
        source_image=handle_file(str(source)),
        target_image=handle_file(str(target)),
        api_name=API_NAME,
    )

    if isinstance(result, str) and Path(result).exists():
        shutil.copy2(result, output)
        return

    print(f"Raw result: {result}")
    raise RuntimeError("Space returned a result that was not a local file path.")


def main() -> None:
    require_file(SOURCE_IMAGE)
    require_file(TARGET_IMAGE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_copy = OUTPUT_DIR / "01_source_base_model.png"
    source_crop = OUTPUT_DIR / "01b_source_face_crop.png"
    target_copy = OUTPUT_DIR / "02_target_westside_model_wearing_garment.jpg"
    output = OUTPUT_DIR / "03_faceswapped_output.webp"
    metadata_path = OUTPUT_DIR / "metadata.json"

    shutil.copy2(SOURCE_IMAGE, source_copy)
    crop_largest_face(source_copy, source_crop)
    shutil.copy2(TARGET_IMAGE, target_copy)

    call_space(source_crop, target_copy, output)

    metadata = {
        "space": SPACE,
        "api_name": API_NAME,
        "source_model_id": SOURCE_MODEL_ID,
        "source_original": str(SOURCE_IMAGE),
        "target_original": str(TARGET_IMAGE),
        "source_copy": str(source_copy),
        "source_face_crop": str(source_crop),
        "target_copy": str(target_copy),
        "output": str(output),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved folder: {OUTPUT_DIR}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
