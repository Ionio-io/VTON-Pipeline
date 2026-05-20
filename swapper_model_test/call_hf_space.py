"""
Call a Hugging Face Space running swapper_model_test/huggingface_space/app.py.

Install:
    python -m pip install gradio_client

Usage:
    python swapper_model_test/call_hf_space.py ^
      --space YOUR_USERNAME/YOUR_SPACE_NAME ^
      --source generated_images/M-REC_S3.png ^
      --target tryon_output/sample1_M-REC_S3_nuon-dark-brown-relaxed-fit-mid-rise-jea.png
"""

import argparse
import shutil
from pathlib import Path

from gradio_client import Client, handle_file


def main():
    parser = argparse.ArgumentParser(description="Call a Hugging Face Space InSwapper API.")
    parser.add_argument("--space", required=True, help="Space id, e.g. username/inswapper-vton")
    parser.add_argument("--source", required=True, help="Source identity image path")
    parser.add_argument("--target", required=True, help="Target try-on image path")
    parser.add_argument("--api-name", default="/swap", help="Gradio API name")
    parser.add_argument("--output-dir", default="tryon_output/hf_space_test", help="Directory to save the returned image")
    parser.add_argument("--label", default="hf_space_swap", help="Output file label")
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    if not source.exists():
        raise FileNotFoundError(f"Source image not found: {source}")
    if not target.exists():
        raise FileNotFoundError(f"Target image not found: {target}")

    client = Client(args.space)
    result = client.predict(
        source_image=handle_file(str(source)),
        target_image=handle_file(str(target)),
        api_name=args.api_name,
    )

    print(f"Raw result: {result}")

    if isinstance(result, str) and Path(result).exists():
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(result).suffix or ".png"
        output_path = output_dir / f"{args.label}{suffix}"
        shutil.copy2(result, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
