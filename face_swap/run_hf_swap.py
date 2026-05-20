"""
Call the deployed Hugging Face InSwapper Space.

Example:
    python swapper_handoff/run_hf_swap.py ^
      --source generated_images/M-OVL_S5.png ^
      --target westside_dataset/images/men/nuon-black-text-design-oversized-fit-cotton-t-shirt-301066288/1.jpg ^
      --output tryon_output/custom_faceswap.webp
"""

import argparse
import os
import shutil
from pathlib import Path

from gradio_client import Client, handle_file


DEFAULT_SPACE = "manideep-e/vton-inswapper-test"
DEFAULT_API_NAME = "/swap"


def resolve_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Hugging Face Space face swap.")
    parser.add_argument("--space", default=DEFAULT_SPACE, help="HF Space id")
    parser.add_argument("--api-name", default=DEFAULT_API_NAME, help="Gradio API endpoint")
    parser.add_argument("--source", required=True, help="Source identity/base face image")
    parser.add_argument("--target", required=True, help="Target Westside garment/model image")
    parser.add_argument("--output", required=True, help="Output image path")
    args = parser.parse_args()

    source = resolve_path(args.source)
    target = resolve_path(args.target)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN") or None
    client = Client(args.space, hf_token=hf_token)

    result = client.predict(
        source_image=handle_file(str(source)),
        target_image=handle_file(str(target)),
        api_name=args.api_name,
    )

    if isinstance(result, str) and Path(result).exists():
        shutil.copy2(result, output)
        print(f"Saved: {output}")
        return

    print(f"Raw result: {result}")
    raise RuntimeError("Space returned a result that was not a local file path.")


if __name__ == "__main__":
    main()
