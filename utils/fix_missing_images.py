"""
Downloads images for any products in the dataset that have empty local_images.
Run once to backfill products that were scraped with --no-images.
"""
import json
import time
import urllib.request
from pathlib import Path

IMAGES_DIR = Path("westside_dataset/images")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def download_images(product: dict) -> list[str]:
    gender_dir = IMAGES_DIR / ("men" if product["gender"] == "male" else "women")
    folder = gender_dir / product["handle"]
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, url in enumerate(product["image_urls"], 1):
        ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg"
        dest = folder / f"{idx}.{ext}"
        if dest.exists():
            paths.append(str(dest))
            continue
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                dest.write_bytes(r.read())
            paths.append(str(dest))
            time.sleep(0.08)
        except Exception as e:
            print(f"    WARNING?  {idx}: {e}")
    return paths


def main():
    total_fixed = 0
    for fname in ("products_men.json", "products_women.json"):
        path = Path(f"westside_dataset/{fname}")
        products = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for p in products:
            if not p["local_images"]:
                print(f"  Downloading: {p['title'][:60]}")
                p["local_images"] = download_images(p)
                print(f"    -> {len(p['local_images'])} images saved")
                changed = True
                total_fixed += 1
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(products, f, indent=2, ensure_ascii=False)
            print(f"  ? Saved {fname}\n")

    print(f"\n[OK] Fixed {total_fixed} products.")


if __name__ == "__main__":
    main()
