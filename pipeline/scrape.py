"""
Westside Product Scraper -- Fashion Try-On Dataset
==================================================
Scrapes 50 men + 50 women clothing products from westside.com.
Skips footwear, accessories, kids, and anything not try-on relevant.
Downloads all product images and saves two clean JSON datasets.

Usage:
    python scrape_westside.py                   # 50 men + 50 women, with images
    python scrape_westside.py --resume          # safe to re-run, skips existing
    python scrape_westside.py --no-images       # metadata only, very fast

Output:
    westside_dataset/
        products_men.json        <- 50 men's clothing items
        products_women.json      <- 50 women's clothing items
        images/
            {product-handle}/
                1.jpg, 2.jpg, ...

Requirements: no extra packages -- uses stdlib only (urllib, json, pathlib)
"""

import json
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# -- Config --------------------------------------------------------------------

BASE_URL     = "https://www.westside.com/products.json"
OUTPUT_DIR   = Path("westside_dataset")
IMAGES_DIR   = OUTPUT_DIR / "images"
MEN_DIR      = IMAGES_DIR / "men"
WOMEN_DIR    = IMAGES_DIR / "women"
PAGE_DELAY   = 1.5    # seconds between page fetches
IMAGE_DELAY  = 0.08   # seconds between individual image downloads

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# -- Category filters ----------------------------------------------------------

CLOTHING_TYPES = {
    "T-Shirts", "Shirts", "Tops", "Blouses", "Tunics",
    "Kurtas", "Kurta", "Kurtis",
    "Dresses", "Jumpsuits", "Playsuits", "Co-ords", "Co-Ords",
    "Jeans", "Pants", "Trousers", "Joggers", "Shorts", "Cargos",
    "Track Pants", "Sweatpants", "Palazzos", "Skirts", "Leggings",
    "Sweatshirts", "Hoodies", "Jackets", "Blazers", "Suits",
    "Ethnic Suits", "Sarees", "Shrugs", "Formal Shirts",
    "Casual Shirts", "Polos", "Sweaters", "Cardigans",
    "Nightwear", "Loungewear", "Coordinates",
}

EXCLUDE_KEYWORDS = {
    "sandal", "shoe", "heel", "sneaker", "boot", "loafer",
    "slipper", "footwear", "mule", "wedge", "flip flop", "flat",
    "bag", "handbag", "wallet", "purse", "belt", "watch",
    "jewellery", "jewelry", "sunglasses", "cap", "hat", "scarf",
    "mask", "sock", "innerwear", "underwear", "undergarment",
    "accessory", "accessories", "fragrance", "perfume",
    "eyewear", "sportswear equipment",
}

KIDS_TAGS   = {"Kids", "Boy", "Girl", "Boys", "Girls", "Junior", "Baby", "Infant"}
MALE_TAGS   = {"Male", "Man", "Men", "Mens", "Gents"}
FEMALE_TAGS = {"Female", "Woman", "Women", "Womens", "Ladies", "Lady"}

# -- Helpers -------------------------------------------------------------------

def fetch_page(page: int, limit: int) -> list[dict]:
    url = f"{BASE_URL}?page={page}&limit={limit}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("products", [])


def detect_gender(tags: list[str], product_type: str) -> str | None:
    tag_set = set(tags)

    if tag_set & KIDS_TAGS:
        return None

    is_male   = bool(tag_set & MALE_TAGS)
    is_female = bool(tag_set & FEMALE_TAGS)

    if is_male and not is_female:
        return "male"
    if is_female and not is_male:
        return "female"

    pt = product_type.lower()
    if any(w in pt for w in ("saree", "kurta", "kurti", "palazzo", "dupatta")):
        return "female"

    return None


def is_clothing(product_type: str) -> bool:
    pt_lower = product_type.lower()

    if any(kw in pt_lower for kw in EXCLUDE_KEYWORDS):
        return False

    if product_type in CLOTHING_TYPES:
        return True

    clothing_keywords = {
        "shirt", "top", "tee", "tshirt", "t-shirt", "dress", "kurta",
        "kurti", "saree", "jeans", "pant", "trouser", "jogger", "short",
        "skirt", "legging", "sweatshirt", "hoodie", "jacket", "blazer",
        "suit", "co-ord", "jumpsuit", "playsuit", "cargo", "palazzo",
        "polo", "sweater", "cardigan", "shrug", "blouse", "tunic",
        "nightwear", "loungewear", "coord",
    }
    return any(kw in pt_lower for kw in clothing_keywords)


def clean_product(raw: dict, gender: str) -> dict:
    images = [img["src"] for img in raw.get("images", [])]

    price     = None
    available = False
    for v in raw.get("variants", []):
        if v.get("available"):
            available = True
        if price is None:
            price = v.get("price")

    return {
        "id":           raw["id"],
        "title":        raw["title"],
        "handle":       raw["handle"],
        "description":  raw.get("body_html", ""),
        "product_type": raw.get("product_type", ""),
        "vendor":       raw.get("vendor", ""),
        "tags":         raw.get("tags", []),
        "gender":       gender,
        "price":        price,
        "available":    available,
        "image_urls":   images,
        "local_images": [],
    }


def download_images(product: dict, resume: bool) -> list[str]:
    gender_dir = MEN_DIR if product["gender"] == "male" else WOMEN_DIR
    folder = gender_dir / product["handle"]
    folder.mkdir(parents=True, exist_ok=True)

    local_paths = []
    for idx, url in enumerate(product["image_urls"], 1):
        ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg"
        dest = folder / f"{idx}.{ext}"

        if resume and dest.exists():
            local_paths.append(str(dest))
            continue

        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                dest.write_bytes(resp.read())
            local_paths.append(str(dest))
            time.sleep(IMAGE_DELAY)
        except Exception as e:
            print(f"      WARNING?  Image {idx} failed: {e}")

    return local_paths


def load_existing(path: Path) -> tuple[list[dict], set[str]]:
    products = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            products = json.load(f)
    return products, {p["handle"] for p in products}


def save_json(products: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)


def print_summary(men: list[dict], women: list[dict]):
    from collections import Counter

    total_imgs = sum(len(p["local_images"]) for p in men + women)

    print(f"\n{'='*65}")
    print(f"  SCRAPE COMPLETE")
    print(f"{'='*65}")
    print(f"  Men's products   : {len(men):3d}  ->  westside_dataset/products_men.json")
    print(f"  Women's products : {len(women):3d}  ->  westside_dataset/products_women.json")
    print(f"  Total images     : {total_imgs}")
    print(f"  Images folder    : westside_dataset/images/")

    for label, products in [("MEN", men), ("WOMEN", women)]:
        types = Counter(p["product_type"] for p in products)
        print(f"\n  {label} -- top categories:")
        for ptype, count in types.most_common(8):
            print(f"    {count:3d}x  {ptype}")

    print(f"{'='*65}\n")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape Westside.com -- 50 men + 50 women clothing items"
    )
    parser.add_argument(
        "--per-gender", type=int, default=50,
        help="Products to collect per gender (default: 50)"
    )
    parser.add_argument(
        "--per-page", type=int, default=100,
        help="Products per API page request (default: 100)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-downloaded images (safe to re-run)"
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Metadata only -- skip image downloads"
    )
    args = parser.parse_args()

    per_page = min(args.per_page, 250)
    target   = args.per_gender

    men_path   = OUTPUT_DIR / "products_men.json"
    women_path = OUTPUT_DIR / "products_women.json"

    men,   seen_men   = load_existing(men_path)
    women, seen_women = load_existing(women_path)
    seen_all = seen_men | seen_women

    print(f"\n??   Westside Fashion Scraper")
    print(f"    Target      : {target} men + {target} women")
    print(f"    Per page    : {per_page}")
    print(f"    Resume      : {args.resume}")
    print(f"    Images      : {'no' if args.no_images else 'yes'}")
    print(f"    Have so far : {len(men)} men, {len(women)} women\n")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MEN_DIR.mkdir(parents=True, exist_ok=True)
    WOMEN_DIR.mkdir(parents=True, exist_ok=True)

    page = 1
    while len(men) < target or len(women) < target:
        print(
            f"?  Page {page} -- "
            f"men {len(men)}/{target}, women {len(women)}/{target}"
        )
        try:
            raw_page = fetch_page(page, per_page)
        except urllib.error.URLError as e:
            print(f"[ERR]  Network error: {e}")
            break
        except Exception as e:
            print(f"[ERR]  Error: {e}")
            break

        if not raw_page:
            print("[OK]  Catalog exhausted.")
            break

        for raw in raw_page:
            handle = raw.get("handle", "")
            if handle in seen_all:
                continue

            product_type = raw.get("product_type", "")
            if not is_clothing(product_type):
                seen_all.add(handle)
                continue

            tags   = raw.get("tags", [])
            gender = detect_gender(tags, product_type)
            if gender is None:
                seen_all.add(handle)
                continue

            if gender == "male"   and len(men)   >= target:
                continue
            if gender == "female" and len(women) >= target:
                continue

            product = clean_product(raw, gender)

            if not args.no_images and product["image_urls"]:
                local_paths = download_images(product, resume=args.resume)
                product["local_images"] = local_paths

            img_info     = f"{len(product['local_images'])} imgs" if not args.no_images else ""
            gender_label = "? M" if gender == "male" else "? F"
            bucket_n     = len(men) + 1 if gender == "male" else len(women) + 1
            print(
                f"  [{gender_label} {bucket_n:02d}] "
                f"{product['title'][:50]:<50} "
                f"| {product_type:<20} {img_info}"
            )

            if gender == "male":
                men.append(product)
                seen_men.add(handle)
            else:
                women.append(product)
                seen_women.add(handle)
            seen_all.add(handle)

            if len(men) >= target and len(women) >= target:
                break

        save_json(men,   men_path)
        save_json(women, women_path)
        print(
            f"    ? men={len(men)}/{target}  "
            f"women={len(women)}/{target}  saved.\n"
        )

        if len(raw_page) < per_page:
            print("[OK]  End of catalog reached.")
            break

        page += 1
        time.sleep(PAGE_DELAY)

    print_summary(men, women)


if __name__ == "__main__":
    main()
