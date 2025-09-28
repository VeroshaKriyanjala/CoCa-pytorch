import os, json, random, requests
from tqdm import tqdm
from collections import defaultdict

# --- Configuration ---
JSON_PATH = "nocap_val_4500_captions.json" # Your uploaded file
OUT_DIR   = "nocaps_val_images_50"
LIMIT     = 50
TIMEOUT_S = 30

# --- Setup ---
os.makedirs(OUT_DIR, exist_ok=True)
sess = requests.Session()
# Always use a non-default user agent
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

# --- Load JSON ---
# Since you uploaded the file, this will now load successfully.
with open(JSON_PATH, "r") as f:
    data = json.load(f)

images = data["images"]
random.seed(42)
# Select the first LIMIT images for deterministic testing
subset = images[:LIMIT]

# --- Captions map (image_id -> list[captions]) ---
caps = defaultdict(list)
for ann in data["annotations"]:
    caps[ann["image_id"]].append(ann["caption"])

def candidates(img):
    """Generates a list of candidate URLs for the image."""
    fname = img["file_name"]
    urls = []
    # 1. nocaps S3 (likely 403, but try)
    urls.append(f"https://s3.amazonaws.com/nocaps/val/{fname}")
    # 2. original COCO/Appen proxy (often 403 or HTML)
    if "coco_url" in img:
        urls.append(img["coco_url"])
    # 3. OpenImages public mirrors (best chance for validation images)
    if "open_images_id" in img:
        oid = img["open_images_id"]
        urls.append(f"https://storage.googleapis.com/openimages/2018_04/validation/{oid}.jpg")
        urls.append(f"https://storage.googleapis.com/openimages/web/validation/{oid}.jpg")
    return urls

def download_image(url, dest):
    """
    Attempts to download an image from a URL with robust error checking.
    Logs specific HTTP status codes on failure.
    """
    try:
        r = sess.get(url, headers=HEADERS, timeout=TIMEOUT_S, stream=True)
        
        # --- FIX: Raise a more descriptive error if request fails (e.g., 403, 404) ---
        r.raise_for_status() 
        
        ctype = r.headers.get("Content-Type","").lower()
        if not ctype.startswith("image/"):
            raise RuntimeError(f"Non-image content-type: {ctype}")

        # write with size guard
        size = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1<<15):
                if chunk:
                    size += len(chunk)
                    f.write(chunk)
        if size < 10_000:  # tiny files are likely error pages/errors
            raise RuntimeError(f"File too small ({size} bytes)")
        return True
    
    # Catch specific HTTP errors (like 403 Forbidden or 404 Not Found)
    except requests.exceptions.HTTPError as e:
        print(f"[WARN] {url} -> HTTP Error: {e}")
    # Catch other errors (Timeout, Connection, RuntimeError from size/ctype)
    except Exception as e:
        print(f"[WARN] {url} -> {type(e).__name__}: {e}")
    
    # Clean up any partial file on failure
    if os.path.exists(dest):
        try: os.remove(dest)
        except Exception: pass
        
    return False

# --- Main Download Loop ---
ok, fail, failed_ids = 0, 0, []
for img in tqdm(subset, desc=f"Downloading {LIMIT} images"):
    dest = os.path.join(OUT_DIR, img["file_name"])
    
    # Skip if file already exists
    if os.path.exists(dest):
        ok += 1
        continue

    got = False
    for url in candidates(img):
        if download_image(url, dest):
            got = True
            ok += 1
            break # Stop trying URLs for this image once successful
            
    if not got:
        fail += 1
        # Use Open Images ID if available, otherwise the file name
        failed_ids.append(img.get("open_images_id", img["file_name"]))

# --- Summary ---
print(f"\nFinished. ok={ok}, fail={fail}")
if failed_ids:
    print("\n----------------------------------------------------")
    print("⚠️ Download Failures: The sources are largely unreliable.")
    print("----------------------------------------------------")
    print("For a complete dataset, use the official **Open Images Downloader** with these IDs.")
    print(f"\nFailed Open Images IDs (total: {len(failed_ids)}):")
    print(failed_ids[:10], "...")
    print("\n")

# quick caption sanity check
print("\n--- Sanity Check ---")
for img in subset[:3]:
    print(f"{img['file_name']} (ID: {img['id']}) -> {len(caps[img['id']])} captions")