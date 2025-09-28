import json
import os
import requests
from tqdm import tqdm

# Path to your JSON file
json_path = "nocap_val_4500_captions.json"
output_dir = "nocaps_val_images"
os.makedirs(output_dir, exist_ok=True)

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Extract images
images = data["images"][:50]

for img in tqdm(images, desc="Downloading images"):
    url = img["coco_url"]
    filename = os.path.join(output_dir, img["file_name"])
    
    if not os.path.exists(filename):  # skip if already downloaded
        try:
            r = requests.get(url, timeout=10)
            with open(filename, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

from collections import defaultdict

captions_dict = defaultdict(list)
for ann in data["annotations"]:
    captions_dict[ann["image_id"]].append(ann["caption"])

# Example: print captions for first image
first_img_id = data["images"][0]["id"]
print("Captions for first image:")
print(captions_dict[first_img_id])
