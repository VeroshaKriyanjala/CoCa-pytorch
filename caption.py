from collections import defaultdict

captions_dict = defaultdict(list)
for ann in data["annotations"]:
    captions_dict[ann["image_id"]].append(ann["caption"])

# Example: print captions for first image
first_img_id = data["images"][0]["id"]
print("Captions for first image:")
print(captions_dict[first_img_id])
