import json
from collections import defaultdict
import random

# Load original COCO annotation
with open('data/coco_top10_2000/labels.json', 'r') as f:
    coco = json.load(f)

# Set the number of instances per class
N = 150

# Get the actual top 10 category IDs from the annotation file
top10_ids = [cat['id'] for cat in coco['categories']]

# Build a mapping from class_id to annotation indices
class_to_anns = defaultdict(list)
for ann in coco['annotations']:
    if ann['category_id'] in top10_ids:
        class_to_anns[ann['category_id']].append(ann)

# Sample N annotations per class
selected_anns = []
selected_img_ids = set()
for class_id in top10_ids:
    anns = class_to_anns[class_id]
    if len(anns) < N:
        print(f"Warning: class {class_id} has only {len(anns)} instances, using all.")
        sampled = anns
    else:
        sampled = random.sample(anns, N)
    selected_anns.extend(sampled)
    selected_img_ids.update([ann['image_id'] for ann in sampled])

# Filter images to only those used
selected_images = [img for img in coco['images'] if img['id'] in selected_img_ids]

# Build new COCO dict
new_coco = {
    'images': selected_images,
    'annotations': selected_anns,
    'categories': [cat for cat in coco['categories'] if cat['id'] in top10_ids]
}

# Save new annotation file
with open('data/coco_top10_2000/labels_balanced.json', 'w') as f:
    json.dump(new_coco, f)

print("Balanced annotation file created: data/coco_top10_2000/labels_balanced.json")