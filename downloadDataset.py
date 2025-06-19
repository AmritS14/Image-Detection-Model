import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.types as fo_types

# Top 10 COCO classes by frequency (can be customized)
top_10_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light"
]

# Download a subset of COCO-2017 train split for top 10 classes
print("Downloading COCO 2017 subset (2000 images, top 10 classes)...")
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=top_10_classes,
    max_samples=2000,
    only_matching=True,
    dataset_name="coco_top10_2000"
)

print("Sample fields:", dataset.get_field_schema().keys())
print("Total samples in dataset:", len(dataset))

# Filter to samples that have at least one detection
view = dataset.match(fo.ViewField("ground_truth.detections").length() > 0)
print("Samples with at least one detection:", len(view))

# Try exporting without filtering
export_dir = "data/coco_top10_2000"
view.export(
    export_dir=export_dir,
    dataset_type=fo_types.COCODetectionDataset,
    label_field="ground_truth"
)

print(f"Subset exported to {export_dir}")
print("Images in:", f"{export_dir}/data/")
print("Annotations:", f"{export_dir}/labels.json") 