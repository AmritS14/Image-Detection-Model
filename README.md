# ImageDetection: COCO Top-10 Object Detection

This project provides an end-to-end pipeline for object detection using a subset of the COCO dataset (top 10 classes), including dataset preparation, model training, evaluation, and inference with PyTorch and torchvision.

## Features
- Custom COCO-format dataset loader with dynamic class mapping
- Training and validation scripts for Faster R-CNN
- Dataset balancing utility
- Inference and visualization with class names

---

## Setup

1. Install the required Python packages
```
pip install -r requirements.txt
```
2. Run `downloadDataset.py` to get the COCO dataset (subset with top 10 classes consisting of ~2000 images) and store it in the `data/` directory
3. Run `createBalanced.py` to balance the classes in the dataset
3. Run all the scripts in the `notebooks/` directory to train the model, evaluate it, and perform inference

---

## Usage of Main Functions and Scripts

### Dataset Loading
- The dataset loader automatically reads the class names and category mappings from your annotation file. This ensures that the labels and class names always match your data, regardless of the order or IDs in the annotation file.
- Use this loader whenever you need to access your dataset for training, validation, or inference.

### Model Training
- The training script (provided as a Jupyter notebook) sets up the object detection model, loads your dataset, and runs the training and validation loops.
- The model is configured to match the number of classes in your dataset, and the training process will save model checkpoints for later use.

### Dataset Balancing
- If your dataset is imbalanced (some classes have many more samples than others), use the balancing script to create a new annotation file (`labels_balanced.json`)with an equal number of samples per class. This helps prevent the model from being biased toward more frequent classes.
- The script works with any COCO-format annotation file and automatically adapts to your class structure.

### Inference and Visualization
- After training, you can load a saved model checkpoint and run inference on new images.
- The inference process uses the class names from your dataset to display human-readable labels on the detected objects in the images.
- Visualization tools are provided to draw bounding boxes and class names on the images for easy inspection of results.

---

## Troubleshooting

- If you see incorrect class names during inference, make sure you are using the dataset loader's dynamic class mapping and not any hardcoded class lists.
- If your model is biased toward certain classes, rebalance your dataset and retrain.
- Always use the class names provided by your dataset instance for mapping label indices to class names.

---

## Requirements
- Python 3.8 or higher
- torch
- torchvision
- pycocotools
- matplotlib
- numpy

---

## License
MIT License 