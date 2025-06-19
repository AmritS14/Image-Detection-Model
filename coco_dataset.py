import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import os

class COCODetectionDataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        # Dynamically build category mapping and class names from annotation file
        self.cat_ids = [cat['id'] for cat in self.coco.dataset['categories']]
        self.cat_id_to_idx = {cat_id: idx+1 for idx, cat_id in enumerate(self.cat_ids)}
        self.idx_to_cat_id = {idx+1: cat_id for idx, cat_id in enumerate(self.cat_ids)}
        self.class_names = [cat['name'] for cat in self.coco.dataset['categories']]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in self.cat_id_to_idx:
                continue  # skip labels not in selected categories
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[cat_id])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)