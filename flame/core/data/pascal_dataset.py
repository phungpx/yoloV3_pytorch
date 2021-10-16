import cv2
import torch
import random
import numpy as np
import imgaug.augmenters as iaa
import xml.etree.ElementTree as ET

from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Optional
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class PascalDataset(Dataset):
    def __init__(
        self,
        VOC2007: Dict[str, str] = None,
        image_size: int = 416,
        image_extent: str = '.jpg',
        label_extent: str = '.xml',
        classes: Dict[str, int] = None,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        transforms: Optional[List] = None
    ):
        super(PascalDataset, self).__init__()
        self.classes = classes
        self.image_size = image_size
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)
        self.pad_to_square = iaa.PadToSquare(position='right-bottom')
        self.transforms = transforms if transforms else []

        # VOC2007
        # image_dir, label_dir = Path(VOC2007['image_dir']), Path(VOC2007['label_dir'])
        # image_paths = natsorted(list(Path(image_dir).glob(f'*{image_extent}')), key=lambda x: str(x.stem))
        # label_paths = natsorted(list(Path(label_dir).glob(f'*{label_extent}')), key=lambda x: str(x.stem))
        # voc2007_pairs = [[image, label] for image, label in zip(image_paths, label_paths) if image.stem == label.stem]

        # VOC2007
        image_dir, label_dir, txt_path = Path(VOC2012['image_dir']), Path(VOC2012['label_dir']), Path(VOC2012['txt_path'])
        with txt_path.open(mode='r', encoding='utf-8') as fp:
            image_names = fp.read().splitlines()

        voc2007_pairs = []
        for image_name in image_names:
            image_path = image_dir.joinpath(f'{image_name}{image_extent}')
            label_path = label_dir.joinpath(f'{image_name}{label_extent}')
            if image_path.exists() and label_path.exists():
                voc2017_pairs.append([image_path, label_path])

        self.data_pairs = voc2007_pairs


        print(f'- {txt_path.stem}:')
        print(f'\t Total: {len(self.data_pairs)}')

    def __len__(self):
        return len(self.data_pairs)

    def get_annotation_info(self, label_path: str) -> Tuple[dict, dict]:
        tree = ET.parse(str(label_path))
        image_info = {'image_name': tree.find('filename').text,
                      'height': int(tree.find('size').find('height').text),
                      'width': int(tree.find('size').find('width').text),
                      'depth': int(tree.find('size').find('depth').text)}
        annotation_info = []
        objects = tree.findall('object')
        for obj in objects:
            bndbox = obj.find('bndbox')
            bbox = np.int32([bndbox.find('xmin').text, bndbox.find('ymin').text,
                             bndbox.find('xmax').text, bndbox.find('ymax').text])
            label_name = obj.find('name').text
            annotation_info.append({'label': label_name, 'bbox': bbox})

        return image_info, annotation_info

    def __getitem__(self, idx):
        image_path, label_path = self.data_pairs[idx]

        # Get all annotations
        _, annotation_info = self.get_annotation_info(label_path)

        # Get image, and then ensure image is RGB
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get image information with image_path and (width, height)
        image_info = [str(image_path), image.shape[1::-1]]

        if not len(annotation_info):  # Has no annotations
            # Target
            target = {
                'image_id': torch.tensor([idx], dtype=torch.int64),
                'boxes': torch.tensor([[0., 0., 1., 1.]], dtype=torch.float32),
                'areas': torch.tensor([0.], dtype=torch.float32),
                'labels': torch.tensor([-1], dtype=torch.int64),
            }

            # Sample
            image = cv2.resize(image, dsize=(self.image_size, self.image_size))
            sample = torch.from_numpy(np.ascontiguousarray(image))
            sample = sample.permute(2, 0, 1).contiguous()
            sample = (sample.float().div(255.) - self.mean) / self.std

            # Information
            print(f'Sample: {image_info[0]} has no labels')

            return sample, target, image_info

        # Get all boxes, all labels from annotation file
        boxes = [label['bbox'] for label in annotation_info]
        labels = [self.classes[label['label']] for label in annotation_info]

        # Convert to object of imgaug for augmenting image and boxes
        bbs = BoundingBoxesOnImage(
            [
                BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=label)
                for box, label in zip(boxes, labels)
            ],
            shape=image.shape
        )

        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, bbs = transform(image=image, bounding_boxes=bbs)

        # Pad to square image and then Resize image and bounding boxes
        image, bbs = self.pad_to_square(image=image, bounding_boxes=bbs)
        image, bbs = iaa.Resize(size=self.image_size)(image=image, bounding_boxes=bbs)
        bbs = bbs.on(image)

        # Get all boxes, labels and area of boxes
        boxes = [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs.bounding_boxes]
        labels = [bb.label for bb in bbs.bounding_boxes]
        areas = [(box[3] - box[1]) * (box[2] - box[0]) for box in boxes]

        # Target
        target = {
            'image_id': torch.tensor([idx], dtype=torch.int64),
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64),  # suppose all instances are not crowd
        }

        # Sample
        sample = torch.from_numpy(np.ascontiguousarray(image))
        sample = sample.permute(2, 0, 1).contiguous()
        sample = (sample.float().div(255.) - self.mean) / self.std

        return sample, target, image_info
