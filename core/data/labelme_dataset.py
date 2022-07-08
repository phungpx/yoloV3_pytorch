import cv2
import json
import torch
import random
import numpy as np
import imgaug.augmenters as iaa

from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Optional
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class LabelmeDataset(Dataset):
    def __init__(
        self,
        dirnames: List[str] = None,
        image_size: int = 416,
        image_patterns: List[str] = ['*.jpg'],
        label_patterns: List[str] = ['*.json'],
        classes: Dict[str, int] = None,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        transforms: Optional[List] = None,
    ) -> None:
        super(LabelmeDataset, self).__init__()
        self.classes = classes
        self.image_size = image_size
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)       # convert std and mean to tensor then reshape to math with shape of input tensor
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)
        self.pad_to_square = iaa.PadToSquare(position='right-bottom')       # pad 0 values to input image to shape is square. And just pad for right or bottom  
        self.transforms = transforms if transforms else []

        image_paths, label_paths = [], []
        for dirname in dirnames:
            for image_pattern in image_patterns:
                image_paths.extend(Path(dirname).glob('**/{}'.format(image_pattern)))       # add path of all images to image_paths list
            for label_pattern in label_patterns:
                label_paths.extend(Path(dirname).glob('**/{}'.format(label_pattern)))       # add path of all labels to image_paths list

        image_paths = natsorted(image_paths, key=lambda x: str(x.stem))                     # sort list follow name
        label_paths = natsorted(label_paths, key=lambda x: str(x.stem))

        self.data_pairs = [[image, label] for image, label in zip(image_paths, label_paths) if image.stem == label.stem]        # zip each other image and label to data_pair list
                                                                                                                    

    def __len__(self):
        return len(self.data_pairs)

    def _get_label_info(self, lable_path: str, classes: dict) -> Dict:          # this function take information of label (pre-process label).
        with open(file=lable_path, mode='r', encoding='utf-8') as f:
            json_info = json.load(f)

        label_info = []
        for shape in json_info['shapes']:
            label = shape['label']              # take label name
            points = shape['points']            # take point is a list of [x1, y1] and [x2, y2].
            if label in self.classes and len(points) > 0:
                x1 = min([point[0] for point in points])
                y1 = min([point[1] for point in points])
                x2 = max([point[0] for point in points])
                y2 = max([point[1] for point in points])
                bbox = (x1, y1, x2, y2)

                label_info.append({'label': self.classes[label], 'bbox': bbox})         # [{'label': a, 'bbox': (x1, y1, x2, y2)}, ...] 

        if not len(label_info):
            label_info.append({'label': -1, 'bbox': (0, 0, 1, 1)})          # if image has not object label_info will just has a dict {'label': -1, 'bbox': (0, 0, 1, 1)}

        return label_info

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict, Tuple[str, Tuple[int, int]]]:
        image_path, label_path = self.data_pairs[idx]           # take path of a image and a corresponding label
        label_info = self._get_label_info(lable_path=str(label_path), classes=self.classes)     # use _get_label_info function to take info of bboxes in sample
        image = cv2.imread(str(image_path))                     # use cv2 to take info of image (np.array - shape: h, w, c)
        image_info = (str(image_path), image.shape[1::-1])      
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # change channels from BGR to RGB

        if len(label_info) == 1 and label_info[0]['label'] == -1:       # in this case: image has not object

            # Target
            target = {
                'image_id': torch.tensor([idx], dtype=torch.int64),
                'boxes': torch.tensor([[0., 0., 1., 1.]], dtype=torch.float32),
                'labels': torch.tensor([-1], dtype=torch.int64),
            }

            # Sample
            image = cv2.resize(image, dsize=(self.image_size, self.image_size))     # resize to input size
            sample = torch.from_numpy(np.ascontiguousarray(image))                  # convert np.array to torch.tensor
            sample = sample.permute(2, 0, 1).contiguous()                           # permute (h, w, c) to (c, h, w)
            sample = (sample.float().div(255.) - self.mean) / self.std              # do normalize

            return sample, target, image_info

        boxes = [label['bbox'] for label in label_info]         # take bboxes from label_info
        labels = [label['label'] for label in label_info]       # take label from label_info

        # Pad to square to keep object's ratio
        bbs = BoundingBoxesOnImage([BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=label)
                                    for box, label in zip(boxes, labels)], shape=image.shape)           # zip bboxes on image to prepare for transform
        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, bbs = transform(image=image, bounding_boxes=bbs)         # implement transforms (data augementation)

        # Rescale image and bounding boxes
        image, bbs = self.pad_to_square(image=image, bounding_boxes=bbs)        # pad 0 values to input image to shape is square. And just pad for right or bottom 
        image, bbs = iaa.Resize(size=self.image_size)(image=image, bounding_boxes=bbs)      # resize to input size of sample
        bbs = bbs.on(image)

        # Convert from Bouding Box Object to boxes, labels list
        boxes = [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs.bounding_boxes]     # take bboxes after transforms
        labels = [bb.label for bb in bbs.bounding_boxes]                        # take labels

        # Convert to Torch Tensor
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)  # suppose all instances are not crowd
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Boxes Info
        target = {
            'image_id': image_id,
            'boxes': boxes,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd,
        }

        # Sample
        sample = torch.from_numpy(np.ascontiguousarray(image))
        sample = sample.permute(2, 0, 1).contiguous()
        sample = (sample.float().div(255.) - self.mean) / self.std

        return sample, target, image_info
