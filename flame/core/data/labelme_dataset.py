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
        imsize: int = 416,
        image_patterns: List[str] = ['*.jpg'],
        label_patterns: List[str] = ['*.json'],
        classes: Dict[str, int] = None,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        transforms: Optional[List] = None,
        S: List[int] = [13, 26, 52],
        anchors: List[List[List[float]]] = None,
    ) -> None:
        super(LabelmeDataset, self).__init__()
        self.S = S
        self.imsize = imsize
        self.classes = classes
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)
        self.pad_to_square = iaa.PadToSquare(position='right-bottom')

        self.anchors = torch.tensor(
            data=anchors[0] + anchors[1] + anchors[2],
            dtype=torch.float32,
        )  # 9 x 2, 3 anchors for each scale output tensor
        self.anchors /= imsize
        self.num_anchors = self.anchors.shape[0]  # 9
        self.num_anchors_per_scale = self.num_anchors // 3  # 3
        self.IGNORE_IOU_THRESHOLD = 0.5

        self.transforms = transforms if transforms else []

        image_paths, label_paths = [], []
        for dirname in dirnames:
            for image_pattern in image_patterns:
                image_paths.extend(Path(dirname).glob('**/{}'.format(image_pattern)))
            for label_pattern in label_patterns:
                label_paths.extend(Path(dirname).glob('**/{}'.format(label_pattern)))

        image_paths = natsorted(image_paths, key=lambda x: str(x.stem))
        label_paths = natsorted(label_paths, key=lambda x: str(x.stem))

        self.data_pairs = [[image, label] for image, label in zip(image_paths, label_paths) if image.stem == label.stem]

        print(f'{Path(dirnames[0]).parent.stem}: {len(self.data_pairs)}')

    def __len__(self):
        return len(self.data_pairs)

    def _get_label_info(self, lable_path: str, classes: dict) -> Dict:
        with open(file=lable_path, mode='r', encoding='utf-8') as f:
            json_info = json.load(f)

        label_info = []
        for shape in json_info['shapes']:
            label = shape['label']
            points = shape['points']
            if label in self.classes and len(points) > 0:
                x1 = min([point[0] for point in points])
                y1 = min([point[1] for point in points])
                x2 = max([point[0] for point in points])
                y2 = max([point[1] for point in points])
                bbox = (x1, y1, x2, y2)

                label_info.append({'label': self.classes[label], 'bbox': bbox})

        if not len(label_info):
            label_info.append({'label': -1, 'bbox': (0, 0, 1, 1)})

        return label_info

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict, Tuple[str, Tuple[int, int]]]:
        image_path, label_path = self.data_pairs[idx]
        label_info = self._get_label_info(lable_path=str(label_path), classes=self.classes)
        image = cv2.imread(str(image_path))
        image_info = (str(image_path), image.shape[1::-1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(label_info) == 1 and label_info[0]['label'] == -1:
            print(f'Sample {image_info[0]} has no labels')

            # Bbox Info
            boxes_info = {
                'image_id': torch.tensor([idx], dtype=torch.int64),
                'boxes': torch.tensor([[0., 0., 1., 1.]], dtype=torch.float32),
                'labels': torch.tensor([-1], dtype=torch.int64),
            }

            # Target
            # initialize targets: shape [probability, x, y, h, w, class_idx]
            targets = [torch.zeros(size=(self.num_anchors_per_scale, S, S, 6)) for S in self.S]
            for target in targets:
                target[:, :, :, 0] = -1  # probability

            # Image
            image = cv2.resize(image, dsize=(self.imsize, self.imsize))
            sample = torch.from_numpy(np.ascontiguousarray(image))
            sample = sample.permute(2, 0, 1).contiguous()
            sample = (sample.float().div(255.) - self.mean) / self.std

            return sample, tuple(targets), image_info, boxes_info

        boxes = [label['bbox'] for label in label_info]
        labels = [label['label'] for label in label_info]

        # Pad to square to keep object's ratio
        bbs = BoundingBoxesOnImage([BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=label)
                                    for box, label in zip(boxes, labels)], shape=image.shape)
        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, bbs = transform(image=image, bounding_boxes=bbs)

        # Rescale image and bounding boxes
        image, bbs = self.pad_to_square(image=image, bounding_boxes=bbs)
        image, bbs = iaa.Resize(size=self.imsize)(image=image, bounding_boxes=bbs)
        bbs = bbs.on(image)

        # Convert from Bouding Box Object to boxes, labels list
        boxes = [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs.bounding_boxes]
        labels = [bb.label for bb in bbs.bounding_boxes]

        # Convert to Torch Tensor
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)  # suppose all instances are not crowd
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Boxes Info
        boxes_info = {
            'image_id': image_id,
            'boxes': boxes,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd,
        }

        # Target
        # convert from Bouding Box Object to boxes (x1, y1, x2, y2, label)
        bboxes = [[bb.x1, bb.y1, bb.x2, bb.y2, bb.label] for bb in bbs.bounding_boxes]

        # convert box type: [x1, y1, x2, y2, class] to box type: [bx/w, by/h, bw/w, bh/h, class]
        height, width = image.shape[:2]
        bboxes = self.xyxy2dcxdcydhdw(bboxes=bboxes, image_height=height, image_width=width)

        # initialize targets: shape [probability, x, y, h, w, class_idx]
        targets = [torch.zeros(size=(self.num_anchors_per_scale, S, S, 6)) for S in self.S]

        for box in bboxes:
            iou_anchors = self.iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            center_x, center_y, width, height, class_idx = box
            has_anchor = [False, False, False]  # set for 3 scales

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale  # scale_idx = 0 (13x13), 1 (26x26), 2 (52x52)
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale  # anchor_on_scale = 0, 1, 2
                S = self.S[scale_idx]
                i, j = min(int(S * center_y), S - 1), min(int(S * center_x), S - 1)  # which cell? Ex: S=13, center_x=0.5 --> i=int(13 * 0.5)=6
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1  # probability

                    cell_w, cell_h = S * width, S * height
                    cell_x, cell_y = S * center_x - j, S * center_y - i  # both are between [0, 1]
                    bounding_box = torch.tensor([cell_x, cell_y, cell_w, cell_h])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = bounding_box  # [x, y, w, h]

                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_idx)  # class_idx
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.IGNORE_IOU_THRESHOLD:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        # Sample
        sample = torch.from_numpy(np.ascontiguousarray(image))
        sample = sample.permute(2, 0, 1).contiguous()
        sample = (sample.float().div(255.) - self.mean) / self.std

        return sample, tuple(targets), image_info, boxes_info

    def xyxy2dcxdcydhdw(self, bboxes, image_height, image_width):
        '''
        Args:
            bboxes: list([[x1, y1, x2, y2, class_idx], ...])
            image_height: int
            image_width: int
        Returns:
            bboxes: list([[bx / iw, by / ih, bw / iw, bh / ih, class_idx], ...])
        '''
        bboxes = np.asarray(bboxes)
        bboxes[:, [2, 3]] = bboxes[:, [2, 3]] - bboxes[:, [0, 1]]
        bboxes[:, [0, 1]] = bboxes[:, [0, 1]] + bboxes[:, [2, 3]] / 2
        bboxes[:, [0, 2]] /= image_width
        bboxes[:, [1, 3]] /= image_height

        return bboxes.tolist()

    def iou_width_height(self, boxes1, boxes2):
        '''
        Args:
            boxes1: Tensor width and height of the first bounding boxes
            boxes2: Tensor width and height of the second bounding boxes
        Returns:
            ious: Tensor Intersection over Union of the corresponding boxes
        '''
        inter_area = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(boxes1[..., 1], boxes2[..., 1])
        union_area = boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - inter_area

        return inter_area / union_area
