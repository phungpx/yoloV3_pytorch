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
        VOC2012: Dict[str, str] = None,
        image_extent: str = '.jpg',
        label_extent: str = '.xml',
        classes: Dict[str, int] = None,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        imsize: int = 416,
        S: List[int] = [13, 26, 52],
        anchors: List[List[List[float]]] = None,
        transforms: Optional[List] = None
    ):
        super(PascalDataset, self).__init__()
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

        # VOC2007
        image_dir, label_dir = Path(VOC2007['image_dir']), Path(VOC2007['label_dir'])
        image_paths = natsorted(list(Path(image_dir).glob(f'*{image_extent}')), key=lambda x: str(x.stem))
        label_paths = natsorted(list(Path(label_dir).glob(f'*{label_extent}')), key=lambda x: str(x.stem))
        voc2007_pairs = [[image, label] for image, label in zip(image_paths, label_paths) if image.stem == label.stem]

        # VOC2012
        image_dir, label_dir, txt_path = Path(VOC2012['image_dir']), Path(VOC2012['label_dir']), Path(VOC2012['txt_path'])
        with txt_path.open(mode='r', encoding='utf-8') as fp:
            image_names = fp.read().splitlines()

        voc2012_pairs = []
        for image_name in image_names:
            image_path = image_dir.joinpath(f'{image_name}{image_extent}')
            label_path = label_dir.joinpath(f'{image_name}{label_extent}')
            if image_path.exists() and label_path.exists():
                voc2012_pairs.append([image_path, label_path])

        self.data_pairs = voc2007_pairs + voc2012_pairs

        print(f'- {txt_path.stem}:')
        print(f'\t VOC2007: {len(voc2007_pairs)}')
        print(f'\t VOC2012: {len(voc2012_pairs)}')
        print(f'\t Total: {len(self.data_pairs)}')

    def __len__(self):
        return len(self.data_pairs)

    def _get_label_info(self, label_path: str) -> Tuple[dict, dict]:
        tree = ET.parse(str(label_path))
        image_info = {'image_name': tree.find('filename').text,
                      'height': int(tree.find('size').find('height').text),
                      'width': int(tree.find('size').find('width').text),
                      'depth': int(tree.find('size').find('depth').text)}
        label_info = []
        objects = tree.findall('object')
        for obj in objects:
            bndbox = obj.find('bndbox')
            bbox = np.int32([bndbox.find('xmin').text, bndbox.find('ymin').text,
                             bndbox.find('xmax').text, bndbox.find('ymax').text])
            label_name = obj.find('name').text
            label_info.append({'label': label_name, 'bbox': bbox})

        return image_info, label_info

    def __getitem__(self, idx):
        image_path, label_path = self.data_pairs[idx]
        _, label_info = self._get_label_info(label_path)

        image = cv2.imread(str(image_path))
        image_info = [str(image_path), image.shape[1::-1]]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = [label['bbox'] for label in label_info]
        labels = [self.classes[label['label']] for label in label_info]

        # Pad to square to keep object's ratio
        bbs = BoundingBoxesOnImage(
            [
                BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=label)
                for box, label in zip(boxes, labels)
            ],
            shape=image.shape
        )

        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, bbs = transform(image=image, bounding_boxes=bbs)

        # Rescale image and bounding boxes
        image, bbs = self.pad_to_square(image=image, bounding_boxes=bbs)
        image, bbs = iaa.Resize(size=self.imsize)(image=image, bounding_boxes=bbs)
        bbs = bbs.on(image)

        # get boxes_info
        boxes = [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs.bounding_boxes]
        labels = [bb.label for bb in bbs.bounding_boxes]

        # Convert to Torch Tensor
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)  # suppose all instances are not crowd
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Target
        boxes_info = {
            'image_id': image_id,
            'boxes': boxes,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd,
        }

        # convert from Bouding Box Object to boxes (x1, y1, x2, y2, label)
        bboxes = [[bb.x1, bb.y1, bb.x2, bb.y2, bb.label] for bb in bbs.bounding_boxes]

        # convert box type: [x1, y1, x2, y2, class] to box type: [bx/w, by/h, bw/w, bh/h, class]
        height, width = image.shape[:2]
        bboxes = self._xyxy2dcxdcydhdw(bboxes=bboxes, image_height=height, image_width=width)

        # initialize targets: shape [probability, x, y, h, w, class_idx]
        targets = [torch.zeros(size=(self.num_anchors_per_scale, S, S, 6)) for S in self.S]

        for box in bboxes:
            iou_anchors = self._iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            center_x, center_y, width, height, class_idx = box
            has_anchor = [False, False, False]  # set for 3 scales

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale  # scale_idx = 0 (13x13), 1 (26x26), 2 (52x52)
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale  # anchor_on_scale = 0, 1, 2
                S = self.S[scale_idx]
                i, j = int(S * center_y), int(S * center_x)  # which cell? Ex: S=13, center_x=0.5 --> i=int(13 * 0.5)=6
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

        # normalize image
        sample = torch.from_numpy(np.ascontiguousarray(image))
        sample = sample.permute(2, 0, 1).contiguous()
        sample = (sample.float().div(255.) - self.mean) / self.std

        return sample, tuple(targets), image_info, boxes_info

    def _dcxdcydhdw2xyxy(self, bboxes, image_height, image_width):
        '''
        Args:
            bboxes: list([[bx / iw, by / ih, bw / iw, bh / ih, class_idx], ...])
            image_height: int
            image_width: int
        Returns:
            bboxes: list([[x1, y1, x2, y2, class_idx], ...])
        '''
        bboxes = np.asarray(bboxes)
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * image_width
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * image_height
        bboxes[:, [0, 1]] = bboxes[:, [0, 1]] - bboxes[:, [2, 3]] / 2
        bboxes[:, [2, 3]] = bboxes[:, [0, 1]] + bboxes[:, [2, 3]]

        return bboxes.tolist()

    def _xyxy2dcxdcydhdw(self, bboxes, image_height, image_width):
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

    def _iou_width_height(self, boxes1, boxes2):
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
