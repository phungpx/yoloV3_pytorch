import cv2
import torch
import random
import numpy as np
import imgaug.augmenters as iaa

from pathlib import Path
from typing import Tuple, List
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class CoCoDataset(Dataset):
    def __init__(
        self,
        image_size: int,
        image_dir: str,
        label_path: str,
        mean: Tuple[float],
        std: Tuple[float],
        transforms: list = None,
        S: List[int] = [13, 26, 52],
        anchors: List[List[List[float]]] = None,
    ) -> None:
        super(CoCoDataset, self).__init__()
        self.S = S
        self.image_size = image_size
        self.transforms = transforms if transforms else []
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)
        self.anchors = torch.tensor(
            data=anchors[0] + anchors[1] + anchors[2],
            dtype=torch.float32,
        )  # 9 x 2, 3 anchors for each scale output tensor
        self.anchors /= image_size
        self.num_anchors = self.anchors.shape[0]  # 9
        self.num_anchors_per_scale = self.num_anchors // 3  # 3
        self.IGNORE_IOU_THRESHOLD = 0.5

        self.image_dir = Path(image_dir)
        self.coco = COCO(annotation_file=label_path)
        self.image_indices = self.coco.getImgIds()

        self.class2idx = dict()
        self.coco_label_to_label = dict()
        self.label_to_coco_label = dict()

        categories = self.coco.loadCats(ids=self.coco.getCatIds())
        categories = sorted(categories, key=lambda x: x['id'])
        for category in categories:
            self.label_to_coco_label[len(self.class2idx)] = category['id']
            self.coco_label_to_label[category['id']] = len(self.class2idx)
            self.class2idx[category['name']] = len(self.class2idx)

        self.idx2class = {class_idx: class_name for class_name, class_idx in self.class2idx.items()}

        self.pad_to_square = iaa.PadToSquare(position='right-bottom')

        print(f'{self.image_dir.stem}: {len(self.image_indices)}')
        print(f'All Classes: {self.idx2class}')

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        image, image_info = self.load_image(image_idx=idx)
        boxes, labels = self.load_annot(image_idx=idx)
        if not len(boxes) and not len(labels):
            print(f'Sample {image_info[0]} has no labels')

            # Bbox Info
            boxes_info = {
                'image_id': torch.tensor([idx], dtype=torch.int64),
                'boxes': torch.tensor([[0., 0., 1., 1.]], dtype=torch.float32),
                'labels': torch.FloatTensor([-1], dtype=torch.int64),
            }

            # Target
            # initialize targets: shape [probability, x, y, h, w, class_idx]
            targets = [torch.zeros(size=(self.num_anchors_per_scale, S, S, 6)) for S in self.S]
            for target in targets:
                target[:, :, :, 0] = -1  # probability

            # Image
            image = cv2.resize(image, dsize=(self.image_size, self.image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            sample = torch.from_numpy(np.ascontiguousarray(image))
            sample = sample.permute(2, 0, 1).contiguous()
            sample = (sample.float().div(255.) - self.mean) / self.std

            return sample, tuple(targets), image_info, boxes_info

        bboxes = [BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=label)
                  for box, label in zip(boxes, labels)]
        bboxes = BoundingBoxesOnImage(bounding_boxes=bboxes, shape=image.shape)
        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, bboxes = transform(image=image, bounding_boxes=bboxes)

        # Rescale image and bounding boxes
        image, bboxes = self.pad_to_square(image=image, bounding_boxes=bboxes)
        image, bboxes = iaa.Resize(size=self.image_size)(image=image, bounding_boxes=bboxes)

        bboxes = bboxes.on(image)

        boxes = [[bbox.x1, bbox.y1, bbox.x2, bbox.y2] for bbox in bboxes.bounding_boxes]
        labels = [bbox.label for bbox in bboxes.bounding_boxes]

        # Convert to Torch Tensor
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)  # suppose all instances are not crowd
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx], dtype=torch.int64)
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Bbox Info
        boxes_info = {
            'image_id': image_id,
            'boxes': boxes,
            'labels': labels,
            # 'area': area,
            'iscrowd': iscrowd,
        }

        # Target
        # convert from Bouding Box Object to boxes (x1, y1, x2, y2, label)
        bboxes = [[bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.label] for bbox in bboxes.bounding_boxes]

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

    def load_image(self, image_idx):
        image_info = self.coco.loadImgs(ids=self.image_indices[image_idx])[0]
        image_path = str(self.image_dir.joinpath(image_info['file_name']))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image_info = [image_path, image.shape[1::-1]]

        return image, image_info

    def load_annot(self, image_idx):
        boxes, labels = [], []
        annot_indices = self.coco.getAnnIds(imgIds=self.image_indices[image_idx], iscrowd=False)
        if not len(annot_indices):
            labels, boxes = [[-1]], [[0, 0, 1, 1]]
            return boxes, labels

        annot_infos = self.coco.loadAnns(ids=annot_indices)
        for idx, annot_info in enumerate(annot_infos):
            # some annotations have basically no width or height, skip them.
            if annot_info['bbox'][2] < 1 or annot_info['bbox'][3] < 1:
                continue

            bbox = self.xywh2xyxy(annot_info['bbox'])
            label = self.coco_label_to_label[annot_info['category_id']]
            boxes.append(bbox)
            labels.append(label)

        return boxes, labels

    def xywh2xyxy(self, box):
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        return box

    def image_aspect_ratio(self, image_idx):
        image_info = self.coco.loadImgs(self.image_indices[image_idx])[0]
        return float(image_info['width']) / float(image_info['height'])

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

    @property
    def num_classes(self):
        return len(list(self.idx2class.keys()))
