import torch
import random
import numpy as np
import pandas as pd
import imgaug.augmenters as iaa

from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLOv3Dataset(Dataset):
    def __init__(self, image_dir, label_dir, csv_path, image_size, anchors, S, C, transforms=None):
        super(YOLOv3Dataset, self).__init__()
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.annotations = pd.read_csv(str(csv_path))
        self.transforms = transforms if transforms is not None else []

        self.S = S  # [13, 26, 52]
        self.C = C  # 20 if using VOC dataset
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # 3 anchors for each scale output tensor
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3

        self.image_size = image_size
        self.IGNORE_IOU_THRESHOLD = 0.5
        self.pad_to_square = iaa.PadToSquare(position='right-bottom')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_path = str(self.image_dir.joinpath(self.annotations.iloc[idx, 0]))
        label_path = str(self.label_dir.joinpath(self.annotations.iloc[idx, 1]))
        image = np.array(Image.open(fp=image_path, mode='r').convert('RGB'))
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), shift=4, axis=1).tolist()  # [bx/w, by/h, bw/w, bh/h, class]

        # get infomation of image
        height, width = image.shape[:2]
        # print(f'height: {height}, width: {width}')
        image_info = [image_path, (width, height)]

        # convert box type: [bx/w, by/h, bw/w, bh/h, class] to box type: [x1, y1, x2, y2, class]
        bboxes = self._dcxdcydhdw2xyxy(bboxes=bboxes, image_height=height, image_width=width)

        # Pad to square to keep object's ratio
        bbs = BoundingBoxesOnImage([BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3], label=bbox[4])
                                    for bbox in bboxes], shape=image.shape)
        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, bbs = transform(image=image, bounding_boxes=bbs)

        # Rescale image and bounding boxes
        image, bbs = self.pad_to_square(image=image, bounding_boxes=bbs)
        sample, bbs = iaa.Resize(size=self.image_size)(image=image, bounding_boxes=bbs)
        bbs = bbs.on(sample)

        # convert from Bouding Box Object to boxes
        bboxes = [[bb.x1, bb.y1, bb.x2, bb.y2, bb.label] for bb in bbs.bounding_boxes]

        # convert box type: [x1, y1, x2, y2, class] to box type: [bx/w, by/h, bw/w, bh/h, class]
        height, width = sample.shape[:2]
        bboxes = self._xyxy2dcxdcydhdw(bboxes=bboxes, image_height=height, image_width=width)

        # initialize targets: shape [probability, x, y, h, w, class_idx]
        targets = [torch.zeros(size=(self.num_anchors // 3, S, S, 6)) for S in self.S]

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
                # print(f'S = {S}, i = {i}, j = {j}, cx = {center_x}, cy = {center_y}')
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
        sample = torch.from_numpy(np.ascontiguousarray(sample))
        sample = sample.permute(2, 0, 1).contiguous()
        sample = sample.float().div(255.)

        return sample, tuple(targets), image_info

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
