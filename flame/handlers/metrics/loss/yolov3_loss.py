import torch
import torch.nn as nn

from collections import defaultdict
from typing import List, Tuple, Dict


class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        lambda_obj: int = 1,
        lambda_noobj: int = 10,
        lambda_bbox: int = 1,
        lambda_class: int = 1,
        image_size: int = 416,
        scales: List[int] = [13, 26, 52],
        anchor_sizes: List[List[Tuple[float, float]]] = None,
    ):
        super(YOLOv3Loss, self).__init__()
        self.lambda_obj = lambda_obj
        self.lambda_bbox = lambda_bbox
        self.lambda_class = lambda_class
        self.lambda_noobj = lambda_noobj

        self.scales = scales
        self.image_size = image_size
        self.anchor_sizes = torch.tensor(anchor_sizes, dtype=torch.float32)
        self.ignore_iou_threshold = 0.5

    def forward(
        self,
        preds: Tuple[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[int, float]:
        '''
        Args:
            preds: Tuple(Tensor)
                N x 3 x S1 x S1 x (5 + C)
                N x 3 x S2 x S2 x (5 + C)
                N x 3 x S3 x S3 x (5 + C)
            targets: Tuple(Tensor)
                N x 3 x S1 x S1 x 6
                N x 3 x S2 x S2 x 6
                N x 3 x S3 x S3 x 6
        Outputs:
            losses: dictionary
            {
                13: float,
                26: float,
                52: float,
            }
        '''
        self.device = preds[0].device
        self.anchor_sizes = self.anchor_sizes.to(self.device)

        scale_targets = defaultdict(list)
        for target in targets:
            for i, scale_target in enumerate(self.convert_target(target=target)):
                scale_targets[i].append(scale_target)

        targets = [torch.stack(scale_target, dim=0) for scale_target in scale_targets.values()]

        losses = {}
        for pred, target, anchor in zip(preds, targets, self.anchor_sizes):
            losses[pred.shape[2]] = self.compute_loss(pred, target, anchor)

        return losses

    def xyxy2xywh(self, boxes):
        '''convert box type from x1, y1, x2, y2 to cx, cy, w, h
        Args:
            boxes: [num_boxes, 4], type of box: (x1, y1, x2, y2)
        Output:
            boxes: [num_boxes, 4], type of box: (cx, cy, w, h)
        '''
        boxes[..., [2, 3]] = torch.clamp(boxes[..., [2, 3]] - boxes[..., [0, 1]], min=1.)
        boxes[..., [0, 1]] = boxes[..., [0, 1]] + boxes[..., [2, 3]] / 2
        return boxes

    def xywh2xyxy(self, boxes):
        '''convert box type from cx, cy, w, h to x1, y1, x2, y2
        Args:
            boxes: [num_boxes, 4], type of box: (cx, cy, w, h)
        Outputs:
            boxes: [num_boxes, 4], type of box: (x1, y1, x2, y2)
        '''
        boxes[..., [0, 1]] = torch.clamp(boxes[..., [0, 1]] - boxes[..., [2, 3]] / 2, min=0.)
        boxes[..., [2, 3]] = boxes[..., [0, 1]] + boxes[..., [2, 3]]
        return boxes

    def convert_target(self, target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Args:
            target: dictionary
                {
                    'boxes': Float32 Tensor[N, 4], box type [x1, y1, x2, y2],
                    'labels': Int32 Tensor[N],
                    'areas': Float32 Tensor[N],
                    'image_id': Int32 Tensor[N],
                    'is_crowed': Int32 Tensor[N],
                }
        Outputs: tuple of three tensors
            (
                num_anchors_per_scale x S1 x S1 x 6,  6: [prob, x, y, h, w, label]
                num_anchors_per_scale x S2 x S2 x 6,
                num_anchors_per_scale x S2 x S2 x 6,
            )
        '''
        # Get all boxes, labels from target
        boxes, labels = target['boxes'], target['labels']
        boxes = self.xyxy2xywh(boxes)

        # initialize targets: Tuple[Tensor[num_anchors_per_scale, S, S, 6]]
        targets = [
            torch.zeros(size=(self.anchor_sizes[i].shape[0], self.scales[i], self.scales[i], 6), device=self.device)
            for i in range(len(self.scales))
        ]

        for scale_id in range(len(self.scales)):
            anchor = self.anchor_sizes[scale_id]  # num_anchors_per_scale x 2
            pw, ph = anchor[:, 0], anchor[:, 1]
            grid_size = self.image_size // self.scales[scale_id]

            for box, label in zip(boxes, labels):
                bx, by, bw, bh = box[0], box[1], box[2], box[3]
                inter_area = torch.min(bw, pw) * torch.min(bh, ph)
                union_area = bw * bh + pw * ph - inter_area
                ious = inter_area / union_area
                anchor_indices = ious.argsort(descending=True, dim=0)  # anchor_indices = 0, 1, 2

                for anchor_id in anchor_indices:
                    j, i = int(bx // grid_size), int(by // grid_size)  # which cell? Ex: S=13, cx=0.5 --> i=int(13 * 0.5)=6
                    anchor_taken = targets[scale_id][anchor_id, i, j, 0]

                    if not anchor_taken:
                        cell_w, cell_h = bw / grid_size, bh / grid_size
                        cell_x, cell_y = bx / grid_size - j, by / grid_size - i  # both are between [0, 1]
                        cell_box = torch.tensor([cell_x, cell_y, cell_w, cell_h])

                        targets[scale_id][anchor_id, i, j, 0] = 1  # probability
                        targets[scale_id][anchor_id, i, j, 1:5] = cell_box  # [x, y, w, h]
                        targets[scale_id][anchor_id, i, j, 5] = label.item()  # class_idx

                    elif not anchor_taken and ious[anchor_id] > self.ignore_iou_threshold:
                        targets[scale_id][anchor_id, i, j, 0] = -1  # ignore prediction

        return targets

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, anchor: torch.Tensor) -> float:
        '''
        Args:
            pred: N x 3 x S x S x (5 + C), 5: tp, tx, ty, tw, th
            target: N x 3 x S x S x 6, (score(-1 or 0 or 1), x, y, w, h)
            anchor: 3 x 2, relative to image_size
        Outputs:
            loss: float
        '''
        scale = pred.shape[2]
        grid_size = self.image_size / scale
        anchor = anchor / grid_size

        # check where obj and noobj (ignore if target == -1)
        obj = target[..., 0] == 1  # Iobj_i
        noobj = target[..., 0] == 0  # Inoobj_i
        anchor = anchor.reshape(1, 3, 1, 1, 2)  # 1 x 3 x 1 x 1 x 2

        # no object loss
        noobj_loss = nn.BCEWithLogitsLoss()(pred[..., 0:1][noobj], target[..., 0:1][noobj])

        # object loss
        bxy = torch.sigmoid(pred[..., 1:3])
        bwh = torch.exp(pred[..., 3:5]) * anchor[..., 1:3]
        pred_boxes = torch.cat([bxy, bwh], dim=-1)
        true_boxes = target[..., 1:5]
        ious = self.compute_iou(
            boxes1=self.xywh2xyxy(pred_boxes[obj]),
            boxes2=self.xywh2xyxy(true_boxes[obj])
        )
        obj_loss = nn.MSELoss()(torch.sigmoid(pred[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # coordinate loss
        pred[..., 1:3] = torch.sigmoid(pred[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(1e-16 + target[..., 3:5] / anchor)  # width, height coordinates
        bbox_loss = nn.MSELoss()(pred[..., 1:5][obj], target[..., 1:5][obj])

        # class loss
        class_loss = nn.CrossEntropyLoss()(pred[..., 5:][obj], target[..., 5][obj].long())

        # combination loss
        loss = (
            self.lambda_bbox * bbox_loss
            + self.lambda_obj * obj_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_class * class_loss
        )

        return loss

    def compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        '''
        args:
            boxes1: [N, 4], box_type: [x1, y1, x2, y2]
            boxes2: [N, 4], box_type: [x1, y1, x2, y2]
        output:
            ious: [N, 1]
        references: https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        '''
        # calculate intersection areas of boxes1 and boxes2
        inter_w = torch.min(boxes1[..., 2], boxes2[..., 2]) - torch.max(boxes1[..., 0], boxes2[..., 0])
        inter_h = torch.min(boxes1[..., 3], boxes2[..., 3]) - torch.max(boxes1[..., 1], boxes2[..., 1])
        inter_w = torch.clamp(inter_w, min=0.)  # N
        inter_h = torch.clamp(inter_h, min=0.)  # N
        inter_areas = inter_w * inter_h  # N

        # calculate union areas of boxes1 and boxes2
        area_boxes1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # N
        area_boxes2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # N
        union_areas = area_boxes1 + area_boxes2 - inter_w * inter_h  # N
        union_areas = torch.clamp(union_areas, min=1e-8)

        # calculate ious of boxes1 and boxes2
        ious = inter_areas / union_areas

        return ious.unsqueeze(dim=1)
