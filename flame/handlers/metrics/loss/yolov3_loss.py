import torch
import torch.nn as nn
from typing import List, Tuple, Dict


class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        lambda_obj: int = 1,
        lambda_noobj: int = 10,
        lambda_bbox: int = 1,
        lambda_class: int = 1,
        input_size: int = 416,
        anchors: List[List[Tuple[float, float]]] = None,
    ):
        super(YOLOv3Loss, self).__init__()
        self.anchors = anchors
        self.input_size = input_size
        self.lambda_obj = lambda_obj
        self.lambda_bbox = lambda_bbox
        self.lambda_class = lambda_class
        self.lambda_noobj = lambda_noobj

    def forward(
        self,
        preds: Tuple[torch.Tensor],
        targets: Tuple[torch.Tensor]
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
        losses = {}
        for pred, target, anchor in zip(preds, targets, self.anchors):
            scale = pred.shape[2]
            anchor = torch.tensor(anchor, dtype=torch.float, device=pred.device)
            anchor /= (self.input_size / scale)  # grid_size = input_size / scale
            losses[scale] = self.compute_loss(pred, target, anchor)

        return losses

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        anchor: torch.Tensor
    ) -> float:
        '''
        Args:
            pred: N x 3 x S x S x (5 + C), 5: tp, tx, ty, tw, th
            target: N x 3 x S x S x 6, (score(-1 or 0 or 1), x, y, w, h)
            anchor: 3 x 2, relative to input_size
        Outputs:
            loss: float
        '''
        device = pred.device

        # check where obj and noobj (ignore if target == -1)
        obj = target[..., 0] == 1  # Iobj_i
        noobj = target[..., 0] == 0  # Inoobj_i
        anchor = anchor.reshape(1, 3, 1, 1, 2)  # 1 x 3 x 1 x 1 x 2

        # no object loss
        noobj_loss = nn.BCEWithLogitsLoss()(pred[..., 0:1][noobj], target[..., 0:1][noobj])

        # object loss
        bxy = torch.sigmoid(pred[..., 1:3])
        bwh = torch.exp(pred[..., 3:5]) * anchor[..., 1:3]
        pred_boxes = torch.cat([bxy, bwh], dim=-1).to(device)
        true_boxes = target[..., 1:5]
        ious = self.compute_iou(pred_boxes[obj], true_boxes[obj])

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

    def compute_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        box_format: str = "midpoint"
    ) -> torch.Tensor:
        """This function calculates intersection over union (iou) given pred boxes and target boxes.
        Args:
            boxes1 (tensor): Prediction of Bounding Boxes (BATCH_SIZE, 4)
            boxes2 (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
            box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        Output:
            tensor: Intersection over union for all examples (BATCH_SIZE, 1)
        """
        if box_format == "midpoint":
            boxes1_x1y1 = boxes1[..., 0:2] - boxes1[..., 2:4] / 2
            boxes1_x2y2 = boxes1[..., 0:2] + boxes1[..., 2:4] / 2
            boxes2_x1y1 = boxes2[..., 0:2] - boxes2[..., 2:4] / 2
            boxes2_x2y2 = boxes2[..., 0:2] + boxes2[..., 2:4] / 2

        if box_format == "corners":
            boxes1_x1y1 = boxes1[..., 0:2]
            boxes1_x2y2 = boxes1[..., 2:4]
            boxes2_x1y1 = boxes2[..., 0:2]
            boxes2_x2y2 = boxes2[..., 2:4]

        x1y1 = torch.max(boxes1_x1y1, boxes2_x1y1)
        x2y2 = torch.min(boxes1_x2y2, boxes2_x2y2)

        inter_areas = torch.prod((x2y2 - x1y1).clamp(min=0), dim=-1, keepdims=True)

        boxes1_area = torch.prod((boxes1_x2y2 - boxes1_x1y1), dim=-1, keepdims=True).abs()
        boxes2_area = torch.prod((boxes2_x2y2 - boxes2_x1y1), dim=-1, keepdims=True).abs()
        union_areas = boxes1_area + boxes2_area - inter_areas

        return inter_areas / (union_areas + 1e-6)
