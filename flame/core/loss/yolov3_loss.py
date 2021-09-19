import torch
import torch.nn as nn


class YOLOv3Loss(nn.Module):
    def __init__(self, lambda_obj, lambda_noobj, lambda_bbox, lambda_class):
        super(YOLOv3Loss, self).__init__()
        self.lambda_obj = lambda_obj
        self.lambda_bbox = lambda_bbox
        self.lambda_class = lambda_class
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets, anchors):
        '''
        Parameters:
            predictions: (Tensor) B x 3 x S x S x [5 + num_classes]  # 5 + num_classes: P0, x, y, w, h, num_classes
            targets: (Tensor) B x 3 x S x S x 6                      # 6: P0, x, y, h, w, class_idx
            anchors: (Tensor) 3 x 2                                  # 2: w_anchor, h_anchor
        Outputs:
            loss
        '''
        # check where obj and noobj (ignore if target == -1)
        obj = targets[..., 0] == 1  # Iobj_i
        noobj = targets[..., 0] == 0  # Inoobj_i

        # no object loss
        noobj_loss = nn.BCEWithLogitsLoss()(predictions[..., 0:1][noobj], targets[..., 0:1][noobj])

        # object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)  # 1 x 3 x 1 x 1 x 2
        bxy, bwh = torch.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors[..., 1:3]
        pred_boxes = torch.cat([bxy, bwh], dim=-1).to(predictions.device)
        true_boxes = targets[..., 1:5]
        ious = self._compute_iou(pred_boxes[obj], true_boxes[obj])

        obj_loss = nn.MSELoss()(torch.sigmoid(predictions[..., 0:1][obj]), ious * targets[..., 0:1][obj])

        # coordinate loss
        predictions[..., 1:3] = torch.sigmoid(predictions[..., 1:3])  # x,y coordinates
        targets[..., 3:5] = torch.log(1e-16 + targets[..., 3:5] / anchors)  # width, height coordinates
        bbox_loss = nn.MSELoss()(predictions[..., 1:5][obj], targets[..., 1:5][obj])

        # class loss
        class_loss = nn.CrossEntropyLoss()(predictions[..., 5:][obj], targets[..., 5][obj].long())

        # combination loss
        loss = self.lambda_bbox * bbox_loss + self.lambda_obj * obj_loss + self.lambda_noobj * noobj_loss + self.lambda_class * class_loss

        return loss

    def _compute_iou(self, boxes1, boxes2, box_format="midpoint"):
        """This function calculates intersection over union (iou) given pred boxes and target boxes.
        Parameters:
            boxes1 (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
            boxes2 (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
            box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        Returns:
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
