import torch
import torch.nn as nn


import torch
from . import loss


class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        scales: List[int] = [13, 26, 52],
        input_size: Tuple[int, int] = (416, 416),  # W x H
        anchors: List[List[Tuple[float, float]]] = [
            [[0.28, 0.22], [0.38, 0.48], [0.9, 0.78]],  # S1 = 13
            [[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]],  # S2 = 26
            [[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]],  # S3 = 52
        ],
        lambda_obj: int = 1,
        lambda_noobj: int = 10,
        lambda_bbox: int = 1,
        lambda_class: int = 1,
        device: str = 'cpu',
    ):
        super(YOLOv3Loss, self).__init__()
        self.scales = scales
        self.anchors = anchors
        self.input_size = input_size
        self.lambda_obj = lambda_obj
        self.lambda_bbox = lambda_bbox
        self.lambda_class = lambda_class
        self.lambda_noobj = lambda_noobj

    def convert_targets(self, target, anchors):
        anchors = []
        # convert box type: [x1, y1, x2, y2, class] to box type: [bx/w, by/h, bw/w, bh/h, class]
        boxes = self._xyxy2dcxdcydhdw(
            boxes=target['boxes'], image_height=self.input_size[1], image_width=self.input_size[0]
        )

        # initialize targets: shape [probability, x, y, h, w, class_idx]
        targets = [torch.zeros(size=(3, S, S, 6)) for S in self.S]

        for box in boxes:
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


    def forward(self, preds, targets):
        '''
        Args:
            preds: Tuple(Tensor)
                N x 3 x S1 x S1 x (5 + C), 5: tp, tx, ty, tw, th
                N x 3 x S1 x S1 x (5 + C)
                N x 3 x S1 x S1 x (5 + C)
            targets: List(Dict)
                {'boxes': Tensor[n x 4], 'labels': Tensor[n]}
        Outputs:
            loss: float
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

    def _dcxdcydhdw2xyxy(self, boxes, image_height, image_width):
        '''
        Args:
            boxes: list([[bx / iw, by / ih, bw / iw, bh / ih, class_idx], ...])
            image_height: int
            image_width: int
        Returns:
            boxes: list([[x1, y1, x2, y2, class_idx], ...])
        '''
        boxes = np.asarray(boxes)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * image_width
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * image_height
        boxes[:, [0, 1]] = boxes[:, [0, 1]] - boxes[:, [2, 3]] / 2
        boxes[:, [2, 3]] = boxes[:, [0, 1]] + boxes[:, [2, 3]]

        return boxes.tolist()

    def _xyxy2dcxdcydhdw(self, boxes: torch.Tensor, image_height: int, image_width: int):
        '''
        Args:
            boxes: list([[x1, y1, x2, y2, class_idx], ...])
            image_height: int
            image_width: int
        Returns:
            boxes: list([[bx / iw, by / ih, bw / iw, bh / ih, class_idx], ...])
        '''
        boxes[:, [2, 3]] = boxes[:, [2, 3]] - boxes[:, [0, 1]]
        boxes[:, [0, 1]] = boxes[:, [0, 1]] + boxes[:, [2, 3]] / 2
        boxes[:, [0, 2]] /= image_width
        boxes[:, [1, 3]] /= image_height

        return boxes

    def _iou_width_height(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
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
