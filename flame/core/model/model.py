import torch
from torch import nn
from torchvision import ops
from typing import Optional, List, Tuple

from .darknet53 import YOLOv3


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 80,
        weight_path: Optional[str] = None,  # using for loading pretrained weight
        anchors: List[List[Tuple[float, float]]] = None,  # related to input_size (416 x 416)
        score_threshold: float = 0.5,  # using for removing objects with score lower than score_threshold
        iou_threshold: float = 0.5,  # using for non-max suppression
    ) -> None:
        super(Model, self).__init__()
        self.model = YOLOv3(
            in_channels=in_channels,
            num_classes=num_classes
        )

        # use for inference mode
        self.anchors = anchors
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        if weight_path is not None:
            state_dict = torch.load(f=weight_path, map_location='cpu')['state_dict']

            state_dict.pop('layers.15.pred.1.conv.weight')
            state_dict.pop('layers.15.pred.1.conv.bias')
            state_dict.pop('layers.15.pred.1.bn.weight')
            state_dict.pop('layers.15.pred.1.bn.bias')
            state_dict.pop('layers.15.pred.1.bn.running_mean')
            state_dict.pop('layers.15.pred.1.bn.running_var')

            state_dict.pop('layers.22.pred.1.conv.weight')
            state_dict.pop('layers.22.pred.1.conv.bias')
            state_dict.pop('layers.22.pred.1.bn.weight')
            state_dict.pop('layers.22.pred.1.bn.bias')
            state_dict.pop('layers.22.pred.1.bn.running_mean')
            state_dict.pop('layers.22.pred.1.bn.running_var')

            state_dict.pop('layers.29.pred.1.conv.weight')
            state_dict.pop('layers.29.pred.1.conv.bias')
            state_dict.pop('layers.29.pred.1.bn.weight')
            state_dict.pop('layers.29.pred.1.bn.bias')
            state_dict.pop('layers.29.pred.1.bn.running_mean')
            state_dict.pop('layers.29.pred.1.bn.running_var')

            self.model.load_state_dict(state_dict, strict=False)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, inputs):
        return self.model(inputs)

    def predict(self, inputs):
        '''
        Arg:
            inputs: N x 3 x 416 x 416
        Output:
            output: dictionary[str, torch.Tensor]
            [
                {
                    'boxes': torch.Tensor(int64),
                    'label': torch.Tensor(int64),
                    'score': torch.Tensor(float),
                },
                {
                    .....
                },
                ....
            ]
        '''
        preds = self.forward(inputs)  # Tuple[N x 3 x S x S x (5 + C)]
        device = inputs.device
        input_size = inputs.shape[2]  # 416
        num_samples = preds[0].shape[0]  # N

        batch_boxes, batch_labels, batch_scores = [], [], []

        for i, pred in enumerate(preds):
            S = pred.shape[2]  # 13 or 26 or 52

            # anchor: 1 x 3 x 1 x 1 x 2
            anchor = torch.tensor(
                self.anchors[i],
                device=device,
                dtype=torch.float
            )  # anchor: 3 x 2
            anchor = anchor.reshape(1, 3, 1, 1, 2)

            # cx, cy: N x 3 x S x S
            cx = torch.arange(S).repeat(num_samples, 3, S, 1).to(device)
            cy = cx.permute(0, 1, 3, 2)

            # N x 3 x S x S -> reshape: N x (3 * S * S)
            # score = sigmoid(tp)
            scores = torch.sigmoid(pred[..., 0]).reshape(num_samples, -1)

            # N x 3 x S x S -> reshape: N x (3 * S * S)
            labels = torch.argmax(pred[..., 5:], dim=-1).reshape(num_samples, -1)

            # xy: N x 3 x S x S x 2 (center of bboxes)
            # bx = sigmoid(tx) + cx, by = sigmoid(ty) + cy
            bx = (torch.sigmoid(pred[..., 1]) + cx) * (input_size / S)  # grid_size = input_size / S
            by = (torch.sigmoid(pred[..., 2]) + cy) * (input_size / S)  # grid_size = input_size / S
            bxy = torch.stack([bx, by], dim=-1)

            # wh: N x 3 x S x S x 2 (width, height of bboxes)
            # bw = pw * e ^ tw, bh = ph * e ^ th
            bwh = anchor * torch.exp(pred[..., 3:5])

            # boxes (x1 y1 x2 y2 type): N x (3 * S * S) x 4
            x1y1, x2y2 = bxy - bwh / 2, bxy + bwh / 2  # convert xywh to x1y1x2y2
            boxes = torch.cat([x1y1, x2y2], dim=-1).reshape(num_samples, -1, 4)
            boxes = torch.clamp(boxes, min=0, max=input_size)  # clip box within [0, input_size]

            batch_boxes.append(boxes)
            batch_labels.append(labels)
            batch_scores.append(scores)

        batch_labels = torch.cat(batch_labels, dim=1)  # [N x (3 * (S1 * S1 + S2 * S2 + S3 * S3))]
        batch_scores = torch.cat(batch_scores, dim=1)  # [N x (3 * (S1 * S1 + S2 * S2 + S3 * S3))]
        batch_boxes = torch.cat(batch_boxes, dim=1)  # [N x (3 * (S1 * S1 + S2 * S2 + S3 * S3)) x 4]

        outputs = []

        for sample_id in range(num_samples):
            score_indices = batch_scores[sample_id, :] > self.score_threshold

            if score_indices.sum() == 0:
                outputs.append({'boxes': torch.FloatTensor([[0, 0, 1, 1]]),
                                'labels': torch.FloatTensor([-1]),
                                'scores': torch.FloatTensor([0])})

                continue

            boxes = batch_boxes[sample_id, score_indices, :]  # [n x 4]
            labels = batch_labels[sample_id, score_indices]   # [n]
            scores = batch_scores[sample_id, score_indices]   # [n]

            nms_indices = ops.boxes.batched_nms(
                boxes=boxes,
                scores=scores,
                idxs=labels,
                iou_threshold=self.iou_threshold,
            )

            if nms_indices.shape[0] != 0:
                outputs.append(
                    {
                        'boxes': boxes[nms_indices, :],
                        'labels': labels[nms_indices],
                        'scores': scores[nms_indices]
                    }
                )
            else:
                outputs.append({'boxes': torch.FloatTensor([[0, 0, 1, 1]]),
                                'labels': torch.FloatTensor([-1]),
                                'scores': torch.FloatTensor([0])})

        return outputs
