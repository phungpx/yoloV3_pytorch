import torch
from torchvision import ops
from typing import Tuple, List


def inference(
        predictions: Tuple[torch.Tensor],
        anchors: List[List[Tuple[float, float]]],
        image_size: int = 416,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.05
    ):
    '''get all boxes at gird S x S (grid_size = image_size / S)
    Args:
        preds: Tuple[[N x 3 x S x S x (tp, tx, ty, tw, th, n_classes)]] with S = [13, 26, 52]
        anchors: [3 x 3 x 2] (pw, ph with size in [0, 1])  (relative to image_size)
    Outputs:
        scores: [N x (3 * (S1 * S1 + S2 * S2 + S3 * S3))]
        labels: [N x (3 * (S1 * S1 + S2 * S2 + S3 * S3))]
        bboxes: [N x (3 * (S1 * S1 + S2 * S2 + S3 * S3)) x 4], with [x1 y1 x2 y2].
    '''
    device = predictions[0].device
    batch_size = predictions[0].shape[0]

    batch_boxes, batch_labels, batch_scores = [], [], []

    for i, pred in enumerate(predictions):
        S = pred.shape[0]

        # anchor: 1 x 3 x 1 x 1 x 2
        anchor = torch.tensor(anchors[i], device=device, dtype=torch.float)  # anchor: 3 x 2
        anchor = anchor.reshape(1, 3, 1, 1, 2)

        # N x 3 x S x S x 1
        x_indices = torch.arange(S).repeat(batch_size, 3, S, 1)
        x_indices = x_indices.unsqueeze(dim=-1).to(device)
        y_indices = x_indices.permute(0, 1, 3, 2, 4)

        # N x 3 x S x S -> reshape: N x (3 * S * S)
        # score = sigmoid(tp)
        scores = torch.sigmoid(pred[..., 0]).reshape(batch_size, -1)

        # N x 3 x S x S -> reshape: N x (3 * S * S)
        labels = torch.argmax(pred[..., 5:], dim=-1).reshape(batch_size, -1)

        # xy: N x 3 x S x S x 2 (top-left corner of bboxes)
        # bx = sigmoid(tx) + cx, by = sigmoid(ty) + cy
        bx = (torch.sigmoid(pred[..., 1]) + x_indices) * (image_size / S)
        by = (torch.sigmoid(pred[..., 2]) + y_indices) * (image_size / S)
        bxy = torch.cat([bx, by], dim=-1)

        # wh: N x 3 x S x S x 2 (width, height of bboxes)
        # bw = pw * e ^ tw, bh = ph * e ^ th
        bwh = (image_size * anchor) * torch.exp(pred[..., 3:5])

        # boxes (x1 y1 x2 y2 type): N x (3 * S * S) x 4
        boxes = torch.cat([bxy, bwh], dim=-1).reshape(batch_size, -1, 4)  # boxes (x1 y1 w h type)
        boxes = ops.box_convert(boxes=boxes, in_fmt='xywh', out_fmt='xyxy')
        boxes = ops.clip_boxes_to_image(boxes=boxes, size=(image_size, image_size))

        batch_boxes.append(boxes)
        batch_labels.append(labels)
        batch_scores.append(scores)

    batch_labels = torch.cat(batch_labels, dim=1)  # [N x (3 * (S1 * S1 + S2 * S2 + S3 * S3))]
    batch_scores = torch.cat(batch_scores, dim=1)  # [N x (3 * (S1 * S1 + S2 * S2 + S3 * S3))]
    batch_boxes = torch.cat(batch_boxes, dim=1)  # [N x (3 * (S1 * S1 + S2 * S2 + S3 * S3)) x 4]

    predictions = []

    for batch_id in range(batch_size):
        score_indices = batch_scores[batch_id, :] > score_threshold

        if score_indices.sum() == 0:
            predictions.append(
                {
                    'boxes': torch.tensor([[0, 0, 1, 1]], dtype=torch.float, device=device),
                    'labels': torch.tensor([-1], dtype=torch.int64, device=device),
                    'scores': torch.tensor([0], dtype=torch.float, device=device)
                }
            )

            continue

        boxes = batch_boxes[batch_id, score_indices, :]
        labels = batch_labels[batch_id, score_indices]
        scores = batch_scores[batch_id, score_indices]

        nms_indices = ops.boxes.batched_nms(
            boxes=boxes, scores=scores, idxs=labels,
            iou_threshold=iou_threshold
        )

        if nms_indices.shape[0] != 0:
            predictions.append(
                {
                    'boxes': boxes[nms_indices, :],
                    'labels': labels[nms_indices],
                    'scores': scores[nms_indices]
                }
            )
        else:
            predictions.append(
                {
                    'boxes': torch.tensor([[0, 0, 1, 1]], dtype=torch.float, device=device),
                    'labels': torch.tensor([-1], dtype=torch.int64, device=device),
                    'scores': torch.tensor([0], dtype=torch.float, device=device)
                }
            )

    return predictions
