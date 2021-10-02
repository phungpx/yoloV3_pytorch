import torch


def compute_iou(anchors: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    '''compute IoU between each anchor boxes and groundtruth boxes
    Args:
        anchors: [num_anchors_per_scale, S, S, 4], S = 13 or 26 or 52
        box_type: [y1, x1, y2, x2]
        boxes: [num_boxes, 4]
        box_type: [x1, y1, x2, y2]
    Output:
        ious: [num_anchors_per_scale, S, S, num_boxes]
    references: https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
    '''
    # get params and reshape anchors form [3 x S x S x 4] to [(3 * S * S), 4]
    num_boxes = boxes.shape[0]
    num_anchors_per_scale, S = anchors.shape[0], anchors.shape[1]
    anchors = anchors.reshape(num_anchors_per_scale * S * S, 4)  # num_anchors_per_scale * S * S, 4

    # calculate intersection areas of anchors and target boxes
    # num_anchors = num_anchors_per_scale * S * S
    inter_width = torch.min(anchors[:, 3].unsqueeze(dim=1), boxes[:, 2]) - torch.max(anchors[:, 1].unsqueeze(dim=1), boxes[:, 0])
    inter_height = torch.min(anchors[:, 2].unsqueeze(dim=1), boxes[:, 3]) - torch.max(anchors[:, 0].unsqueeze(dim=1), boxes[:, 1])
    inter_width = torch.clamp(inter_width, min=0.)  # num_anchors x num_boxes
    inter_height = torch.clamp(inter_height, min=0.)  # num_anchors x num_boxes
    inter_areas = inter_width * inter_height  # num_anchors x num_boxes

    # calculate union areas of anchors and target boxes
    # num_anchors = num_anchors_per_scale * S * S
    area_anchors = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])  # num_anchors
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # num_boxes
    union_areas = area_anchors.unsqueeze(dim=1) + area_boxes - inter_width * inter_height  # num_anchors x num_boxes
    union_areas = torch.clamp(union_areas, min=1e-8)

    # calculate ious of anchors and target boxes
    # shape of ious is [(num_anchors_per_scale * S * S) x num_boxes]
    ious = inter_areas / union_areas

    # reshape ious from [(num_anchors_per_scale * S * S) x num_boxes] to [num_anchors_per_scale x S x S x num_boxes]
    ious = ious.reshape(num_anchors_per_scale, S, S, num_boxes)

    return ious


if __name__ == "__main__":
    anchors = torch.FloatTensor(3, 13, 13, 4)  # num_anchors_per_scale x S x S x 4
    boxes = torch.FloatTensor(2, 4)  # num_boxes x 4
    ious = compute_iou(anchors, boxes)  # num_anchors_per_scale x S x S x num_boxes

    print(f'anchors at scale #{anchors.shape[1]} shape: {anchors.shape}')
    print(f'groundtruth boxes shape: {boxes.shape}')
    print(f'iou shape: {ious.shape}')
