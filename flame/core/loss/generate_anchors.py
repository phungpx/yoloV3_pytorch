from torch import nn, Tensor
from typing import List, Tuple, Dict


class AnchorGeneration(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int] = (416, 416),
        scales: List[int] = [13, 26, 52],
        anchor_sizes: Dict[int, List[Tuple[float, float]]] = {
            13: [(116, 90), (156, 198), (373, 326)],
            26: [(30, 61), (62, 45), (59, 119)],
            52: [(10, 13), (16, 30), (33, 23)]
        },
        device: str = 'cpu',
    ):
        super(AnchorGeneration, self).__init__()
        self.scales = scales
        self.input_size = input_size
        self.anchor_sizes = anchor_sizes
        self.device = device

    def forward(self) -> Dict[int, Tensor]:
        anchor_boxes = {}

        for scale in self.scales:
            grid_size_x, grid_size_y = (self.input_size[0] / scale, self.input_size[1] / scale)

            # num_anchors_per_scale x 2
            anchor_size = torch.tensor(self.anchor_sizes[scale], dtype=torch.float, device=device)

            w = anchor_size[:, 0].view(anchor_size.shape[0], 1, 1)  # num_anchors_per_scale x 1 x 1
            h = anchor_size[:, 1].view(anchor_size.shape[0], 1, 1)  # num_anchors_per_scale x 1 x 1

            # all center of each anchor boxes
            x = torch.arange(start=grid_size_x / 2, end=image_size[0], step=grid_size_x)  # scale
            y = torch.arange(start=grid_size_y / 2, end=image_size[1], step=grid_size_y)  # scale

            cx, cy = torch.meshgrid(x, y)  # cx: scale x scale, cy: scale x scale  (coordinates)
            cx, cy = cx.unsqueeze(dim=0), cy.unsqueeze(dim=0)  # 1 x scale x scale

            # all coordinates of anchor boxes at scale
            x1, y1 = cx - w / 2, cy - h / 2  # num_anchors_per_scale x scale x scale
            x2, y2 = cx + w / 2, cy + h / 2  # num_anchors_per_scale x scale x scale

            # num_anchors_per_scale x scale x scale x 4
            anchor_boxes[scale] = torch.stack([x1, y1, x2, y2], dim=3)

        return anchor_boxes


# visualization for all anchor boxes at each scales
if __name__ == "__main__"
    import cv2
    import numpy as np

    anchor_generator = AnchorGeneration(
        input_size=(416, 416),
        scales=[13, 26, 52],
        anchor_sizes={
            13: [(116, 90), (156, 198), (373, 326)],
            26: [(30, 61), (62, 45), (59, 119)],
            52: [(10, 13), (16, 30), (33, 23)]
        },
        device='cpu',
    )

    anchor_boxes = anchor_generator()

    scale = 13  # 13 / 26 / 52
    image = np.zeros(
        shape=(anchor_generator.input_size[1], anchor_generator.input_size[0], 3), dtype=np.uint8
    )

    boxes = boxes[scale].numpy().reshape(-1, 4)

    for boxes in boxes:
        box = np.int32(box)
        cv2.rectangle(
            img=image,
            pt1=tuple(box[:2]),
            pt2=tuple(box[2:]),
            color=(0, 0, 255),
            thickness=1
        )
        cv2.circle(
            img=image,
            center=(int((box[0] + box[2]) / 2),
                    int((box[1] + box[3]) / 2)),
            radius=1,
            color=(0, 255, 0),
            thickness=-1
        )

    cv2.imshow('a', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
