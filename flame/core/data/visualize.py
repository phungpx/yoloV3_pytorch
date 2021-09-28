import cv2
import torch
import numpy as np
from typing import List, Tuple


def to_image(
    sample: torch.Tensor = None,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    device: str = 'cpu',
) -> np.ndarray:
    mean = torch.tensor(mean, dtype=torch.float, device=device).view(3, 1, 1)
    std = torch.tensor(std, dtype=torch.float, device=device).view(3, 1, 1)
    sample = (sample * std + mean) * 255
    sample = sample.permute(1, 2, 0).contiguous()
    image = sample.to(torch.uint8).cpu().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def draw_box(
    image: np.ndarray, box: List[int], name: str,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    '''
    Args:
        box: x1 y1 x2 y2 format
        name: name of class
    Output:
        image
    '''
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    thickness = max(height, width) // 500
    fontscale = max(height, width) / 1000

    cv2.rectangle(
        img=image,
        pt1=(x1, y1),
        pt2=(x2, y2),
        color=box_color,
        thickness=thickness
    )

    cv2.putText(
        img=image, text=name,
        org=(x1, y1),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontscale,
        thickness=thickness,
        color=text_color,
        lineType=cv2.LINE_AA
    )

    return image
