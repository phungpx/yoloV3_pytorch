import cv2
import torch
import numpy as np

from torchvision import ops
from torch import nn, Tensor
from typing import List, Tuple, Dict, Union, Optional, Generator

import utils


def chunks(lst: list, size: Optional[int] = None) -> Union[List, Generator]:
    if size is None:
        yield lst
    else:
        for i in range(0, len(lst), size):
            yield lst[i:i + size]


class Predictor:
    def __init__(
        self,
        model: dict = None,
        weight_path: str = None,
        batch_size: Optional[str] = None,
        image_size: int = 416,
        classes: Dict[str, int] = None,
        mean: Tuple[float, float, float] = (0., 0., 0.),
        std: Tuple[float, float, float] = (1., 1., 1.),
        anchors: List[List[Tuple[float, float]]] = None,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        device: str = 'cpu',
    ):
        super(Predictor, self).__init__()
        self.device = device
        self.anchors = anchors
        self.batch_size = batch_size
        self.image_size = image_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.mean = torch.tensor(mean, dtype=torch.float, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float, device=device).view(1, 3, 1, 1)
        self.classes = {class_id: class_name for class_name, class_id in classes.items()}

        self.model = utils.create_instance(model)
        state_dict = torch.load(f=weight_path, map_location=device)
        self.model.load_state_dict(state_dict=state_dict)
        self.model.eval().to(device)

    def __call__(self, images: List[np.ndarray]) -> List[Dict[str, Tensor]]:
        samples = self.preprocess(images)
        preds = self.process(samples)
        outputs = self.postprocess(preds)

        for i in range(len(images)):
            if outputs[i]['labels'] is not None:
                ratio = max(images[i].shape[:2]) / self.image_size
                outputs[i]['boxes'] *= ratio
                outputs[i]['names'] = [self.classes[label.item()] for label in outputs[i]['labels']]

        return outputs

    def preprocess(self, images: List[np.ndarray]) -> List[np.ndarray]:
        '''
        Args:
            images: list of images (image: H x W x C)
        Outputs:
            samples: list of processed images (sample: 416 x 416 x 3)
        '''
        samples = []
        for image in images:
            sample = self._resize(image, imsize=self.image_size)
            sample = self._pad_to_square(sample)
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

            samples.append(sample)

        return samples

    def process(self, samples: List[np.ndarray]) -> Tuple[Tensor, Tensor, Tensor]:
        '''
        Args:
            samples: list of processed images (sample: 416 x 416 x 3)
        Outputs:
            preds: list of three tensor for three scales (S1=13, S2=26, S3=52)
                . N x 3 x S1 x S1 x (5 + C)
                . N x 3 x S2 x S2 x (5 + C)
                . N x 3 x S3 x S3 x (5 + C)
            with:
                N: num of input images
                3: num of anchors for each scales,
                5: num pf predicted values for each boxes (tp, tx, ty, tw, th),
                C: num classes,
        '''
        s1_preds, s2_preds, s3_preds = [], [], []

        for batch in chunks(samples, size=self.batch_size):
            batch = [torch.from_numpy(sample) for sample in batch]
            batch = torch.stack(batch, dim=0).to(self.device)
            batch = batch.permute(0, 3, 1, 2).contiguous()
            batch = (batch.float().div(255.) - self.mean) / self.std

            with torch.no_grad():
                s1, s2, s3 = self.model(batch)

                s1_preds += torch.split(tensor=s1, split_size_or_sections=1, dim=0)
                s2_preds += torch.split(tensor=s2, split_size_or_sections=1, dim=0)
                s3_preds += torch.split(tensor=s3, split_size_or_sections=1, dim=0)

        preds = (
            torch.cat(s1_preds, dim=0),  # N x 3 x S1 x S1 x (5 + C)
            torch.cat(s2_preds, dim=0),  # N x 3 x S1 x S1 x (5 + C)
            torch.cat(s3_preds, dim=0),  # N x 3 x S1 x S1 x (5 + C)
        )

        return preds

    def postprocess(self, preds: Tuple[Tensor, Tensor, Tensor]) -> List[Dict[str, Tensor]]:
        '''
        Args:
            preds: tuple of three tensor for three scales (S1=13, S2=26, S3=52),
                .1: N x 3 x S1 x S1 x (5 + C)
                .2: N x 3 x S2 x S2 x (5 + C)
                .3: N x 3 x S3 x S3 x (5 + C)
        Outputs:
            outputs: list of final information of each images,
                     each values in list is a dict included informations
                     ('boxes', 'labels', 'scores') of image.
        '''
        num_samples = preds[0].shape[0]
        batch_boxes, batch_labels, batch_scores = [], [], []

        for i, pred in enumerate(preds):
            S = pred.shape[2]  # 13, 26, 52

            # anchor: 1 x 3 x 1 x 1 x 2
            anchor = torch.tensor(self.anchors[i], device=self.device, dtype=torch.float)  # anchor: 3 x 2
            anchor = anchor.reshape(1, 3, 1, 1, 2)

            # cx, cy: N x 3 x S x S
            cx = torch.arange(S).repeat(num_samples, 3, S, 1).to(self.device)
            cy = cx.permute(0, 1, 3, 2)

            # N x 3 x S x S -> reshape: N x (3 * S * S)
            # score = sigmoid(tp)
            scores = torch.sigmoid(pred[..., 0]).reshape(num_samples, -1)

            # N x 3 x S x S -> reshape: N x (3 * S * S)
            labels = torch.argmax(pred[..., 5:], dim=-1).reshape(num_samples, -1)

            # xy: N x 3 x S x S x 2 (center of bboxes)
            # bx = sigmoid(tx) + cx, by = sigmoid(ty) + cy
            bx = (torch.sigmoid(pred[..., 1]) + cx) * (self.image_size / S)
            by = (torch.sigmoid(pred[..., 2]) + cy) * (self.image_size / S)
            bxy = torch.stack([bx, by], dim=-1)

            # wh: N x 3 x S x S x 2 (width, height of bboxes)
            # bw = pw * e ^ tw, bh = ph * e ^ th
            # bwh = (self.image_size * anchor) * torch.exp(pred[..., 3:5])
            bwh = anchor * torch.exp(pred[..., 3:5])

            # boxes (x1 y1 x2 y2 type): N x (3 * S * S) x 4
            x1y1, x2y2 = bxy - bwh / 2, bxy + bwh / 2  # convert xywh to x1y1x2y2
            boxes = torch.cat([x1y1, x2y2], dim=-1).reshape(num_samples, -1, 4)
            boxes = torch.clamp(boxes, min=0, max=self.image_size)  # clip box within [0, image_size]

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
                outputs.append({'boxes': None, 'labels': None, 'scores': None})

                continue

            boxes = batch_boxes[sample_id, score_indices, :]  # [n x 4]
            labels = batch_labels[sample_id, score_indices]   # [n]
            scores = batch_scores[sample_id, score_indices]   # [n]

            nms_indices = ops.boxes.batched_nms(
                boxes=boxes, scores=scores, idxs=labels,
                iou_threshold=self.iou_threshold
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
                outputs.append({'boxes': None, 'labels': None, 'scores': None})

        return outputs

    def _resize(self, image: np.ndarray, imsize=416) -> Tuple[np.ndarray, float]:
        ratio = imsize / max(image.shape)
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
        return image

    def _pad_to_square(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        max_size = max(height, width)
        image = np.pad(
            image, ((0, max_size - height), (0, max_size - width), (0, 0))
        )
        return image
