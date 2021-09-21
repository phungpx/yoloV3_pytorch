import cv2
import torch
import numpy as np


def chunks(lst: list, size: Optional[int] = None) -> Union[List, Generator]:
    if size is None:
        yield lst
    else:
        for i in range(0, len(lst), size):
            yield lst[i:i + size]


class Predictor:
    def __init__(
        self,
        model,
        weight_path,
        batch_size,
        image_size,
        mean,
        std,
        anchors,
        score_threshold,
        iou_threshold,
        device
    ):
        super(Predictor, self).__init__()
        self.device
        self.anchors = anchors
        self.batch_size = batch_size
        self.image_size = image_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.mean = torch.tensor(mean, dtype=torch.float, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float, device=device).view(1, 3, 1, 1)

        state_dict = torch.load(f=weight_path, map_location=device)
        self.model = model.load_state_dict(state_dict=state_dict['state_dict'])
        self.model.to(device).eval()


    def __call__(self, images: List[np.ndarray]) -> List[Dict[str, torch.Tensor]]:
        samples = self.preprocess(images)
        preds = self.process(samples)
        outputs = self.postprocess(preds)

        return outputs

    def preprocess(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
        '''
        Args:
            images: list of images (image: H x W x C)
        Outputs:
            samples: list of processed images (sample: 416 x 416 x 3)
        '''
        samples = [], []
        for image in images:
            sample = self._resize(image, imsize=self.image_size)
            sample = self._pad_to_square(sample)
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

            samples.append(sample)

        return samples

    def process(self, samples):
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
        preds = []

        for batch in chunks(samples, size=self.batch_size):
            batch = [torch.from_numpy(batch) for batch in batchs]
            batch = torch.stack(batch, dim=0).to(self.device)
            batch = batch.permute(0, 3, 1, 2).contiguous()
            batch = (batch.float().div(255.) - self.mean) / self.std

            with torch.no_grad():
                preds += self.model(batch)

        return preds

    def postprocess(self, preds: Tuple[torch.Tensor]):
        '''
        Args:
            preds: list of three tensor for three scales (S1=13, S2=26, S3=52),
                .1: N x 3 x S1 x S1 x (5 + C)
                .2: N x 3 x S2 x S2 x (5 + C)
                .3: N x 3 x S3 x S3 x (5 + C)
        Outputs:
            outputs: list of final information of each images,
                     each values in list is a dict included informations
                     ('boxes', 'labels', 'scores') of image.
        '''
        batch_boxes, batch_labels, batch_scores = [], [], []

        for i, pred in enumerate(preds):
            S = pred.shape[2]  # 13, 26, 52

            # anchor: 1 x 3 x 1 x 1 x 2
            anchor = torch.tensor(self.anchors[i], device=device, dtype=torch.float)  # anchor: 3 x 2
            anchor = anchor.reshape(1, 3, 1, 1, 2)

            # cx, cy: N x 3 x S x S
            cx = torch.arange(S).repeat(batch_size, 3, S, 1).to(device)
            cy = cx.permute(0, 1, 3, 2)

            # N x 3 x S x S -> reshape: N x (3 * S * S)
            # score = sigmoid(tp)
            scores = torch.sigmoid(pred[..., 0]).reshape(batch_size, -1)

            # N x 3 x S x S -> reshape: N x (3 * S * S)
            labels = torch.argmax(pred[..., 5:], dim=-1).reshape(batch_size, -1)

            # xy: N x 3 x S x S x 2 (center of bboxes)
            # bx = sigmoid(tx) + cx, by = sigmoid(ty) + cy
            bx = (torch.sigmoid(pred[..., 1]) + cx) * (image_size / S)
            by = (torch.sigmoid(pred[..., 2]) + cy) * (image_size / S)
            bxy = torch.stack([bx, by], dim=-1)

            # wh: N x 3 x S x S x 2 (width, height of bboxes)
            # bw = pw * e ^ tw, bh = ph * e ^ th
            bwh = (image_size * anchor) * torch.exp(pred[..., 3:5])

            # boxes (x1 y1 x2 y2 type): N x (3 * S * S) x 4
            boxes = torch.cat([bxy - bwh / 2, bxy + bwh / 2], dim=-1).reshape(batch_size, -1, 4)
            boxes = torch.clamp(boxes, min=0, max=image_size)

            batch_boxes.append(boxes)
            batch_labels.append(labels)
            batch_scores.append(scores)

        batch_labels = torch.cat(batch_labels, dim=1)  # [N x (3 * (S1 * S1 + S2 * S2 + S3 * S3))]
        batch_scores = torch.cat(batch_scores, dim=1)  # [N x (3 * (S1 * S1 + S2 * S2 + S3 * S3))]
        batch_boxes = torch.cat(batch_boxes, dim=1)  # [N x (3 * (S1 * S1 + S2 * S2 + S3 * S3)) x 4]

        outputs = []

        for batch_id in range(batch_size):
            score_indices = batch_scores[batch_id, :] > score_threshold

            if score_indices.sum() == 0:
                outputs.append(
                    {
                        'boxes': torch.tensor([[0, 0, 1, 1]], dtype=torch.float, device=device),
                        'labels': torch.tensor([-1], dtype=torch.int64, device=device),
                        'scores': torch.tensor([0], dtype=torch.float, device=device)
                    }
                )

                continue

            boxes = batch_boxes[batch_id, score_indices, :]  # n_boxes x 4
            labels = batch_labels[batch_id, score_indices]  # n_labels
            scores = batch_scores[batch_id, score_indices]  # n_scores

            nms_indices = ops.boxes.batched_nms(
                boxes=boxes, scores=scores, idxs=labels,
                iou_threshold=iou_threshold
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
                outputs.append(
                    {
                        'boxes': torch.tensor([[0, 0, 1, 1]], dtype=torch.float, device=device),
                        'labels': torch.tensor([-1], dtype=torch.int64, device=device),
                        'scores': torch.tensor([0], dtype=torch.float, device=device)
                    }
                )

        return outputs

    def _resize(self, image: np.ndarray, imsize=416) -> Tuple[np.ndarray, float]:
        ratio = imsize / max(image.shape)
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
        return image, ratio

    def _pad_to_square(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        max_size = max(height, width)
        image = np.pad(
            image, ((0, max_size - height), (0, max_size - width), (0, 0))
        )
        return image
