import torch
import torchvision

from ignite import engine as e
from abc import abstractmethod

from ...module import Module
# from torch.cuda.amp import autocast, GradScaler


torch.backends.cudnn.benchmark = True


class Engine(Module):
    '''
        Base class for all engines. Your engine should subclass this class.
        Class Engine contains an Ignite Engine that controls running process over a dataset.
        Method _update is a function receiving the running Ignite Engine and the current batch in each iteration and returns data to be stored in the Ignite Engine's state.
        Parameters:
            dataset_name (str): dataset which engine run over.
            device (str): device on which model and tensor is allocated.
            max_epochs (int): number of epochs training process runs.
    '''

    def __init__(self, dataset, device, max_epochs=1):
        super(Engine, self).__init__()
        self.dataset = dataset
        self.device = device
        self.max_epochs = max_epochs
        self.engine = e.Engine(self._update)

    def run(self):
        return self.engine.run(self.dataset, self.max_epochs)

    @abstractmethod
    def _update(self, engine, batch):
        pass


class Trainer(Engine):
    '''
        Engine controls training process.
        See Engine documentation for more details about parameters.
    '''

    def __init__(self, dataset, device, scales, based_anchors, max_epochs=1):
        super(Trainer, self).__init__(dataset, device, max_epochs)
        anchors = torch.tensor(based_anchors)
        scales = torch.tensor(scales).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, 3, 2)
        self.scaled_anchors = anchors * scales
        self.scaled_anchors = self.scaled_anchors.to(self.device)
        # self.scaler = GradScaler()

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        assert 'optim' in self.frame, 'The frame does not have optim.'
        assert 'loss' in self.frame, 'The frame does not have loss.'
        self.model = self.frame['model'].to(self.device)
        self.optimizer = self.frame['optim']
        self.loss = self.frame['loss']

    # def _update(self, engine, batch):
    #     self.model.train()
    #     self.optimizer.zero_grad()
    #     params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
    #     params[1] = [param.to(self.device) for param in params[1]]
    #     with autocast():
    #         params[0] = self.model(params[0])
    #         loss = torch.tensor([self.loss(pred, target, anchors) for pred, target, anchors in zip(params[0], params[1], self.scaled_anchors)])
    #         loss = loss.sum()
    #         print(f'loss={loss}')
    #     self.scaler.scale(loss).backward()
    #     self.scaler.step(self.optimizer)
    #     self.scaler.update()
    #     return loss.item()

    def _update(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()
        params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
        params[1] = [param.to(self.device) for param in params[1]]
        params[0] = self.model(params[0])
        loss = sum([self.loss(pred, target, anchors) for pred, target, anchors in zip(params[0], params[1], self.scaled_anchors)])
        loss = loss.sum()
        # print(f'loss={loss}')
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Evaluator(Engine):
    '''
        Engine controls evaluating process.
        See Engine documentation for more details about parameters.
    '''

    def __init__(self, dataset, device, scales, based_anchors, max_epochs=1):
        super(Evaluator, self).__init__(dataset, device, max_epochs)
        anchors = torch.tensor(based_anchors)
        scales = torch.tensor(scales).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, 3, 2)
        self.multi_scaled_anchors = anchors * scales
        self.multi_scaled_anchors = self.multi_scaled_anchors.to(self.device)

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        self.model = self.frame['model'].to(self.device)

    def _update(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            batch_samples, tuple_batch_targets, batch_infos = batch
            batch_size = batch_samples.shape[0]

            batch_samples = batch_samples.to(self.device)
            batch_multi_scale_trues = [batch_targets.to(self.device) for batch_targets in tuple_batch_targets]
            batch_multi_scale_preds = self.model(batch_samples)

            batch_multi_scale_true_bboxes = []
            batch_multi_scale_pred_bboxes = []
            for scale_idx, (batch_scale_trues, batch_scale_preds) in enumerate(zip(batch_multi_scale_trues, batch_multi_scale_preds)):
                chosen_scale_anchors = self.multi_scaled_anchors[scale_idx]
                scale = batch_scale_preds.shape[2]
                batch_scale_true_bboxes = self.cells_to_bboxes(bboxes=batch_scale_trues, is_prediction=False,
                                                               S=scale, scale_anchors=chosen_scale_anchors)
                batch_scale_pred_bboxes = self.cells_to_bboxes(bboxes=batch_scale_preds, is_prediction=True,
                                                               S=scale, scale_anchors=chosen_scale_anchors)
                batch_multi_scale_true_bboxes.append(batch_scale_true_bboxes)
                batch_multi_scale_pred_bboxes.append(batch_scale_pred_bboxes)

            batch_multi_scale_true_bboxes = torch.cat(batch_multi_scale_true_bboxes, dim=1)
            batch_multi_scale_pred_bboxes = torch.cat(batch_multi_scale_pred_bboxes, dim=1)

            true_bboxes, pred_bboxes = [], []
            for sample_idx in range(batch_size):
                sample_height, sample_width = batch_samples[sample_idx].shape[1:]

                multi_scale_true_bboxes = batch_multi_scale_true_bboxes[sample_idx]
                multi_scale_pred_bboxes = batch_multi_scale_pred_bboxes[sample_idx]

                multi_scale_true_bboxes = self.convert_cxcywh2xyxy(bboxes=multi_scale_true_bboxes,
                                                                   image_height=sample_height, image_width=sample_width)

                multi_scale_pred_bboxes = self.convert_cxcywh2xyxy(bboxes=multi_scale_pred_bboxes,
                                                                   image_height=sample_height, image_width=sample_width)

                multi_scale_true_bboxes = self.postprocess_batched_nms(bboxes=multi_scale_true_bboxes,
                                                                       iou_threshold=0., score_threshold=1.)

                multi_scale_pred_bboxes = self.postprocess_batched_nms(bboxes=multi_scale_pred_bboxes,
                                                                       iou_threshold=0.5, score_threshold=0.2)

                true_bboxes.append(self.convert_evaluation_bboxes(bboxes=multi_scale_true_bboxes, sample_idx=sample_idx))
                pred_bboxes.append(self.convert_evaluation_bboxes(bboxes=multi_scale_pred_bboxes, sample_idx=sample_idx))

            return pred_bboxes, true_bboxes

    def convert_evaluation_bboxes(self, bboxes: torch.Tensor, sample_idx: int) -> torch.Tensor:
        labels = bboxes[:, 0].to(torch.int64)
        scores = bboxes[:, 1].to(torch.float32)
        boxes = bboxes[:, 2:6].to(torch.float32)
        image_id = torch.tensor([sample_idx])
        converted_bboxes = {'boxes': boxes, 'scores': scores, 'labels': labels, 'image_id': image_id}

        return converted_bboxes

    def convert_cxcywh2xyxy(self, bboxes: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
        '''convert (center_x / image_width, center_y / image_height, w / image_width, h / image_height) to (x1, y1, x2, y2)
        '''
        bboxes[:, [2, 4]] = bboxes[:, [2, 4]] * image_width
        bboxes[:, [3, 5]] = bboxes[:, [3, 5]] * image_height
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 4] / 2
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 5] / 2
        bboxes[:, 4] = bboxes[:, 2] + bboxes[:, 4]
        bboxes[:, 5] = bboxes[:, 3] + bboxes[:, 5]

        return bboxes

    def cells_to_bboxes(self, bboxes: torch.Tensor, scale_anchors: torch.Tensor,
                        S: int, is_prediction: bool = True) -> torch.Tensor:
        """
        Scales the predictions coming from the model to be relative to the entire image.
        Args:
            bboxes (Tensor[N, 3, S, S, num_classes + 5])
            anchors: the anchors used for the predictions
            S: the number of cells the image is divided in on the width (and height)
            is_preds: whether the input is predictions or the true bounding boxes
        Returns:
            converted_bboxes: the converted boxes of sizes (N, num_anchors * S * S, 1+5)
                              with class index, object score, bounding box coordinates.
        """
        BATCH_SIZE = bboxes.shape[0]
        num_anchors = scale_anchors.shape[0]
        box_coords = bboxes[..., 1:5]
        if is_prediction:
            scale_anchors = scale_anchors.reshape(1, num_anchors, 1, 1, 2)
            box_coords[..., 0:2] = torch.sigmoid(box_coords[..., 0:2])  # x, y coordinates
            box_coords[..., 2:4] = torch.exp(box_coords[..., 2:4]) * scale_anchors  # w, h
            scores = torch.sigmoid(bboxes[..., 0:1])
            labels = torch.argmax(bboxes[..., 5:], dim=-1).unsqueeze(dim=-1)
        else:
            scores = bboxes[..., 0:1]
            labels = bboxes[..., 5:6]

        # BATCH_SIZE x 3 x S x S x 1
        cell_indices = torch.arange(S).repeat(BATCH_SIZE, 3, S, 1).unsqueeze(dim=-1).to(bboxes.device)

        x = 1 / S * (box_coords[..., 0:1] + cell_indices)
        y = 1 / S * (box_coords[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
        w = 1 / S * box_coords[..., 2:3]
        h = 1 / S * box_coords[..., 3:4]

        bboxes = torch.cat([labels, scores, x, y, w, h], dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)

        return bboxes

    def batched_nms(self, boxes: torch.Tensor, scores: torch.Tensor,
                    idxs: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """Performs non-maximum suppression in a batched fashion.
        Each index value correspond to a category, and NMS will not be applied between elements of different categories.
        Args:
            boxes (Tensor[N, 4]): boxes where NMS will be performed. (x1, y1, x2, y2) with 0 <= x1 < x2 and 0 <= y1 < y2
            scores (Tensor[N]): scores for each one of the boxes
            idxs (Tensor[N]): indices of the categories for each one of the boxes.
            iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold
        Returns:
            keep (Tensor): int64 tensor with the indices of the elements that have been kept by NMS, sorted in decreasing order of scores
        """
        if boxes.numel() == 0:
            return torch.empty(size=(0,), dtype=torch.int64, device=boxes.device)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(data=1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]
            keep = torchvision.ops.nms(boxes=boxes_for_nms, scores=scores, iou_threshold=iou_threshold)
            return keep

    def postprocess_batched_nms(self, bboxes: torch.Tensor, iou_threshold: float, score_threshold: float) -> torch.Tensor:
        '''
        Args:
            bboxes (Tensor[N, 6]): (class_id, score, x1, y1, x2, y2)
            iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold
            score_threshold (float): discards all boxes with score < score_threshold
        Returns:
            bboxes (Tensor[M, 6]): (class_id, score, x1, y1, x2, y2)
        '''
        # bboxes = torch.from_numpy(np.asarray(bboxes))
        idxs, scores, boxes = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2:6]
        indices = scores >= score_threshold
        idxs, scores, boxes = idxs[indices], scores[indices], boxes[indices]
        indices = self.batched_nms(boxes=boxes, scores=scores, idxs=idxs, iou_threshold=iou_threshold)
        idxs, scores, boxes = idxs[indices], scores[indices], boxes[indices]
        bboxes = torch.cat([idxs[:, None], scores[:, None], boxes], dim=-1)

        return bboxes
