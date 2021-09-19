import cv2
import torch
import torchvision
import numpy as np

from pathlib import Path
from ..module import Module
from ignite.engine import Events


class RegionPredictor(Module):
    def __init__(self, evaluator_name,
                 output_dir,
                 output_img_ext,
                 output_mask_ext,
                 classes,
                 image_size,
                 use_pad_to_square,
                 thresh_score,
                 thresh_iou_nms,
                 ratio_color_mask_on_image=None,
                 output_transform=lambda x: x):
        super(RegionPredictor, self).__init__()
        self.evaluator_name = evaluator_name
        self.output_dir = Path(output_dir)
        self.output_img_ext = output_img_ext
        self.output_mask_ext = output_mask_ext
        self.classes = classes
        self.image_size = image_size
        self.use_pad_to_square = use_pad_to_square
        self.thresh_score = thresh_score
        self.thresh_iou_nms = thresh_iou_nms
        self.ratio_color_mask_on_image = ratio_color_mask_on_image
        self._output_transform = output_transform

    def init(self):
        assert self.evaluator_name in self.frame, f'The frame does not have {self.evaluator_name}'
        self._attach(self.frame[self.evaluator_name].engine)

    def reset(self):
        pass

    def update(self, output):
        preds, image_infos = output
        image_names = [image_info[0] for image_info in image_infos]
        image_sizes = [image_info[1] for image_info in image_infos]
        for pred, image_name, image_size in zip(preds, image_names, image_sizes):
            class_name = str(Path(image_name).parent.stem)
            save_dir = self.output_dir.joinpath(class_name)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

            image_path = '{}/{}{}'.format(save_dir, Path(image_name).stem, self.output_img_ext)
            image = cv2.imread(image_name)
            labels, boxes, scores, masks = pred['labels'], pred['boxes'], pred['scores'], pred.get('masks', None)

            indices = torchvision.ops.nms(boxes, scores, self.thresh_iou_nms)
            if masks is not None:
                labels, boxes, scores, masks = labels[indices], boxes[indices], scores[indices], masks[indices]
            else:
                labels, boxes, scores = labels[indices], boxes[indices], scores[indices]

            indices = scores > self.thresh_score
            if masks is not None:
                labels, boxes, scores, masks = labels[indices], boxes[indices], scores[indices], masks[indices]
            else:
                labels, boxes, scores = labels[indices], boxes[indices], scores[indices]

            labels = labels.detach().cpu().numpy()
            boxes = boxes.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            if masks is not None:
                masks = masks.round().to(torch.float).squeeze(dim=1).detach().cpu().numpy()
            else:
                masks = [None] * len(labels)

            classes = {label: {'class': cls_name, 'color': color} for cls_name, (color, label) in self.classes.items()}
            max_dim = max(image.shape[0], image.shape[1])
            image_size = (max_dim, max_dim) if self.use_pad_to_square else image.shape[:2]
            rw, rh = image_size[1] / self.image_size[1], image_size[0] / self.image_size[0]
            for (label, box, score, mask) in zip(labels, boxes, scores, masks):
                if mask is not None:
                    mask = cv2.resize(mask, dsize=image_size[::-1], interpolation=cv2.INTER_NEAREST)
                    mask = mask[:image.shape[0], :image.shape[1]]
                box = np.int32([box[0] * rw, box[1] * rh, box[2] * rw, box[3] * rh])
                cv2.rectangle(img=image, pt1=tuple(box[:2]), pt2=tuple(box[2:]), color=classes[label]['color'], thickness=3)
                cv2.putText(img=image, text=(classes[label]['class'] + ': ' + str(score)),
                            org=(box[0], box[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.001 * max(image.shape[0], image.shape[1]),
                            color=classes[label]['color'], thickness=3, lineType=cv2.LINE_AA)
                if mask is not None:
                    image[mask.astype(dtype=bool)] = (image[mask.astype(dtype=bool)] * (1. - self.ratio_color_mask_on_image) \
                            + np.array([classes[label]['color']], dtype=float) * self.ratio_color_mask_on_image).astype(np.uint8)

            cv2.imwrite(image_path, image)

    def compute(self):
        pass

    def started(self, engine):
        self.reset()

    @torch.no_grad()
    def iteration_completed(self, engine):
        output = self._output_transform(engine.state.output)
        self.update(output)

    def completed(self, engine):
        self.compute()

    def _attach(self, engine):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed)
        if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
