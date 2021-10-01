import torch

from ...module import Module
from ignite import engine as e


class Evaluator(Module):
    '''
        Engine controls evaluating process.
        See Engine documentation for more details about parameters.
    '''
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
        super(Evaluator, self).__init__()
        self.device = device
        self.dataset = dataset
        self.max_epochs = max_epochs
        self.engine = e.Engine(self._update)

    def run(self):
        return self.engine.run(self.dataset, self.max_epochs)

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        self.model = self.frame['model'].to(self.device)

    def _update(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
            predictions = self.model.inference(params[0])
            groundtruths = self.get_groundtruths(
                samples=params[0], targets=params[1], image_info=params[2]
            )

            return predictions, groundtruths

    def get_groundtruths(
        self,
        samples: torch.Tensor,  # N x 3 x 416 x 416
        targets: Tuple[torch.Tensor],
        image_info: dict
    ) -> List[Dict[str, torch.Tensor]]:

        groundtruths = []

        s1_targets = targets[0]  # N x 3 x 13 x 13 x (5 + C)
        # s2_targets = targets[1]  # N x 3 x 26 x 26 x (5 + C)
        # s3_targets = targets[2]  # N x 3 x 52 x 52 x (5 + C)

        num_samples = targets[0].shape[0]  # N
        device = samples.device

        for i in range(num_samples):
            grid_size = samples.shape[2] / target.shape[2]

            indices = target[:, :, :, 0] == 1  # 3 x 13 x 13 x 6
            boxes = target[indices]  # n_boxes x 6

            x = (target[..., 1:2] == boxes[:, 1]).nonzero(as_tuple=True)[2]  # columns
            y = (target[..., 2:3] == boxes[:, 2]).nonzero(as_tuple=True)[1]  # rows

            boxes[:, 1] += x
            boxes[:, 2] += y
            boxes[:, 1:5] *= grid_size
            boxes[:, [1, 2]] -= boxes[:, [3, 4]] / 2  # x1, y1 = x - w / 2, y - h / 2
            boxes[:, [3, 4]] += boxes[:, [1, 2]]  # x2 = x1 + w, y2 = y1 + h

            groundtruth = {
                'image_id': torch.tensor(
                    [image_info['image_id'][i].item()], dtype=torch.int64, device=device
                ),
                'labels': torch.tensor(
                    boxes[:, 5], dtype=torch.int64, device=device
                ),
                'boxes': torch.tensor(
                    boxes[:, 1:5], dtype=torch.float32, device=device
                ),
            }

            groundtruths.append(groundtruth)

        return groundtruths
