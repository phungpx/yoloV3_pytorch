import torch

from ignite import engine as e
from abc import abstractmethod

from ...module import Module


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

    def __init__(self, dataset, device, max_epochs=1):
        super(Trainer, self).__init__(dataset, device, max_epochs)

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        assert 'optim' in self.frame, 'The frame does not have optim.'
        assert 'loss' in self.frame, 'The frame does not have loss.'
        self.model = self.frame['model'].to(self.device)
        self.optimizer = self.frame['optim']
        self.loss = self.frame['loss']

    def _update(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()
        samples = torch.stack(batch[0], dim=0).to(self.device)
        targets = [torch.stack(target, dim=0).to(self.device) for target in zip(*batch[1])]  # Tuple of target Tensors
        preds = self.model(samples)  # Tuple of prediction Tensors

        losses = self.loss(preds, targets)

        loss = losses[13] + losses[26] + losses[52]
        loss.backward()

        self.optimizer.step()

        return loss.item()


class Evaluator(Engine):
    '''
        Engine controls evaluating process.
        See Engine documentation for more details about parameters.
    '''

    def __init__(self, dataset, device, max_epochs=1):
        super(Evaluator, self).__init__(dataset, device, max_epochs)

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        self.model = self.frame['model'].to(self.device)

    def _update(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            batch[0] = torch.stack(batch[0], dim=0).to(self.device)
            batch[1] = [torch.stack(target, dim=0).to(self.device) for target in zip(*batch[1])]  # Tuple of target Tensors
            batch[0] = self.model(batch[0])  # Tuple of prediction Tensors

            return batch
