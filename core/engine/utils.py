import time
from typing import List, Tuple

import torch
from torch import nn


def prepare_device(n_gpu_use: int = 0) -> Tuple[str, List[int]]:
    n_gpu = torch.cuda.device_count()  # get all GPU indices of system.

    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0

    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are available on this machine.")
        n_gpu_use = n_gpu

    device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
    gpu_indices = list(range(n_gpu_use))

    return device, gpu_indices


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class CustomDataParallel(nn.DataParallel):
    """force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """
    def __init__(self, module, num_gpus):
        super().__init__(module)
        self.num_gpus = num_gpus

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = [f'cuda:{x}' for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus

        if splits == 0:
            raise Exception('Batchsize must be greater than num_gpus.')

        return (
            [
                (
                    inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
                    inputs[1][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True)
                )
                for device_idx in range(len(devices))
            ],
            [kwargs] * len(devices)
        )
