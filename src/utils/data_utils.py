import random
import numpy as np

import torch
from torch.utils.data import Dataset



def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DLoader(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(self.data)


    def __getitem__(self, idx):
        return self.data[idx], -1

    
    def __len__(self):
        return self.length