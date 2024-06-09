import os

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, distributed, random_split

from models import VAE
from utils import RANK
from utils.data_utils import DLoader, seed_worker

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def get_model(config, device):
    model = VAE(config).to(device)
    return model


def build_dataset(config, modes):
    dataset_dict = {}
    if config.MNIST_train:
        # set to MNIST size
        config.width, config.height = 28, 28

        # init train, validation, test sets
        mnist_path = config.MNIST.path
        mnist_valset_proportion = config.MNIST.MNIST_valset_proportion
        trainset = dsets.MNIST(root=mnist_path, transform=transforms.ToTensor(), train=True, download=True)
        valset_l = int(len(trainset) * mnist_valset_proportion)
        trainset_l = len(trainset) - valset_l
        trainset, valset = random_split(trainset, [trainset_l, valset_l])
        testset = dsets.MNIST(root=mnist_path, transform=transforms.ToTensor(), train=False, download=True)
        tmp_dsets = {'train': trainset, 'validation': valset, 'test': testset}
        for mode in modes:
            dataset_dict[mode] = tmp_dsets[mode]
    else:
        for mode in modes:
            dataset_dict[mode] = DLoader(config.CUSTOM.get(f'{mode}_data_path'))
    return dataset_dict


def build_dataloader(dataset, batch, workers, shuffle=True, is_ddp=False):
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if not is_ddp else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)


def get_data_loader(config, modes, is_ddp=False):
    datasets = build_dataset(config, modes)
    dataloaders = {m: build_dataloader(datasets[m], 
                                       config.batch_size, 
                                       config.workers, 
                                       shuffle=(m == 'train'), 
                                       is_ddp=is_ddp) for m in modes}

    return dataloaders