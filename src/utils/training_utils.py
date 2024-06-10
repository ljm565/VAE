import os
import numpy as np

import torch

from utils import LOGGER, TQDM



def init_progress_bar(dloader, is_rank_zero, loss_names, nb):
    if is_rank_zero:
        header = tuple(['Epoch'] + loss_names)
        LOGGER.info(('\n' + '%15s' * (1 + len(loss_names))) % header)
        pbar = TQDM(enumerate(dloader), total=nb)
    else:
        pbar = enumerate(dloader)
    return pbar


def choose_proper_resume_model(resume_dir, type):
    weights_dir = os.listdir(os.path.join(resume_dir, 'weights'))
    try:
        weight = list(filter(lambda x: type in x, weights_dir))[0]
        return os.path.join(resume_dir, 'weights', weight)
    except IndexError:
        raise IndexError(f"There's no model path in {weights_dir} of type {type}")
    

def vae_loss(x, output, mu, log_var, decoder_loss):
    batch_size = x.size(0)
    x = x.view(batch_size, -1)

    BCE_loss = decoder_loss(output, x)
    KLD_loss = 0.5 * torch.sum((torch.square(mu) + torch.exp(log_var) - log_var - 1))

    return BCE_loss + KLD_loss


def make_z(mu, log_var):
    std = np.exp(0.5 * log_var)
    eps = torch.randn_like(torch.Tensor(std))
    return mu + std*eps.numpy()
