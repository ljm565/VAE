import torch
import os
from PIL import Image
from tqdm import tqdm
import numpy as np


def save_checkpoint(file, model, optimizer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')


def make_img_data(path, trans):
    files = os.listdir(path)
    data = [trans(Image.open(path+file)) for file in tqdm(files) if not file.startswith('.')]
    return data    


def VAE_loss(x, output, mu, log_var, decoder_loss):
    batch_size = x.size(0)
    x = x.view(batch_size, -1)

    BCE_loss = decoder_loss(output, x)
    KLD_loss = 0.5 * torch.sum((torch.square(mu) + torch.exp(log_var) - log_var - 1))

    return BCE_loss + KLD_loss


def make_z(mu, log_var):
    std = np.exp(0.5 * log_var)
    eps = torch.randn_like(torch.Tensor(std))
    return mu + std*eps.numpy()



