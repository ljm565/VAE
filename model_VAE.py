import torch
import torch.nn as nn
from config import Config

class VAE(nn.Module):
    def __init__(self, config:Config, color_channel:int):
        super(VAE, self).__init__()
        self.height = config.height
        self.width = config.width
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.dropout = config.dropout
        self.color_channel = color_channel
        
        self.encoder = nn.Sequential(
            nn.Linear(self.height*self.width*self.color_channel, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim//2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim, self.height*self.width*self.color_channel),
            nn.Sigmoid()
        )
        self.fc_mu = nn.Linear(self.hidden_dim//2, self.latent_dim)
        self.fc_log_var = nn.Linear(self.hidden_dim//2, self.latent_dim)

    
    def reparameterization_trick(self, encoded):
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)

        return mu + std*eps, mu, log_var


    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = self.encoder(x)
        z, mu, log_var = self.reparameterization_trick(output)
        output = self.decoder(z)

        return output, mu, log_var