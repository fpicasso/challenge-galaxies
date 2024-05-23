import os
import h5py

import torch
from torch import nn
import torchvision
from torchvision.transforms import v2

import numpy as np
import matplotlib.pyplot as plt


from pathlib import Path



class Autoencoder(nn.Module):

    def __init__(self, image_size=64,num_channels=3, latent_dims=128, num_filters=32, do_sampling=False):
        super(Autoencoder, self).__init__()

        self.latent_dims  = latent_dims
        self.image_size   = image_size
        self.num_channels = num_channels
        self.num_filters  = num_filters
        self.do_sampling  = do_sampling

        # Encoder
        self.conv_encoder = nn.Sequential(
            # TODO: Build the convolutional layers (torch.nn.Conv2d) here
            torch.nn.Conv2d(self.num_channels, self.num_channels, (4,4), 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.num_channels,self.num_channels, (4,4), 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.num_channels,self.num_channels, (4,4), 2, 1),
            torch.nn.ReLU(),
        )

        # Linear Encoder
        # TODO: Match the dimensionality of the first and last layer here!
        self.fc_lin_down = nn.Linear(64*self.num_filters, 8 * self.num_filters)
        self.fc_mu       = nn.Linear(8 * self.num_filters, self.latent_dims)
        self.fc_logvar   = nn.Linear(self.latent_dims, self.latent_dims)
        self.fc_z        = nn.Linear(self.latent_dims, 8 * self.num_filters)
        self.fc_lin_up   = nn.Linear(8 * self.num_filters, 64*self.num_filters)

        # Decoder
        self.conv_decoder = nn.Sequential(
            # TODO: Implement the reverse of the encoder here using torch.nn.ConvTranspose2d layers
            # The last activation here should be a sigmoid to keep the pixel values clipped in [0, 1)
            torch.nn.Conv2d(self.num_channels, self.num_channels, (4,4), 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.num_channels,self.num_channels, (4,4), 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.num_channels,self.num_channels, (4,4), 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        ''' Encoder: output is (mean, log(variance))'''
        x       = self.conv_encoder(x)
        # Here, we resize the convolutional output appropriately for a linear layer
        # TODO: Fill in the correct dimensionality for the reordering
        x       = x.view(-1, self.num_filters * 8 * 8)
        x       = self.fc_lin_down(x)
        x       = nn.functional.relu(x)
        mu      = self.fc_mu(x)
        logvar  = self.fc_logvar(x)
        return mu, logvar

    def sample(self, mu, logvar):
        ''' Sample from Gaussian with mean `mu` and SD `sqrt(exp(logvarz))`'''
        # Only use the full mean/stddev procedure if we want to later do sampling
        # And only reparametrise if we are in training mode
        if self.training and self.do_sampling:
            std = torch.exp(logvar * 0.5)
            eps = torch.randn_like(std)
            sample = mu + (eps * std)
            return sample
        else:
            return mu

    def decode(self, z):
        '''Decoder: produces reconstruction from sample of latent z'''
        z = self.fc_z(z)
        z = nn.functional.relu(z)
        z = self.fc_lin_up(z)
        z = nn.functional.relu(z)
        # TODO: Fill in the correct dimensionality for the reordering here again
        z = z.view(-1, self.num_filters, 8, 8)
        z = self.conv_decoder(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_hat = self.decode(z)
        if self.do_sampling:
            return x_hat, mu, logvar
        else:
            return x_hat, None, None


