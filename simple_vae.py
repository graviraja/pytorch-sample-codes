'''This code contains the implementation of simple VAE

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np

from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data', one_hot=True)
batch_size = 64
z_dim = 100
x_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]

        hidden = F.relu(self.linear(x))
        z_mu = self.mu(hidden)
        z_var = self.var(hidden)

        return z_mu, z_var


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()

        self.linear = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = F.relu(self.linear(x))
        predicted = F.sigmoid(self.out(hidden))
        return predicted


class VAE(nn.Module):
    def __init__(self, enc, dec, z_dim):
        super().__init__()

        self.z_dim = z_dim
        self.enc = enc
        self.dec = dec

    def sample(self, mu, log_var, batch_size):
        eps = Variable(torch.randn(batch_size, self.z_dim))
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x):
        batch_size = x.shape[0]
        z_mu, z_var = self.enc(x)
        x_sample = self.sample(z_mu, z_var, batch_size)
        predicted = self.dec(x_sample)
        return predicted, z_mu, z_var

encoder = Encoder(x_dim, h_dim, z_dim)
decoder = Decoder(z_dim, h_dim, x_dim)
model = VAE(encoder, decoder, z_dim)

optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(100000):
    X, _ = mnist.train.next_batch(batch_size)
    X = Variable(torch.from_numpy(X))

    optimizer.zero_grad()
    X_sample, z_mu, z_var = model(X)

    recon_loss = F.binary_cross_entropy(X_sample, X)
    kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)

    loss = recon_loss + kl_loss

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"epoch : {epoch}, loss: {loss.item()}")
