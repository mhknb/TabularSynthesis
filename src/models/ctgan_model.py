
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from src.models.base_gan import BaseGAN

class Residual(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input_):
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)

class Generator(nn.Module):
    def __init__(self, embedding_dim, generator_dim, data_dim):
        super().__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, input_):
        return self.seq(input_)

class Discriminator(nn.Module):
    def __init__(self, input_dim, discriminator_dim, pac=10):
        super().__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item
        seq += [nn.Linear(dim, 1)]
        self.seq = nn.Sequential(*seq)

    def forward(self, input_):
        return self.seq(input_.view(-1, self.pacdim))

class CTGANModel(BaseGAN):
    def __init__(self, input_dim: int, hidden_dim: int = 256, epochs=300, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.embedding_dim = hidden_dim
        self.generator_dim = (hidden_dim, hidden_dim)
        self.discriminator_dim = (hidden_dim, hidden_dim)
        self.epochs = epochs
        self.batch_size = 500
        self.pac = 10

        self.generator = Generator(
            self.embedding_dim,
            self.generator_dim,
            input_dim
        ).to(device)

        self.discriminator = Discriminator(
            input_dim,
            self.discriminator_dim,
            self.pac
        ).to(device)

        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.9)
        )

        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.9)
        )

    def train_step(self, real_data: torch.Tensor) -> dict:
        batch_size = min(self.batch_size, len(real_data))
        real_data = real_data.to(self.device)

        # Train Discriminator
        for _ in range(5):  # n_critic=5
            self.d_optimizer.zero_grad()
            noise = torch.randn(batch_size, self.embedding_dim).to(self.device)
            fake_data = self.generator(noise)
            
            fake_validity = self.discriminator(fake_data.detach())
            real_validity = self.discriminator(real_data)
            
            d_loss = -(torch.mean(real_validity) - torch.mean(fake_validity))
            d_loss.backward()
            self.d_optimizer.step()

            # Weight clipping for WGAN
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

        # Train Generator
        self.g_optimizer.zero_grad()
        noise = torch.randn(batch_size, self.embedding_dim).to(self.device)
        fake_data = self.generator(noise)
        fake_validity = self.discriminator(fake_data)
        
        g_loss = -torch.mean(fake_validity)
        g_loss.backward()
        self.g_optimizer.step()

        return {
            'generator_loss': g_loss.item(),
            'discriminator_loss': d_loss.item()
        }

    def generate_samples(self, num_samples: int) -> torch.Tensor:
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.embedding_dim).to(self.device)
            generated = self.generator(noise)
        self.generator.train()
        return generated

    def state_dict(self):
        return {
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim,
            'device': self.device
        }

    def load_state_dict(self, state_dict):
        self.generator.load_state_dict(state_dict['generator_state'])
        self.discriminator.load_state_dict(state_dict['discriminator_state'])
        self.g_optimizer.load_state_dict(state_dict['g_optimizer_state'])
        self.d_optimizer.load_state_dict(state_dict['d_optimizer_state'])
        self.input_dim = state_dict['input_dim']
        self.embedding_dim = state_dict['embedding_dim']
        self.device = state_dict['device']
