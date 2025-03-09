import torch
import torch.nn as nn
from src.models.base_gan import BaseGAN

class TableGAN(BaseGAN):
    """TableGAN implementation for tabular data"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.generator = self.build_generator().to(device)
        self.discriminator = self.build_discriminator().to(device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def build_generator(self) -> nn.Module:
        """Build generator network with improved architecture"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.ReLU(),

            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.ReLU(),

            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Tanh()
        )

    def build_discriminator(self) -> nn.Module:
        """Build discriminator network with improved architecture"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # Add dropout for regularization

            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(self.hidden_dim, 1)
            #Removed Sigmoid, using BCEWithLogitsLoss instead.
        )

    def train_step(self, real_data: torch.Tensor) -> dict:
        """Perform one training step"""
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)

        # Train Discriminator
        self.d_optimizer.zero_grad()

        label_real = torch.ones(batch_size, 1).to(self.device)
        label_fake = torch.zeros(batch_size, 1).to(self.device)

        # Use BCEWithLogitsLoss which combines sigmoid and BCE
        criterion = nn.BCEWithLogitsLoss()

        output_real = self.discriminator(real_data)
        d_loss_real = criterion(output_real, label_real)

        noise = torch.randn(batch_size, self.input_dim).to(self.device)
        fake_data = self.generator(noise)
        output_fake = self.discriminator(fake_data.detach())
        d_loss_fake = criterion(output_fake, label_fake)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()
        output_fake = self.discriminator(fake_data)
        g_loss = criterion(output_fake, label_real)
        g_loss.backward()
        self.g_optimizer.step()

        return {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item()
        }

    def state_dict(self):
        """Get state dict for model persistence"""
        return {
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'device': self.device
        }

    def load_state_dict(self, state_dict):
        """Load state dict for model persistence"""
        self.generator.load_state_dict(state_dict['generator_state'])
        self.discriminator.load_state_dict(state_dict['discriminator_state'])
        self.g_optimizer.load_state_dict(state_dict['g_optimizer_state'])
        self.d_optimizer.load_state_dict(state_dict['d_optimizer_state'])
        self.input_dim = state_dict['input_dim']
        self.hidden_dim = state_dict['hidden_dim']
        self.device = state_dict['device']