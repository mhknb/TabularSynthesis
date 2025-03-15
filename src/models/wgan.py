import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from src.models.base_gan import BaseGAN

class WGAN(BaseGAN):
    """WGAN implementation for tabular data"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, clip_value: float = 0.01, n_critic: int = 5, 
             lr_g: float = 0.0001, lr_d: float = 0.0001, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.clip_value = clip_value
        self.n_critic = n_critic

        self.generator = self.build_generator().to(device)
        self.discriminator = self.build_discriminator().to(device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))

        # Dynamic learning rate scheduling
        self.scheduler_g = torch.optim.lr_scheduler.StepLR(self.g_optimizer, step_size=10, gamma=0.9)
        self.scheduler_d = torch.optim.lr_scheduler.StepLR(self.d_optimizer, step_size=10, gamma=0.9)

    def build_generator(self) -> nn.Module:
        """Build generator network with dynamic layer sizes"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Tanh()
        )

    def build_discriminator(self) -> nn.Module:
        """Build discriminator network with spectral normalization"""
        return nn.Sequential(
            spectral_norm(nn.Linear(self.input_dim, self.hidden_dim)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim * 2)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Linear(self.hidden_dim * 2, self.hidden_dim)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Linear(self.hidden_dim, 1))
        )

    def train_step(self, real_data: torch.Tensor) -> dict:
        """Perform one training step"""
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)

        # Variables to store metrics
        total_disc_loss = 0
        wasserstein_distance = 0

        # Train Discriminator/Critic
        for _ in range(self.n_critic):
            self.d_optimizer.zero_grad()

            # Generate fake data
            noise = torch.randn(batch_size, self.input_dim).to(self.device)
            fake_data = self.generator(noise)

            # Extract features and get discriminator scores
            disc_real = self.discriminator(real_data)
            disc_fake = self.discriminator(fake_data.detach())

            # Wasserstein loss with spectral normalization
            curr_wasserstein_distance = torch.mean(disc_real) - torch.mean(disc_fake)
            disc_loss = -curr_wasserstein_distance

            # Update metrics
            total_disc_loss += disc_loss.item()
            wasserstein_distance = curr_wasserstein_distance.item()

            # Backward and optimize
            disc_loss.backward()
            self.d_optimizer.step()

            # Apply weight clipping to discriminator parameters
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)

        # Calculate average discriminator loss
        avg_disc_loss = total_disc_loss / self.n_critic

        # Train Generator
        self.g_optimizer.zero_grad()

        # Generate new fake data
        noise = torch.randn(batch_size, self.input_dim).to(self.device)
        fake_data = self.generator(noise)
        disc_fake = self.discriminator(fake_data)

        # Generator loss
        gen_loss = -torch.mean(disc_fake)
        gen_loss.backward()
        self.g_optimizer.step()

        # Return metrics dictionary
        return {
            'disc_loss': avg_disc_loss,
            'gen_loss': gen_loss.item(),
            'wasserstein_distance': wasserstein_distance
        }

    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate synthetic samples"""
        with torch.no_grad():
            noise = torch.randn(num_samples, self.input_dim).to(self.device)
            samples = self.generator(noise)
            return samples

    def state_dict(self):
        """Get state dict for model persistence"""
        return {
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
            'g_scheduler_state': self.scheduler_g.state_dict(),
            'd_scheduler_state': self.scheduler_d.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'clip_value': self.clip_value,
            'n_critic': self.n_critic,
            'device': self.device
        }

    def load_state_dict(self, state_dict):
        """Load state dict for model persistence"""
        # Initialize new model with the same architecture
        self.input_dim = state_dict['input_dim']
        self.hidden_dim = state_dict['hidden_dim']
        self.clip_value = state_dict['clip_value']
        self.n_critic = state_dict['n_critic']
        self.device = state_dict['device']

        # Rebuild networks with correct dimensions
        self.generator = self.build_generator().to(self.device)
        self.discriminator = self.build_discriminator().to(self.device)

        # Load the state dictionaries
        self.generator.load_state_dict(state_dict['generator_state'])
        self.discriminator.load_state_dict(state_dict['discriminator_state'])
        self.g_optimizer.load_state_dict(state_dict['g_optimizer_state'])
        self.d_optimizer.load_state_dict(state_dict['d_optimizer_state'])

        if 'g_scheduler_state' in state_dict:
            self.scheduler_g.load_state_dict(state_dict['g_scheduler_state'])
        if 'd_scheduler_state' in state_dict:
            self.scheduler_d.load_state_dict(state_dict['d_scheduler_state'])