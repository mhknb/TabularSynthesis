import torch
import torch.nn as nn
from src.models.base_gan import BaseGAN

class WGAN(BaseGAN):
    """WGAN implementation for tabular data"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, clip_value: float = 0.01, n_critic: int = 5, 
             lr_g: float = 0.0001, lr_d: float = 0.0001, lambda_gp: float = 10.0, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.clip_value = clip_value
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        self.generator = self.build_generator().to(device)
        self.discriminator = self.build_discriminator().to(device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))

        # Metrics for optimization
        self.eval_metrics = {
            'gen_loss': [],
            'disc_loss': [],
            'wasserstein_distance': []
        }

    def build_generator(self) -> nn.Module:
        """Build generator network with enhanced architecture"""
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
        """Build discriminator network with enhanced architecture"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_dim, 1)
        )

    def train_step(self, real_data: torch.Tensor) -> dict:
        """Perform one training step"""
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)

        # Train Discriminator
        self.d_optimizer.zero_grad()

        # Generate fake data
        noise = torch.randn(batch_size, self.input_dim).to(self.device)
        fake_data = self.generator(noise)

        # Discriminator scores
        disc_real = self.discriminator(real_data)
        disc_fake = self.discriminator(fake_data.detach())

        # Wasserstein loss
        wasserstein_distance = torch.mean(disc_real) - torch.mean(disc_fake)
        disc_loss = -wasserstein_distance

        # Gradient penalty
        alpha = torch.rand(batch_size, 1, device=self.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        disc_interpolated = self.discriminator(interpolated)
        grad_outputs = torch.ones_like(disc_interpolated)
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_norm = gradients.norm(2, 1)
        grad_penalty = ((grad_norm - 1) ** 2).mean()
        disc_loss += self.lambda_gp * grad_penalty

        disc_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()
        fake_data = self.generator(noise)
        disc_fake = self.discriminator(fake_data)
        gen_loss = -torch.mean(disc_fake)
        gen_loss.backward()
        self.g_optimizer.step()


        self.eval_metrics['gen_loss'].append(gen_loss.item())
        self.eval_metrics['disc_loss'].append(disc_loss.item())
        self.eval_metrics['wasserstein_distance'].append(wasserstein_distance.item())

        return {
            'disc_loss': disc_loss.item(),
            'gen_loss': gen_loss.item(),
            'wasserstein_distance': wasserstein_distance.item()
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
            'clip_value': self.clip_value,
            'n_critic': self.n_critic,
            'lambda_gp': self.lambda_gp,
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
        self.clip_value = state_dict['clip_value']
        self.n_critic = state_dict['n_critic']
        self.lambda_gp = state_dict['lambda_gp']
        self.device = state_dict['device']