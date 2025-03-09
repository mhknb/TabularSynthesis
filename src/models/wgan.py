import torch
import torch.nn as nn
from src.models.base_gan import BaseGAN

class WGAN(BaseGAN):
    """WGAN implementation for tabular data"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, clip_value: float = 0.01, n_critic: int = 5, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.clip_value = clip_value
        self.n_critic = n_critic
        self.generator = self.build_generator().to(device)
        self.critic = self.build_critic().to(device)

        self.g_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=0.00005)
        self.c_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=0.00005)
        
    def build_discriminator(self) -> nn.Module:
        """Alias for build_critic to satisfy BaseGAN interface"""
        return self.build_critic()

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

    def build_critic(self) -> nn.Module:
        """Build critic network with enhanced architecture"""
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

        # Train Critic
        critic_losses = []
        for _ in range(self.n_critic):
            self.c_optimizer.zero_grad()

            # Generate fake data
            noise = torch.randn(batch_size, self.input_dim).to(self.device)
            fake_data = self.generator(noise)

            # Critic scores
            critic_real = self.critic(real_data)
            critic_fake = self.critic(fake_data.detach())

            # Wasserstein loss
            critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic_loss.backward()
            self.c_optimizer.step()

            # Clip critic weights
            for p in self.critic.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)

            critic_losses.append(critic_loss.item())

        # Train Generator
        self.g_optimizer.zero_grad()
        fake_data = self.generator(noise)
        critic_fake = self.critic(fake_data)
        g_loss = -torch.mean(critic_fake)
        g_loss.backward()
        self.g_optimizer.step()

        return {
            'critic_loss': sum(critic_losses) / len(critic_losses),
            'generator_loss': g_loss.item()
        }

    def state_dict(self):
        """Get state dict for model persistence"""
        return {
            'generator_state': self.generator.state_dict(),
            'critic_state': self.critic.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'c_optimizer_state': self.c_optimizer.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'clip_value': self.clip_value,
            'n_critic': self.n_critic,
            'device': self.device
        }

    def load_state_dict(self, state_dict):
        """Load state dict for model persistence"""
        self.generator.load_state_dict(state_dict['generator_state'])
        self.critic.load_state_dict(state_dict['critic_state'])
        self.g_optimizer.load_state_dict(state_dict['g_optimizer_state'])
        self.c_optimizer.load_state_dict(state_dict['c_optimizer_state'])
        self.input_dim = state_dict['input_dim']
        self.hidden_dim = state_dict['hidden_dim']
        self.clip_value = state_dict['clip_value']
        self.n_critic = state_dict['n_critic']
        self.device = state_dict['device']
