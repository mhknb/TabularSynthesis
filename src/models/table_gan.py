import torch
import torch.nn as nn
from src.models.base_gan import BaseGAN

class TableGAN(BaseGAN):
    """TableGAN implementation for tabular data"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, device: str = 'cpu', min_batch_size: int = 2):
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.min_batch_size = min_batch_size
        self.generator = self.build_generator().to(device)
        self.discriminator = self.build_discriminator().to(device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def build_generator(self) -> nn.Module:
        """Build generator network with improved architecture and batch handling"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim, momentum=0.01),  # Reduced momentum for better small batch handling
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2, momentum=0.01),
            nn.ReLU(),

            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2, momentum=0.01),
            nn.ReLU(),

            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim, momentum=0.01),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Tanh()
        )

    def build_discriminator(self) -> nn.Module:
        """Build discriminator network with improved architecture"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def validate_batch(self, batch: torch.Tensor) -> bool:
        """Validate batch size is sufficient for training"""
        return batch.size(0) >= self.min_batch_size

    def train_step(self, real_data: torch.Tensor) -> dict:
        """Perform one training step with batch size validation"""
        # Validate batch size
        if not self.validate_batch(real_data):
            raise ValueError(f"Batch size {real_data.size(0)} is too small. Minimum required: {self.min_batch_size}")

        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)

        # Train Discriminator
        self.d_optimizer.zero_grad()

        label_real = torch.ones(batch_size, 1).to(self.device)
        label_fake = torch.zeros(batch_size, 1).to(self.device)

        output_real = self.discriminator(real_data)
        d_loss_real = nn.BCELoss()(output_real, label_real)

        noise = torch.randn(batch_size, self.input_dim).to(self.device)
        fake_data = self.generator(noise)
        output_fake = self.discriminator(fake_data.detach())
        d_loss_fake = nn.BCELoss()(output_fake, label_fake)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()
        output_fake = self.discriminator(fake_data)
        g_loss = nn.BCELoss()(output_fake, label_real)
        g_loss.backward()
        self.g_optimizer.step()

        return {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item()
        }

    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate synthetic samples"""
        with torch.no_grad():
            # Generate in batches to avoid memory issues
            batch_size = min(self.min_batch_size * 4, num_samples)
            num_batches = (num_samples + batch_size - 1) // batch_size
            samples_list = []

            for i in range(num_batches):
                current_batch_size = min(batch_size, num_samples - i * batch_size)
                if current_batch_size < self.min_batch_size:
                    current_batch_size = self.min_batch_size
                noise = torch.randn(current_batch_size, self.input_dim).to(self.device)
                samples = self.generator(noise)
                samples_list.append(samples)

            all_samples = torch.cat(samples_list, dim=0)
            # Trim excess samples if needed
            return all_samples[:num_samples]

    def state_dict(self):
        """Get state dict for model persistence"""
        return {
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'device': self.device,
            'min_batch_size': self.min_batch_size
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
        self.min_batch_size = state_dict.get('min_batch_size', 2)  # Default for backward compatibility