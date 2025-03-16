import torch
import torch.nn as nn
from src.models.base_gan import BaseGAN

class ResidualBlock(nn.Module):
    """Residual block for generator"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = self.activation(out)
        return out

class CTGANGenerator(nn.Module):
    """Generator network for CTGAN"""
    def __init__(self, input_dim, hidden_dim, num_residual_blocks=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim)
            for _ in range(num_residual_blocks)
        ])

        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.fc2(x)
        x = self.output_activation(x)
        return x

class CTGANDiscriminator(nn.Module):
    """Discriminator network for CTGAN with PatchGAN-style output"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class CTGAN(BaseGAN):
    """Conditional Tabular GAN implementation"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_residual_blocks: int = 5, 
                 device: str = 'cpu', lr_g: float = 0.0002, lr_d: float = 0.0002):
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.num_residual_blocks = num_residual_blocks
        
        # Initialize networks using the build methods
        self.generator = self.build_generator().to(device)
        self.discriminator = self.build_discriminator().to(device)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr_g,
            betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(0.5, 0.999)
        )

        # Initialize criterion
        self.criterion = nn.BCELoss()
    
    def build_generator(self) -> nn.Module:
        """Build generator network with residual blocks"""
        return CTGANGenerator(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_residual_blocks=self.num_residual_blocks
        )
    
    def build_discriminator(self) -> nn.Module:
        """Build discriminator network"""
        return CTGANDiscriminator(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        )
    
    def train_step(self, real_data: torch.Tensor) -> dict:
        """Perform one training step"""
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)

        # Labels for real and fake data
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)

        # Train Discriminator
        self.d_optimizer.zero_grad()

        # Train with real data
        d_real_output = self.discriminator(real_data)
        d_real_loss = self.criterion(d_real_output, real_labels)

        # Train with fake data
        noise = torch.randn(batch_size, self.input_dim).to(self.device)
        fake_data = self.generator(noise)
        d_fake_output = self.discriminator(fake_data.detach())
        d_fake_loss = self.criterion(d_fake_output, fake_labels)

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()
        g_output = self.discriminator(fake_data)
        g_loss = self.criterion(g_output, real_labels)
        g_loss.backward()
        self.g_optimizer.step()

        return {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item(),
            'd_real_loss': d_real_loss.item(),
            'd_fake_loss': d_fake_loss.item(),
            'd_real_mean': d_real_output.mean().item(),
            'd_fake_mean': d_fake_output.mean().item()
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
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_residual_blocks': self.num_residual_blocks,
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
        self.num_residual_blocks = state_dict.get('num_residual_blocks', 5)
        self.device = state_dict['device']