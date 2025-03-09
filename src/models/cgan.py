
import torch
import torch.nn as nn
from src.models.base_gan import BaseGAN

class CGAN(BaseGAN):
    """Conditional GAN implementation for tabular data"""

    def __init__(self, input_dim: int, condition_dim: int, hidden_dim: int = 256, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.generator = self.build_generator().to(device)
        self.discriminator = self.build_discriminator().to(device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def build_generator(self) -> nn.Module:
        """Build generator network with condition input"""
        class ConditionalGenerator(nn.Module):
            def __init__(self, input_dim, condition_dim, hidden_dim):
                super().__init__()
                self.input_dim = input_dim
                self.condition_dim = condition_dim
                
                # Noise processing layers
                self.noise_fc = nn.Linear(input_dim, hidden_dim)
                self.noise_bn = nn.BatchNorm1d(hidden_dim)
                
                # Condition processing layers
                self.cond_fc = nn.Linear(condition_dim, hidden_dim)
                self.cond_bn = nn.BatchNorm1d(hidden_dim)
                
                # Combined processing layers
                self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
                self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
                
                self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
                self.bn2 = nn.BatchNorm1d(hidden_dim)
                
                self.fc3 = nn.Linear(hidden_dim, input_dim - condition_dim)
                
                self.relu = nn.ReLU()
                self.tanh = nn.Tanh()
                
            def forward(self, noise, condition):
                # Process noise
                n = self.relu(self.noise_bn(self.noise_fc(noise)))
                
                # Process condition
                c = self.relu(self.cond_bn(self.cond_fc(condition)))
                
                # Combine and process
                x = torch.cat([n, c], dim=1)
                x = self.relu(self.bn1(self.fc1(x)))
                x = self.relu(self.bn2(self.fc2(x)))
                x = self.tanh(self.fc3(x))
                
                return x
        
        return ConditionalGenerator(self.input_dim, self.condition_dim, self.hidden_dim)

    def build_discriminator(self) -> nn.Module:
        """Build discriminator network with condition input"""
        class ConditionalDiscriminator(nn.Module):
            def __init__(self, input_dim, condition_dim, hidden_dim):
                super().__init__()
                self.input_dim = input_dim - condition_dim
                self.condition_dim = condition_dim
                
                # Data processing layers
                self.data_fc = nn.Linear(self.input_dim, hidden_dim)
                
                # Condition processing layers
                self.cond_fc = nn.Linear(condition_dim, hidden_dim)
                
                # Combined processing layers
                self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
                self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, 1)
                
                self.leaky_relu = nn.LeakyReLU(0.2)
                self.dropout = nn.Dropout(0.3)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, data, condition):
                # Process data
                d = self.leaky_relu(self.data_fc(data))
                d = self.dropout(d)
                
                # Process condition
                c = self.leaky_relu(self.cond_fc(condition))
                c = self.dropout(c)
                
                # Combine and process
                x = torch.cat([d, c], dim=1)
                x = self.leaky_relu(self.fc1(x))
                x = self.dropout(x)
                x = self.leaky_relu(self.fc2(x))
                x = self.dropout(x)
                x = self.sigmoid(self.fc3(x))
                
                return x
        
        return ConditionalDiscriminator(self.input_dim, self.condition_dim, self.hidden_dim)

    def train_step(self, real_data: torch.Tensor, condition: torch.Tensor) -> dict:
        """Perform one training step with condition"""
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)
        condition = condition.to(self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()

        # Real data
        label_real = torch.ones(batch_size, 1).to(self.device)
        label_fake = torch.zeros(batch_size, 1).to(self.device)
        
        output_real = self.discriminator(real_data, condition)
        d_loss_real = nn.BCELoss()(output_real, label_real)

        # Fake data
        noise = torch.randn(batch_size, self.input_dim).to(self.device)
        fake_data = self.generator(noise, condition)
        output_fake = self.discriminator(fake_data.detach(), condition)
        d_loss_fake = nn.BCELoss()(output_fake, label_fake)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()
        output_fake = self.discriminator(fake_data, condition)
        g_loss = nn.BCELoss()(output_fake, label_real)
        g_loss.backward()
        self.g_optimizer.step()

        return {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item()
        }
        
    def generate_samples(self, num_samples: int, conditions: torch.Tensor = None) -> torch.Tensor:
        """Generate synthetic samples with conditions"""
        with torch.no_grad():
            if conditions is None:
                # Use random conditions if none provided
                conditions = torch.rand(num_samples, self.condition_dim).to(self.device)
                
            noise = torch.randn(num_samples, self.input_dim).to(self.device)
            fake_data = self.generator(noise, conditions)
            # Combine generated data with conditions
            return torch.cat([fake_data, conditions], dim=1)
            
    def state_dict(self):
        """Get state dict for model persistence"""
        return {
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
            'input_dim': self.input_dim,
            'condition_dim': self.condition_dim,
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
        self.condition_dim = state_dict['condition_dim']
        self.hidden_dim = state_dict['hidden_dim']
        self.device = state_dict['device']
import torch
import torch.nn as nn
from src.models.base_gan import BaseGAN

class CGAN(BaseGAN):
    """Conditional GAN implementation for tabular data"""

    def __init__(self, input_dim: int, condition_dim: int, hidden_dim: int = 256, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.generator = self.build_generator().to(device)
        self.discriminator = self.build_discriminator().to(device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def build_generator(self) -> nn.Module:
        """Build generator network with condition input"""
        class ConditionalGenerator(nn.Module):
            def __init__(self, input_dim, condition_dim, hidden_dim):
                super().__init__()
                self.input_dim = input_dim
                self.condition_dim = condition_dim
                
                # Noise processing layers
                self.noise_fc = nn.Linear(input_dim, hidden_dim)
                self.noise_bn = nn.BatchNorm1d(hidden_dim)
                
                # Condition processing layers
                self.cond_fc = nn.Linear(condition_dim, hidden_dim)
                self.cond_bn = nn.BatchNorm1d(hidden_dim)
                
                # Combined processing layers
                self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
                self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
                
                self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
                self.bn2 = nn.BatchNorm1d(hidden_dim)
                
                self.fc3 = nn.Linear(hidden_dim, input_dim - condition_dim)
                
                self.relu = nn.ReLU()
                self.tanh = nn.Tanh()
                
            def forward(self, noise, condition):
                # Process noise
                n = self.relu(self.noise_bn(self.noise_fc(noise)))
                
                # Process condition
                c = self.relu(self.cond_bn(self.cond_fc(condition)))
                
                # Combine and process
                x = torch.cat([n, c], dim=1)
                x = self.relu(self.bn1(self.fc1(x)))
                x = self.relu(self.bn2(self.fc2(x)))
                x = self.tanh(self.fc3(x))
                
                return x
        
        return ConditionalGenerator(self.input_dim, self.condition_dim, self.hidden_dim)

    def build_discriminator(self) -> nn.Module:
        """Build discriminator network with condition input"""
        class ConditionalDiscriminator(nn.Module):
            def __init__(self, input_dim, condition_dim, hidden_dim):
                super().__init__()
                self.input_dim = input_dim - condition_dim
                self.condition_dim = condition_dim
                
                # Data processing layers
                self.data_fc = nn.Linear(self.input_dim, hidden_dim)
                
                # Condition processing layers
                self.cond_fc = nn.Linear(condition_dim, hidden_dim)
                
                # Combined processing layers
                self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
                self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, 1)
                
                self.leaky_relu = nn.LeakyReLU(0.2)
                self.dropout = nn.Dropout(0.3)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, data, condition):
                # Process data
                d = self.leaky_relu(self.data_fc(data))
                d = self.dropout(d)
                
                # Process condition
                c = self.leaky_relu(self.cond_fc(condition))
                c = self.dropout(c)
                
                # Combine and process
                x = torch.cat([d, c], dim=1)
                x = self.leaky_relu(self.fc1(x))
                x = self.dropout(x)
                x = self.leaky_relu(self.fc2(x))
                x = self.dropout(x)
                x = self.sigmoid(self.fc3(x))
                
                return x
        
        return ConditionalDiscriminator(self.input_dim, self.condition_dim, self.hidden_dim)

    def train_step(self, real_data: torch.Tensor) -> dict:
        """Perform one training step with conditions"""
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)
        
        # Split data into features and condition
        condition = real_data[:, -self.condition_dim:].detach()
        real_features = real_data[:, :-self.condition_dim]

        # Train Discriminator
        self.d_optimizer.zero_grad()

        label_real = torch.ones(batch_size, 1).to(self.device)
        label_fake = torch.zeros(batch_size, 1).to(self.device)

        output_real = self.discriminator(real_features, condition)
        d_loss_real = nn.BCELoss()(output_real, label_real)

        noise = torch.randn(batch_size, self.input_dim).to(self.device)
        fake_features = self.generator(noise, condition)
        output_fake = self.discriminator(fake_features.detach(), condition)
        d_loss_fake = nn.BCELoss()(output_fake, label_fake)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()
        output_fake = self.discriminator(fake_features, condition)
        g_loss = nn.BCELoss()(output_fake, label_real)
        g_loss.backward()
        self.g_optimizer.step()

        return {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item()
        }

    def generate_samples(self, num_samples: int, conditions=None) -> torch.Tensor:
        """Generate synthetic samples with provided conditions"""
        with torch.no_grad():
            if conditions is None:
                # Generate random conditions if none provided
                conditions = torch.rand(num_samples, self.condition_dim).to(self.device)
            else:
                conditions = conditions.to(self.device)
                
            noise = torch.randn(num_samples, self.input_dim).to(self.device)
            fake_features = self.generator(noise, conditions)
            
            # Combine generated features with conditions
            return torch.cat([fake_features, conditions], dim=1)

    def state_dict(self):
        """Get state dict for model persistence"""
        return {
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
            'input_dim': self.input_dim,
            'condition_dim': self.condition_dim,
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
        self.condition_dim = state_dict['condition_dim']
        self.hidden_dim = state_dict['hidden_dim']
        self.device = state_dict['device']
