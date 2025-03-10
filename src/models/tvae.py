import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_gan import BaseGAN

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)  # mu
        self.fc32 = nn.Linear(hidden_dim, latent_dim)  # logvar

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        return self.fc31(h), self.fc32(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, z):
        h = F.relu(self.bn1(self.fc1(z)))
        h = F.relu(self.bn2(self.fc2(h)))
        return torch.tanh(self.fc3(h))

class TVAE(BaseGAN):
    """Tabular Variational Autoencoder implementation"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 128, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Create encoder and decoder instances
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim).to(device)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim).to(device)

        # We don't need discriminator for VAE
        self.discriminator = None

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=0.001
        )

    def build_generator(self):
        """Required by BaseGAN but not used in TVAE"""
        return self.encoder

    def build_discriminator(self):
        """Required by BaseGAN but not used in TVAE"""
        return self.decoder

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def train_step(self, real_data: torch.Tensor) -> dict:
        """Perform one training step"""
        self.optimizer.zero_grad()

        # Move data to device
        real_data = real_data.to(self.device)

        # Encode
        mu, logvar = self.encoder(real_data)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        recon = self.decoder(z)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, real_data, reduction='sum')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        loss = recon_loss + kl_loss

        loss.backward()
        self.optimizer.step()

        # Return losses in the format expected by training progress
        return {
            'generator_loss': recon_loss.item() / len(real_data),
            'discriminator_loss': kl_loss.item() / len(real_data)
        }

    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate synthetic samples"""
        with torch.no_grad():
            # Sample from normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            # Decode
            samples = self.decoder(z)
            return samples

    def state_dict(self):
        """Get state dict for model persistence"""
        return {
            'encoder_state': self.encoder.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'device': self.device
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict for model persistence"""
        self.encoder.load_state_dict(state_dict['encoder_state'])
        self.decoder.load_state_dict(state_dict['decoder_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.input_dim = state_dict['input_dim']
        self.hidden_dim = state_dict['hidden_dim']
        self.latent_dim = state_dict['latent_dim']
        self.device = state_dict['device']