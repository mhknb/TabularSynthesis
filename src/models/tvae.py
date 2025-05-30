import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_gan import BaseGAN

class TVAE(BaseGAN):
    """Tabular Variational Autoencoder implementation"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 128, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder and Decoder replace Generator and Discriminator
        self.encoder = self.build_encoder().to(device)
        self.decoder = self.build_decoder().to(device)

        # We don't need discriminator for VAE
        self.discriminator = None

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=0.001
        )

    def build_encoder(self) -> nn.Module:
        """Build encoder network (replaces generator in base class)"""
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
                h = F.leaky_relu(self.bn1(self.fc1(x)), 0.2)
                h = F.relu(self.bn2(self.fc2(h)))
                return self.fc31(h), self.fc32(h)

        return Encoder(self.input_dim, self.hidden_dim, self.latent_dim)

    def build_decoder(self) -> nn.Module:
        """Build decoder network (replaces discriminator in base class)"""
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

        return Decoder(self.latent_dim, self.hidden_dim, self.input_dim)

    def build_generator(self) -> nn.Module:
        """Implement required method from BaseGAN"""
        return self.encoder

    def build_discriminator(self) -> nn.Module:
        """Implement required method from BaseGAN"""
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

        # Calculate per sample losses for consistent reporting
        metrics = {
            'total_loss': loss.item() / len(real_data),
            'reconstruction_loss': recon_loss.item() / len(real_data),
            'kl_loss': kl_loss.item() / len(real_data)
        }

        return metrics

    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate synthetic samples"""
        with torch.no_grad():
            # Sample from normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            # Decode
            samples = self.decoder(z)
            return samples

    def optimize_hyperparameters(self, train_loader, n_epochs=50, n_iterations=10):
        """
        Perform Bayesian optimization of hyperparameters

        Args:
            train_loader: DataLoader with training data
            n_epochs: Number of epochs to train for each iteration
            n_iterations: Number of optimization iterations

        Returns:
            best_params: Dictionary of best parameters
            history_df: DataFrame with optimization history
        """
        from src.utils.optimization import BayesianOptimizer
        import pandas as pd
        import numpy as np
        import wandb
        import matplotlib.pyplot as plt

        # Define parameter ranges
        param_ranges = {
            'learning_rate': (0.00001, 0.001),
            'kl_weight': (0.1, 1.0),
            'latent_dim': (64, 256)
        }

        # Define objective function
        def objective_function(params):
            try:
                # Create temporary model with new parameters
                latent_dim = int(params['latent_dim'])
                temp_model = TVAE(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    latent_dim=latent_dim,
                    device=self.device
                )

                # Update optimizer with new learning rate
                temp_model.optimizer = torch.optim.Adam(
                    list(temp_model.encoder.parameters()) + list(temp_model.decoder.parameters()),
                    lr=params['learning_rate']
                )

                # KL weight for balancing reconstruction and KL loss
                kl_weight = params['kl_weight']

                # Train for a few epochs
                metrics_history = []
                for epoch in range(n_epochs):
                    epoch_metrics = {'epoch': epoch}
                    epoch_loss = 0.0
                    batch_count = 0

                    for i, real_data in enumerate(train_loader):
                        try:
                            # Custom training step with KL weight
                            temp_model.optimizer.zero_grad()

                            # Move data to device
                            real_data = real_data.to(temp_model.device)

                            # Encode
                            mu, logvar = temp_model.encoder(real_data)

                            # Reparameterize
                            z = temp_model.reparameterize(mu, logvar)

                            # Decode
                            recon = temp_model.decoder(z)

                            # Reconstruction loss (MSE)
                            recon_loss = F.mse_loss(recon, real_data, reduction='sum')

                            # KL divergence
                            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                            # Total loss with weight
                            loss = recon_loss + kl_weight * kl_loss

                            loss.backward()
                            temp_model.optimizer.step()

                            metrics = {
                                'total_loss': loss.item() / len(real_data),
                                'reconstruction_loss': recon_loss.item() / len(real_data),
                                'kl_loss': kl_loss.item() / len(real_data)
                            }

                            # Log metrics to wandb
                            if wandb.run is not None:
                                wandb.log({
                                    'batch_loss': metrics['total_loss'],
                                    'reconstruction_loss': metrics['reconstruction_loss'],
                                    'kl_divergence': metrics['kl_loss'],
                                    'epoch': epoch,
                                    'batch': i,
                                })

                            epoch_loss += metrics['total_loss']
                            batch_count += 1

                            # Generate and log synthetic data evaluation every few batches
                            if i % 10 == 0:
                                with torch.no_grad():
                                    # Generate synthetic samples
                                    synthetic_data = temp_model.generate_samples(len(real_data))

                                    # Calculate statistical similarity
                                    real_mean = real_data.mean(dim=0)
                                    real_std = real_data.std(dim=0)
                                    synth_mean = synthetic_data.mean(dim=0)
                                    synth_std = synthetic_data.std(dim=0)

                                    mean_diff = torch.mean(torch.abs(real_mean - synth_mean))
                                    std_diff = torch.mean(torch.abs(real_std - synth_std))

                                    if wandb.run is not None:
                                        # Create distribution plots
                                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                                        # Real data distribution
                                        axes[0].hist(real_data[:, 0].cpu().numpy(), bins=30, alpha=0.5, label='Real')
                                        axes[0].set_title('Real Data Distribution')

                                        # Synthetic data distribution
                                        axes[1].hist(synthetic_data[:, 0].cpu().numpy(), bins=30, alpha=0.5, label='Synthetic')
                                        axes[1].set_title('Synthetic Data Distribution')

                                        # Log evaluation metrics and plot
                                        wandb.log({
                                            'mean_difference': mean_diff.item(),
                                            'std_difference': std_diff.item(),
                                            'distribution_plot': wandb.Image(fig),
                                            'epoch': epoch,
                                            'batch': i,
                                        })

                                        plt.close(fig)

                        except Exception as e:
                            # Skip problematic batches
                            print(f"Error in batch: {str(e)}")
                            continue

                    if batch_count > 0:
                        epoch_metrics['total_loss'] = epoch_loss / batch_count
                        metrics_history.append(epoch_metrics)

                        if wandb.run is not None:
                            wandb.log({
                                'epoch_loss': epoch_metrics['total_loss'],
                                'epoch': epoch,
                            })

                # Calculate score - negative of average total loss in last 10% of epochs
                last_n = max(1, int(n_epochs * 0.1))
                if len(metrics_history) < last_n:
                    return None  # Not enough data points

                last_metrics = metrics_history[-last_n:]
                avg_loss = np.mean([m['total_loss'] for m in last_metrics])

                # Return negative loss as score (since we want to minimize loss)
                return -avg_loss

            except Exception as e:
                import traceback
                print(f"Error in objective function: {e}")
                print(traceback.format_exc())
                return None

        # Create and run optimizer
        optimizer = BayesianOptimizer(param_ranges, objective_function, n_iterations=n_iterations)

        # Define callback for Streamlit progress and WandB logging
        def callback(i, params, score):
            import streamlit as st
            if 'optimization_progress' not in st.session_state:
                st.session_state.optimization_progress = []

            progress_info = {
                'iteration': i+1,
                'params': params,
                'score': score
            }
            st.session_state.optimization_progress.append(progress_info)

            # Log optimization progress to WandB
            if wandb.run is not None:
                wandb.log({
                    'optimization_iteration': i+1,
                    'optimization_score': -score if score is not None else None,  # Convert back to loss
                    'learning_rate': params['learning_rate'],
                    'kl_weight': params['kl_weight'],
                    'latent_dim': params['latent_dim'],
                })

        best_params, _, history_df = optimizer.optimize(callback=callback)

        # Update model with best parameters
        self.latent_dim = int(best_params['latent_dim'])
        # Rebuild encoder and decoder with new latent_dim
        self.encoder = self.build_encoder().to(self.device)
        self.decoder = self.build_decoder().to(self.device)

        # Update optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=best_params['learning_rate']
        )

        return best_params, history_df

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