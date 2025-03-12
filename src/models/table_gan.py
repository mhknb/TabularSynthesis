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
        """Build generator network"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim, momentum=0.01),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2, momentum=0.01),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2, momentum=0.01),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim, momentum=0.01),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Tanh()
        )

    def build_discriminator(self) -> nn.Module:
        """Build discriminator network"""
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
        """Perform one training step"""
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

        # Return metrics
        return {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item(),
            'd_real_loss': d_loss_real.item(),
            'd_fake_loss': d_loss_fake.item(),
            'd_real_mean': output_real.mean().item(),
            'd_fake_mean': output_fake.mean().item()
        }

    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate synthetic samples"""
        with torch.no_grad():
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

        # Define parameter ranges
        param_ranges = {
            'lr_d': (0.00001, 0.001),
            'lr_g': (0.00001, 0.001),
            'dropout_rate': (0.1, 0.5)
        }

        # Define objective function
        def objective_function(params):
            try:
                # Create temporary model with new parameters
                temp_model = TableGAN(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    device=self.device,
                    min_batch_size=self.min_batch_size
                )

                # Update optimizers with new learning rates
                temp_model.g_optimizer = torch.optim.Adam(
                    temp_model.generator.parameters(), 
                    lr=params['lr_g'], 
                    betas=(0.5, 0.999)
                )
                temp_model.d_optimizer = torch.optim.Adam(
                    temp_model.discriminator.parameters(), 
                    lr=params['lr_d'], 
                    betas=(0.5, 0.999)
                )

                # Train for a few epochs
                metrics_history = []
                for epoch in range(n_epochs):
                    epoch_metrics = {'epoch': epoch}
                    epoch_loss_g = 0.0
                    epoch_loss_d = 0.0
                    batch_count = 0

                    for i, real_data in enumerate(train_loader):
                        try:
                            metrics = temp_model.train_step(real_data)
                            epoch_loss_g += metrics['generator_loss']
                            epoch_loss_d += metrics['discriminator_loss']
                            batch_count += 1
                        except Exception as e:
                            # Skip problematic batches
                            continue

                    if batch_count > 0:
                        epoch_metrics['generator_loss'] = epoch_loss_g / batch_count
                        epoch_metrics['discriminator_loss'] = epoch_loss_d / batch_count
                        metrics_history.append(epoch_metrics)

                # Calculate score - negative of average generator loss in last 10% of epochs
                # Lower generator loss indicates better performance
                last_n = max(1, int(n_epochs * 0.1))
                if len(metrics_history) < last_n:
                    return None  # Not enough data points

                last_metrics = metrics_history[-last_n:]
                avg_gen_loss = np.mean([m['generator_loss'] for m in last_metrics])

                # Return negative loss as score (since we want to minimize loss)
                return -avg_gen_loss

            except Exception as e:
                import traceback
                print(f"Error in objective function: {e}")
                print(traceback.format_exc())
                return None

        # Create and run optimizer
        optimizer = BayesianOptimizer(param_ranges, objective_function, n_iterations=n_iterations)

        # Define callback for Streamlit progress
        def callback(i, params, score):
            import streamlit as st
            if 'optimization_progress' not in st.session_state:
                st.session_state.optimization_progress = []
            st.session_state.optimization_progress.append({
                'iteration': i+1, 
                'params': params,
                'score': score
            })

        best_params, _, history_df = optimizer.optimize(callback=callback)

        # Update model with best parameters
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), 
            lr=best_params['lr_g'], 
            betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=best_params['lr_d'], 
            betas=(0.5, 0.999)
        )

        return best_params, history_df