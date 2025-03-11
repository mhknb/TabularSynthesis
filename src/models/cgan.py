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
            'noise_scale': (0.5, 1.5)
        }

        # Define objective function
        def objective_function(params):
            try:
                # Create temporary model with new parameters
                temp_model = CGAN(
                    input_dim=self.input_dim,
                    condition_dim=self.condition_dim,
                    hidden_dim=self.hidden_dim,
                    device=self.device
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

                # Custom noise scale for generator
                noise_scale = params['noise_scale']

                # Train for a few epochs
                metrics_history = []
                for epoch in range(n_epochs):
                    epoch_metrics = {'epoch': epoch}
                    epoch_loss_g = 0.0
                    epoch_loss_d = 0.0
                    batch_count = 0

                    for i, batch in enumerate(train_loader):
                        try:
                            if isinstance(batch, list) and len(batch) == 2:
                                real_data, condition = batch
                            else:
                                # If no explicit condition in dataloader, extract the first column as condition
                                real_data = batch
                                condition = batch[:, :self.condition_dim]

                            # Move data to device
                            real_data = real_data.to(temp_model.device)
                            condition = condition.to(temp_model.device)

                            batch_size = real_data.size(0)

                            # Train Discriminator
                            temp_model.d_optimizer.zero_grad()

                            label_real = torch.ones(batch_size, 1).to(temp_model.device)
                            label_fake = torch.zeros(batch_size, 1).to(temp_model.device)

                            # Real data with real condition
                            output_real = temp_model.discriminator(real_data[:, :-temp_model.condition_dim], condition)
                            d_loss_real = nn.BCELoss()(output_real, label_real)

                            # Generate fake data with real condition
                            noise = torch.randn(batch_size, temp_model.input_dim - temp_model.condition_dim).to(temp_model.device) * noise_scale
                            fake_data = temp_model.generator(noise, condition)

                            # Fake data with real condition
                            output_fake = temp_model.discriminator(fake_data.detach(), condition)
                            d_loss_fake = nn.BCELoss()(output_fake, label_fake)

                            d_loss = d_loss_real + d_loss_fake
                            d_loss.backward()
                            temp_model.d_optimizer.step()

                            # Train Generator
                            temp_model.g_optimizer.zero_grad()
                            output_fake = temp_model.discriminator(fake_data, condition)
                            g_loss = nn.BCELoss()(output_fake, label_real)
                            g_loss.backward()
                            temp_model.g_optimizer.step()

                            epoch_loss_g += g_loss.item()
                            epoch_loss_d += d_loss.item()
                            batch_count += 1

                        except Exception as e:
                            # Skip problematic batches
                            continue

                    if batch_count > 0:
                        epoch_metrics['generator_loss'] = epoch_loss_g / batch_count
                        epoch_metrics['discriminator_loss'] = epoch_loss_d / batch_count
                        metrics_history.append(epoch_metrics)

                # Calculate score - negative of average generator loss in last 10% of epochs
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