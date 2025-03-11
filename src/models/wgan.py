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
            'lambda_gp': (1.0, 20.0)
        }
        
        # Define objective function
        def objective_function(params):
            try:
                # Create temporary model with new parameters
                temp_model = WGAN(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    clip_value=self.clip_value,
                    n_critic=self.n_critic,
                    lr_g=params['lr_g'],
                    lr_d=params['lr_d'],
                    lambda_gp=params['lambda_gp'],
                    device=self.device
                )
                
                # Train for a few epochs
                metrics_history = []
                for epoch in range(n_epochs):
                    epoch_metrics = {'epoch': epoch}
                    for i, real_data in enumerate(train_loader):
                        metrics = temp_model.train_step(real_data)
                        epoch_metrics.update(metrics)
                    metrics_history.append(epoch_metrics)
                
                # Calculate score - negative wasserstein distance (since we want to maximize)
                # Use the average of the last 10% of epochs to reduce noise
                last_n = max(1, int(n_epochs * 0.1))
                last_metrics = metrics_history[-last_n:]
                avg_wasserstein = np.mean([m['wasserstein_distance'] for m in last_metrics])
                
                # Return negative wasserstein distance as score (since we want to minimize wasserstein distance)
                return -avg_wasserstein
            
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
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=best_params['lr_g'], betas=(0.5, 0.9))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=best_params['lr_d'], betas=(0.5, 0.9))
        self.lambda_gp = best_params['lambda_gp']
        
        return best_params, history_df

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