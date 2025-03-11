import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from src.models.base_gan import BaseGAN

class WGAN(BaseGAN):
    """WGAN implementation for tabular data"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, clip_value: float = 0.01, n_critic: int = 5, 
             lr_g: float = 0.0001, lr_d: float = 0.0001, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.clip_value = clip_value
        self.n_critic = n_critic
        # No need for lambda_gp with spectral normalization

        self.generator = self.build_generator().to(device)
        self.discriminator = self.build_discriminator().to(device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))
        
        # Dynamic learning rate scheduling
        self.scheduler_g = torch.optim.lr_scheduler.StepLR(self.g_optimizer, step_size=10, gamma=0.9)
        self.scheduler_d = torch.optim.lr_scheduler.StepLR(self.d_optimizer, step_size=10, gamma=0.9)

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
        """Build discriminator network with spectral normalization for Lipschitz constraint"""
        return nn.Sequential(
            spectral_norm(nn.Linear(self.input_dim, self.hidden_dim)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim * 2)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Linear(self.hidden_dim * 2, self.hidden_dim)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Linear(self.hidden_dim, 1))
        )

    def train_step(self, real_data: torch.Tensor, current_step: int = 0) -> dict:
        """Perform one training step with proper critic iterations"""
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)
        
        # Variables to store metrics
        total_disc_loss = 0
        gen_loss = 0
        wasserstein_distance = 0
        
        # Train Discriminator/Critic for n_critic steps
        for _ in range(self.n_critic):
            self.d_optimizer.zero_grad()
            
            # Generate fake data
            noise = torch.randn(batch_size, self.input_dim).to(self.device)
            fake_data = self.generator(noise)
            
            # Extract features and get discriminator scores
            intermediate_layers = self.discriminator[:-1]
            final_layer = self.discriminator[-1]
            
            # Forward pass through intermediate layers and final layer separately
            feat_real = intermediate_layers(real_data)
            feat_fake = intermediate_layers(fake_data.detach())
            
            # Get final discriminator scores
            disc_real = final_layer(feat_real)
            disc_fake = final_layer(feat_fake)
            
            # Wasserstein loss with spectral normalization
            curr_wasserstein_distance = torch.mean(disc_real) - torch.mean(disc_fake)
            disc_loss = -curr_wasserstein_distance
            
            # Update metrics
            total_disc_loss += disc_loss.item()
            wasserstein_distance = curr_wasserstein_distance.item()  # Store the last one
            
            # Backward and optimize
            disc_loss.backward()
            self.d_optimizer.step()
        
        # Calculate average discriminator loss
        avg_disc_loss = total_disc_loss / self.n_critic
        
        # Train Generator (only once per n_critic iterations)
        self.g_optimizer.zero_grad()
        
        # Generate new fake data
        noise = torch.randn(batch_size, self.input_dim).to(self.device)
        fake_data = self.generator(noise)
        
        # Extract features from intermediate layers for feature matching
        intermediate_layers = self.discriminator[:-1]
        final_layer = self.discriminator[-1]
        
        with torch.no_grad():
            feat_real = intermediate_layers(real_data)
        
        feat_fake = intermediate_layers(fake_data)
        disc_fake = final_layer(feat_fake)
        
        # Wasserstein loss + Feature matching loss
        wasserstein_loss = -torch.mean(disc_fake)
        feature_matching_loss = F.mse_loss(feat_real, feat_fake)
        
        gen_loss = wasserstein_loss + feature_matching_loss * 0.5
        gen_loss.backward()
        self.g_optimizer.step()
        
        # Step the learning rate schedulers (typically done once per epoch, but can be configured per step)
        if current_step > 0 and current_step % 100 == 0:  # Step scheduler every 100 iterations
            self.scheduler_d.step()
            self.scheduler_g.step()
        
        # Record metrics
        self.eval_metrics['gen_loss'].append(gen_loss.item())
        self.eval_metrics['disc_loss'].append(avg_disc_loss)
        self.eval_metrics['wasserstein_distance'].append(wasserstein_distance)
        
        return {
            'disc_loss': avg_disc_loss,
            'gen_loss': gen_loss.item(),
            'wasserstein_distance': wasserstein_distance
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
            'lr_g': (0.00001, 0.001)
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
                    device=self.device
                )
                
                # Train for a few epochs
                metrics_history = []
                total_steps = 0
                for epoch in range(n_epochs):
                    epoch_metrics = {'epoch': epoch}
                    for i, real_data in enumerate(train_loader):
                        metrics = temp_model.train_step(real_data, current_step=total_steps)
                        total_steps += 1
                        epoch_metrics.update(metrics)
                    # Step schedulers at the end of each epoch
                    temp_model.scheduler_g.step()
                    temp_model.scheduler_d.step()
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
        
        return best_params, history_df

    def state_dict(self):
        """Get state dict for model persistence"""
        return {
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
            'g_scheduler_state': self.scheduler_g.state_dict(),
            'd_scheduler_state': self.scheduler_d.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'clip_value': self.clip_value,
            'n_critic': self.n_critic,
            'device': self.device
        }

    def load_state_dict(self, state_dict):
        """Load state dict for model persistence"""
        self.generator.load_state_dict(state_dict['generator_state'])
        self.discriminator.load_state_dict(state_dict['discriminator_state'])
        self.g_optimizer.load_state_dict(state_dict['g_optimizer_state'])
        self.d_optimizer.load_state_dict(state_dict['d_optimizer_state'])
        
        # Load scheduler states if they exist (for backward compatibility)
        if 'g_scheduler_state' in state_dict:
            self.scheduler_g.load_state_dict(state_dict['g_scheduler_state'])
        if 'd_scheduler_state' in state_dict:
            self.scheduler_d.load_state_dict(state_dict['d_scheduler_state'])
            
        self.input_dim = state_dict['input_dim']
        self.hidden_dim = state_dict['hidden_dim']
        self.clip_value = state_dict['clip_value']
        self.n_critic = state_dict['n_critic']
        # For backward compatibility with older state dicts that may have lambda_gp
        if 'lambda_gp' in state_dict:
            pass  # We ignore it now as we're using spectral normalization
        self.device = state_dict['device']