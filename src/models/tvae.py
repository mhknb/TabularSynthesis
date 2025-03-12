import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_gan import BaseGAN

class TVAE(BaseGAN):
    """Tabular Variational Autoencoder implementation"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 128, device: str = 'cpu', use_wandb: bool = False):
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_wandb = use_wandb
        
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
        
        # Initialized flag to track if wandb has been initialized
        self.wandb_initialized = False

        if self.use_wandb:
            try:
                import wandb
                import os
                import time
                
                # Check if wandb is already initialized 
                if wandb.run is None:
                    # Ensure environment variables are set
                    if not os.environ.get("WANDB_ENTITY"):
                        os.environ["WANDB_ENTITY"] = "smilai"
                    if not os.environ.get("WANDB_PROJECT"):
                        os.environ["WANDB_PROJECT"] = "sd1"
                    
                    wandb.init(
                        project="sd1", 
                        name=f"tvae-run-{int(time.time())}", 
                        entity="smilai",
                        config={
                            "hidden_dim": self.hidden_dim,
                            "latent_dim": self.latent_dim,
                            "input_dim": self.input_dim,
                        }
                    )
                    self.wandb_initialized = True
                    print(f"Wandb initialized successfully with entity: smilai, project: sd1")
                else:
                    self.wandb_initialized = True
                    print(f"Using existing wandb run: {wandb.run.name}")
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                self.use_wandb = False
        
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
        
    def train_step(self, real_data: torch.Tensor, current_step: int = 0) -> dict:
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
        total_loss_per_sample = loss.item() / len(real_data)
        recon_loss_per_sample = recon_loss.item() / len(real_data)
        kl_loss_per_sample = kl_loss.item() / len(real_data)
        
        # Log to wandb if enabled and initialized
        if self.use_wandb and (self.wandb_initialized or wandb.run is not None):
            try:
                wandb.log({
                    "Total Loss": total_loss_per_sample,
                    "Reconstruction Loss": recon_loss_per_sample,
                    "KL Divergence Loss": kl_loss_per_sample,
                    "Learning Rate": self.optimizer.param_groups[0]['lr'],
                    "Step": current_step
                })
            except Exception as e:
                print(f"Error logging to wandb: {e}")
        
        return {
            'total_loss': total_loss_per_sample,
            'reconstruction_loss': recon_loss_per_sample,
            'kl_loss': kl_loss_per_sample
        }
        
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
                            
                            epoch_loss += metrics['total_loss']
                            batch_count += 1
                        except Exception as e:
                            # Skip problematic batches
                            continue
                    
                    if batch_count > 0:
                        epoch_metrics['total_loss'] = epoch_loss / batch_count
                        metrics_history.append(epoch_metrics)
                
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
            'device': self.device,
            'use_wandb': self.use_wandb
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
        self.use_wandb = state_dict.get('use_wandb', False)
    
    def finish_wandb(self):
        """Finish the wandb run when training is complete"""
        if self.use_wandb and (self.wandb_initialized or wandb.run is not None):
            try:
                import wandb
                # Only finish if this class initialized wandb
                if self.wandb_initialized:
                    wandb.finish()
                    self.wandb_initialized = False
                    print("WandB run finished")
            except Exception as e:
                print(f"Error finishing wandb run: {e}")
                self.wandb_initialized = False
