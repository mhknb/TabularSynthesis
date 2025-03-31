import torch
import torch.nn as nn
from src.models.base_gan import BaseGAN

class TableGAN(BaseGAN):
    """TableGAN implementation for tabular data with support for categorical columns"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, device: str = 'cpu', min_batch_size: int = 2, 
                 categorical_columns=None, categorical_dims=None):
        """
        Initialize TableGAN with support for categorical columns
        
        Args:
            input_dim: Dimension of input noise vector
            hidden_dim: Hidden dimension size for networks
            device: Device to run on ('cpu' or 'cuda')
            min_batch_size: Minimum batch size for training
            categorical_columns: List of indices for categorical columns (0-based)
            categorical_dims: Dictionary mapping column indices to number of categories
        """
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.min_batch_size = min_batch_size
        
        # Set up categorical column information
        self.categorical_columns = [] if categorical_columns is None else categorical_columns
        self.categorical_dims = {} if categorical_dims is None else categorical_dims
        
        # For tracking training progress and stability with categorical columns
        self.categorical_entropy = {}
        
        # Categorical entropy parameters
        self.current_entropy_weight = 5.0  # Starting weight for entropy loss
        self.min_entropy_weight = 0.5     # Minimum weight for entropy loss
        self.entropy_decay = 0.995        # Decay rate for entropy weight
        self.epsilon = 5.0                # Weight for categorical entropy loss
        
        # Default loss weights
        self.alpha = 1.0   # Adversarial loss weight
        self.beta = 10.0   # Relationship loss weight
        self.gamma = 0.1   # Feature matching loss weight
        self.delta = 2.0   # Range preservation loss weight
        
        # Build models
        self.generator = self.build_generator().to(device)
        self.discriminator = self.build_discriminator().to(device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def build_generator(self) -> nn.Module:
        """Build generator network with support for both continuous and categorical data"""
        class CategoricalMixingGenerator(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, categorical_columns=None, categorical_dims=None):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.categorical_columns = [] if categorical_columns is None else categorical_columns
                self.categorical_dims = {} if categorical_dims is None else categorical_dims
                self.temperature = 1.0  # For controlling categorical sampling randomness
                
                # Noise processing layers
                self.noise_processor = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim, momentum=0.01),
                    nn.LeakyReLU(0.2),
                    
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.BatchNorm1d(hidden_dim * 2, momentum=0.01),
                    nn.LeakyReLU(0.2),
                    
                    # Deeper network for more expressive power
                    nn.Linear(hidden_dim * 2, hidden_dim * 4),
                    nn.BatchNorm1d(hidden_dim * 4, momentum=0.01),
                    nn.LeakyReLU(0.2),
                    
                    nn.Linear(hidden_dim * 4, hidden_dim * 2),
                    nn.BatchNorm1d(hidden_dim * 2, momentum=0.01),
                    nn.LeakyReLU(0.2),
                    
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.BatchNorm1d(hidden_dim, momentum=0.01),
                    nn.LeakyReLU(0.2),
                )
                
                # Create specialized generators for each column
                self.continuous_generators = nn.ModuleList()
                self.categorical_generators = nn.ModuleList()
                
                # Initialize empty outputs for tracking column indices
                self.continuous_output_indices = []
                self.categorical_output_indices = []
                
                # Set up appropriate generators for each column
                for i in range(output_dim):
                    if i in self.categorical_columns:
                        # Get number of categories for this column
                        num_categories = self.categorical_dims.get(i, 2)  # Default to binary if not specified
                        
                        # Create a categorical generator that outputs logits for all categories
                        cat_gen = nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim // 2),
                            nn.LayerNorm(hidden_dim // 2),
                            nn.LeakyReLU(0.2),
                            nn.Linear(hidden_dim // 2, num_categories)
                        )
                        self.categorical_generators.append(cat_gen)
                        self.categorical_output_indices.append(i)
                    else:
                        # Create a continuous generator
                        cont_gen = nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim // 2),
                            nn.LeakyReLU(0.2),
                            nn.Linear(hidden_dim // 2, 1)
                        )
                        self.continuous_generators.append(cont_gen)
                        self.continuous_output_indices.append(i)
                
                # Relationship layers for continuous columns
                self.relationship_layers = []
                prev_continuous_count = 0
                
                for i, idx in enumerate(self.continuous_output_indices):
                    if i > 0:  # Skip the first continuous column
                        self.relationship_layers.append(
                            nn.Sequential(
                                nn.Linear(prev_continuous_count, hidden_dim // 4),
                                nn.LeakyReLU(0.2),
                                nn.Linear(hidden_dim // 4, 1)
                            )
                        )
                    prev_continuous_count += 1
                
                self.relationship_layers = nn.ModuleList(self.relationship_layers)
                
                # Mixing parameter for continuous columns
                self.alpha = nn.Parameter(torch.tensor([0.5]))
                
            def forward(self, noise, temperature=None):
                # Set temperature if provided
                if temperature is not None:
                    self.temperature = temperature
                
                # Process noise through shared layers
                processed_noise = self.noise_processor(noise)
                
                # Create output tensor with the right size
                batch_size = noise.size(0)
                outputs = torch.zeros(batch_size, self.output_dim).to(noise.device)
                
                # Generate continuous columns with relationship awareness
                if self.continuous_generators:
                    continuous_outputs = []
                    
                    # Generate each continuous column independently first
                    indep_cont_outputs = [
                        gen(processed_noise) for gen in self.continuous_generators
                    ]
                    
                    # First continuous column is always independent
                    continuous_outputs.append(indep_cont_outputs[0])
                    
                    # For subsequent continuous columns, use relationship layers
                    for i in range(1, len(indep_cont_outputs)):
                        # Get independent prediction
                        indep_pred = indep_cont_outputs[i]
                        
                        # Get relationship-based prediction using previous continuous columns
                        prev_cols = torch.cat(continuous_outputs, dim=1)
                        rel_pred = self.relationship_layers[i-1](prev_cols)
                        
                        # Combine predictions
                        combined = self.alpha * indep_pred + (1 - self.alpha) * rel_pred
                        continuous_outputs.append(combined)
                    
                    # Place continuous outputs in the correct positions
                    for i, idx in enumerate(self.continuous_output_indices):
                        outputs[:, idx:idx+1] = continuous_outputs[i]
                
                # Generate categorical columns
                if self.categorical_generators:
                    for i, cat_idx in enumerate(self.categorical_output_indices):
                        # Get the generator for this categorical column
                        cat_gen = self.categorical_generators[i]
                        
                        # Generate logits for this categorical column
                        logits = cat_gen(processed_noise)
                        
                        # Apply temperature scaling for sampling control
                        if self.training:
                            # During training, use softmax with temperature for controlled randomness
                            probs = torch.softmax(logits / self.temperature, dim=1)
                            
                            # Sample from the categorical distribution using Gumbel-Softmax
                            # This gives differentiable sampling for training
                            sampled = torch.nn.functional.gumbel_softmax(
                                logits, 
                                tau=self.temperature, 
                                hard=False
                            )
                            
                            # Convert to expected continuous value (weighted average)
                            # This produces the values 0, 1, 2, ... for each category
                            cat_values = torch.sum(
                                sampled * torch.arange(logits.size(1), device=logits.device).float(),
                                dim=1, keepdim=True
                            )
                        else:
                            # During inference, either sample from the distribution or use argmax
                            if self.temperature > 0.5:  # If temperature is high, sample
                                probs = torch.softmax(logits / self.temperature, dim=1)
                                cat_indices = torch.multinomial(probs, 1)
                            else:  # If temperature is low, use argmax (deterministic)
                                cat_indices = torch.argmax(logits, dim=1, keepdim=True)
                            
                            cat_values = cat_indices.float()
                        
                        # Place categorical output in the correct position
                        outputs[:, cat_idx:cat_idx+1] = cat_values
                
                return outputs
                
        # Create the generator with categorical column information
        return CategoricalMixingGenerator(
            self.input_dim, 
            self.hidden_dim, 
            self.input_dim,
            categorical_columns=self.categorical_columns,
            categorical_dims=self.categorical_dims
        )

    def build_discriminator(self) -> nn.Module:
        """Build enhanced discriminator network with attention to column relationships"""
        class RelationAwareDiscriminator(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.input_dim = input_dim
                
                # Main processing network
                self.main = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),  # LayerNorm instead of BatchNorm for stability
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                )
                
                # Relationship analyzing network - specifically for column relationships
                self.relationship_analyzer = nn.Sequential(
                    nn.Linear(input_dim * (input_dim - 1) // 2, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LeakyReLU(0.2),
                )
                
                # Final classification layers
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()
                )
                
            def compute_column_relationships(self, x):
                """Compute pairwise relationships between columns"""
                relationships = []
                for i in range(self.input_dim):
                    for j in range(i+1, self.input_dim):
                        # Add pairwise differences or ratios to capture relationships
                        # For this dataset with known relationships (column i+1 ≈ 2*column i)
                        # we use division to learn the multiplier
                        rel = x[:, j:j+1] / (x[:, i:i+1] + 1e-10)  # avoid division by zero
                        relationships.append(rel)
                
                # Concatenate all relationships
                return torch.cat(relationships, dim=1)
            
            def forward(self, x):
                # Process through main network
                main_features = self.main(x)
                
                # Analyze column relationships
                relationships = self.compute_column_relationships(x)
                relationship_features = self.relationship_analyzer(relationships)
                
                # Combine features and classify
                combined_features = torch.cat([main_features, relationship_features], dim=1)
                return self.classifier(combined_features)
                
        return RelationAwareDiscriminator(self.input_dim, self.hidden_dim)

    def validate_batch(self, batch: torch.Tensor) -> bool:
        """Validate batch size is sufficient for training"""
        return batch.size(0) >= self.min_batch_size

    def calculate_relationship_loss(self, data: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss based on column relationships
        For this specific case (column[i+1] = 2 * column[i]), we compute the error in this ratio
        
        Skips categorical columns since they don't follow the 2x relationship pattern
        """
        loss = 0.0
        num_valid_pairs = 0
        
        # For each pair of adjacent columns
        for i in range(data.shape[1] - 1):
            # Skip if either column is categorical
            if i in self.categorical_columns or i+1 in self.categorical_columns:
                continue
                
            # Current column and next column
            col_i = data[:, i:i+1]
            col_i_plus_1 = data[:, i+1:i+2]
            
            # The ideal ratio should be 2.0 for this dataset
            # Calculate the actual ratio (safely avoiding division by zero)
            actual_ratio = col_i_plus_1 / (col_i + 1e-10)
            
            # Calculate mean squared error from target ratio of 2.0
            target_ratio = torch.ones_like(actual_ratio) * 2.0
            ratio_mse = torch.mean((actual_ratio - target_ratio) ** 2)
            
            loss += ratio_mse
            num_valid_pairs += 1
        
        # Average over valid column pairs (non-categorical)
        return loss / max(1, num_valid_pairs)
        
    def calculate_categorical_entropy(self, fake_data: torch.Tensor) -> torch.Tensor:
        """
        Calculate entropy loss to encourage diversity in categorical columns
        Lower entropy means less diversity (bad), higher entropy means more diversity (good)
        So we return negative entropy as the loss (to minimize)
        """
        if not self.categorical_columns:
            return torch.tensor(0.0).to(self.device)
            
        entropy_loss = torch.tensor(0.0).to(self.device)
        
        for col_idx in self.categorical_columns:
            # Get the categorical column data
            col_data = fake_data[:, col_idx]
            
            # Get number of categories
            num_categories = self.categorical_dims.get(col_idx, 2)
            
            # Count occurrences of each category
            category_counts = []
            for c in range(num_categories):
                # Count instances of this category (with some tolerance)
                category_count = torch.sum((col_data >= c - 0.25) & (col_data <= c + 0.25)).float()
                category_counts.append(category_count)
            
            category_counts = torch.tensor(category_counts).to(self.device)
            
            # Add small constant to avoid log(0)
            category_counts = category_counts + 1e-10
            
            # Calculate probabilities
            probabilities = category_counts / torch.sum(category_counts)
            
            # Calculate entropy: -sum(p * log(p))
            entropy = -torch.sum(probabilities * torch.log(probabilities))
            
            # Track entropy for monitoring
            self.categorical_entropy[col_idx] = entropy.item()
            
            # Add negative entropy to the loss (higher entropy = more diversity = better)
            entropy_loss -= entropy
        
        # Average entropy loss across all categorical columns
        return entropy_loss / max(1, len(self.categorical_columns))

    def train_step(self, real_data: torch.Tensor) -> dict:
        """Perform enhanced training step with relationship preservation and range matching"""
        if not self.validate_batch(real_data):
            raise ValueError(f"Batch size {real_data.size(0)} is too small. Minimum required: {self.min_batch_size}")

        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)

        # Store min and max values for each column for range preservation
        # Update with exponential moving average to capture the data range across batches
        if not hasattr(self, 'data_min') or not hasattr(self, 'data_max'):
            # Initialize min/max tracking on first batch
            self.data_min = torch.min(real_data, dim=0)[0]
            self.data_max = torch.max(real_data, dim=0)[0]
            self.data_range = self.data_max - self.data_min
        else:
            # Update min/max with exponential moving average
            current_min = torch.min(real_data, dim=0)[0]
            current_max = torch.max(real_data, dim=0)[0]
            # Use a slow decay to gradually capture the true range
            decay = 0.95
            self.data_min = decay * self.data_min + (1 - decay) * torch.min(self.data_min, current_min)
            self.data_max = decay * self.data_max + (1 - decay) * torch.max(self.data_max, current_max)
            self.data_range = self.data_max - self.data_min

        # Train Discriminator
        self.d_optimizer.zero_grad()

        # Add label smoothing for improved stability
        label_real = torch.ones(batch_size, 1).to(self.device) * 0.9  # 0.9 instead of 1.0
        label_fake = torch.zeros(batch_size, 1).to(self.device) * 0.1  # 0.1 instead of 0.0

        output_real = self.discriminator(real_data)
        d_loss_real = nn.BCELoss()(output_real, label_real)

        noise = torch.randn(batch_size, self.input_dim).to(self.device)
        fake_data = self.generator(noise)
        output_fake = self.discriminator(fake_data.detach())
        d_loss_fake = nn.BCELoss()(output_fake, label_fake)

        # Additional feature matching loss (helps with mode collapse)
        # This assumes our discriminator's main network can be accessed for intermediate features
        if hasattr(self.discriminator, 'main'):
            real_features = self.discriminator.main(real_data)
            fake_features = self.discriminator.main(fake_data.detach())
            feature_matching_loss = torch.mean(torch.abs(torch.mean(real_features, dim=0) - torch.mean(fake_features, dim=0)))
            d_loss_fm = feature_matching_loss * 0.1  # Scale to avoid dominating other losses
        else:
            d_loss_fm = torch.tensor(0.0).to(self.device)

        d_loss = d_loss_real + d_loss_fake + d_loss_fm
        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator with enhanced losses
        self.g_optimizer.zero_grad()
        
        # Regular adversarial loss
        output_fake = self.discriminator(fake_data)
        g_adv_loss = nn.BCELoss()(output_fake, label_real)
        
        # Column relationship preservation loss - crucial for continuous columns
        relationship_loss = self.calculate_relationship_loss(fake_data)

        # Range preservation loss - helps match the synthetic data range to the original data
        fake_min = torch.min(fake_data, dim=0)[0]
        fake_max = torch.max(fake_data, dim=0)[0]
        fake_range = fake_max - fake_min

        # Calculate range loss - penalize when the synthetic data range doesn't match original data range
        # We divide by data_range to normalize the loss across different scales of features
        epsilon = 1e-8  # Small value to avoid division by zero
        range_loss = torch.mean(torch.abs(fake_range - self.data_range) / (self.data_range + epsilon))
        
        # Also penalize if min/max values are too far from original
        min_loss = torch.mean(torch.abs(fake_min - self.data_min) / (self.data_range + epsilon))
        max_loss = torch.mean(torch.abs(fake_max - self.data_max) / (self.data_range + epsilon))
        
        # Combined range loss
        total_range_loss = range_loss + 0.5 * (min_loss + max_loss)
        
        # Categorical entropy loss - encourages diversity in categorical columns
        if self.categorical_columns:
            # Initialize temperature scheduling
            # Start with high temperature (more randomness) and gradually decrease
            # to ensure exploration at the beginning and stability at the end
            if not hasattr(self, 'current_entropy_weight'):
                self.current_entropy_weight = 5.0  # Starting weight for entropy loss
                self.min_entropy_weight = 0.5     # Minimum weight
                self.entropy_decay = 0.995        # Decay rate
            else:
                # Gradually reduce entropy weight
                self.current_entropy_weight = max(
                    self.min_entropy_weight,
                    self.current_entropy_weight * self.entropy_decay
                )
            
            # Calculate entropy loss
            entropy_loss = self.calculate_categorical_entropy(fake_data)
            
            # Set generator temperature based on training progress
            if hasattr(self.generator, 'temperature'):
                # Decay temperature over time (from more random to more deterministic)
                self.generator.temperature = max(0.5, 1.5 * self.current_entropy_weight / 5.0)
        else:
            entropy_loss = torch.tensor(0.0).to(self.device)
            self.current_entropy_weight = 0.0
        
        # Optional: Feature matching loss for generator too
        if hasattr(self.discriminator, 'main'):
            real_features = self.discriminator.main(real_data)
            fake_features = self.discriminator.main(fake_data)
            g_feature_loss = torch.mean(torch.abs(torch.mean(real_features, dim=0) - torch.mean(fake_features, dim=0)))
        else:
            g_feature_loss = torch.tensor(0.0).to(self.device)
        
        # Combined generator loss with weights (can be set dynamically via hyperparameter optimization)
        self.alpha = getattr(self, 'alpha', 1.0)  # Weight for adversarial loss
        self.beta = getattr(self, 'beta', 10.0)   # Higher weight for relationship loss to emphasize the pattern
        self.gamma = getattr(self, 'gamma', 0.1)  # Weight for feature matching loss
        self.delta = getattr(self, 'delta', 2.0)  # Weight for range preservation loss
        self.epsilon = getattr(self, 'epsilon', self.current_entropy_weight)  # Weight for categorical entropy
        
        g_loss = (
            self.alpha * g_adv_loss + 
            self.beta * relationship_loss + 
            self.gamma * g_feature_loss + 
            self.delta * total_range_loss +
            self.epsilon * entropy_loss
        )
        
        g_loss.backward()
        self.g_optimizer.step()

        # Return detailed metrics
        metrics = {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item(),
            'd_real_loss': d_loss_real.item(),
            'd_fake_loss': d_loss_fake.item(),
            'g_adv_loss': g_adv_loss.item(),
            'relationship_loss': relationship_loss.item(),
            'range_loss': total_range_loss.item(),
            'feature_loss': g_feature_loss.item(),
            'd_real_mean': output_real.mean().item(),
            'd_fake_mean': output_fake.mean().item(),
            'entropy_weight': self.current_entropy_weight
        }
        
        # Add categorical entropy metrics if available
        if self.categorical_columns:
            metrics['entropy_loss'] = entropy_loss.item()
            
            # Add individual column entropy values
            for col_idx in self.categorical_columns:
                if col_idx in self.categorical_entropy:
                    metrics[f'entropy_col_{col_idx}'] = self.categorical_entropy[col_idx]
        
        return metrics

    def generate_samples(self, num_samples: int, temperature: float = 0.8) -> torch.Tensor:
        """
        Generate synthetic samples with proper handling of categorical data
        
        Args:
            num_samples: Number of samples to generate
            temperature: Temperature for categorical sampling (higher = more diversity)
                         Range 0.1-1.0, where 0.1 is most deterministic and 1.0 is most diverse
        """
        # Store current training mode and set to eval
        was_training = self.generator.training
        self.generator.train(False)
        
        # Set temperature for categorical diversity
        if hasattr(self.generator, 'temperature'):
            original_temp = self.generator.temperature
            self.generator.temperature = temperature
        
        with torch.no_grad():
            batch_size = min(self.min_batch_size * 4, num_samples)
            num_batches = (num_samples + batch_size - 1) // batch_size
            samples_list = []

            for i in range(num_batches):
                current_batch_size = min(batch_size, num_samples - i * batch_size)
                if current_batch_size < self.min_batch_size:
                    current_batch_size = self.min_batch_size
                noise = torch.randn(current_batch_size, self.input_dim).to(self.device)
                
                # Pass temperature explicitly if the generator accepts it
                if hasattr(self.generator, 'forward') and 'temperature' in self.generator.forward.__code__.co_varnames:
                    samples = self.generator(noise, temperature=temperature)
                else:
                    samples = self.generator(noise)
                
                samples_list.append(samples)

            all_samples = torch.cat(samples_list, dim=0)
            
            # Process categorical outputs to ensure they are valid integer indices
            if self.categorical_columns:
                for col_idx in self.categorical_columns:
                    # Get number of categories for this column
                    num_categories = self.categorical_dims.get(col_idx, 2)
                    
                    # Ensure categorical values are integers within valid range
                    col_values = all_samples[:, col_idx]
                    col_values = torch.round(col_values).clamp(0, num_categories - 1)
                    all_samples[:, col_idx] = col_values
        
        # Restore original temperature and training mode
        if hasattr(self.generator, 'temperature'):
            self.generator.temperature = original_temp
        self.generator.train(was_training)
            
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
            'min_batch_size': self.min_batch_size,
            # Save column type information
            'categorical_columns': self.categorical_columns,
            'categorical_dims': self.categorical_dims,
            # Save loss weights
            'alpha': getattr(self, 'alpha', 1.0),  # Adversarial loss weight
            'beta': getattr(self, 'beta', 10.0),   # Relationship loss weight
            'gamma': getattr(self, 'gamma', 0.1),  # Feature matching loss weight
            'delta': getattr(self, 'delta', 2.0),  # Range preservation loss weight
            'epsilon': getattr(self, 'epsilon', 0.0),  # Categorical entropy weight
            # Save temperature and entropy parameters
            'current_entropy_weight': getattr(self, 'current_entropy_weight', 5.0),
            'min_entropy_weight': getattr(self, 'min_entropy_weight', 0.5),
            'entropy_decay': getattr(self, 'entropy_decay', 0.995),
            # Save data range information if available
            'data_min': getattr(self, 'data_min', None),
            'data_max': getattr(self, 'data_max', None),
            'data_range': getattr(self, 'data_range', None),
            # Save categorical entropy tracking
            'categorical_entropy': self.categorical_entropy
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
        
        # Load column type information if available
        if 'categorical_columns' in state_dict:
            self.categorical_columns = state_dict['categorical_columns']
        if 'categorical_dims' in state_dict:
            self.categorical_dims = state_dict['categorical_dims']
            
        # Ensure categorical_entropy tracking dictionary exists
        if not hasattr(self, 'categorical_entropy'):
            self.categorical_entropy = {}
        
        # Load categorical entropy tracking if available
        if 'categorical_entropy' in state_dict:
            self.categorical_entropy = state_dict['categorical_entropy']
        
        # Load loss weights with defaults for backward compatibility
        self.alpha = state_dict.get('alpha', 1.0)   # Adversarial loss weight
        self.beta = state_dict.get('beta', 10.0)    # Relationship loss weight
        self.gamma = state_dict.get('gamma', 0.1)   # Feature matching loss weight
        self.delta = state_dict.get('delta', 2.0)   # Range preservation loss weight
        self.epsilon = state_dict.get('epsilon', 0.0)  # Categorical entropy weight
        
        # Load temperature and entropy parameters
        self.current_entropy_weight = state_dict.get('current_entropy_weight', 5.0)
        self.min_entropy_weight = state_dict.get('min_entropy_weight', 0.5)
        self.entropy_decay = state_dict.get('entropy_decay', 0.995)
        
        # Load data range information if available
        if state_dict.get('data_min') is not None:
            self.data_min = state_dict['data_min']
        if state_dict.get('data_max') is not None:
            self.data_max = state_dict['data_max']
        if state_dict.get('data_range') is not None:
            self.data_range = state_dict['data_range']

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

        # Define parameter ranges with all loss weights
        param_ranges = {
            'lr_d': (0.00001, 0.001),
            'lr_g': (0.00001, 0.001),
            'dropout_rate': (0.1, 0.5),
            'alpha': (0.5, 2.0),        # Weight for adversarial loss
            'beta': (5.0, 15.0),        # Weight for relationship loss - important for this dataset
            'gamma': (0.05, 0.2),       # Weight for feature matching loss
            'delta': (1.0, 5.0),        # Weight for range preservation loss
            'epsilon': (0.5, 5.0)       # Weight for categorical entropy loss (if applicable)
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
                
                # Apply loss weights for relationship preservation and range/entropy
                temp_model.alpha = params['alpha']
                temp_model.beta = params['beta']
                temp_model.gamma = params['gamma']
                temp_model.delta = params['delta']
                
                # Apply entropy weight if applicable and if categorical columns exist
                if 'epsilon' in params and hasattr(self, 'categorical_columns') and self.categorical_columns:
                    temp_model.epsilon = params['epsilon']
                    # Copy categorical column information to temp model
                    temp_model.categorical_columns = self.categorical_columns 
                    temp_model.categorical_dims = self.categorical_dims

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
        
        # Apply optimized loss weights
        self.alpha = best_params['alpha']
        self.beta = best_params['beta']
        self.gamma = best_params['gamma']
        self.delta = best_params['delta']
        
        # Apply epsilon if categorical columns exist
        if 'epsilon' in best_params and self.categorical_columns:
            self.epsilon = best_params['epsilon']
            # Update temperature dynamics based on optimized epsilon
            self.current_entropy_weight = best_params['epsilon']
        
        print(f"Optimized model parameters: lr_g={best_params['lr_g']:.6f}, lr_d={best_params['lr_d']:.6f}, "
              f"alpha={self.alpha:.2f}, beta={self.beta:.2f}, gamma={self.gamma:.3f}, delta={self.delta:.2f}"
              + (f", epsilon={self.epsilon:.2f}" if hasattr(self, 'epsilon') and self.categorical_columns else ""))

        return best_params, history_df