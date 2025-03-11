
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import time
import pandas as pd

class BayesianOptimizer:
    """Bayesian Optimization for hyperparameter tuning"""
    
    def __init__(self, param_ranges, objective_function, n_iterations=10, exploration_weight=0.01):
        """
        Initialize the Bayesian Optimizer
        
        Args:
            param_ranges: Dictionary of parameter names and their ranges (min, max)
            objective_function: Function to maximize/minimize, takes parameters dict as input
            n_iterations: Number of optimization iterations
            exploration_weight: Weight for exploration vs exploitation (higher = more exploration)
        """
        self.param_ranges = param_ranges
        self.objective_function = objective_function
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight
        
        # Initialize Gaussian Process with Matern kernel
        self.kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer=5)
        
        # History of evaluations
        self.X_samples = []  # Parameter combinations
        self.y_samples = []  # Results
        self.best_params = None
        self.best_score = float('-inf')  # For maximization
        
    def _convert_params_to_array(self, params):
        """Convert parameters dictionary to normalized array for GP"""
        x = []
        for param_name in self.param_ranges:
            param_min, param_max = self.param_ranges[param_name]
            normalized_value = (params[param_name] - param_min) / (param_max - param_min)
            x.append(normalized_value)
        return np.array(x).reshape(1, -1)
    
    def _convert_array_to_params(self, x):
        """Convert normalized array back to parameters dictionary"""
        params = {}
        for i, param_name in enumerate(self.param_ranges):
            param_min, param_max = self.param_ranges[param_name]
            params[param_name] = param_min + x[i] * (param_max - param_min)
        return params
    
    def _acquisition_function(self, x, gp, best_f):
        """Expected Improvement acquisition function"""
        x = x.reshape(1, -1)
        mu, sigma = gp.predict(x, return_std=True)
        
        # Handle zero/near-zero variance case
        if sigma < 1e-6:
            return 0
        
        z = (mu - best_f - self.exploration_weight) / sigma
        return (mu - best_f - self.exploration_weight) * norm.cdf(z) + sigma * norm.pdf(z)
    
    def _suggest_next_params(self):
        """Suggest next parameters to evaluate using acquisition function"""
        if len(self.X_samples) < 2:
            # Random sampling for first few iterations
            x_random = np.random.rand(len(self.param_ranges))
            return self._convert_array_to_params(x_random)
        
        # Fit GP with existing data
        X = np.array(self.X_samples)
        y = np.array(self.y_samples)
        self.gp.fit(X, y)
        
        # Find parameters that maximize acquisition function
        best_acquisition = float('-inf')
        best_x = None
        
        # Grid search for acquisition function maximization
        n_random_points = 10000
        random_points = np.random.rand(n_random_points, len(self.param_ranges))
        
        for x in random_points:
            acq_value = self._acquisition_function(x, self.gp, np.max(y))
            if acq_value > best_acquisition:
                best_acquisition = acq_value
                best_x = x
        
        return self._convert_array_to_params(best_x)
    
    def optimize(self, callback=None, verbose=True):
        """Run the optimization process"""
        start_time = time.time()
        history = []
        
        for i in range(self.n_iterations):
            # Suggest parameters
            params = self._suggest_next_params()
            
            # Evaluate objective function
            score = self.objective_function(params)
            
            # Store results
            self.X_samples.append(self._convert_params_to_array(params)[0])
            self.y_samples.append(score)
            
            # Update best score and parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
            
            # Log progress
            if verbose:
                print(f"Iteration {i+1}/{self.n_iterations}, Score: {score:.4f}, Params: {params}")
            
            # Record history
            history_entry = {
                'iteration': i+1,
                'score': score,
                **params
            }
            history.append(history_entry)
            
            # Callback for external use
            if callback:
                callback(i, params, score)
                
        # Create DataFrame with history
        history_df = pd.DataFrame(history)
        
        if verbose:
            elapsed_time = time.time() - start_time
            print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
            print(f"Best score: {self.best_score:.4f}")
            print(f"Best parameters: {self.best_params}")
            
        return self.best_params, self.best_score, history_df
