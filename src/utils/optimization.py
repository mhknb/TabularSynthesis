
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import time
import pandas as pd

import numpy as np
import pandas as pd
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from typing import Dict, Callable, List, Tuple, Any, Optional

class BayesianOptimizer:
    """Bayesian Optimization for hyperparameter tuning"""
    
    def __init__(self, param_ranges: Dict[str, Tuple[float, float]], 
                 objective_function: Callable, 
                 n_iterations: int = 10, 
                 exploration_weight: float = 0.01):
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
        self.X_samples: List[np.ndarray] = []  # Parameter combinations
        self.y_samples: List[float] = []  # Results
        self.best_params: Optional[Dict[str, float]] = None
        self.best_score: float = float('-inf')  # For maximization
        
    def _convert_params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameters dictionary to normalized array for GP"""
        x = []
        for param_name in self.param_ranges:
            param_min, param_max = self.param_ranges[param_name]
            normalized_value = (params[param_name] - param_min) / (param_max - param_min)
            x.append(normalized_value)
        return np.array(x).reshape(1, -1)
    
    def _convert_array_to_params(self, x: np.ndarray) -> Dict[str, float]:
        """Convert normalized array back to parameters dictionary"""
        params = {}
        for i, param_name in enumerate(self.param_ranges):
            param_min, param_max = self.param_ranges[param_name]
            params[param_name] = param_min + x[i] * (param_max - param_min)
        return params
    
    def _acquisition_function(self, x: np.ndarray, gp: GaussianProcessRegressor, best_f: float) -> float:
        """Expected Improvement acquisition function"""
        x = x.reshape(1, -1)
        mu, sigma = gp.predict(x, return_std=True)
        
        # Handle zero/near-zero variance case
        if sigma < 1e-6:
            return 0.0
        
        # Calculate the improvement over the current best
        z = (mu - best_f - self.exploration_weight) / sigma
        return (mu - best_f - self.exploration_weight) * norm.cdf(z) + sigma * norm.pdf(z)
    
    def _suggest_next_params(self) -> Dict[str, float]:
        """Suggest next parameters to evaluate using acquisition function"""
        if len(self.X_samples) < 2:
            # Random sampling for first few iterations
            x_random = np.random.rand(len(self.param_ranges))
            return self._convert_array_to_params(x_random)
        
        try:
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
            
            current_best = np.max(y) if len(y) > 0 else 0.0
            
            for x in random_points:
                acq_value = self._acquisition_function(x, self.gp, current_best)
                if acq_value > best_acquisition:
                    best_acquisition = acq_value
                    best_x = x
            
            # Fallback to random if we couldn't find a good point
            if best_x is None:
                best_x = np.random.rand(len(self.param_ranges))
                
            return self._convert_array_to_params(best_x)
            
        except Exception as e:
            # Fallback to random sampling if GP fitting fails
            print(f"GP fitting failed: {e}. Falling back to random sampling.")
            x_random = np.random.rand(len(self.param_ranges))
            return self._convert_array_to_params(x_random)
    
    def optimize(self, callback: Optional[Callable] = None, verbose: bool = True) -> Tuple[Dict[str, float], float, pd.DataFrame]:
        """Run the optimization process"""
        start_time = time.time()
        history = []
        
        try:
            for i in range(self.n_iterations):
                # Suggest parameters
                params = self._suggest_next_params()
                
                # Evaluate objective function with error handling
                try:
                    score = self.objective_function(params)
                    if score is None:
                        print(f"Warning: Objective function returned None for params {params}. Using default score.")
                        score = float('-inf')
                except Exception as e:
                    print(f"Error in objective function: {e}. Using default score.")
                    score = float('-inf')
                
                # Store results
                self.X_samples.append(self._convert_params_to_array(params)[0])
                self.y_samples.append(score)
                
                # Update best score and parameters
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()  # Make a copy to avoid reference issues
                
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
            
            # If we didn't find any good parameters, generate a random set
            if self.best_params is None:
                self.best_params = self._convert_array_to_params(np.random.rand(len(self.param_ranges)))
                self.best_score = float('-inf')
                
            if verbose:
                elapsed_time = time.time() - start_time
                print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
                print(f"Best score: {self.best_score:.4f}")
                print(f"Best parameters: {self.best_params}")
                
            return self.best_params, self.best_score, history_df
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            # Return a fallback result
            random_params = self._convert_array_to_params(np.random.rand(len(self.param_ranges)))
            history_df = pd.DataFrame(history) if history else pd.DataFrame()
            return random_params, float('-inf'), history_df
