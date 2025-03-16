"""
Utility module for PyTorch initialization and configuration
"""
import torch
import os
import warnings

def init_torch():
    """Initialize PyTorch with proper configuration"""
    # Set environment variables
    os.environ['PYTORCH_JIT'] = '0'  # Disable JIT to avoid class registration issues
    
    # Configure PyTorch
    torch.backends.cudnn.enabled = False  # Disable CUDNN
    torch.set_grad_enabled(True)  # Ensure gradients are enabled
    
    # Configure device
    device = 'cpu'  # Default to CPU
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            device = 'cuda'
        except Exception as e:
            warnings.warn(f"CUDA initialization failed: {str(e)}")
    
    return device

def get_device():
    """Get the current device"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'