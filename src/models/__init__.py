"""
GAN models module containing base class and implementations
"""
from .base_gan import BaseGAN
from .table_gan import TableGAN
from .wgan import WGAN
from .cgan import CGAN
from .tvae import TVAE
from .ctgan import CTGAN

__all__ = ['BaseGAN', 'TableGAN', 'WGAN', 'CGAN', 'TVAE', 'CTGAN']