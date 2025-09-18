# Model architectures for CINIC-10 classification

from .base import BaseModel
from .mlp import MLP
from .cnn import CNN

__all__ = ['BaseModel', 'MLP', 'CNN']