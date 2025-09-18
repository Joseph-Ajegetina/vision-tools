# CINIC-10 MLP vs CNN Comparison Project
# Ashesi University - ICS553 Deep Learning - PROSIT 1

__version__ = "1.0.0"
__author__ = "Ashesi University Student"
__description__ = "Comparing MLP and CNN architectures for image classification on CINIC-10"

# Import main classes for easy access
from .models.mlp import MLP
from .models.cnn import CNN
from .data.dataset import CINIC10DataModule
from .training.trainer import ModelTrainer
from .training.evaluator import ModelEvaluator
from .utils.export import ModelExporter

__all__ = [
    'MLP',
    'CNN',
    'CINIC10DataModule',
    'ModelTrainer',
    'ModelEvaluator',
    'ModelExporter'
]