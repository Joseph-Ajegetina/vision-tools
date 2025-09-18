import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models in the comparison study.

    This class provides a common interface for MLP and CNN models,
    ensuring consistent implementation and evaluation.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.model_type = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns model information including architecture details.

        Returns:
            Dictionary containing model metadata
        """
        pass

    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_size_mb(self) -> float:
        """
        Calculate model size in megabytes.

        Returns:
            Model size in MB
        """
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def summary(self) -> str:
        """
        Generate a summary of the model.

        Returns:
            String containing model summary
        """
        info = self.get_model_info()
        summary_str = f"""
Model Summary:
=============
Type: {info.get('type', 'Unknown')}
Architecture: {info.get('architecture', 'Not specified')}
Total Parameters: {self.count_parameters():,}
Model Size: {self.get_parameter_size_mb():.2f} MB
Input Shape: {info.get('input_shape', 'Not specified')}
Output Classes: {self.num_classes}
        """
        return summary_str.strip()

    def save_model(self, path: str):
        """
        Save model state dict.

        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'num_parameters': self.count_parameters(),
            'model_size_mb': self.get_parameter_size_mb()
        }, path)

    def load_model(self, path: str):
        """
        Load model state dict.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint