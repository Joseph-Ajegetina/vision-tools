import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from .base import BaseModel


class CNN(BaseModel):
    """
    Convolutional Neural Network for image classification.

    This model uses convolutional layers to extract spatial features from images,
    followed by fully connected layers for classification. CNNs are particularly
    effective for image tasks due to their ability to capture local patterns
    and spatial hierarchies.

    Architecture:
    - Convolutional blocks: Conv2d -> BatchNorm -> ReLU -> MaxPool
    - Adaptive pooling: Ensures consistent feature map size
    - Fully connected layers: For final classification

    Mathematical Foundation:
    - Convolution: (f * g)(t) = ∫ f(τ)g(t-τ)dτ (discrete version for images)
    - Feature maps: Detect edges, textures, and complex patterns
    - Pooling: Reduces spatial dimensions while retaining important features
    - Batch normalization: Normalizes inputs to improve training stability
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        conv_layers: List[Dict] = None,
        fc_layers: List[int] = [256, 128],
        dropout: float = 0.5,
        pool_size: int = 2,
        batch_norm: bool = True
    ):
        """
        Initialize the CNN model.

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB)
            conv_layers: List of conv layer configs [{'out_channels': 32, 'kernel_size': 3, 'padding': 1}]
            fc_layers: List of fully connected layer sizes
            dropout: Dropout probability
            pool_size: MaxPool kernel size
            batch_norm: Whether to use batch normalization
        """
        super().__init__(num_classes)
        self.model_type = "CNN"
        self.input_channels = input_channels
        self.dropout_prob = dropout
        self.batch_norm = batch_norm

        # Default conv layer configuration
        if conv_layers is None:
            conv_layers = [
                {'out_channels': 32, 'kernel_size': 3, 'padding': 1},
                {'out_channels': 64, 'kernel_size': 3, 'padding': 1},
                {'out_channels': 128, 'kernel_size': 3, 'padding': 1}
            ]

        self.conv_config = conv_layers
        self.fc_config = fc_layers

        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        in_channels = input_channels
        for conv_config in conv_layers:
            # Convolutional layer
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_config['out_channels'],
                kernel_size=conv_config['kernel_size'],
                padding=conv_config.get('padding', 0),
                stride=conv_config.get('stride', 1)
            )
            self.conv_layers.append(conv_layer)

            # Batch normalization (optional)
            if batch_norm:
                self.bn_layers.append(nn.BatchNorm2d(conv_config['out_channels']))

            # Max pooling
            self.pool_layers.append(nn.MaxPool2d(
                kernel_size=pool_size,
                stride=pool_size
            ))

            in_channels = conv_config['out_channels']

        # Activation and dropout
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Adaptive pooling to ensure consistent feature map size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Calculate the size of features after conv layers
        self.feature_size = conv_layers[-1]['out_channels']

        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = self.feature_size

        for fc_size in fc_layers:
            self.fc_layers.append(nn.Linear(in_features, fc_size))
            in_features = fc_size

        # Output layer
        self.output_layer = nn.Linear(in_features, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using He initialization for ReLU networks.

        He initialization is particularly effective for ReLU networks as it
        accounts for the fact that ReLU zeros out half of the activations.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.

        Mathematical operations:
        1. Convolution: output[i,j] = Σ(input[i+m,j+n] * kernel[m,n])
        2. Batch normalization: y = γ * (x - μ) / σ + β
        3. ReLU activation: f(x) = max(0, x)
        4. Max pooling: output[i,j] = max(input[i*s:i*s+k, j*s:j*s+k])
        5. Adaptive pooling: Reduces feature maps to consistent size
        6. Fully connected: y = Wx + b

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional feature extraction
        for i, conv_layer in enumerate(self.conv_layers):
            # Convolution operation
            x = conv_layer(x)

            # Batch normalization (if enabled)
            if self.batch_norm and i < len(self.bn_layers):
                x = self.bn_layers[i](x)

            # ReLU activation
            x = self.relu(x)

            # Max pooling
            x = self.pool_layers[i](x)

        # Adaptive pooling to ensure consistent size
        x = self.adaptive_pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Output layer (no activation for logits)
        x = self.output_layer(x)

        return x

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary containing model architecture details
        """
        conv_arch = []
        for i, config in enumerate(self.conv_config):
            layer_desc = f"Conv2d({config['out_channels']}, k={config['kernel_size']}"
            if config.get('padding'):
                layer_desc += f", p={config['padding']}"
            layer_desc += ")"
            if self.batch_norm:
                layer_desc += " -> BatchNorm -> ReLU -> MaxPool"
            else:
                layer_desc += " -> ReLU -> MaxPool"
            conv_arch.append(layer_desc)

        fc_arch = " -> ".join([f"FC({size})" for size in self.fc_config])

        return {
            "type": "Convolutional Neural Network (CNN)",
            "architecture": {
                "convolutional": conv_arch,
                "fully_connected": fc_arch,
                "output": f"FC({self.num_classes})"
            },
            "input_shape": f"({self.input_channels}, 32, 32)",
            "conv_layers": len(self.conv_layers),
            "fc_layers": len(self.fc_layers),
            "batch_norm": self.batch_norm,
            "dropout": self.dropout_prob,
            "feature_size": self.feature_size,
            "mathematical_foundation": {
                "convolution": "output[i,j] = Σ(input[i+m,j+n] * kernel[m,n])",
                "pooling": "max(input[region])",
                "batch_norm": "y = γ * (x - μ) / σ + β",
                "activation": "ReLU(x) = max(0, x)",
                "loss": "CrossEntropyLoss"
            }
        }

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract feature maps from each convolutional layer for visualization.

        Args:
            x: Input tensor

        Returns:
            Dictionary mapping layer names to feature maps
        """
        feature_maps = {}
        feature_maps["input"] = x

        # Process through conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            feature_maps[f"conv_{i}_raw"] = x

            if self.batch_norm and i < len(self.bn_layers):
                x = self.bn_layers[i](x)
                feature_maps[f"conv_{i}_bn"] = x

            x = self.relu(x)
            feature_maps[f"conv_{i}_relu"] = x

            x = self.pool_layers[i](x)
            feature_maps[f"conv_{i}_pool"] = x

        feature_maps["final_features"] = x
        return feature_maps

    def count_conv_parameters(self) -> int:
        """Count parameters in convolutional layers only."""
        return sum(p.numel() for layer in self.conv_layers for p in layer.parameters())

    def count_fc_parameters(self) -> int:
        """Count parameters in fully connected layers only."""
        fc_params = sum(p.numel() for layer in self.fc_layers for p in layer.parameters())
        fc_params += sum(p.numel() for p in self.output_layer.parameters())
        return fc_params