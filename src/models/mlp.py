import torch
import torch.nn as nn
from typing import List, Dict, Any
from .base import BaseModel


class MLP(BaseModel):
    """
    Multi-Layer Perceptron (Deep Feed-forward Neural Network) for image classification.

    This model flattens the input images and processes them through a series of
    fully connected layers with dropout for regularization.

    Architecture:
    - Input: Flattened image (32x32x3 = 3072 features for CINIC-10)
    - Hidden layers: Configurable fully connected layers with ReLU activation
    - Dropout: Applied between layers for regularization
    - Output: num_classes logits

    Mathematical Foundation:
    - Forward propagation: y = f(Wx + b) where f is activation function
    - Backpropagation: Gradient descent to minimize loss function
    - Regularization: Dropout randomly sets neurons to zero during training
    """

    def __init__(
        self,
        input_size: int = 3072,  # 32x32x3 for CINIC-10
        hidden_layers: List[int] = [512, 256, 128],
        num_classes: int = 10,
        dropout: float = 0.5,
        activation: str = "relu"
    ):
        """
        Initialize the MLP model.

        Args:
            input_size: Size of flattened input (height * width * channels)
            hidden_layers: List of hidden layer sizes
            num_classes: Number of output classes
            dropout: Dropout probability for regularization
            activation: Activation function ("relu", "tanh", "sigmoid")
        """
        super().__init__(num_classes)
        self.model_type = "MLP"
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.dropout_prob = dropout

        # Select activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build the network layers
        self.layers = nn.ModuleList()

        # Input layer
        prev_size = input_size
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(prev_size, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Initialize weights using Xavier/Glorot initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier initialization.

        Xavier initialization helps with gradient flow by setting initial
        weights based on the number of input and output neurons.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Mathematical operation:
        1. Flatten input: x_flat = x.view(batch_size, -1)
        2. For each layer: x = activation(Wx + b)
        3. Apply dropout for regularization
        4. Final layer: output = W_out * x + b_out (no activation for logits)

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten the input image to a 1D vector
        # Shape: (batch_size, channels, height, width) -> (batch_size, input_size)
        x = x.view(x.size(0), -1)

        # Pass through hidden layers with activation and dropout
        for layer in self.layers:
            x = layer(x)  # Linear transformation: Wx + b
            x = self.activation(x)  # Non-linear activation
            x = self.dropout(x)  # Regularization

        # Output layer (no activation for logits)
        x = self.output_layer(x)

        return x

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary containing model architecture details
        """
        return {
            "type": "Multi-Layer Perceptron (MLP)",
            "architecture": f"Input({self.input_size}) -> " +
                          " -> ".join([f"Hidden({size})" for size in self.hidden_layers]) +
                          f" -> Output({self.num_classes})",
            "input_shape": f"({self.input_size},) [Flattened from (3, 32, 32)]",
            "hidden_layers": self.hidden_layers,
            "dropout": self.dropout_prob,
            "activation": str(self.activation),
            "total_layers": len(self.hidden_layers) + 1,
            "mathematical_foundation": {
                "forward_pass": "y = f(Wx + b)",
                "activation": "ReLU(x) = max(0, x)",
                "loss": "CrossEntropyLoss",
                "optimization": "Adam with backpropagation"
            }
        }

    def get_layer_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate layer outputs for analysis.

        Args:
            x: Input tensor

        Returns:
            Dictionary mapping layer names to their outputs
        """
        outputs = {}
        x = x.view(x.size(0), -1)
        outputs["input_flattened"] = x

        for i, layer in enumerate(self.layers):
            x = layer(x)
            outputs[f"linear_{i}"] = x
            x = self.activation(x)
            outputs[f"activation_{i}"] = x
            x = self.dropout(x)

        x = self.output_layer(x)
        outputs["output_logits"] = x

        return outputs