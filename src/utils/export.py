import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import logging
import warnings

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Export trained PyTorch models for deployment in production environments.

    Supports multiple export formats:
    - ONNX: Cross-platform neural network representation
    - TorchScript: PyTorch's JIT compilation for deployment
    - State Dict: PyTorch native format

    This enables deployment in various environments including:
    - FastAPI backends
    - Next.js frontend (via ONNX.js)
    - Mobile applications
    - Cloud inference services
    """

    def __init__(self, export_dir: str = "./exported_models"):
        """
        Initialize the model exporter.

        Args:
            export_dir: Directory to save exported models
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_to_onnx(
        self,
        model: nn.Module,
        model_name: str,
        input_shape: Tuple[int, ...] = (1, 3, 32, 32),
        opset_version: int = 11,
        dynamic_axes: Optional[Dict[str, Any]] = None,
        verify: bool = True
    ) -> Dict[str, Any]:
        """
        Export PyTorch model to ONNX format.

        ONNX (Open Neural Network Exchange) is an open format for representing
        machine learning models, enabling interoperability between different
        frameworks and deployment platforms.

        Args:
            model: Trained PyTorch model
            model_name: Name for the exported model
            input_shape: Shape of input tensor (batch_size, channels, height, width)
            opset_version: ONNX opset version to use
            dynamic_axes: Dynamic axes for variable input sizes
            verify: Whether to verify the exported model

        Returns:
            Dictionary containing export information
        """
        logger.info(f"Exporting {model_name} to ONNX format...")

        # Set model to evaluation mode
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(input_shape)

        # Default dynamic axes for batch dimension
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }

        # Export path
        onnx_path = self.export_dir / f"{model_name}.onnx"

        try:
            # Export to ONNX
            torch.onnx.export(
                model,                          # Model to export
                dummy_input,                    # Model input
                str(onnx_path),                # Export path
                export_params=True,             # Store trained parameters
                opset_version=opset_version,    # ONNX version
                do_constant_folding=True,       # Optimize constant folding
                input_names=['input'],          # Input names
                output_names=['output'],        # Output names
                dynamic_axes=dynamic_axes       # Dynamic axes
            )

            logger.info(f"Model exported to {onnx_path}")

            # Verify the exported model
            if verify:
                verification_result = self._verify_onnx_model(
                    str(onnx_path), dummy_input, model
                )
            else:
                verification_result = {"verified": False, "message": "Verification skipped"}

            # Get model information
            model_info = self._get_onnx_model_info(str(onnx_path))

            return {
                "format": "ONNX",
                "export_path": str(onnx_path),
                "input_shape": input_shape,
                "opset_version": opset_version,
                "dynamic_axes": dynamic_axes,
                "verification": verification_result,
                "model_info": model_info,
                "file_size_mb": onnx_path.stat().st_size / (1024 * 1024)
            }

        except Exception as e:
            logger.error(f"Failed to export to ONNX: {str(e)}")
            return {
                "format": "ONNX",
                "error": str(e),
                "export_path": str(onnx_path)
            }

    def export_to_torchscript(
        self,
        model: nn.Module,
        model_name: str,
        input_shape: Tuple[int, ...] = (1, 3, 32, 32),
        method: str = "trace",
        verify: bool = True
    ) -> Dict[str, Any]:
        """
        Export PyTorch model to TorchScript format.

        TorchScript is PyTorch's way to create serializable and optimizable
        models from PyTorch code, enabling deployment without Python dependencies.

        Args:
            model: Trained PyTorch model
            model_name: Name for the exported model
            input_shape: Shape of input tensor
            method: Export method ("trace" or "script")
            verify: Whether to verify the exported model

        Returns:
            Dictionary containing export information
        """
        logger.info(f"Exporting {model_name} to TorchScript format using {method}...")

        # Set model to evaluation mode
        model.eval()

        # Export path
        torchscript_path = self.export_dir / f"{model_name}_torchscript.pt"

        try:
            if method == "trace":
                # Trace method: Record operations during forward pass
                dummy_input = torch.randn(input_shape)
                traced_model = torch.jit.trace(model, dummy_input)
                traced_model.save(str(torchscript_path))

            elif method == "script":
                # Script method: Compile the model directly
                scripted_model = torch.jit.script(model)
                scripted_model.save(str(torchscript_path))

            else:
                raise ValueError(f"Unsupported method: {method}. Use 'trace' or 'script'")

            logger.info(f"Model exported to {torchscript_path}")

            # Verify the exported model
            if verify:
                verification_result = self._verify_torchscript_model(
                    str(torchscript_path), input_shape, model
                )
            else:
                verification_result = {"verified": False, "message": "Verification skipped"}

            return {
                "format": "TorchScript",
                "export_path": str(torchscript_path),
                "input_shape": input_shape,
                "method": method,
                "verification": verification_result,
                "file_size_mb": torchscript_path.stat().st_size / (1024 * 1024)
            }

        except Exception as e:
            logger.error(f"Failed to export to TorchScript: {str(e)}")
            return {
                "format": "TorchScript",
                "error": str(e),
                "export_path": str(torchscript_path)
            }

    def export_state_dict(
        self,
        model: nn.Module,
        model_name: str,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Export model state dictionary (native PyTorch format).

        Args:
            model: Trained PyTorch model
            model_name: Name for the exported model
            include_metadata: Whether to include model metadata

        Returns:
            Dictionary containing export information
        """
        logger.info(f"Exporting {model_name} state dictionary...")

        # Export path
        state_dict_path = self.export_dir / f"{model_name}_state_dict.pth"

        try:
            # Prepare checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__
            }

            # Add metadata if requested
            if include_metadata and hasattr(model, 'get_model_info'):
                checkpoint['model_info'] = model.get_model_info()

            if hasattr(model, 'count_parameters'):
                checkpoint['total_parameters'] = model.count_parameters()

            # Save checkpoint
            torch.save(checkpoint, state_dict_path)

            logger.info(f"State dictionary exported to {state_dict_path}")

            return {
                "format": "PyTorch State Dict",
                "export_path": str(state_dict_path),
                "includes_metadata": include_metadata,
                "file_size_mb": state_dict_path.stat().st_size / (1024 * 1024)
            }

        except Exception as e:
            logger.error(f"Failed to export state dictionary: {str(e)}")
            return {
                "format": "PyTorch State Dict",
                "error": str(e),
                "export_path": str(state_dict_path)
            }

    def export_all_formats(
        self,
        model: nn.Module,
        model_name: str,
        input_shape: Tuple[int, ...] = (1, 3, 32, 32),
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Export model in all supported formats.

        Args:
            model: Trained PyTorch model
            model_name: Name for the exported model
            input_shape: Shape of input tensor
            config: Export configuration dictionary

        Returns:
            Dictionary containing all export results
        """
        logger.info(f"Exporting {model_name} in all formats...")

        # Default configuration
        if config is None:
            config = {
                'onnx': {'opset_version': 11, 'verify': True},
                'torchscript': {'method': 'trace', 'verify': True},
                'state_dict': {'include_metadata': True}
            }

        results = {
            'model_name': model_name,
            'input_shape': input_shape,
            'exports': {}
        }

        # Export to ONNX
        try:
            onnx_config = config.get('onnx', {})
            onnx_result = self.export_to_onnx(
                model, model_name, input_shape, **onnx_config
            )
            results['exports']['onnx'] = onnx_result
        except Exception as e:
            logger.error(f"ONNX export failed: {str(e)}")
            results['exports']['onnx'] = {'error': str(e)}

        # Export to TorchScript
        try:
            torchscript_config = config.get('torchscript', {})
            torchscript_result = self.export_to_torchscript(
                model, model_name, input_shape, **torchscript_config
            )
            results['exports']['torchscript'] = torchscript_result
        except Exception as e:
            logger.error(f"TorchScript export failed: {str(e)}")
            results['exports']['torchscript'] = {'error': str(e)}

        # Export state dictionary
        try:
            state_dict_config = config.get('state_dict', {})
            state_dict_result = self.export_state_dict(
                model, model_name, **state_dict_config
            )
            results['exports']['state_dict'] = state_dict_result
        except Exception as e:
            logger.error(f"State dict export failed: {str(e)}")
            results['exports']['state_dict'] = {'error': str(e)}

        # Save export summary
        self._save_export_summary(results)

        return results

    def _verify_onnx_model(
        self,
        onnx_path: str,
        dummy_input: torch.Tensor,
        original_model: nn.Module
    ) -> Dict[str, Any]:
        """
        Verify ONNX model against original PyTorch model.

        Args:
            onnx_path: Path to ONNX model
            dummy_input: Input tensor for testing
            original_model: Original PyTorch model

        Returns:
            Verification results
        """
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(onnx_path)

            # Get PyTorch model output
            original_model.eval()
            with torch.no_grad():
                pytorch_output = original_model(dummy_input).numpy()

            # Get ONNX model output
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            onnx_output = ort_session.run(None, ort_inputs)[0]

            # Compare outputs
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

            is_close = np.allclose(pytorch_output, onnx_output, atol=1e-5)

            return {
                "verified": True,
                "outputs_match": is_close,
                "max_difference": float(max_diff),
                "mean_difference": float(mean_diff),
                "tolerance": 1e-5
            }

        except Exception as e:
            return {
                "verified": False,
                "error": str(e)
            }

    def _verify_torchscript_model(
        self,
        torchscript_path: str,
        input_shape: Tuple[int, ...],
        original_model: nn.Module
    ) -> Dict[str, Any]:
        """
        Verify TorchScript model against original PyTorch model.

        Args:
            torchscript_path: Path to TorchScript model
            input_shape: Input shape for testing
            original_model: Original PyTorch model

        Returns:
            Verification results
        """
        try:
            # Load TorchScript model
            loaded_model = torch.jit.load(torchscript_path)
            loaded_model.eval()

            # Create test input
            dummy_input = torch.randn(input_shape)

            # Get outputs
            original_model.eval()
            with torch.no_grad():
                pytorch_output = original_model(dummy_input)
                torchscript_output = loaded_model(dummy_input)

            # Compare outputs
            max_diff = torch.max(torch.abs(pytorch_output - torchscript_output)).item()
            mean_diff = torch.mean(torch.abs(pytorch_output - torchscript_output)).item()

            is_close = torch.allclose(pytorch_output, torchscript_output, atol=1e-5)

            return {
                "verified": True,
                "outputs_match": is_close,
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "tolerance": 1e-5
            }

        except Exception as e:
            return {
                "verified": False,
                "error": str(e)
            }

    def _get_onnx_model_info(self, onnx_path: str) -> Dict[str, Any]:
        """Get information about the ONNX model."""
        try:
            onnx_model = onnx.load(onnx_path)

            # Get input/output information
            inputs = []
            for input_tensor in onnx_model.graph.input:
                shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
                inputs.append({
                    'name': input_tensor.name,
                    'shape': shape,
                    'type': input_tensor.type.tensor_type.elem_type
                })

            outputs = []
            for output_tensor in onnx_model.graph.output:
                shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
                outputs.append({
                    'name': output_tensor.name,
                    'shape': shape,
                    'type': output_tensor.type.tensor_type.elem_type
                })

            return {
                'producer_name': onnx_model.producer_name,
                'producer_version': onnx_model.producer_version,
                'opset_version': onnx_model.opset_import[0].version,
                'inputs': inputs,
                'outputs': outputs,
                'num_nodes': len(onnx_model.graph.node)
            }

        except Exception as e:
            return {'error': str(e)}

    def _save_export_summary(self, results: Dict[str, Any]):
        """Save export summary to JSON file."""
        summary_path = self.export_dir / f"{results['model_name']}_export_summary.json"

        # Convert any non-serializable objects
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj

        # Deep copy and convert results
        import copy
        serializable_results = copy.deepcopy(results)

        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(item) for item in d]
            else:
                return convert_to_serializable(d)

        serializable_results = recursive_convert(serializable_results)

        with open(summary_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Export summary saved to {summary_path}")


def create_deployment_package(
    model_exports: Dict[str, Any],
    class_names: List[str],
    dataset_stats: Dict[str, Any],
    output_dir: str = "./deployment_package"
) -> str:
    """
    Create a complete deployment package with model and metadata.

    Args:
        model_exports: Model export results
        class_names: List of class names
        dataset_stats: Dataset statistics (mean, std, etc.)
        output_dir: Output directory for deployment package

    Returns:
        Path to deployment package
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create metadata file
    metadata = {
        'class_names': class_names,
        'num_classes': len(class_names),
        'dataset_stats': dataset_stats,
        'model_info': model_exports,
        'deployment_info': {
            'input_shape': [3, 32, 32],
            'input_format': 'RGB image, normalized',
            'output_format': 'Class probabilities',
            'preprocessing_required': True
        }
    }

    metadata_path = output_path / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Deployment package created at {output_path}")
    return str(output_path)