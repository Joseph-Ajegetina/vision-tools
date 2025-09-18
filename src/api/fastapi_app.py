from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import json
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CINIC-10 Classification API",
    description="Deep Learning API for tool classification using MLP and CNN models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API responses
class PredictionResponse(BaseModel):
    """Response model for image classification predictions."""
    success: bool
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    model_used: str

class ModelInfo(BaseModel):
    """Response model for model information."""
    model_name: str
    model_type: str
    num_classes: int
    class_names: List[str]
    input_shape: List[int]
    preprocessing_info: Dict[str, Any]

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    models_loaded: List[str]
    timestamp: str

# Global variables for model loading
models = {}
model_metadata = {}
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Image preprocessing transforms
def get_preprocessing_transforms(mean: List[float], std: List[float]):
    """
    Get image preprocessing transforms for model inference.

    Args:
        mean: Dataset mean values for normalization
        std: Dataset standard deviation values for normalization

    Returns:
        Composed transforms for preprocessing
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

# Default preprocessing (CINIC-10 statistics)
DEFAULT_MEAN = [0.47889522, 0.47227842, 0.43047404]
DEFAULT_STD = [0.24205776, 0.23828046, 0.25874835]
preprocess_transform = get_preprocessing_transforms(DEFAULT_MEAN, DEFAULT_STD)

def load_model_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load model metadata from JSON file."""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load metadata: {str(e)}")
        return {}

def load_onnx_model(model_path: str, model_name: str) -> Optional[ort.InferenceSession]:
    """
    Load ONNX model for inference.

    Args:
        model_path: Path to ONNX model file
        model_name: Name of the model

    Returns:
        ONNX Runtime inference session or None if failed
    """
    try:
        # Create inference session
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # Use CPU for broader compatibility
        )

        logger.info(f"Successfully loaded ONNX model: {model_name}")
        logger.info(f"Input shape: {session.get_inputs()[0].shape}")
        logger.info(f"Output shape: {session.get_outputs()[0].shape}")

        return session

    except Exception as e:
        logger.error(f"Failed to load ONNX model {model_name}: {str(e)}")
        return None

def load_torchscript_model(model_path: str, model_name: str) -> Optional[torch.jit.ScriptModule]:
    """
    Load TorchScript model for inference.

    Args:
        model_path: Path to TorchScript model file
        model_name: Name of the model

    Returns:
        TorchScript model or None if failed
    """
    try:
        # Load TorchScript model
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()

        logger.info(f"Successfully loaded TorchScript model: {model_name}")
        return model

    except Exception as e:
        logger.error(f"Failed to load TorchScript model {model_name}: {str(e)}")
        return None

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess uploaded image for model inference.

    Args:
        image: PIL Image object

    Returns:
        Preprocessed image as numpy array
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply preprocessing transforms
    image_tensor = preprocess_transform(image)

    # Add batch dimension and convert to numpy
    image_batch = image_tensor.unsqueeze(0)
    return image_batch.numpy()

def predict_with_onnx(session: ort.InferenceSession, image_array: np.ndarray) -> np.ndarray:
    """
    Make prediction using ONNX model.

    Args:
        session: ONNX Runtime inference session
        image_array: Preprocessed image array

    Returns:
        Prediction probabilities
    """
    # Get input name
    input_name = session.get_inputs()[0].name

    # Run inference
    result = session.run(None, {input_name: image_array})

    # Apply softmax to get probabilities
    logits = result[0]
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    return probabilities

def predict_with_torchscript(model: torch.jit.ScriptModule, image_array: np.ndarray) -> np.ndarray:
    """
    Make prediction using TorchScript model.

    Args:
        model: TorchScript model
        image_array: Preprocessed image array

    Returns:
        Prediction probabilities
    """
    # Convert to tensor
    image_tensor = torch.from_numpy(image_array)

    # Run inference
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

    return probabilities.numpy()

# Startup event to load models
@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting CINIC-10 Classification API...")

    # Define model paths (these would be set via environment variables in production)
    model_dir = Path("./exported_models")

    # Try to load available models
    model_files = {
        "mlp_onnx": model_dir / "MLP.onnx",
        "cnn_onnx": model_dir / "CNN.onnx",
        "mlp_torchscript": model_dir / "MLP_torchscript.pt",
        "cnn_torchscript": model_dir / "CNN_torchscript.pt"
    }

    # Load ONNX models
    for model_name, model_path in model_files.items():
        if model_path.exists():
            if "onnx" in model_name:
                model = load_onnx_model(str(model_path), model_name)
                if model:
                    models[model_name] = model
            elif "torchscript" in model_name:
                model = load_torchscript_model(str(model_path), model_name)
                if model:
                    models[model_name] = model

    # Load metadata if available
    metadata_path = model_dir / "model_metadata.json"
    if metadata_path.exists():
        global model_metadata
        model_metadata = load_model_metadata(str(metadata_path))

    logger.info(f"Loaded {len(models)} models: {list(models.keys())}")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys()),
        timestamp=str(time.time())
    )

# Model information endpoint
@app.get("/models", response_model=List[ModelInfo])
async def get_model_info():
    """Get information about loaded models."""
    model_info_list = []

    for model_name in models.keys():
        # Determine model type
        if "mlp" in model_name.lower():
            model_type = "Multi-Layer Perceptron (MLP)"
        elif "cnn" in model_name.lower():
            model_type = "Convolutional Neural Network (CNN)"
        else:
            model_type = "Unknown"

        model_info = ModelInfo(
            model_name=model_name,
            model_type=model_type,
            num_classes=len(class_names),
            class_names=class_names,
            input_shape=[3, 32, 32],
            preprocessing_info={
                "resize": [32, 32],
                "normalize": {
                    "mean": DEFAULT_MEAN,
                    "std": DEFAULT_STD
                },
                "format": "RGB"
            }
        )
        model_info_list.append(model_info)

    return model_info_list

# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = "cnn_onnx"
):
    """
    Classify an uploaded image using the specified model.

    Args:
        file: Uploaded image file
        model_name: Name of the model to use for prediction

    Returns:
        Prediction results
    """
    start_time = time.time()

    # Validate model availability
    if model_name not in models:
        available_models = list(models.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. Available models: {available_models}"
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = preprocess_image(image)

        # Make prediction based on model type
        model = models[model_name]

        if "onnx" in model_name:
            probabilities = predict_with_onnx(model, image_array)
        elif "torchscript" in model_name:
            probabilities = predict_with_torchscript(model, image_array)
        else:
            raise HTTPException(
                status_code=500,
                detail="Unsupported model format"
            )

        # Process results
        probabilities_flat = probabilities[0]  # Remove batch dimension
        predicted_class_idx = np.argmax(probabilities_flat)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(probabilities_flat[predicted_class_idx])

        # Create probability dictionary
        class_probabilities = {
            class_names[i]: float(probabilities_flat[i])
            for i in range(len(class_names))
        }

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return PredictionResponse(
            success=True,
            prediction=predicted_class,
            confidence=confidence,
            probabilities=class_probabilities,
            processing_time_ms=processing_time,
            model_used=model_name
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = "cnn_onnx"
):
    """
    Classify multiple uploaded images.

    Args:
        files: List of uploaded image files
        model_name: Name of the model to use for prediction

    Returns:
        List of prediction results
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 10 images"
        )

    results = []

    for i, file in enumerate(files):
        try:
            prediction = await predict_image(file, model_name)
            prediction_dict = prediction.dict()
            prediction_dict["image_index"] = i
            prediction_dict["filename"] = file.filename
            results.append(prediction_dict)

        except HTTPException as e:
            results.append({
                "image_index": i,
                "filename": file.filename,
                "success": False,
                "error": e.detail
            })

    return {"results": results}

# Compare models endpoint
@app.post("/compare")
async def compare_models(
    file: UploadFile = File(...),
    models_to_compare: Optional[List[str]] = None
):
    """
    Compare predictions from multiple models on the same image.

    Args:
        file: Uploaded image file
        models_to_compare: List of model names to compare

    Returns:
        Comparison results
    """
    if models_to_compare is None:
        models_to_compare = list(models.keys())

    # Validate requested models
    invalid_models = [m for m in models_to_compare if m not in models]
    if invalid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid models: {invalid_models}. Available: {list(models.keys())}"
        )

    results = {}

    for model_name in models_to_compare:
        try:
            # Reset file pointer
            await file.seek(0)
            prediction = await predict_image(file, model_name)
            results[model_name] = prediction.dict()

        except Exception as e:
            results[model_name] = {
                "success": False,
                "error": str(e)
            }

    return {"comparison_results": results}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )