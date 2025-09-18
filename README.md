# CINIC-10 MLP vs CNN Comparison Project

A comprehensive deep learning project comparing Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) architectures for image classification on the CINIC-10 dataset.

## 🎯 Project Overview

This project implements a tool classification system for the CINIC-10 dataset, comparing the performance of MLP and CNN architectures. It's designed as a complete solution for the Ashesi University Deep Learning course (ICS553) PROSIT 1.

### Key Features

- **Dual Architecture Comparison**: Side-by-side evaluation of MLP vs CNN performance
- **Mathematical Foundation Documentation**: Detailed explanations of underlying mathematical concepts
- **Production-Ready Deployment**: FastAPI backend and model export functionality
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Cloud Integration**: Google Drive dataset management
- **Modular Design**: Easy to modify for different datasets

### Dataset: CINIC-10

CINIC-10 is an augmented extension of CIFAR-10 with additional data from ImageNet, containing:
- **Classes**: 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Images**: 270,000 total images (90,000 train, 90,000 validation, 90,000 test)
- **Resolution**: 32×32 pixels, RGB color

## 🏗️ Project Structure

```
cinic10-mlp-cnn-comparison/
├── src/
│   ├── models/
│   │   ├── base.py              # Abstract base model class
│   │   ├── mlp.py               # Multi-Layer Perceptron implementation
│   │   └── cnn.py               # Convolutional Neural Network implementation
│   ├── data/
│   │   ├── download.py          # Google Drive dataset downloader
│   │   ├── dataset.py           # CINIC-10 data module
│   │   └── transforms.py        # Data augmentation and preprocessing
│   ├── training/
│   │   ├── trainer.py           # Model training pipeline
│   │   └── evaluator.py         # Comprehensive evaluation suite
│   ├── utils/
│   │   └── export.py            # Model export for deployment
│   └── api/
│       └── fastapi_app.py       # Production API server
├── notebooks/                   # Jupyter notebooks for experiments
├── configs/
│   └── config.yaml             # Configuration parameters
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── CLAUDE.md                   # Claude Code guidance
```

## 📋 Requirements

### Core Dependencies
- Python 3.7+
- PyTorch 1.11.0+
- torchvision 0.12.0+
- FastAPI 0.70.0+
- NumPy, Matplotlib, Pandas
- ONNX and ONNX Runtime
- scikit-learn

### Development Tools
- Jupyter Lab
- pytest (testing)
- tensorboard (visualization)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create --name cinic10_comparison python=3.7.6
conda activate cinic10_comparison

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup (Automatic!)

**🚀 Quick Setup (Recommended):**
```python
from src.utils.setup import setup_env

# This automatically downloads CINIC-10 and sets up everything!
setup_info = setup_env()
print("✅ Ready to train!")
```

**📋 Manual Setup (if needed):**
```python
from src.data.download import DataDownloader

# Initialize downloader
downloader = DataDownloader(data_dir="./data")

# Setup dataset with Google Drive URL (automatically extracts file ID)
google_drive_url = "https://drive.google.com/file/d/1s5fGcJNGwUbujBxtTXcMN6YAYSVZHvAC/view?usp=drive_link"
success = downloader.setup_dataset(google_drive_id=google_drive_url)

if success:
    print("Dataset ready!")
```

**🧪 Test Setup:**
```bash
# Test that everything works
python test_setup.py
```

### 3. Train Models

```python
import torch
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.data.dataset import CINIC10DataModule
from src.training.trainer import ModelTrainer
import yaml

# Load configuration
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup data
data_module = CINIC10DataModule(
    data_dir=config['dataset']['data_dir'],
    batch_size=config['data_loader']['batch_size']
)
data_loaders = data_module.setup_data_loaders()

# Train MLP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = MLP(
    input_size=3072,  # 32*32*3
    hidden_layers=config['models']['mlp']['hidden_layers'],
    num_classes=10,
    dropout=config['models']['mlp']['dropout']
)

mlp_trainer = ModelTrainer(mlp_model, device, config, "MLP_experiment")
mlp_history = mlp_trainer.train(
    data_loaders['train'],
    data_loaders['val']
)

# Train CNN
cnn_model = CNN(
    num_classes=10,
    conv_layers=config['models']['cnn']['conv_layers'],
    fc_layers=config['models']['cnn']['fc_layers'],
    dropout=config['models']['cnn']['dropout']
)

cnn_trainer = ModelTrainer(cnn_model, device, config, "CNN_experiment")
cnn_history = cnn_trainer.train(
    data_loaders['train'],
    data_loaders['val']
)
```

### 4. Evaluate and Compare Models

```python
from src.training.evaluator import ModelEvaluator

evaluator = ModelEvaluator(
    class_names=data_module.class_names,
    device=device
)

# Evaluate both models
mlp_results = evaluator.evaluate_model(mlp_model, data_loaders['test'], "MLP")
cnn_results = evaluator.evaluate_model(cnn_model, data_loaders['test'], "CNN")

# Compare models
comparison = evaluator.compare_models(mlp_results, cnn_results)

# Generate comprehensive report
report = evaluator.generate_report([mlp_results, cnn_results], comparison)
print(report)

# Create visualizations
evaluator.plot_confusion_matrix(mlp_results)
evaluator.plot_confusion_matrix(cnn_results)
evaluator.plot_model_comparison(comparison)
```

### 5. Export Models for Deployment

```python
from src.utils.export import ModelExporter

exporter = ModelExporter(export_dir="./exported_models")

# Export both models in all formats
mlp_exports = exporter.export_all_formats(mlp_model, "MLP")
cnn_exports = exporter.export_all_formats(cnn_model, "CNN")
```

### 6. Start API Server

```bash
# Start FastAPI server
cd src/api
python fastapi_app.py

# Or using uvicorn directly
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

## 🧠 Mathematical Foundation

### Multi-Layer Perceptron (MLP)

**Architecture**: Fully connected layers with ReLU activation
```
Input (3072) → Hidden₁ (512) → Hidden₂ (256) → Hidden₃ (128) → Output (10)
```

**Mathematical Operations**:
- Forward pass: `y = f(Wx + b)` where `f` is ReLU activation
- ReLU activation: `f(x) = max(0, x)`
- Loss function: `CrossEntropyLoss = -Σ(y_true * log(y_pred))`
- Backpropagation: `∂L/∂w = gradient computation for weight updates`

**Key Characteristics**:
- Treats images as flattened vectors (loses spatial information)
- Uses dropout for regularization
- Xavier initialization for stable training

### Convolutional Neural Network (CNN)

**Architecture**: Convolutional blocks + fully connected layers
```
Input (3×32×32) → Conv₁ (32) → Conv₂ (64) → Conv₃ (128) → FC (256) → FC (128) → Output (10)
```

**Mathematical Operations**:
- Convolution: `output[i,j] = Σ(input[i+m,j+n] * kernel[m,n])`
- Max pooling: `output[i,j] = max(input[region])`
- Batch normalization: `y = γ * (x - μ) / σ + β`

**Key Characteristics**:
- Preserves spatial relationships in images
- Uses batch normalization for training stability
- He initialization optimized for ReLU networks
- Adaptive pooling for consistent feature map sizes

## 📊 Performance Metrics

The project evaluates models using comprehensive metrics:

### Classification Metrics
- **Accuracy**: Overall classification correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Top-k Accuracy**: Accuracy when correct class is in top-k predictions

### Model Analysis
- **Confusion Matrix**: Detailed classification breakdown
- **Per-class Performance**: Individual class accuracies
- **Parameter Count**: Model complexity comparison
- **Inference Time**: Speed comparison
- **Model Size**: Storage requirements

## 🚀 Deployment

### FastAPI Backend

The project includes a production-ready FastAPI backend with:

**Endpoints**:
- `POST /predict`: Single image classification
- `POST /predict/batch`: Batch image processing
- `POST /compare`: Compare multiple models on same image
- `GET /models`: List available models
- `GET /health`: Health check

**Example Usage**:
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg" \
  -F "model_name=cnn_onnx"

# Model comparison
curl -X POST "http://localhost:8000/compare" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

### Next.js Integration

The API is designed for easy integration with Next.js frontends:

```javascript
// Example Next.js integration
const predictImage = async (imageFile, modelName = 'cnn_onnx') => {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('model_name', modelName);

  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData,
  });

  return await response.json();
};
```

## 🔬 Experimental Results

### Expected Performance Comparison

Based on the architectural differences, we expect:

**MLP Performance**:
- Lower accuracy due to loss of spatial information
- Faster training per epoch (simpler architecture)
- More parameters in fully connected layers
- Better performance on globally distributed features

**CNN Performance**:
- Higher accuracy due to spatial feature extraction
- Slower training (more complex operations)
- More efficient parameter usage
- Better performance on local pattern recognition

### Key Insights

1. **Spatial Awareness**: CNNs leverage spatial relationships that MLPs ignore
2. **Parameter Efficiency**: CNNs achieve better performance with fewer parameters
3. **Feature Learning**: CNNs learn hierarchical features (edges → textures → objects)
4. **Generalization**: CNNs typically generalize better to new images

## 📈 Usage Examples

### Training from Scratch

```python
# Complete training pipeline
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.data.dataset import CINIC10DataModule
from src.training.trainer import ModelTrainer
from src.training.evaluator import ModelEvaluator

# Setup data
data_module = CINIC10DataModule()
data_loaders = data_module.setup_data_loaders()

# Initialize models
mlp = MLP(num_classes=10)
cnn = CNN(num_classes=10)

# Train and evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model, name in [(mlp, "MLP"), (cnn, "CNN")]:
    trainer = ModelTrainer(model, device, config, f"{name}_experiment")
    trainer.train(data_loaders['train'], data_loaders['val'])

    evaluator = ModelEvaluator(data_module.class_names, device)
    results = evaluator.evaluate_model(model, data_loaders['test'], name)
    print(f"{name} Results: {results['overall_metrics']['accuracy']:.2f}% accuracy")
```

### Custom Configuration

```yaml
# config.yaml customization
models:
  mlp:
    hidden_layers: [1024, 512, 256]  # Larger MLP
    dropout: 0.3

  cnn:
    conv_layers:
      - {out_channels: 64, kernel_size: 3, padding: 1}
      - {out_channels: 128, kernel_size: 3, padding: 1}
      - {out_channels: 256, kernel_size: 3, padding: 1}

training:
  epochs: 150
  learning_rate: 0.0005
  batch_size: 64
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit a pull request with detailed description

## 📄 License

This project is created for educational purposes as part of the Ashesi University Deep Learning course.

## 🔗 References

- CINIC-10 Dataset: [GitHub Repository](https://github.com/BayesWatch/cinic-10)
- PyTorch Documentation: [pytorch.org](https://pytorch.org/docs/)
- FastAPI Documentation: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)

## 📞 Support

For questions or issues:
1. Check the documentation in `CLAUDE.md`
2. Review the example notebooks
3. Check the API documentation at `http://localhost:8000/docs`

---

**Ashesi University - ICS553 Deep Learning - PROSIT 1**
*Comparing MLP and CNN Architectures for Image Classification*