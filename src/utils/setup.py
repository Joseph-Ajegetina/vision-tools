"""
Environment setup utilities for CINIC-10 project.

This module provides setup functions similar to the landmark identifier project,
automatically downloading and preparing the dataset.
"""

import os
import torch
import numpy as np
import random
from pathlib import Path
import logging
import yaml
from typing import Optional, Dict, Any

from ..data.download import DataDownloader

logger = logging.getLogger(__name__)


def setup_env(config_path: Optional[str] = None, force_download: bool = False) -> Dict[str, Any]:
    """
    Set up the environment for CINIC-10 MLP vs CNN comparison.

    This function mimics the setup_env from the landmark identifier project,
    automatically downloading the dataset and setting up the environment.

    Args:
        config_path: Path to configuration file
        force_download: Force re-download even if dataset exists

    Returns:
        Dictionary containing setup information
    """
    print("🚀 Setting up CINIC-10 environment...")

    # Check GPU availability
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("✅ GPU available")
        device = torch.device("cuda")
    else:
        print("⚠️ GPU *NOT* available. Will use CPU (slow)")
        device = torch.device("cpu")

    # Load configuration
    config = load_config(config_path)

    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    set_random_seeds(seed)
    print(f"🎲 Random seed set to: {seed}")

    # Download and setup dataset
    setup_dataset(config, force_download)

    # Create necessary directories
    create_directories()

    # Setup environment variables
    setup_environment_vars()

    print("✅ Environment setup complete!")

    return {
        'device': device,
        'config': config,
        'cuda_available': use_cuda,
        'seed': seed
    }


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Try to find config file relative to this module
        current_dir = Path(__file__).parent.parent.parent
        config_path = current_dir / "configs" / "config.yaml"

    config_path = Path(config_path)

    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"📋 Configuration loaded from: {config_path}")
            return config
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
    else:
        print("⚠️ Config file not found. Using default configuration.")

    # Default configuration
    return {
        'seed': 42,
        'dataset': {
            'name': 'CINIC-10',
            'num_classes': 10,
            'data_dir': './data/cinic10',
            'google_drive_id': 'https://drive.google.com/file/d/1s5fGcJNGwUbujBxtTXcMN6YAYSVZHvAC/view?usp=drive_link'
        },
        'data_loader': {
            'batch_size': 128,
            'num_workers': 4
        },
        'training': {
            'epochs': 50,
            'learning_rate': 0.001
        }
    }


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_dataset(config: Dict[str, Any], force_download: bool = False):
    """Setup CINIC-10 dataset with automatic download."""
    dataset_config = config.get('dataset', {})
    data_dir = dataset_config.get('data_dir', './data/cinic10')
    google_drive_url = dataset_config.get('google_drive_id')

    # Initialize data downloader
    downloader = DataDownloader(data_dir=Path(data_dir).parent)

    # Check if dataset already exists
    if downloader._is_dataset_organized() and not force_download:
        print("✅ Dataset already downloaded and organized")

        # Display dataset info
        try:
            dataset_info = downloader.get_dataset_info()
            if 'error' not in dataset_info:
                print(f"📊 Dataset information:")
                print(f"   Classes: {len(dataset_info.get('classes', []))}")
                for split, info in dataset_info.get('splits', {}).items():
                    total_images = info.get('total_images', 0)
                    if total_images > 0:
                        print(f"   {split.capitalize()}: {total_images:,} images")
        except Exception as e:
            print(f"⚠️ Could not load dataset info: {e}")

        return

    # Dataset doesn't exist, download it
    if google_drive_url:
        print(f"📥 Downloading CINIC-10 dataset from Google Drive...")
        print("⏳ This may take a while (dataset is ~1.5GB)...")

        try:
            success = downloader.setup_dataset(google_drive_id=google_drive_url)

            if success:
                print("✅ Dataset downloaded and organized successfully!")

                # Display dataset info
                dataset_info = downloader.get_dataset_info()
                if 'error' not in dataset_info:
                    print(f"📊 Dataset information:")
                    print(f"   Classes: {len(dataset_info.get('classes', []))}")
                    for split, info in dataset_info.get('splits', {}).items():
                        total_images = info.get('total_images', 0)
                        if total_images > 0:
                            print(f"   {split.capitalize()}: {total_images:,} images")
            else:
                print("❌ Dataset download failed")
                print("🔗 Please download CINIC-10 manually from:")
                print("   https://www.kaggle.com/datasets/mengcius/cinic10")
                print("   Or update the google_drive_id in config.yaml")

        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("🔗 Please download CINIC-10 manually from:")
            print("   https://www.kaggle.com/datasets/mengcius/cinic10")
            print("   Or check your Google Drive link in config.yaml")
    else:
        print("⚠️ No Google Drive URL provided in configuration")
        print("🔗 Please download CINIC-10 from:")
        print("   https://www.kaggle.com/datasets/mengcius/cinic10")
        print("   And update the google_drive_id in config.yaml")


def create_directories():
    """Create necessary directories for the project."""
    directories = [
        "checkpoints",
        "exported_models",
        "evaluation_results",
        "plots",
        "logs"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print(f"📁 Created directories: {', '.join(directories)}")


def setup_environment_vars():
    """Setup environment variables for the project."""
    # Add local bin to PATH if it exists (for workspace compatibility)
    local_bin = "/root/.local/bin"
    if os.path.exists(local_bin) and local_bin not in os.environ.get('PATH', ''):
        os.environ['PATH'] = f"{os.environ['PATH']}:{local_bin}"

    # Set other useful environment variables
    os.environ['PYTHONPATH'] = os.getcwd()

    # Disable some warnings for cleaner output
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'


def get_data_location() -> str:
    """
    Get the location of the CINIC-10 dataset.

    Returns:
        Path to the dataset directory
    """
    possible_locations = [
        "./data/cinic10",
        "../data/cinic10",
        "/data/cinic10",
        os.path.expanduser("~/data/cinic10")
    ]

    for location in possible_locations:
        if os.path.exists(location):
            return location

    raise IOError("CINIC-10 dataset not found. Please run setup_env() first.")


def verify_setup() -> bool:
    """
    Verify that the environment is properly set up.

    Returns:
        True if setup is valid, False otherwise
    """
    try:
        # Check if dataset exists
        data_location = get_data_location()

        # Check if required splits exist
        splits = ['train', 'test', 'valid']
        for split in splits:
            split_dir = Path(data_location) / split
            if not split_dir.exists():
                print(f"❌ Missing {split} split in dataset")
                return False

        # Check if classes exist
        train_dir = Path(data_location) / 'train'
        class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        if len(class_dirs) != 10:
            print(f"❌ Expected 10 classes, found {len(class_dirs)}")
            return False

        print("✅ Environment verification passed")
        return True

    except Exception as e:
        print(f"❌ Environment verification failed: {e}")
        return False


def print_setup_instructions():
    """Print setup instructions for users."""
    print("📋 CINIC-10 Setup Instructions:")
    print("=" * 50)
    print()
    print("🚀 Quick Setup:")
    print("   from src.utils.setup import setup_env")
    print("   setup_info = setup_env()")
    print()
    print("📁 Manual Setup (if automatic download fails):")
    print("   1. Download CINIC-10 from: https://www.kaggle.com/datasets/mengcius/cinic10")
    print("   2. Extract to: ./data/cinic10/")
    print("   3. Ensure structure: data/cinic10/{train,test,valid}/{class_names}/")
    print()
    print("🔧 Configuration:")
    print("   - Update configs/config.yaml with your Google Drive file ID")
    print("   - Adjust batch size and other hyperparameters as needed")
    print()
    print("✅ Verification:")
    print("   from src.utils.setup import verify_setup")
    print("   verify_setup()")


if __name__ == "__main__":
    # If run directly, print setup instructions
    print_setup_instructions()