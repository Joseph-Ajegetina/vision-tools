import os
import gdown
import zipfile
from pathlib import Path
import shutil
from typing import Optional
import logging
import urllib.request
import re

logger = logging.getLogger(__name__)


class DataDownloader:
    """
    Handles downloading and extracting CINIC-10 dataset from Google Drive.

    This class provides functionality to download the dataset from Google Drive,
    extract it, and organize it in the expected directory structure for PyTorch
    data loaders.
    """

    def __init__(self, data_dir: str = "../data/cinic10", google_drive_id: Optional[str] = None):
        """
        Initialize the data downloader.

        Args:
            data_dir: Directory to store the dataset
            google_drive_id: Google Drive file ID for the dataset
        """
        self.data_dir = Path(data_dir)
        self.google_drive_id = google_drive_id
        self.dataset_dir = self.data_dir / "cinic10"
        self.zip_path = self.data_dir / "cinic10.zip"

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def extract_file_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract Google Drive file ID from various URL formats.

        Args:
            url: Google Drive URL (sharing link or direct link)

        Returns:
            File ID if found, None otherwise
        """
        # Pattern for different Google Drive URL formats
        patterns = [
            r'/file/d/([a-zA-Z0-9-_]+)',  # Standard sharing URL
            r'id=([a-zA-Z0-9-_]+)',       # Direct download URL
            r'/d/([a-zA-Z0-9-_]+)/',      # Alternative format
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        # If no pattern matches, maybe it's already just the file ID
        if re.match(r'^[a-zA-Z0-9-_]+$', url):
            return url

        return None

    def download_from_google_drive(self, file_id_or_url: Optional[str] = None) -> bool:
        """
        Download CINIC-10 dataset from Google Drive.

        Args:
            file_id_or_url: Google Drive file ID or sharing URL (optional, uses instance variable if not provided)

        Returns:
            True if download successful, False otherwise
        """
        if file_id_or_url is None:
            file_id_or_url = self.google_drive_id

        if file_id_or_url is None:
            logger.error("No Google Drive file ID or URL provided")
            return False

        # Extract file ID if a URL was provided
        if 'drive.google.com' in file_id_or_url:
            file_id = self.extract_file_id_from_url(file_id_or_url)
            if file_id is None:
                logger.error(f"Could not extract file ID from URL: {file_id_or_url}")
                return False
        else:
            file_id = file_id_or_url

        try:
            logger.info(f"Downloading dataset from Google Drive (ID: {file_id})")

            # Download the file using gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(self.zip_path), quiet=False)

            logger.info(f"Dataset downloaded successfully to {self.zip_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download dataset: {str(e)}")
            return False

    def download_from_url(self, url: str) -> bool:
        """
        Download dataset from a direct URL.

        Args:
            url: Direct download URL

        Returns:
            True if download successful, False otherwise
        """
        try:
            logger.info(f"Downloading dataset from URL: {url}")
            gdown.download(url, str(self.zip_path), quiet=False)
            logger.info(f"Dataset downloaded successfully to {self.zip_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download dataset from URL: {str(e)}")
            return False

    def extract_dataset(self) -> bool:
        """
        Extract the downloaded zip file.

        Returns:
            True if extraction successful, False otherwise
        """
        if not self.zip_path.exists():
            logger.error(f"Zip file not found: {self.zip_path}")
            return False

        try:
            logger.info(f"Extracting dataset to {self.dataset_dir}")

            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

            logger.info("Dataset extracted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to extract dataset: {str(e)}")
            return False

    def organize_dataset(self) -> bool:
        """
        Organize the extracted dataset into train/test/validation structure.

        Expected CINIC-10 structure after extraction:
        cinic10/
        ├── train/
        │   ├── class0/
        │   ├── class1/
        │   └── ...
        ├── test/
        │   ├── class0/
        │   ├── class1/
        │   └── ...
        └── valid/
            ├── class0/
            ├── class1/
            └── ...

        Returns:
            True if organization successful, False otherwise
        """
        try:
            # Check if dataset is already organized
            if self._is_dataset_organized():
                logger.info("Dataset is already properly organized")
                return True

            logger.info("Organizing dataset structure")

            # CINIC-10 typically comes with train/test/valid splits
            # Verify the structure exists
            splits = ['train', 'test', 'valid']
            for split in splits:
                split_dir = self.dataset_dir / split
                if not split_dir.exists():
                    logger.error(f"Expected split directory not found: {split_dir}")
                    return False

            logger.info("Dataset organization verified")
            return True

        except Exception as e:
            logger.error(f"Failed to organize dataset: {str(e)}")
            return False

    def _is_dataset_organized(self) -> bool:
        """
        Check if the dataset is properly organized.

        Returns:
            True if dataset structure is valid
        """
        if not self.dataset_dir.exists():
            return False

        # Check for train/test/valid splits
        splits = ['train', 'test', 'valid']
        for split in splits:
            split_dir = self.dataset_dir / split
            if not split_dir.exists():
                return False

            # Check if split contains class directories
            class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            if len(class_dirs) == 0:
                return False

        return True

    def get_dataset_info(self) -> dict:
        """
        Get information about the downloaded dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        if not self._is_dataset_organized():
            return {"error": "Dataset not properly organized"}

        info = {
            "dataset_path": str(self.dataset_dir),
            "splits": {},
            "classes": []
        }

        splits = ['train', 'test', 'valid']
        for split in splits:
            split_dir = self.dataset_dir / split
            if split_dir.exists():
                class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                class_names = [d.name for d in class_dirs]

                # Count images in each class
                class_counts = {}
                total_images = 0
                for class_dir in class_dirs:
                    image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
                    class_counts[class_dir.name] = len(image_files)
                    total_images += len(image_files)

                info["splits"][split] = {
                    "total_images": total_images,
                    "num_classes": len(class_names),
                    "class_distribution": class_counts
                }

                # Store class names (should be same across splits)
                if not info["classes"]:
                    info["classes"] = sorted(class_names)

        return info

    def setup_dataset(self, google_drive_id: Optional[str] = None,
                     download_url: Optional[str] = None) -> bool:
        """
        Complete dataset setup: download, extract, and organize.

        Args:
            google_drive_id: Google Drive file ID
            download_url: Direct download URL (alternative to Google Drive)

        Returns:
            True if setup successful, False otherwise
        """
        # Check if dataset already exists
        if self._is_dataset_organized():
            logger.info("Dataset already exists and is properly organized")
            return True

        # Download dataset
        if download_url:
            if not self.download_from_url(download_url):
                return False
        elif google_drive_id:
            if not self.download_from_google_drive(google_drive_id):
                return False
        else:
            logger.error("No download source provided (Google Drive ID or URL)")
            return False

        # Extract dataset
        if not self.extract_dataset():
            return False

        # Organize dataset
        if not self.organize_dataset():
            return False

        # Clean up zip file
        if self.zip_path.exists():
            self.zip_path.unlink()
            logger.info("Cleaned up zip file")

        logger.info("Dataset setup completed successfully")
        return True

    def cleanup(self):
        """Clean up downloaded files."""
        if self.zip_path.exists():
            self.zip_path.unlink()
            logger.info("Cleaned up zip file")


# Example usage and instructions
def get_cinic10_instructions() -> str:
    """
    Get instructions for setting up CINIC-10 dataset.

    Returns:
        String with setup instructions
    """
    instructions = """
CINIC-10 Dataset Setup Instructions:

1. Download CINIC-10 from Kaggle or other source
2. Upload the dataset zip file to your Google Drive
3. Get the Google Drive sharing link and extract the file ID
4. Use the DataDownloader with your file ID:

Example:
```python
from src.data.download import DataDownloader

# Initialize downloader
downloader = DataDownloader(data_dir="./data")

# Setup dataset with Google Drive ID
file_id = "your_google_drive_file_id_here"
success = downloader.setup_dataset(google_drive_id=file_id)

if success:
    print("Dataset ready for use!")
    print(downloader.get_dataset_info())
else:
    print("Dataset setup failed")
```

Alternative sources for CINIC-10:
- Kaggle: https://www.kaggle.com/datasets/mengcius/cinic10
- Official: https://github.com/BayesWatch/cinic-10

Note: Make sure the dataset follows the expected structure:
cinic10/
├── train/
├── test/
└── valid/
    """
    return instructions