import torch
import torch.utils.data
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import logging
from .transforms import get_transforms

logger = logging.getLogger(__name__)


class CINIC10DataModule:
    """
    Data module for handling CINIC-10 dataset loading, preprocessing, and splitting.

    This class provides a clean interface for loading the CINIC-10 dataset with
    appropriate transforms and creating data loaders for training, validation, and testing.

    CINIC-10 Classes:
    0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
    5: dog, 6: frog, 7: horse, 8: ship, 9: truck
    """

    def __init__(
        self,
        data_dir: str = "./data/cinic10",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        validation_split: float = 0.2,
        seed: int = 42
    ):
        """
        Initialize the CINIC-10 data module.

        Args:
            data_dir: Path to the CINIC-10 dataset directory
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            validation_split: Fraction of training data to use for validation
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.validation_split = validation_split
        self.seed = seed

        # Class names for CINIC-10
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.num_classes = len(self.class_names)

        # Data loaders will be initialized later
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Dataset statistics (will be computed)
        self.mean = None
        self.std = None

    def compute_dataset_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and standard deviation of the dataset for normalization.

        Returns:
            Tuple of (mean, std) tensors for each channel
        """
        logger.info("Computing dataset statistics...")

        # Use minimal transforms for stats computation
        temp_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Load training data for stats computation
        train_dataset = datasets.ImageFolder(
            root=self.data_dir / "train",
            transform=temp_transform
        )

        # Create a data loader for stats computation
        stats_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=self.num_workers
        )

        # Compute mean and std
        mean = torch.zeros(3)
        std = torch.zeros(3)
        total_samples = 0

        for images, _ in stats_loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_samples += batch_samples

        mean /= total_samples
        std /= total_samples

        self.mean = mean
        self.std = std

        logger.info(f"Dataset statistics computed - Mean: {mean}, Std: {std}")
        return mean, std

    def setup_data_loaders(self, use_augmentation: bool = True) -> Dict[str, DataLoader]:
        """
        Setup data loaders for training, validation, and testing.

        Args:
            use_augmentation: Whether to apply data augmentation to training data

        Returns:
            Dictionary containing train, val, and test data loaders
        """
        logger.info("Setting up data loaders...")

        # Compute dataset statistics if not already done
        if self.mean is None or self.std is None:
            self.compute_dataset_stats()

        # Get transforms
        train_transform, test_transform = get_transforms(
            mean=self.mean.tolist(),
            std=self.std.tolist(),
            use_augmentation=use_augmentation
        )

        # Load datasets
        # Check if CINIC-10 has separate validation set
        val_dir = self.data_dir / "valid"
        if val_dir.exists():
            # CINIC-10 has separate validation set
            train_dataset = datasets.ImageFolder(
                root=self.data_dir / "train",
                transform=train_transform
            )
            val_dataset = datasets.ImageFolder(
                root=self.data_dir / "valid",
                transform=test_transform
            )
        else:
            # Split training set into train and validation
            full_train_dataset = datasets.ImageFolder(
                root=self.data_dir / "train",
                transform=train_transform
            )

            # Calculate split sizes
            total_size = len(full_train_dataset)
            val_size = int(total_size * self.validation_split)
            train_size = total_size - val_size

            # Split dataset
            torch.manual_seed(self.seed)
            train_dataset, val_dataset = random_split(
                full_train_dataset, [train_size, val_size]
            )

            # Apply different transforms to validation set
            val_dataset.dataset = datasets.ImageFolder(
                root=self.data_dir / "train",
                transform=test_transform
            )

        # Test dataset
        test_dataset = datasets.ImageFolder(
            root=self.data_dir / "test",
            transform=test_transform
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  # For consistent batch sizes
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        logger.info(f"Data loaders created:")
        logger.info(f"  Train: {len(self.train_loader)} batches, {len(train_dataset)} samples")
        logger.info(f"  Val: {len(self.val_loader)} batches, {len(val_dataset)} samples")
        logger.info(f"  Test: {len(self.test_loader)} batches, {len(test_dataset)} samples")

        return {
            'train': self.train_loader,
            'val': self.val_loader,
            'test': self.test_loader
        }

    def get_data_loaders(self) -> Dict[str, DataLoader]:
        """
        Get existing data loaders or create them if they don't exist.

        Returns:
            Dictionary containing train, val, and test data loaders
        """
        if self.train_loader is None:
            return self.setup_data_loaders()

        return {
            'train': self.train_loader,
            'val': self.val_loader,
            'test': self.test_loader
        }

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.

        Returns:
            Dictionary containing dataset information
        """
        info = {
            "name": "CINIC-10",
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "image_shape": (3, 32, 32),
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "data_dir": str(self.data_dir)
        }

        # Add loader information if available
        if self.train_loader is not None:
            info.update({
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "train_batches": len(self.train_loader),
                "val_batches": len(self.val_loader),
                "test_batches": len(self.test_loader),
                "train_samples": len(self.train_loader.dataset),
                "val_samples": len(self.val_loader.dataset),
                "test_samples": len(self.test_loader.dataset)
            })

        return info

    def visualize_samples(self, num_samples: int = 8, split: str = "train") -> None:
        """
        Visualize random samples from the dataset.

        Args:
            num_samples: Number of samples to visualize
            split: Dataset split to sample from ("train", "val", "test")
        """
        import matplotlib.pyplot as plt

        # Get appropriate data loader
        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        elif split == "test":
            loader = self.test_loader
        else:
            raise ValueError(f"Invalid split: {split}")

        if loader is None:
            logger.error("Data loaders not initialized. Call setup_data_loaders() first.")
            return

        # Get a batch of data
        data_iter = iter(loader)
        images, labels = next(data_iter)

        # Select random samples
        indices = torch.randperm(len(images))[:num_samples]
        sample_images = images[indices]
        sample_labels = labels[indices]

        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()

        for i in range(num_samples):
            img = sample_images[i]
            label = sample_labels[i]

            # Denormalize image for visualization
            if self.mean is not None and self.std is not None:
                for c in range(3):
                    img[c] = img[c] * self.std[c] + self.mean[c]

            # Convert to numpy and transpose for matplotlib
            img_np = img.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)

            axes[i].imshow(img_np)
            axes[i].set_title(f'{self.class_names[label]}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def get_class_distribution(self, split: str = "train") -> Dict[str, int]:
        """
        Get class distribution for a specific split.

        Args:
            split: Dataset split ("train", "val", "test")

        Returns:
            Dictionary mapping class names to counts
        """
        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        elif split == "test":
            loader = self.test_loader
        else:
            raise ValueError(f"Invalid split: {split}")

        if loader is None:
            logger.error("Data loaders not initialized.")
            return {}

        class_counts = torch.zeros(self.num_classes, dtype=torch.long)

        for _, labels in loader:
            for label in labels:
                class_counts[label] += 1

        return {
            self.class_names[i]: count.item()
            for i, count in enumerate(class_counts)
        }