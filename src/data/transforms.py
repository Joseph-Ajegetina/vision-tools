from torchvision import transforms
from typing import Tuple, List
import torch


def get_transforms(
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    use_augmentation: bool = True,
    image_size: int = 32
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and testing transforms for CINIC-10 dataset.

    Mathematical Foundation:
    - Normalization: x_normalized = (x - mean) / std
    - Random crop: Improves generalization by training on different parts
    - Horizontal flip: Data augmentation to increase dataset diversity
    - ToTensor: Converts PIL Image (0-255) to Tensor (0.0-1.0)

    Args:
        mean: Mean values for each channel [R, G, B]
        std: Standard deviation values for each channel [R, G, B]
        use_augmentation: Whether to apply data augmentation
        image_size: Target image size (CINIC-10 is 32x32)

    Returns:
        Tuple of (train_transform, test_transform)
    """

    # Base transforms for both train and test
    base_transforms = [
        transforms.ToTensor(),  # Convert PIL Image to Tensor, scale to [0, 1]
        transforms.Normalize(mean=mean, std=std)  # Normalize with dataset statistics
    ]

    if use_augmentation:
        # Training transforms with augmentation
        train_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip with 50% probability
            transforms.RandomCrop(
                size=image_size,
                padding=4,  # Add 4 pixels padding then crop to original size
                padding_mode='reflect'  # Reflect padding at borders
            ),
            transforms.ColorJitter(
                brightness=0.2,  # ±20% brightness variation
                contrast=0.2,    # ±20% contrast variation
                saturation=0.2,  # ±20% saturation variation
                hue=0.1         # ±10% hue variation
            ),
            transforms.RandomRotation(degrees=15),  # Random rotation ±15 degrees
        ] + base_transforms
    else:
        # Simple training transforms without augmentation
        train_transforms = [
            transforms.Resize((image_size, image_size)),
        ] + base_transforms

    # Test transforms (no augmentation)
    test_transforms = [
        transforms.Resize((image_size, image_size)),
    ] + base_transforms

    return (
        transforms.Compose(train_transforms),
        transforms.Compose(test_transforms)
    )


def get_mlp_transforms(
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    use_augmentation: bool = True,
    image_size: int = 32
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transforms specifically optimized for MLP models.

    MLPs process flattened images, so we can use more aggressive augmentation
    since spatial relationships are not preserved anyway.

    Args:
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        use_augmentation: Whether to apply augmentation
        image_size: Target image size

    Returns:
        Tuple of (train_transform, test_transform)
    """

    # Base transforms
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    if use_augmentation:
        # More aggressive augmentation for MLPs since spatial structure is lost anyway
        train_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),  # Sometimes useful for certain classes
            transforms.RandomCrop(
                size=image_size,
                padding=4,
                padding_mode='reflect'
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.15
            ),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Random translation
                scale=(0.9, 1.1),      # Random scaling
                shear=5                # Random shear
            ),
        ] + base_transforms
    else:
        train_transforms = [
            transforms.Resize((image_size, image_size)),
        ] + base_transforms

    # Test transforms
    test_transforms = [
        transforms.Resize((image_size, image_size)),
    ] + base_transforms

    return (
        transforms.Compose(train_transforms),
        transforms.Compose(test_transforms)
    )


def get_cnn_transforms(
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    use_augmentation: bool = True,
    image_size: int = 32
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transforms specifically optimized for CNN models.

    CNNs preserve spatial relationships, so we use augmentations that
    maintain the spatial structure while increasing diversity.

    Args:
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        use_augmentation: Whether to apply augmentation
        image_size: Target image size

    Returns:
        Tuple of (train_transform, test_transform)
    """

    # Base transforms
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    if use_augmentation:
        # Spatial-aware augmentation for CNNs
        train_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(
                size=image_size,
                padding=4,
                padding_mode='reflect'
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(degrees=10),  # Less rotation to preserve spatial features
            # Add cutout for regularization
            transforms.RandomErasing(
                p=0.1,              # 10% probability
                scale=(0.02, 0.1),  # Erase 2-10% of image
                ratio=(0.3, 3.3),   # Aspect ratio range
                value=0             # Fill with zeros
            ),
        ] + base_transforms
    else:
        train_transforms = [
            transforms.Resize((image_size, image_size)),
        ] + base_transforms

    # Test transforms
    test_transforms = [
        transforms.Resize((image_size, image_size)),
    ] + base_transforms

    return (
        transforms.Compose(train_transforms),
        transforms.Compose(test_transforms)
    )


class ToFlatVector:
    """
    Custom transform to flatten images for MLP models.

    This transform reshapes the image tensor from (C, H, W) to (C*H*W,)
    which is required for fully connected layers.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Flatten the tensor.

        Args:
            tensor: Input tensor of shape (C, H, W)

        Returns:
            Flattened tensor of shape (C*H*W,)
        """
        return tensor.view(-1)


def get_mlp_flatten_transform() -> transforms.Compose:
    """
    Get transform that includes flattening for MLP models.

    Returns:
        Transform that converts image to flat vector
    """
    return transforms.Compose([
        ToFlatVector()
    ])


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.

    Mathematical operation: x_denorm = x_norm * std + mean

    Args:
        tensor: Normalized tensor
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization

    Returns:
        Denormalized tensor
    """
    # Convert to tensor if needed
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean).view(-1, 1, 1)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std).view(-1, 1, 1)

    return tensor * std + mean


def visualize_transforms(dataset, transform, num_samples: int = 4):
    """
    Visualize the effect of transforms on sample images.

    Args:
        dataset: Dataset to sample from
        transform: Transform to apply
        num_samples: Number of samples to show
    """
    import matplotlib.pyplot as plt
    import random

    # Get random samples
    indices = random.sample(range(len(dataset)), num_samples)

    fig, axes = plt.subplots(2, num_samples, figsize=(12, 6))

    for i, idx in enumerate(indices):
        # Original image
        original_img, label = dataset[idx]

        # Apply transform
        if hasattr(dataset, 'transform'):
            # Temporarily remove transform
            old_transform = dataset.transform
            dataset.transform = None
            original_img, _ = dataset[idx]
            dataset.transform = old_transform

        transformed_img = transform(original_img)

        # Plot original
        if original_img.dim() == 3:
            axes[0, i].imshow(original_img.permute(1, 2, 0))
        else:
            axes[0, i].imshow(original_img, cmap='gray')
        axes[0, i].set_title(f'Original - Class {label}')
        axes[0, i].axis('off')

        # Plot transformed
        if transformed_img.dim() == 3:
            # Denormalize for visualization
            denorm_img = denormalize_tensor(transformed_img)
            denorm_img = torch.clamp(denorm_img, 0, 1)
            axes[1, i].imshow(denorm_img.permute(1, 2, 0))
        else:
            axes[1, i].imshow(transformed_img, cmap='gray')
        axes[1, i].set_title('Transformed')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()