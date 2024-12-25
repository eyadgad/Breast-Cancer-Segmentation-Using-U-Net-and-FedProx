import os
import cv2
import numpy as np
import glob
from augmentations import augment_images_and_masks  # Assuming augmentations function is in a separate module

# Load and preprocess image and mask data
def load_and_resize_images(image_paths, mask_paths, image_size=(256, 256)):
    """
    Load and resize images and masks.

    Args:
        image_paths (list): List of paths to image files.
        mask_paths (list): List of paths to mask files.
        image_size (tuple): Desired size of the resized images.

    Returns:
        tuple: Arrays of resized images and masks.
    """
    images = [
        cv2.resize(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), image_size, interpolation=cv2.INTER_NEAREST)
        for img in image_paths
    ]
    masks = [
        cv2.resize(cv2.imread(mask, cv2.IMREAD_GRAYSCALE), image_size, interpolation=cv2.INTER_NEAREST)
        for mask in mask_paths
    ]
    return np.array(images), np.array(masks)

# Load data paths and sort
def load_data_paths(base_path):
    """
    Load and sort data paths for images and masks.

    Args:
        base_path (str): Base directory containing the data.

    Returns:
        dict: Dictionary with sorted image and mask paths for each class.
    """
    data_paths = {}
    for cls in ["benign", "malignant", "normal"]:
        scans = glob.glob(os.path.join(base_path, cls, "scans", "*.png"))
        masks = glob.glob(os.path.join(base_path, cls, "masks", "*.png"))
        scans.sort()
        masks.sort()
        data_paths[cls] = {"scans": scans, "masks": masks}
    return data_paths

# Shuffle data
def shuffle_data(images, masks, seed=42):
    """
    Shuffle images and masks in unison.

    Args:
        images (np.ndarray): Array of images.
        masks (np.ndarray): Array of masks.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Shuffled images and masks.
    """
    np.random.seed(seed)
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    return images[indices], masks[indices]

# Split data into training and testing sets
def split_data(images, masks, split_ratio=0.9):
    """
    Split data into training and testing sets.

    Args:
        images (np.ndarray): Array of images.
        masks (np.ndarray): Array of masks.
        split_ratio (float): Ratio of training data.

    Returns:
        tuple: Training and testing images and masks.
    """
    split_idx = int(len(images) * split_ratio)
    return (
        images[:split_idx],
        masks[:split_idx],
        images[split_idx:],
        masks[split_idx:]
    )

# Augment and distribute data
def prepare_client_data(images, masks, augmentation_fn, split_ratio=0.9):
    """
    Augment and split data for federated clients.

    Args:
        images (np.ndarray): Array of images.
        masks (np.ndarray): Array of masks.
        augmentation_fn (function): Function to apply augmentations.
        split_ratio (float): Ratio of training data.

    Returns:
        tuple: Augmented training images/masks and testing images/masks.
    """
    train_images, train_masks, test_images, test_masks = split_data(images, masks, split_ratio)
    aug_images, aug_masks = augmentation_fn(train_images, train_masks)
    aug_images, aug_masks = shuffle_data(aug_images, aug_masks)
    return aug_images, aug_masks, test_images, test_masks



def prepare_federated_data(dataset_paths, image_size=(256, 256)):
    """
    Prepare federated data from multiple dataset paths.

    Args:
        dataset_paths (list): List of dataset paths. Each dataset should have subdirectories 
                              for "scans" and "masks", or class-specific subdirectories.
        image_size (tuple): Target image size (height, width) for resizing.

    Returns:
        tuple: (client_data, global_test_images, global_test_masks)
            - client_data: List of client-specific data [(train_images, train_masks, test_images, test_masks), ...].
            - global_test_images: Concatenated test images for all clients.
            - global_test_masks: Concatenated test masks for all clients.
    """
    # Placeholder for client data
    client_data = []

    # Loop through the dataset paths
    for dataset_path in dataset_paths:
        if isinstance(dataset_path, dict):
            # Structured datasets with class-specific subdirectories
            data_paths = load_data_paths(dataset_path["path"])
            normal_images, normal_masks = load_and_resize_images(
                data_paths["normal"]["scans"], data_paths["normal"]["masks"], image_size
            )
            benign_images, benign_masks = load_and_resize_images(
                data_paths["benign"]["scans"], data_paths["benign"]["masks"], image_size
            )
            malignant_images, malignant_masks = load_and_resize_images(
                data_paths["malignant"]["scans"], data_paths["malignant"]["masks"], image_size
            )

            # Distribute data to clients
            client_data.append(prepare_client_data(
                np.concatenate((benign_images, normal_images[: len(normal_images) // 2]), axis=0),
                np.concatenate((benign_masks, normal_masks[: len(normal_masks) // 2]), axis=0),
                augment_images_and_masks
            ))
            client_data.append(prepare_client_data(
                np.concatenate((malignant_images, normal_images[len(normal_images) // 2:]), axis=0),
                np.concatenate((malignant_masks, normal_masks[len(normal_masks) // 2:]), axis=0),
                augment_images_and_masks
            ))
        else:
            # Unstructured datasets with flat "scans" and "masks" directories
            scans = glob.glob(os.path.join(dataset_path, "scans", "*.png"))
            masks = glob.glob(os.path.join(dataset_path, "masks", "*.png"))
            scans.sort()
            masks.sort()

            images, masks = load_and_resize_images(scans, masks, image_size)
            client_data.append(prepare_client_data(images, masks, augment_images_and_masks))

    # Prepare global test data
    global_test_images = np.concatenate([client[2] for client in client_data], axis=0) / 255.0
    global_test_masks = np.concatenate([client[3] for client in client_data], axis=0) / 255.0

    return client_data, global_test_images, global_test_masks
