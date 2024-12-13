import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.errors import RasterioIOError

import numpy as np
from torchvision.transforms.functional import center_crop, pad

class batch_process:
    def __init__(self, *args, mode: str = 'concat'):
        self.process_fns = args
        self.mode = mode
    def __call__(self, batch):
        for fn in self.process_fns:
            batch = [fn(image, label) for image, label in batch]
        # unzip the batch
        images, labels = tuple(zip(*batch))
        # convert to tensor
        if self.mode == 'concat':
            images = torch.cat(images, dim=0)
            labels = torch.cat(labels, dim=0)
        elif self.mode == 'stack':
            images = torch.stack(images, dim=0)
            labels = torch.stack(labels, dim=0)
        return images, labels

def resize_to_224(image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resize tensors to 224x224 using torchvision transforms.
    Args:
        image (torch.Tensor): [C, H, W]
        label (torch.Tensor): [H, W] or [1, H, W]
    """
    target_size = 224
    _, h, w = image.shape
    
    # For very small images, repeat the image content instead of padding
    if w < target_size // 2 or h < target_size // 2:
        # Calculate how many times to repeat
        repeat_h = max(1, target_size // h + (1 if target_size % h else 0))
        repeat_w = max(1, target_size // w + (1 if target_size % w else 0))
        
        # Repeat the tensors
        image = image.repeat(1, repeat_h, repeat_w)
        label = label.repeat(repeat_h, repeat_w) if label.dim() == 2 else label.repeat(1, repeat_h, repeat_w)
        
        # Now we can safely crop to exact size
        image = center_crop(image, [target_size, target_size])
        label = center_crop(label if label.dim() == 3 else label.unsqueeze(0), [target_size, target_size])
    else:
        # Original logic for images closer to target size
        if h > target_size or w > target_size:
            image = center_crop(image, [target_size, target_size])
            label = center_crop(label if label.dim() == 3 else label.unsqueeze(0), [target_size, target_size])
        elif h < target_size or w < target_size:
            padding = [
                (target_size - w) // 2, (target_size - h) // 2,
                (target_size - w + 1) // 2, (target_size - h + 1) // 2
            ]
            image = pad(image, padding, padding_mode='reflect')
            label = pad(label if label.dim() == 3 else label.unsqueeze(0), padding, padding_mode='reflect')
    
    return image, label.squeeze(0) if label.dim() == 3 else label
def shuffle_pixel(image, label):
    """
    Shuffles the pixels of the image and label in the same order.

    Args:
        image (np.ndarray): Image data.
        label (np.ndarray): Label data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Shuffled image and label.
    """
    # Flatten the image and label
    image_flat = image.reshape(image.shape[0], -1)
    label_flat = label.reshape(label.shape[0], -1)
    
    # mask the no data values
    mask = (label_flat == -1).squeeze()
    # mask pixels where either label or any band in image is -1
    image_mask = (image_flat <= 0).any(dim=0)
    mask = mask | image_mask
    image_flat = image_flat[...,~mask]
    label_flat = label_flat[...,~mask]

    # Shuffle the pixels
    perm = np.random.permutation(image_flat.shape[-1])
    image_shuffled = image_flat[...,perm].permute(1,0)
    label_shuffled = label_flat[...,perm].permute(1,0)
    
    return image_shuffled, label_shuffled
class TiffDataset(Dataset):


    def __init__(
        self,
        base_dir: str,
        split: str = 'train',
        transform_funcs: Optional[Callable] = None,
        selected_bands = None,
        image_suffix: str = '.tif*',
        # label_suffix: str = '_label.tif*'
        label_suffix: str = '.tif*'
    ):
        """
        Initializes the dataset by listing image and label file paths.

        Args:
            base_dir (str): The base directory containing the dataset folders.
            split (str, optional): One of 'train', 'val', or 'test'. Defaults to 'train'.
            transform (Callable, optional): Transformations to apply to the images. Defaults to None.
            label_transform (Callable, optional): Transformations to apply to the labels. Defaults to None.
            image_suffix (str, optional): Suffix of image files. Defaults to '.tif'.
            label_suffix (str, optional): Suffix of label files. Defaults to '_label.tif'.
        """
        assert split in ['train', 'val', 'test'], "split must be one of 'train', 'val', or 'test'"
        self.base_dir = Path(base_dir)
        self.split = split
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix

        # Store transform functions
        self.transform_funcs = transform_funcs
        if self.transform_funcs is not None and not isinstance(self.transform_funcs, list):
            self.transform_funcs = [self.transform_funcs]
            
        # Define image and label directories
        self.image_dir = self.base_dir / f"{split}_images"
        self.label_dir = self.base_dir / f"{split}_labels"

        # Verify directories exist
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        # List all image files
        self.image_paths = sorted(self.image_dir.glob(f"*{self.image_suffix}"))
        if not self.image_paths:
            raise FileNotFoundError(f"No image files found in {self.image_dir}")

        # List all label files and ensure they correspond to images
        self.label_paths = []
        for img_path in self.image_paths:
            label_filename = str(img_path.name).replace( self.image_suffix, self.label_suffix)
            if '_img' in label_filename:
                label_filename = label_filename.replace('_img', '_label')
            else:
                label_filename = label_filename.replace('.tiff', '_label.tiff')
            label_path = self.label_dir / label_filename
            if not label_path.exists():
                raise FileNotFoundError(f"Label file not found for image {img_path.name}: {label_path}")
            self.label_paths.append(label_path)

        assert len(self.image_paths) == len(
            self.label_paths
        ), "Number of images and labels must be the same."
        
            # Handle band selection
        self.selected_band_indices = None  # To store indices of selected bands
        if selected_bands is None:
            UNUSED_BAND = "-1"
            
            selected_bands = [UNUSED_BAND, "Blue", "Green", "Red", "NIR_Narrow", "SWIR1", "SWIR2", UNUSED_BAND, UNUSED_BAND]
            # selected_bands = [UNUSED_BAND,"BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2",UNUSED_BAND,UNUSED_BAND,UNUSED_BAND]
            self._process_selected_bands(selected_bands)
        else:
            assert isinstance(selected_bands, list), "selected_bands must be a list of int."
            assert all(isinstance(band, int) for band in selected_bands), "selected_bands must be a list of int."
            self.selected_band_indices = selected_bands
        # if self.split == 'train':
        #     self._filter_range(num_files_per_range=9)
    def _filter_range(self, num_files_per_range: int):
        """
        Filters the dataset to keep only a specified number of files per range.
        
        Args:
            num_files_per_range (int): Maximum number of files to keep for each range
        """
        # Group files by their range
        range_groups = {}
        for img_path, label_path in zip(self.image_paths, self.label_paths):
            # Extract range from filename (e.g., "biomass_0-1_img1.tiff" -> "0-1")
            range_str = img_path.stem.split('_')[1]
            if range_str not in range_groups:
                range_groups[range_str] = []
            range_groups[range_str].append((img_path, label_path))
        
        # Filter each group to keep only num_files_per_range files
        filtered_images = []
        filtered_labels = []
        for range_files in range_groups.values():
            selected_files = range_files[:num_files_per_range]
            images, labels = zip(*selected_files)
            filtered_images.extend(images)
            filtered_labels.extend(labels)
        
        self.image_paths = filtered_images
        self.label_paths = filtered_labels
    def _process_selected_bands(self, selected_bands: List[str]):
        """
        Processes the selected_bands list to determine which band indices to include.

        Args:
            selected_bands (List[str]): List specifying which bands to select.
                                        Use "UNUSED_BAND" to exclude a band.
        """
        if not isinstance(selected_bands, list):
            raise ValueError("selected_bands must be a list of strings.")

        # Find out the number of bands from the first image
        with rasterio.open(self.image_paths[0]) as img:
            num_bands = img.count

        if len(selected_bands) != num_bands:
            if len(selected_bands) < num_bands:
                # Append -1 until lengths match
                selected_bands.extend(["-1"] * (num_bands - len(selected_bands)))
            else:
                raise ValueError(
                    f"Length of selected_bands ({len(selected_bands)}) does not match number of bands in images ({num_bands})."
                )

        # Determine indices of bands to include
        self.selected_band_indices = [
            idx for idx, band in enumerate(selected_bands) if band != "-1"
        ]

        if not self.selected_band_indices:
            raise ValueError("All bands are marked as UNUSED_BAND. At least one band must be used.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the data point.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (image, label)
        """
        # Convert to float32 (for regression) or long (for classification)

        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load image
        try:
            with rasterio.open(image_path) as img:
                image = img.read()  # Shape: (channels, height, width)
            image = image.astype(np.float32)
            image = torch.from_numpy(image)
            
            # Apply mask from the last band
            mask = (image[-1] != 0).float()  # Convert boolean mask to float (1 for valid, 0 for invalid)
            image = torch.where(mask.unsqueeze(0) == 1, image, -1)  # Set masked values to -1
            # select bands
            if self.selected_band_indices is not None:
                image = image[self.selected_band_indices]
        except RasterioIOError as e:
            raise FileNotFoundError(f"Unable to read image file {image_path}: {e}")


        try:
            with rasterio.open(label_path) as lbl:
                label = lbl.read()  # Assuming label is single-channel
            label = label.astype(np.float32)
            label = torch.from_numpy(label)
        except RasterioIOError as e:
            raise FileNotFoundError(f"Unable to read label file {label_path}: {e}")
        
        # Shuffle the pixels
        if self.transform_funcs is not None:
            for t in self.transform_funcs:
                image, label = t(image, label)
        return image, label
