import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.errors import RasterioIOError

import numpy as np

class batch_process:
    def __init__(self, *args):
        self.process_fns = args
    def __call__(self, batch):
        for fn in self.process_fns:
            batch = [fn(image, label) for image, label in batch]
        # unzip the batch
        images, labels = tuple(zip(*batch))
        # convert to tensor
        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        return images, labels
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
        image_suffix: str = '.tiff',
        label_suffix: str = '_label.tiff'
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
