import os
from pathlib import Path
import shutil
import random

import rasterio
import numpy as np
def split_dataset(
    data_dir: str,
    label_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    image_suffix: str = '_img.tif',
    label_suffix: str = '_label.tif',
    random_seed: int = 42
):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Create output directories
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_path / f"{split}_images").mkdir(parents=True, exist_ok=True)
        (output_path / f"{split}_labels").mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(Path(data_dir).glob(f"*{image_suffix}"))
    filtered_image_files = []

    # Filter images with negative values
    for img_path in image_files:
        with rasterio.open(img_path) as src:
            image_data = src.read()

            # if np.all(image_data[np.array([1,2,3,4,5,6])] > 0):
            if image_data[np.array([1,2,3,4,5,6])].min() > 0:
                filtered_image_files.append(img_path)
            else:
                print(f"Skipping {img_path.name} - contains negative values")

    random.seed(random_seed)
    random.shuffle(filtered_image_files)
    
    # Calculate split indices
    n_samples = len(filtered_image_files)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # Split files
    splits = {
        'train': filtered_image_files[:train_end],
        'val': filtered_image_files[train_end:val_end],
        'test': filtered_image_files[val_end:]
    }
    
    # Copy files to respective directories
    for split, files in splits.items():
        print(f"Processing {split} split: {len(files)} files")
        for img_path in files:
            # Get corresponding label file
            label_filename = img_path.name.replace(image_suffix, label_suffix)
            label_path = Path(label_dir) / label_filename
            
            if not label_path.exists():
                print(f"Warning: Label file not found for {img_path.name}")
                continue
            
            # Read and process image
            with rasterio.open(img_path) as src:
                image_data = src.read()
                c,h,w = image_data.shape
                if image_data[np.array([1,2,3,4,5,6])].min() == 0:
                    print('Skipping', img_path.name)
                    continue
                if h < 224 - 5 or w < 224 - 5:
                    print('Skipping', img_path.name)
                    continue
            
            # Copy label file
            shutil.copy2(img_path, output_path / f"{split}_images" / img_path.name)
            shutil.copy2(label_path, output_path / f"{split}_labels" / label_path.name)


if __name__ == "__main__":
    # Example usage
    split_dataset(
        data_dir="Dataset/V2/data",
        label_dir="Dataset/V2/labels",
        output_dir="Running_Dataset/V2/raw_224",
        train_ratio=0.9,
        val_ratio=0.05,
        test_ratio=0.05,
        image_suffix='.tiff',
        label_suffix='_label.tiff'
    )