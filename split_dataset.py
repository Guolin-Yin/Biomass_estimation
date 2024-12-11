import os
from pathlib import Path
import shutil
import random

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
    random.seed(random_seed)
    random.shuffle(image_files)
    
    # Calculate split indices
    n_samples = len(image_files)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # Split files
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
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
            
            # Copy image and label to respective directories
            shutil.copy2(img_path, output_path / f"{split}_images" / img_path.name)
            shutil.copy2(label_path, output_path / f"{split}_labels" / label_path.name)

if __name__ == "__main__":
    # Example usage
    split_dataset(
        data_dir="Dataset/V2/data",
        label_dir="Dataset/V2/labels",
        output_dir="Running_Dataset/V2/raw",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        image_suffix='.tiff',
        label_suffix='_label.tiff'
    )