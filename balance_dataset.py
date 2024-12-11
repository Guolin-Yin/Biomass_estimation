import os
import numpy as np
import rasterio
from tqdm import tqdm
from pathlib import Path
for mode in [ "test", "train", "val",]:
    # Directory containing training images and labels
    train_images_dir = f"Running_Dataset/V2/raw/{mode}_images"
    train_labels_dir = f"Running_Dataset/V2/raw/{mode}_labels"
    # Directory to save the balanced dataset
    balanced_dataset_dir = f"Running_Dataset/V2/balanced/{mode}_images"
    balanced_labels_dir = f"Running_Dataset/V2/balanced/{mode}_labels"

    # Create the balanced dataset directories if they don't exist
    os.makedirs(balanced_dataset_dir, exist_ok=True)
    os.makedirs(balanced_labels_dir, exist_ok=True)

    # Dictionary to store pixels for each biomass range
    # Format: {range_key: [(pixel_value, image_file, x, y, original_biomass), ...]}
    biomass_range_pixels = {
        "0-1": [], "1-2": [], "2-3": [], "3-4": [], "4-5": [], "5-6": []
    }

    def get_range_key(biomass_value):
        if biomass_value > 6:  # Skip outliers
            return None
        return f"{int(biomass_value)}-{int(biomass_value)+1}"

    # First pass: collect all pixels and their locations
    print("Collecting pixels for each biomass range...")
    for filename in tqdm(os.listdir(train_labels_dir)):
        if filename.endswith(".tiff"):
            label_filepath = os.path.join(train_labels_dir, filename)
            image_filepath = os.path.join(train_images_dir, filename.replace("_label", ""))
            
            with rasterio.open(label_filepath) as label_src, rasterio.open(image_filepath) as image_src:
                label_data = label_src.read(1)
                image_data = image_src.read()
                
                # Get coordinates of non-zero pixels
                rows, cols = np.nonzero(label_data)
                for row, col in zip(rows, cols):
                    biomass_value = label_data[row, col]
                    if biomass_value <= 0:  # Skip invalid values
                        continue
                        
                    range_key = get_range_key(biomass_value)
                    if range_key is None:
                        continue
                        
                    pixel_values = image_data[:, row, col]
                    biomass_range_pixels[range_key].append(
                        (pixel_values, image_filepath, row, col, biomass_value)
                    )

    # Set target pixels to match the 5-6 range (approximately 14000)
    target_pixels = 14000

    print(f"\nPixels per biomass range before balancing:")
    for range_key, pixels in biomass_range_pixels.items():
        print(f"Range {range_key}: {len(pixels)} pixels")
    print(f"Target pixels per range: {target_pixels}")

    # Create new balanced images and labels
    print("\nCreating balanced dataset...")
    
    # Specify the number of valid pixels you want per image
    valid_pixels_per_image = 10000  # You can adjust this number
    
    # Calculate how many pixels we need from each range per image
    num_ranges = len(biomass_range_pixels)
    pixels_per_range_per_image = valid_pixels_per_image // num_ranges
    
    # Create new balanced images
    output_shape = (224, 224)
    total_pixels_available = min(len(pixels) for pixels in biomass_range_pixels.values())
    # Calculate number of images we can create with the available pixels
    num_images = total_pixels_available // pixels_per_range_per_image
    
    print(f"\nCreating {num_images} balanced images with {valid_pixels_per_image} valid pixels each...")
    for image_idx in tqdm(range(num_images)):
        new_label = np.full(output_shape, -1, dtype=np.float32)
        new_image = np.full((image_data.shape[0],) + output_shape, -1, dtype=np.float32)
        
        current_position = 0
        # Fill arrays with selected pixels from each range
        for range_key, pixels in biomass_range_pixels.items():
            indices = np.arange(len(pixels))
            selected_indices = np.random.choice(
                indices, 
                pixels_per_range_per_image, 
                replace=False  # Changed to False to avoid duplicates within an image
            )
            
            for idx, selected_idx in enumerate(selected_indices):
                pixel_values, _, _, _, original_biomass = pixels[selected_idx]
                if pixel_values.sum() == 0:
                    continue
                    
                new_row = current_position // output_shape[1]
                new_col = current_position % output_shape[1]
                new_label[new_row, new_col] = original_biomass
                new_image[:, new_row, new_col] = pixel_values
                current_position += 1
        
        # Save new image and label
        output_filename = f"balanced_data_part_{image_idx+1}.tiff"
        label_output_path = os.path.join(balanced_labels_dir, f"{Path(output_filename).stem}_label.tiff")
        image_output_path = os.path.join(balanced_dataset_dir, output_filename)
        
        # Save label and image using existing rasterio code
        with rasterio.open(label_output_path, 'w', driver='GTiff',
                        height=output_shape[0], width=output_shape[1],
                        count=1, dtype=new_label.dtype) as dst:
            dst.write(new_label, 1)
        
        with rasterio.open(image_output_path, 'w', driver='GTiff',
                        height=output_shape[0], width=output_shape[1],
                        count=image_data.shape[0], dtype=new_image.dtype) as dst:
            dst.write(new_image)

    print("\nBalanced dataset created successfully!")

# Create a balanced dataset
# for value, files in biomass_dict.items():
#     # Randomly sample from the files to match the minimum number of samples
#     sampled_files = random.sample(files, min_samples)
#     for file in sampled_files:
#         shutil.copy(file, balanced_dataset_dir)

# print(f"Balanced dataset created with {min_samples} samples for each biomass value.")

# import numpy as np
# import rasterio
# import os
# import matplotlib.pyplot as plt
# from collections import Counter

# # Directory containing training labels
# train_labels_dir = "Dataset_splited/test_labels"

# # List to store biomass values
# biomass_values = []

# # Read all label files and collect biomass values
# for filename in os.listdir(train_labels_dir):
#     if filename.endswith(".tiff"):
#         filepath = os.path.join(train_labels_dir, filename)
#         # print(filepath)
#         with rasterio.open(filepath) as src:
#             data = src.read(1)  # Read first band
#             # Collect non-zero biomass values
#             valid_biomass = data[data > 0]
#             biomass_values.extend(valid_biomass.flatten())

# biomass_values = np.array(biomass_values)

# # Calculate distribution statistics
# mean_biomass = np.mean(biomass_values)
# median_biomass = np.median(biomass_values)
# std_biomass = np.std(biomass_values)

# # Count occurrences of each biomass value
# biomass_counts = Counter(biomass_values)

# # Plot histogram of biomass distribution
# plt.figure(figsize=(10, 6))
# plt.hist(biomass_values, bins=50, edgecolor='black')
# plt.title('Biomass Distribution in Training Labels')
# plt.xlabel('Biomass Value')
# plt.ylabel('Frequency')
# plt.axvline(mean_biomass, color='red', linestyle='dashed', label=f'Mean: {mean_biomass:.2f}')
# plt.axvline(median_biomass, color='green', linestyle='dashed', label=f'Median: {median_biomass:.2f}')
# plt.legend()
# plt.grid(True, alpha=0.3)

# print(f"Biomass Distribution Statistics:")
# print(f"Mean: {mean_biomass:.2f}")
# print(f"Median: {median_biomass:.2f}")
# print(f"Standard Deviation: {std_biomass:.2f}")
# print(f"Min: {np.min(biomass_values):.2f}")
# print(f"Max: {np.max(biomass_values):.2f}")


# import torch
# from src.tools.dataset import TiffDataset, shuffle_pixel, batch_process
# from src.tools.trainer import Trainer
# from src.tools.train_utils import load_checkpoint_cfg, EarlyStopping
# from src.model.TFCNN import TFCNN
# from src.utils.config import model_params
# import argparse
# import numpy as np
# def norm(image, label):
#     return image, label
# processor_fns = [shuffle_pixel, norm]
# data_dir = './Dataset_splited'
# train_ds = TiffDataset(data_dir, split='train')
# val_ds = TiffDataset(data_dir, split='val')
# test_ds = TiffDataset(data_dir, split='test')
# train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=batch_process(*processor_fns))
# val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, collate_fn=batch_process(*processor_fns))
# test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, collate_fn=batch_process(*processor_fns))

# biomass_values = []
# for image, label in train_loader:
#     biomass_values.extend(label.flatten())

# biomass_values = np.array(biomass_values)
# max_biomass = np.max(biomass_values)
# min_biomass = np.min(biomass_values)
# print(f"Max biomass: {max_biomass}, Min biomass: {min_biomass}")