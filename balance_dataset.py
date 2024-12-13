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
        # "0-1": [], "1-2": [], "2-3": [], "3-4": [], "4-5": [], "5-6": []
    }

    def get_range_key(biomass_value):
        if biomass_value < 10:  # Skip outliers
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
                    if range_key not in biomass_range_pixels.keys():
                        biomass_range_pixels[range_key] = []
                    biomass_range_pixels[range_key].append(
                        (pixel_values, image_filepath, row, col, biomass_value)
                    )



    ## Modified saving section
    print("\nSaving images for each biomass range...")
    output_shape = (64, 64)

    # Get metadata from the first image/label pair for reference
    first_label = next(iter(os.listdir(train_labels_dir)))
    first_image = first_label.replace("_label", "")
    with rasterio.open(os.path.join(train_labels_dir, first_label)) as label_src, \
        rasterio.open(os.path.join(train_images_dir, first_image)) as image_src:
        label_meta = label_src.meta.copy()
        image_meta = image_src.meta.copy()
        
        # Update the metadata for new dimensions
        label_meta.update({
            'height': output_shape[0],
            'width': output_shape[1]
        })
        image_meta.update({
            'height': output_shape[0],
            'width': output_shape[1]
        })

    for range_key, pixels in biomass_range_pixels.items():
        print(f"\nProcessing range {range_key}...")
        
        total_pixels = len(pixels)
        pixels_per_image = output_shape[0] * output_shape[1]
        num_images = (total_pixels + pixels_per_image - 1) // pixels_per_image
        
        for image_idx in tqdm(range(num_images)):
            new_label = np.full(output_shape, -1, dtype=np.float32)
            new_image = np.full((image_data.shape[0],) + output_shape, -1, dtype=np.float32)
            
            start_idx = image_idx * pixels_per_image
            end_idx = min(start_idx + pixels_per_image, total_pixels)
            selected_pixels = pixels[start_idx:end_idx]
            
            for idx, (pixel_values, _, _, _, original_biomass) in enumerate(selected_pixels):
                if pixel_values.sum() == 0:
                    continue
                    
                new_row = idx // output_shape[1]
                new_col = idx % output_shape[1]
                new_label[new_row, new_col] = original_biomass
                new_image[:, new_row, new_col] = pixel_values
            
            output_filename = f"biomass_{range_key}_{image_idx+1}th_img.tif"
            label_output_path = os.path.join(balanced_labels_dir, f"{Path(output_filename).stem.replace('_img', '_label.tif')}")
            image_output_path = os.path.join(balanced_dataset_dir, output_filename)
            
            # Save with original metadata
            with rasterio.open(label_output_path, 'w', **label_meta) as dst:
                dst.write(new_label, 1)
            
            with rasterio.open(image_output_path, 'w', **image_meta) as dst:
                dst.write(new_image)

    print("\nBalanced dataset created successfully!")