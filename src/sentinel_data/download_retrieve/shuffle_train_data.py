import numpy as np
from pathlib import Path
import random
from collections import defaultdict
from utils.settings.config import LINEAR_ENCODER

# Define paths and parameters
data_dir = Path('./data/dataset/train')
output_dir = Path('./data//dataset/train_shuffled_pixels')
output_dir.mkdir(exist_ok=True, parents=True)


# Filter data function to control max samples per class
def filter_data(data_paths, max_samples_per_class):
    label_dict = defaultdict(list)
    for path in data_paths:
        label = path.parent.name
        label_dict[label].append(path)
    
    # Limit the number of samples per label
    for label, paths in label_dict.items():
        if len(paths) > max_samples_per_class:
            label_dict[label] = random.sample(paths, max_samples_per_class)
    
    # Flatten the dictionary back into a list
    filtered_paths = [path for paths in label_dict.values() for path in paths]
    return filtered_paths

# Collect all data paths
data_paths = list(data_dir.glob('**/*_data.npy'))

# Apply filtering to control the number of samples per class
max_samples_per_class = 2000  # Example value, adjust as needed
filtered_paths = filter_data(data_paths, max_samples_per_class)

# Collect pixels and their corresponding labels
pixel_pool = []

for path in filtered_paths:
    label = path.parent.name
    class_id = LINEAR_ENCODER[label]
    data = np.load(path)  # Assuming data is of shape (75, 15, 12)
    time_path = (path.parent / path.stem.replace('data', 'time')).with_suffix('.npy')
    time_data = np.load(time_path)  # Load time data corresponding to the pixel data

    # Combine pixel data with its corresponding label and time data
    for pixel_idx in range(data.shape[0]):
        pixel_data = data[pixel_idx]  # Shape: (15, 12)
        # time_pixel_data = time_data[pixel_idx]  # Corresponding time data for the pixel
        pixel_pool.append((pixel_data, time_data, class_id))

# Shuffle the entire pool of pixels
random.shuffle(pixel_pool)

# Split the shuffled pool into files with a random number of pixels (35 to 600)
current_index = 0
file_count = 0

while current_index < len(pixel_pool):
    # Randomly determine the number of pixels for this file
    num_pixels = random.randint(35, 600)
    
    # Ensure we do not exceed the available pixels
    end_index = min(current_index + num_pixels, len(pixel_pool))
    
    # Extract the subset of pixels
    subset = pixel_pool[current_index:end_index]
    current_index = end_index

    # Save the subset to a new file
    pixel_data = np.array([item[0] for item in subset])  # Shape: (num_pixels, 15, 12)
    time_data = np.array([item[1] for item in subset])   # Corresponding time data
    pixel_labels = np.array([item[2] for item in subset])  # Shape: (num_pixels,)

    # Save as numpy files
    np.save(output_dir / f'pixels_{file_count}_data.npy', pixel_data)
    np.save(output_dir / f'pixels_{file_count}_time.npy', time_data)
    np.save(output_dir / f'pixels_{file_count}_labels.npy', pixel_labels)
    file_count += 1
    # print the progress evert 100 files, indicate the number of pixel saved
    if end_index % 1000000 == 0:
        print(f'Saved {file_count} files with {end_index} / {len(pixel_pool)} pixels.')

print(f'Saved {file_count} files with shuffled pixels.')