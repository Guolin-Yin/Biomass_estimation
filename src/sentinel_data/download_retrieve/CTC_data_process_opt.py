import random
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import pandas as pd
from scipy.io import loadmat
import datetime
def get_all_data_paths(data_dir):
    all_dataset = {}
    all_data_paths = list(data_dir.rglob("*.mat"))
    for data_path in all_data_paths:
        label = data_path.parent.name
        if label not in all_dataset:
            all_dataset[label] = []
        all_dataset[label].append(data_path)
    return all_dataset
def split_data(all_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("Sum of ratios must be equal to 1")

    # Initialize dictionaries for train, val, and test sets
    train_set = {}
    val_set = {}
    test_set = {}

    for label, paths in all_dataset.items():
        # Shuffle paths to ensure randomness
        random.shuffle(paths)
        
        # Calculate split indices
        n_total = len(paths)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Create splits
        train_paths = paths[:n_train]
        val_paths = paths[n_train:n_train + n_val]
        test_paths = paths[n_train + n_val:]

        # Store paths by label in the respective dictionaries
        train_set[label] = train_paths
        val_set[label] = val_paths
        test_set[label] = test_paths

    return train_set, val_set, test_set

mat_header = ['__header__', '__version__', '__globals__']
data_dir = Path('data/polygon_data')

all_dataset = get_all_data_paths(data_dir)

train_set, val_set, test_set = split_data(all_dataset)

splited_dataset = {'train': train_set, 'val': val_set, 'test': test_set}

for mode, dataset in splited_dataset.items():
    print(f"{mode} set:")
    # labels = list(dataset.keys())
    # train_img = []
    # train_label = []
    # for l in labels:
    #     path = dataset[l]
    #     for p in path:
    #         img = loadmat(p)
    #         train_img.append(img)
    #         l = np.repeat(l, img['data'].shape[0])
    #         train_label.append(l)
    
    
    for label, paths in dataset.items():
        print(f"Label: {label}, Number of samples: {len(paths)}")
        data_save_folder = data_dir.parent / Path('dataset') / mode / label
        # delete the folder if it already exists
        # if data_save_folder.exists():
        #     for file in data_save_folder.glob("*"):
        #         file.unlink()
        #     data_save_folder.rmdir()
        # data_save_folder.mkdir(parents=True, exist_ok=True)

        for path in paths:
            data = loadmat(path)
            data_arrays = [data[key] for key in data.keys() if key not in mat_header]
            stacked_data = np.stack(data_arrays, axis=-1) # shape: (n_pixels, n_bands, n_dates)
            stacked_data = stacked_data[:,:-2,:]
            stacked_data = np.transpose(stacked_data, (0,2,1)) # shape: (n_pixels, n_dates, n_bands)
            
            dates = [key for key in data.keys() if key not in mat_header]
            dates_dt = pd.to_datetime(dates)

            # Calculate the day of the year for each date
            days_of_year = np.array([(date - datetime.datetime(date.year, 1, 1)).days + 1 for date in dates_dt])/366

            # key is date across multiple months, value is the corresponding data
            # Save data to npy file

            assert stacked_data.reshape(-1).shape[0] > 0, "No data"
            np.save(
                (data_save_folder / (path.stem + "_data")).with_suffix('.npy'), stacked_data
                )
            np.save(
                (data_save_folder / (path.stem + "_time")).with_suffix('.npy'), days_of_year
                )
