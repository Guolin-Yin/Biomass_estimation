import random
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import pandas as pd
def interpolate_along_axis(data, axis=1):
    # Interpolate along the specified axis for each slice in the other dimensions
    shape = data.shape
    # Iterate over the first and third dimension, interpolating along the second
    for i in range(shape[0]):
        for j in range(shape[2]):
            # Extract the line to interpolate
            line = data[i, :, j]
            # Find indices where data is non-zero
            if np.any(line):  # Check if there's at least some non-zero data to interpolate
                indices = np.arange(len(line))
                good_data = line != 0
                # Create an interpolation function based on non-zero data
                f = interp1d(indices[good_data], line[good_data], kind='linear', bounds_error=False, fill_value="extrapolate")
                # Interpolate missing data
                data[i, :, j] = f(indices)
            # Optionally handle cases where the entire line is zero by setting a default or carrying forward last known good data

    return data
def extract_interesting_data(df, bands):
    # Extract the starting date from the SourceFile column
    df['StartDate'] = df['SourceFile'].apply(lambda x: x.split('_to_')[0])
    
    # Convert StartDate to datetime to extract the week of the year
    df['WeekOfYear'] = pd.to_datetime(df['StartDate']).dt.isocalendar().week
    
    # Calculate week of the year divided by 52 to get the fraction of the year
    df['WeekFraction'] = df['WeekOfYear'] / 52
    
    df['MonthOfYear'] = pd.to_datetime(df['StartDate']).dt.month
    df['MonthOfYear'] = df['MonthOfYear'] / 12
    # Filter the dataframe to include only the specified bands and the new WeekFraction column
    columns_of_interest = bands + ['WeekFraction'] + ['MonthOfYear'] + ['StartDate']
    filtered_df = df[columns_of_interest]
    
    return filtered_df

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
def parse_filename(file_path):
    parts = file_path.stem.split('_')
    sample_indx = int(parts[0])
    label = parts[1]
    return sample_indx, label

def sum_dataset(dir_data):
    dir_data = Path(dir_data)  # Ensure dir_data is a Path object
    filespath = dir_data.glob('*.csv')
    all_dataset = {}

    for path in filespath:
        _, label= parse_filename(path)
        if label in all_dataset:
            all_dataset[label].append(path)
        else:
            all_dataset[label] = []
    return all_dataset
def remove_zero_rows(df, band_columns ):
    # Specify the band columns to check for zeros
    
    
    # Check if all values in these columns are zero
    mask = (df[band_columns] == 0).all(axis=1)
    
    # Filter out rows where all band values are zero
    filtered_df = df[~mask]
    
    return filtered_df
def compute_monthly_median(df):
    # Create a copy of the dataframe to avoid modifying the original inadvertently
    df_copy = df.copy()

    # Convert the 'StartDate' column to datetime
    df_copy['StartDate'] = pd.to_datetime(df_copy['StartDate'])

    # Set 'StartDate' as the index of the dataframe
    df_copy.set_index('StartDate', inplace=True)
    
    # Resample data by month and calculate the median
    monthly_median = df_copy.resample('M').median()

    return monthly_median
def compute_median_of_each_month(filtered_df, interested_bands):
    allmonth_data = {}
    all_month = []
    for start_date, group in filtered_df.groupby('StartDate'):

        year = pd.to_datetime(start_date).year
        month = pd.to_datetime(start_date).month
        # print(start_date)
        # print(year)
        # print(month)
        # print(group[interested_bands].values.shape)
        key = f'{year}_{month}_{month/12}'
        data = group[interested_bands].values
        if key in allmonth_data:
            allmonth_data[key].append(data)
        else:
            allmonth_data[key] = [data]
    for key, data in allmonth_data.items():
        data = np.stack(data)
        median = np.median(data, axis=0)
        #  replace the value in the key with the median value
        allmonth_data[key] = median
        all_month.append(int(key.split('_')[1])/12)
    return allmonth_data, np.array(all_month)
save_dir = Path('data/ctc_dataset_interpolated')

source_dir = Path('data/processed_merged')
all_dataset = sum_dataset(source_dir)
train_set, val_set, test_set = split_data(all_dataset)

splited_dataset = {'train': train_set, 'val': val_set, 'test': test_set}
for mode in ['train', 'val', 'test']:
    data_paths_dict = splited_dataset[mode]
    all_labels = list(data_paths_dict.keys())
    bar = tqdm(all_labels, desc=f'Processing {mode} data')
    for label in bar:
        data_paths = data_paths_dict[label]
        for i, path in enumerate(data_paths):

            save_path_data = save_dir / mode / label / f'data_{i}'
            save_path_time = save_dir / mode / label / f'time_{i}'
            save_path_data.parent.mkdir(parents=True, exist_ok=True)
            save_path_time.parent.mkdir(parents=True, exist_ok=True)

            if save_path_data.with_suffix('.npy').exists() and save_path_time.with_suffix('.npy').exists():
                continue
            source_df = pd.read_csv(path)
            if source_df.empty:
                continue
            try:
                interested_bands = [f'Band_{i}' for i in range(1, 13)]

                filtered_df = extract_interesting_data(source_df, interested_bands)

                # filtered_df = remove_zero_rows(filtered_df, interested_bands)
                filtered_df, time       = compute_median_of_each_month(filtered_df, interested_bands)
                data = np.stack(list(filtered_df.values()),axis = 1)
                data = interpolate_along_axis(data, axis=1)
                assert data.shape[1:] == (16,12), f"Data for {label} is not complete, as it has shape of {data.shape}"

                # Save the data to .npy files
                np.save(save_path_data.with_suffix('.npy'), data)
                np.save(save_path_time.with_suffix('.npy'), time)
                bar.set_postfix_str(f'Processed {i} data')
            except Exception as e:
                print(f"Error processing {path}")
                print(e)
                continue
