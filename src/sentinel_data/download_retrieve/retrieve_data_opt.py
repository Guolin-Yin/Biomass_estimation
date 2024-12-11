import os
import numpy as np
from shapely.geometry import shape, box
import scipy.io
import json
import csv
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from datetime import datetime
import concurrent.futures
import logging
from pathlib import Path
from tqdm import tqdm
import warnings
# 1. 解析 JSON 文件中的 Polygon 数据
def load_polygons_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)

    polygons = []
    for feature in data['features']:
        geometry = shape(feature['geometry'])
        properties = feature['properties']
        sample_name = f"{feature['id']}_{properties['VARIEDADE']}"
        polygons.append({'geometry': geometry, 'SampleName': sample_name, 'properties': properties})

    return polygons

# 2. 从 .mat 文件中按需加载 tile 信息以及波段表头
# 处理瓦片的 size 数据，提取实际的 height 和 width
def load_tiles_and_band_names_from_mat(path):
    data = scipy.io.loadmat(path)
    tiles = {}
    band_names = []

    for key in data.keys():
        if key.startswith('tile_'):
            try:
                min_x, max_x, min_y, max_y = data[key][0][0][0][0][0]
                size = data[key][0][0][1]

                # 处理嵌套的二维数组，提取实际的 (height, width)
                height, width = size[0][0], size[0][1]

                img = np.transpose(data[key][0][0][2], (0, 2, 1))  # 将图像数据转置为 (bands, height, width)

                # 检查 img 是否是三维数据
                if img.ndim != 3:
                    print(f"Warning: Unexpected image dimensions in {key}: {img.shape}")
                    continue

                tiles[key] = {
                    'bbox': (min_x, max_x, min_y, max_y),
                    'size': (height, width),
                    'img': img
                }
            except IndexError as e:
                print(f"Error processing tile {key} in {path}: {e}")
                continue

    if 'band_names' in data:
        band_names = data['band_names'].flatten().tolist()
    else:
        band_names = [f'Band_{i+1}' for i in range(14)]

    return tiles, band_names


# 4. 栅格化多边形生成掩码
def polygon_to_mask(polygon, tile_bbox, tile_size):
    min_x, max_x, min_y, max_y = tile_bbox

    height, width = tile_size[0], tile_size[1]

    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

    try:
        mask = geometry_mask([polygon], transform=transform, invert=True, out_shape=(height, width))
        if np.any(mask):
            return mask
        else:
            return None
        # if not np.any(mask):
        #     # Compute the intersection area
        #     tile_poly = box(min_x, min_y, max_x, max_y)
        #     intersection = polygon.intersection(tile_poly)
            
            # If the intersection area is significant (you can adjust this threshold)
            # if intersection.area > 1e-8:  # Adjust this threshold as needed
            #     # Force at least one pixel to be True
            #     center_x = int(width / 2)
            #     center_y = int(height / 2)
            #     mask[center_y, center_x] = True
            #     print(f"Forced central pixel to True due to small intersection")
        # else:
        #     print(f"Intersection too small, returning None")
        #     return None
    except Exception as e:
        print(f"Error creating mask: {e}")
        return None

    # return mask

# 5. 预计算样本的瓦片相交关系与掩码
def precompute_tile_masks_for_sample(polygon, tiles):
    tile_masks = {}

    for tile_name, tile_info in tiles.items():
        min_x, max_x, min_y, max_y = tile_info['bbox']
        if polygon.intersects(box(min_x, min_y, max_x, max_y)):
            mask = polygon_to_mask(polygon, tile_info['bbox'], tile_info['size'])
            if mask:
                tile_masks[tile_name] = mask
    # check if tile_masks is empty
    if len(tile_masks) == 0:
        return None
    return tile_masks


def apply_mask_and_extract(tile_info, mask):
    img = tile_info['img']
    height, width = tile_info['size'][0], tile_info['size'][1]
    lon_grid = np.linspace(tile_info['bbox'][0], tile_info['bbox'][1], width)
    lat_grid = np.linspace(tile_info['bbox'][3], tile_info['bbox'][2], height)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    return lon_mesh[mask], lat_mesh[mask], img[:, mask].T


if __name__ == '__main__':

    json_path = '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Shape_labels/filtered_output_data_geo.json'
    polygons  = load_polygons_from_json(json_path)

    # mat_folder = Path('/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023')
    # mat_paths = list(mat_folder.glob('*.mat'))
    # tiles, _ = load_tiles_and_band_names_from_mat(mat_paths[0])

    valid_path_2023 = ['/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-04-01_to_2023-04-07.mat',
                        '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-04-22_to_2023-04-28.mat',
                        '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-04-29_to_2023-05-05.mat',
                        '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-05-13_to_2023-05-19.mat',
                        '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-06-03_to_2023-06-09.mat',
                        '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-06-17_to_2023-06-23.mat',
                        '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-06-24_to_2023-06-30.mat',
                        '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-07-01_to_2023-07-07.mat',
                        '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-07-08_to_2023-07-14.mat',
                        '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-07-15_to_2023-07-21.mat',
                        '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-07-29_to_2023-08-04.mat',
                        '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-08-05_to_2023-08-11.mat',
                        '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-08-26_to_2023-09-01.mat',
                        '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-09-23_to_2023-09-29.mat',
                        '/home/glyin/Projects/Crop_Classification/Sen4AgriNet/S4A-Models/data/Raw_data-2023/Alldata_2023-11-04_to_2023-11-10.mat']
    all_data = {}
    for path in valid_path_2023:
        path = Path(path)
        tiles, _ = load_tiles_and_band_names_from_mat(path)
        file_name = path.stem
        
        start_time = file_name.split('_to_')[0].split('_')[-1]
        end_time = file_name.split('_to_')[1].split('_')[-1]

        all_data[start_time] = tiles
        # mask_dir = Path('data/tile_masks')
        # mask_paths = list(mask_dir.rglob('*.mat'))
        # print(tiles)
        
    label_counter = {}
    bar = tqdm(polygons, desc='Processing polygons')
    for polygon in bar:
        
        label = polygon['SampleName'].split('_')[-1]
        poly = polygon['geometry']

        if label not in label_counter:
            label_counter[label] = 1
        else:
            label_counter[label] += 1

        label_dir = Path('./data/polygon_data') / label
        label_dir.mkdir(parents=True, exist_ok=True)

        data_path = (label_dir / str(label_counter[label])).with_suffix('.mat')


        anchor_img = all_data[list(all_data.keys())[0]]
        tile_mask = precompute_tile_masks_for_sample(poly, anchor_img)
        if tile_mask is None:
            continue
        data_to_save = {}
        for date, tiles in all_data.items():
            current_date = {date: []}
            for keys in list(tile_mask.keys()):
                tile_info = tiles[keys]
                mask      = tile_mask[keys]
                lon_values, lat_values, band_values = apply_mask_and_extract(tile_info, mask)
                current_date[date].append(band_values)
            # check if np.concatenate(current_date[date]) is empty
            if len(np.concatenate(current_date[date])) == 0:
                print(f"Empty data for {data_path}")
                continue
            data_to_save.update(
                { 
                    date:np.concatenate(current_date[date])
                 }
                )
        bar.set_description(f'tiles: {keys}, No of tiles: {len(list(tile_mask.keys()))}, mask: {mask.shape}, band_values: {band_values.shape}')
        bar.update(1)
        scipy.io.savemat(data_path, data_to_save)