# %%

from pathlib import Path
import datetime as dt
import numpy as np
from shapely.geometry import MultiLineString, MultiPolygon, Polygon, box, shape
import scipy.io
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely import wkt
from sentinelhub import (
    CRS,
    BBox,
    bbox_to_dimensions
)
import rasterio
from src.sentinel_data.download_retrieve.data_collection.utils import (
                                        get_sar_and_optical, 
                                        generate_intervals)
import geopandas as gpd
import pandas as pd
import pandas as pd
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import box
import re
from tqdm import tqdm

def plot_gedi_points(filtered):
    latitudes = filtered['lat_lowestmode'].values
    longitudes = filtered['lon_lowestmode'].values
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes, latitudes, color='blue', marker='x')
    plt.gca().set_facecolor('white')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GEDI Data Point Locations')
    plt.show()
def create_gedi_raster(csv_files, hls_image, output_file, if_plot=False):
    """Create a raster of GEDI measurements matching the HLS image extent"""
    
    # Read HLS image properties first
    with rio.open(hls_image) as src:
        crs = src.crs
        width = src.width
        height = src.height
        transform = src.transform
        bound = src.bounds  # Get bounds from the HLS image
    if not isinstance(csv_files, list):
        csv_files = [csv_files]
    # Read and combine CSV files
    l4adf = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    # Filter GEDI data to HLS image extent
    filtered = l4adf[
        (l4adf['lat_lowestmode'] >= bound.bottom) & 
        (l4adf['lat_lowestmode'] <= bound.top) & 
        (l4adf['lon_lowestmode'] >= bound.left) & 
        (l4adf['lon_lowestmode'] <= bound.right) & 
        (l4adf['l4_quality_flag'] > 0) & 
        (l4adf['agbd_se']/l4adf['agbd'] * 100 > 50)
    ]

    # Create raster
    noDataValue = -1
    agbd_raster = np.full((height, width), noDataValue, dtype=np.float32)
    
    # Populate raster
    for _, row in filtered.iterrows():
        x, y = row['lon_lowestmode'], row['lat_lowestmode']
        agbd = row['agbd']
        row_idx, col_idx = ~transform * (x, y)
        row_idx, col_idx = int(row_idx), int(col_idx)
        
        if 0 <= row_idx < height and 0 <= col_idx < width:
            if agbd_raster[row_idx, col_idx] == noDataValue:
                agbd_raster[row_idx, col_idx] = agbd
            else:
                agbd_raster[row_idx, col_idx] = (agbd_raster[row_idx, col_idx] + agbd) / 2

    # Save raster
    with rio.open(
        output_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=agbd_raster.dtype,
        crs=crs,
        transform=transform,
        nodata=noDataValue
    ) as dst:
        dst.write(agbd_raster, 1)
        
    if if_plot and len(filtered) > 0:
        plt.figure(figsize=(10, 6))
        
        # Plot HLS image bounds
        bound_box = box(bound.left, bound.bottom, bound.right, bound.top)
        x, y = bound_box.exterior.xy
        plt.plot(x, y, 'r--', linewidth=2, label='HLS Image Bounds')
        
        # Plot GEDI points
        scatter = plt.scatter(filtered['lon_lowestmode'], filtered['lat_lowestmode'], 
                            c=filtered['agbd'], cmap='viridis', label='GEDI Points')
        plt.colorbar(scatter, label='AGBD')
        
        plt.title('GEDI Points and HLS Image Extent')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    return agbd_raster
def load_polygons(csv_file):
    """
    Load polygon data from CSV file containing WKT geometry strings.
    
    Args:
        csv_file (str): Path to CSV file containing polygon data
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the polygon data
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Convert WKT strings to geometry objects
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.GeoSeries.from_wkt(df['geometry']),
        crs="EPSG:4326"  # Assuming coordinates are in WGS84
    )
    
    return gdf
def get_bboxes_from_polygons(gdf):
    """
    Convert polygons to bounding boxes in the format required for satellite data download.
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing polygon geometries
        
    Returns:
        list: List of bounding boxes in format [(min_lon, min_lat, max_lon, max_lat), ...]
    """
    bboxes = []
    for _, polygon in gdf.iterrows():
        bounds = polygon.geometry.bounds  # Returns (minx, miny, maxx, maxy)
        bbox = (bounds[0], bounds[1], bounds[2], bounds[3])  # (min_lon, min_lat, max_lon, max_lat)
        bboxes.append(bbox)
    return bboxes
def save_multiband_geotiff(img_data, bbox, time_interval, save_path, f_name):
    """
    Save multi-band satellite image as a single GeoTIFF file
    
    Args:
        img_data: numpy array of shape (bands, height, width)
        bbox: tuple of (min_lon, min_lat, max_lon, max_lat)
        time_interval: tuple of (start_date, end_date)
        save_path: Path object for save directory
    """
    try:
        # Create save directory if it doesn't exist
        # save_dir = save_path / '_to_'.join(time_interval)
        # save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare filename
        output_file = save_path / f"{f_name}.tiff"
        
        # Create transform for georeferencing
        transform = rasterio.transform.from_bounds(
            bbox[0], bbox[1], bbox[2], bbox[3],  # west, south, east, north
            width=img_data.shape[2],
            height=img_data.shape[1]
        )
        
        # Save all bands to a single GeoTIFF
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=img_data.shape[1],
            width=img_data.shape[2],
            count=img_data.shape[0],  # Number of bands
            dtype=img_data.dtype,
            crs='+proj=longlat +datum=WGS84 +no_defs',
            transform=transform,
            compress='lzw',  # Add compression
        ) as dst:
            # Write all bands
            dst.write(img_data)
            
            # Add band descriptions (modify according to your bands)
            band_descriptions = [
                'CoastalAerosol', 'Blue', 'Green', 'Red', 
                'NIR_Narrow', 'SWIR1', 'SWIR2', 'Cirrus', 'dataMask'
            ]
            for i, desc in enumerate(band_descriptions, 1):
                dst.set_band_description(i, desc)
            
        print(f"Saved multi-band GeoTIFF to: {output_file}")
        
    except Exception as e:
        print(f"Error saving GeoTIFF: {str(e)}")
        raise
def process_satellite_data(bound, width, height, time_interval, save_dir, f_name):
    """
    Process and save satellite data for given time interval
    """
    try:
        print(f'Processing {time_interval[0]} to {time_interval[1]}')
        
        # Get satellite data
        img = get_sar_and_optical(
            bounds=bound,
            img_size=(width, height),
            time_interval=time_interval
        )
        # print(f"Image shape: {img.shape}")
        

        if isinstance(time_interval[0], dt.datetime):
            time_interval = (time_interval[0].strftime('%Y-%m-%d'), 
                            time_interval[1].strftime('%Y-%m-%d'))
        # Save as GeoTIFF
        save_multiband_geotiff(img, bound, time_interval, save_dir, f_name)
        
    except Exception as e:
        print(f"Error processing time interval {time_interval}: {str(e)}")
        raise
def extract_index_and_date(csv_file_name):
    # name format: sample_0-2022_04_08_to_2022_04_14
    info = csv_file_name.stem.split('-')
    index = int(info[0].split('_')[1])
    date = info[1].split('_to_') # [date 1, date 2]
    # convert to datetime
    date_1 = dt.datetime.strptime(date[0], '%Y_%m_%d')
    date_2 = dt.datetime.strptime(date[1], '%Y_%m_%d')
    return index, date_1, date_2
def extract_gedi_date(filename):
    """
    Extract date from GEDI filename.
    
    Args:
        filename (str): GEDI filename (e.g., 'GEDI04_A_2020094133316_O07413_02_T05062_02_002_02_V002.h5')
        
    Returns:
        datetime.datetime: The extracted date
        
    Example:
        >>> extract_gedi_date('GEDI04_A_2020094133316_O07413_02_T05062_02_002_02_V002.h5')
        datetime.datetime(2020, 4, 3, 13, 33, 16)
    """
    # Split the filename and get the datetime part (e.g., '2020094133316')
    filename = filename.stem
    datetime_str = filename.split('_')[2]
    
    # Extract components
    year = int(datetime_str[:4])           # 2020
    day_of_year = int(datetime_str[4:7])   # 094
    hour = int(datetime_str[7:9])          # 13
    minute = int(datetime_str[9:11])       # 33
    second = int(datetime_str[11:13])      # 16
    
    # Convert to datetime
    date = dt.datetime(year, 1, 1) + dt.timedelta(days=day_of_year-1, 
                                                 hours=hour,
                                                 minutes=minute,
                                                 seconds=second)
    
    return date
def split_bounds_into_patches(bounds, resolution, patch_size=224):
    """
    Split large bounds into smaller patches of specified size.
    
    Args:
        bounds (tuple): (min_lon, min_lat, max_lon, max_lat)
        resolution (int): Resolution in meters per pixel
        patch_size (int): Desired size of each patch in pixels
        
    Returns:
        list: List of (bounds, width, height) for each patch
    """
    bbox = BBox(bbox=bounds, crs=CRS.WGS84)
    total_width, total_height = bbox_to_dimensions(bbox, resolution=resolution)
    
    # Calculate the size of each patch in coordinate units
    lon_range = bounds[2] - bounds[0]
    lat_range = bounds[3] - bounds[1]
    
    lon_per_pixel = lon_range / total_width
    lat_per_pixel = lat_range / total_height
    
    lon_per_patch = lon_per_pixel * patch_size
    lat_per_patch = lat_per_pixel * patch_size
    
    patches = []
    
    # Generate patches
    for i in range(0, total_height, patch_size):
        for j in range(0, total_width, patch_size):
            min_lon = bounds[0] + (j * lon_per_pixel)
            min_lat = bounds[1] + (i * lat_per_pixel)
            max_lon = min(bounds[0] + ((j + patch_size) * lon_per_pixel), bounds[2])
            max_lat = min(bounds[1] + ((i + patch_size) * lat_per_pixel), bounds[3])
            
            patch_bounds = (min_lon, min_lat, max_lon, max_lat)
            patch_bbox = BBox(bbox=patch_bounds, crs=CRS.WGS84)
            patch_width, patch_height = bbox_to_dimensions(patch_bbox, resolution=resolution)
            
            patches.append((patch_bounds, patch_width, patch_height))
    
    return patches
def generate_patches_from_gedi_points(gedi_points, resolution=30, patch_size=224, if_plot=False, csv_file_name=None):

    # Get the overall bounds of all GEDI points
    bound = (gedi_points.lon_lowestmode.min(), 
            gedi_points.lat_lowestmode.min(),
            gedi_points.lon_lowestmode.max(), 
            gedi_points.lat_lowestmode.max())
    

    # Generate all patches
    all_patches = split_bounds_into_patches(bound, resolution, patch_size)
    
    # Filter patches that contain points
    patches_with_points = []
    patch_point_counts = []
    for patch_bounds, width, height in all_patches:
        # Check if any points fall within this patch
        points_in_patch = gedi_points[
            (gedi_points.lon_lowestmode >= patch_bounds[0]) &
            (gedi_points.lon_lowestmode <= patch_bounds[2]) &
            (gedi_points.lat_lowestmode >= patch_bounds[1]) &
            (gedi_points.lat_lowestmode <= patch_bounds[3])
        ]
        
        if not points_in_patch.empty:
            patches_with_points.append((patch_bounds, width, height))
            patch_point_counts.append(len(points_in_patch))  # Store the count

    # For visualization (optional)
    def plot_patches(gedi_points, patches, num_points):
        plt.figure(figsize=(12, 8))
        
        # Plot GEDI points
        plt.scatter(gedi_points.lon_lowestmode, gedi_points.lat_lowestmode, 
                   c='lightgray', s=10, label='GEDI Points')
        
        # Plot each patch
        for i, (patch_bounds, _, _) in enumerate(patches):
            patch_box = box(*patch_bounds)
            plt.plot(*patch_box.exterior.xy, 'r-', linewidth=0.5, alpha=0.5)
            
            # Get points in this patch for coloring
            points_in_patch = gedi_points[
                (gedi_points.lon_lowestmode >= patch_bounds[0]) &
                (gedi_points.lon_lowestmode <= patch_bounds[2]) &
                (gedi_points.lat_lowestmode >= patch_bounds[1]) &
                (gedi_points.lat_lowestmode <= patch_bounds[3])
            ]
            
            # Plot points in this patch with different color
            plt.scatter(points_in_patch.lon_lowestmode, points_in_patch.lat_lowestmode,
                       c='red', s=20, alpha=0.6)
            
            # Add patch number
            center_x = (patch_bounds[0] + patch_bounds[2]) / 2
            center_y = (patch_bounds[1] + patch_bounds[3]) / 2
            plt.text(center_x, center_y, str(i), fontsize=8, ha='center', va='center')
            
            # Optionally print number of points in each patch
            print(f"Patch {i}: {len(points_in_patch)} points")
        
        # Plot overall bounds
        bound_box = box(*bound)
        save_path = Path('./plots')
        save_path.mkdir(parents=True, exist_ok=True)
        plt.plot(*bound_box.exterior.xy, 'b--', linewidth=2, label='Overall Bounds')
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'GEDI Points with {patch_size}x{patch_size} Patches, num_points: {num_points}\n(Only showing patches containing points)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(save_path / f'{csv_file_name.stem}_patches.png')
        plt.show()
    
    # Plot the patches (comment out if not needed)
    if if_plot:
        plot_patches(gedi_points, patches_with_points, np.sum(patch_point_counts))
        # print(f"Patch contains {len(points_in_patch)} points")
        # print(f"Found {len(patches_with_points)} patches containing GEDI points out of {len(all_patches)} total patches")
    
    
    return patches_with_points, np.sum(patch_point_counts)
# Use the functions
if __name__ == "__main__":

# %%
    filtered_bounds = (-8.499590926570734, 38.77199531533364, -6.395956468456142, 40.13990015731868)
    resolution = 30
    csv_folder = Path('./csv_labels')
    img_dir = Path('./Dataset/V2/data')
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_files = list(csv_folder.glob('**/*.csv'))
    counter = 0
    # for csv_file in csv_files:
    #     date = extract_gedi_date(csv_file)

    #     # Example usage:
    #     l4adf = pd.read_csv(csv_file)

    #     # Filter GEDI points
    #     gedi_points = l4adf[
    #         (l4adf['agbd'] > 0) &
    #         (l4adf['lon_lowestmode'] >= filtered_bounds[0]) &
    #         (l4adf['lon_lowestmode'] <= filtered_bounds[2]) &
    #         (l4adf['lat_lowestmode'] >= filtered_bounds[1]) &
    #         (l4adf['lat_lowestmode'] <= filtered_bounds[3]) 
    #         # (l4adf['l4_quality_flag'] > 0) 
    #         # (l4adf['agbd_se']/l4adf['agbd'] * 100 > 50)
    #     ]
    #     # Generate dates for 3 days before and 3 days after
    #     date_start = date - dt.timedelta(days=3)
    #     date_end = date + dt.timedelta(days=3)
        
    #     if len(gedi_points) == 0:
    #         continue
    #     # Get patches instead of full bounds
    #     try:
    #         patches, patch_point_counts = generate_patches_from_gedi_points(gedi_points, if_plot=True, csv_file_name=csv_file)
    #     except Exception as e:
    #         print(f"Error generating patches for {csv_file.stem}: {str(e)}")
    #         continue
        
    #     for patch_idx, (patch_bounds, patch_width, patch_height) in enumerate(patches):
    #         f_name = f"{csv_file.stem}_patch_{patch_idx}"
            
    #         # check files exist
    #         if (img_dir / f"{f_name}.tiff").exists():
    #             continue
    #         try:
    #             process_satellite_data(
    #                 patch_bounds, 
    #                 patch_width, 
    #                 patch_height, 
    #                 (date_start, date_end), 
    #                 img_dir,
    #                 f_name
    #             )
    #             counter += patch_point_counts
    #             print(f"Processed patch {patch_idx} for {csv_file.stem}, num patch points: {patch_point_counts}, total points: {counter}")
    #         except Exception as e:
    #             print(f"Error processing patch {patch_idx}: {str(e)}")
    #             continue


    for i, hls_image in enumerate(img_dir.glob('**/*.tiff')):
        # Extract base name by removing patch number and extension
        base_name = re.sub(r'_patch_\d+\.tiff$', '', str(hls_image))
        # csv_files = Path(base_name.replace('data', 'csv_labels')).with_suffix('.csv')
        csv_files = (Path('./csv_labels') / Path(base_name).stem).with_suffix('.csv')
        output_file = Path('./Dataset/V2/labels') / f"{hls_image.stem}_label.tiff"
        if output_file.exists():
            continue
        output_file.parent.mkdir(parents=True, exist_ok=True)
        create_gedi_raster(csv_files, hls_image, output_file=output_file, if_plot=False)
        
        print(f"Processed {i+1} of {len(list(img_dir.glob('**/*.tiff')))}")