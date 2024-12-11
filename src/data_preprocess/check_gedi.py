
import requests
import affine
import datetime as dt
import pandas as pd
import geopandas as gpd
import rioxarray
import rasterio as rio
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
import h5py, tabulate
import contextily as ctx
import numpy as np
import pyproj
from getpass import getpass
from IPython.display import HTML, display
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import orient
from pyproj import Transformer
import os
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from rasterio.transform import from_bounds

def create_gedi_raster(bound, csv_files, hls_image, output_file="GEDI_1.tif", if_plot=False):
    """
    Creates a raster from GEDI data points matching HLS image properties.
    
    Args:
        bound (tuple): Geographical bounds (min_lon, min_lat, max_lon, max_lat)
        output_folder (str): Path to folder containing GEDI CSV files
        hls_image (str): Path to HLS reference image
        output_file (str): Output GeoTIFF filename
    """
    # Optional: Plot data points
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
    # Extract bounds
    min_lon, min_lat, max_lon, max_lat = bound

    # Read and combine CSV files
    l4adf = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    # Filter GEDI data
    filtered = l4adf[
        (l4adf['lat_lowestmode'] >= min_lat) & 
        (l4adf['lat_lowestmode'] <= max_lat) & 
        (l4adf['lon_lowestmode'] >= min_lon) & 
        (l4adf['lon_lowestmode'] <= max_lon) & 
        (l4adf['l4_quality_flag'] > 0) & 
        (l4adf['agbd_se']/l4adf['agbd'] * 100 > 50)
    ]


    # Create raster
    noDataValue = -1
    
    # Read HLS image properties
    with rio.open(hls_image) as src:
        crs = "EPSG:4326"
        width = src.width
        height = src.height
        transform = from_bounds(bound[0], bound[1], bound[2], bound[3], width, height)
        
    # Initialize and populate raster
    agbd_raster = np.full((height, width), noDataValue, dtype=np.float32)
    
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
    if if_plot and filtered.shape[0] > 0:
        plot_gedi_points(filtered)
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

if __name__ == "__main__":
    from tqdm import tqdm
    # Load polygons
    polygons = load_polygons('species.csv')
    csv_folder = Path('./hf_data')
    # csv_files = list(csv_folder.glob('**/*.csv'))[:1]
    # l4adf = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    # gedi_points = gpd.GeoDataFrame(
    #     l4adf,
    #     geometry=gpd.points_from_xy(l4adf.lon_lowestmode, l4adf.lat_lowestmode),
    #     crs="EPSG:4326"
    # )
    # gedi_points = gedi_points[gedi_points.agbd > 0]
    # Initialize a dictionary to store results for each polygon
    def check_gedi(polygons_folder, gedi_points):
        polygons = load_polygons(polygons_folder)
        polygon_results = {}
        for idx, polygon in polygons.iterrows():
            points_in_poly = gedi_points[gedi_points.geometry.within(polygon.geometry)]
            if not points_in_poly.empty:
                # polygon_info = pd.DataFrame({
                #     'polygon_index': [idx],
                #     'polygon_geometry': [polygon.geometry.wkt],
                #     'csv_file': [csv_file.name],
                # })
                # csv_name = f'polygon_info.csv'
                # # Append to CSV if exists, create new if not
                # if os.path.exists(csv_name):
                #     polygon_info.to_csv(csv_name, mode='a', header=False, index=False)
                # else:
                #     polygon_info.to_csv(csv_name, index=False)
                
                # points_found = []
                return True
        return False
    
    csv_files = list(csv_folder.glob('**/*.csv'))
    bar = tqdm(csv_files, total=len(csv_files))

    all_points = 0
    for csv_file in bar:
        l4adf = pd.read_csv(csv_file)
        gedi_points = gpd.GeoDataFrame(
            l4adf,
            geometry=gpd.points_from_xy(l4adf.lon_lowestmode, l4adf.lat_lowestmode),
            crs="EPSG:4326"
        )
        gedi_points = gedi_points[gedi_points.agbd > 0]
        for idx, polygon in polygons.iterrows():
            points_in_poly = gedi_points[gedi_points.geometry.within(polygon.geometry)]
            
            # If points are found, save to CSV and add to results
            if not points_in_poly.empty:
                # Check for overlapping points with previously found points
                # Save polygon info to CSV with overlap count
                polygon_info = pd.DataFrame({
                    'polygon_index': [idx],
                    'polygon_geometry': [polygon.geometry.wkt],
                    'csv_file': [csv_file.name],
                })
                csv_name = f'polygon_info.csv'
                # Append to CSV if exists, create new if not
                if os.path.exists(csv_name):
                    polygon_info.to_csv(csv_name, mode='a', header=False, index=False)
                else:
                    polygon_info.to_csv(csv_name, index=False)
                
                points_found.append((csv_file.name, points_in_poly))
                
                all_points += len(points_in_poly)
            # print(f"  Found {len(points_in_poly)} points in {csv_file.name}")
                bar.set_postfix(polygon=f"Polygon {idx}", found=len(points_found), all_points=all_points)
        # Store results for this polygon
    if points_found:
        polygon_results[idx] = points_found
        print(f"Total files with points for Polygon {idx}: {len(points_found)}")
    else:
        print(f"No GEDI points found for Polygon {idx}")

    # Print summary of results
    print("\nSummary of Results:")
    for polygon_idx, file_points in polygon_results.items():
        print(f"\nPolygon {polygon_idx}:")
        total_points = 0
        for file_name, points in file_points:
            total_points += len(points)
            print(f"  {file_name}: {len(points)} points")
            # Print some example AGBD values
            print(f"  Example AGBD values: {points['agbd'].head().tolist()}")
        print(f"  Total points: {total_points}")