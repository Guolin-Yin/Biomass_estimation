import itertools
import tempfile
from pathlib import Path

import numpy as np
from shapely.geometry import MultiLineString, MultiPolygon, Polygon, box, shape
import scipy.io

from sentinelhub import (
    CRS,
    BBox,
    BBoxSplitter,
    CustomGridSplitter,
    DataCollection,
    MimeType,
    MosaickingOrder,
    OsmSplitter,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions
)
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely import wkt

from .data_collection.utils import (
                                        get_sar_and_optical, 
                                        extract_coordinates, 
                                        plot_bbox_with_annotation, 
                                        plot_multipolygon, 
                                        generate_intervals)

from pyproj import Transformer
def read_csv2geo(path = './data/tiles.csv'):
    
    
    # Load and prepare data
    df = pd.read_csv(path)

    # Creating a MultiPolygon from the geometries
    area_geo = gpd.GeoDataFrame(df['geometry'].apply(wkt.loads), geometry='geometry').unary_union


    # Convert WKT to properly structured GeoJSON format
    polygons = []
    for wkt_str in df['geometry']:
        polygon = wkt.loads(wkt_str)
        exterior_coords = [list(coord[:2]) for coord in polygon.exterior.coords]
        polygons.append([exterior_coords])  # Note the change here, wrapping in one more list


    # Create the final dictionary correctly
    geojson_dict = {
        'type': 'MultiPolygon',
        'coordinates': polygons  # No additional wrapping in a list
    }
    area_shape = shape(geojson_dict)
    return area_shape, area_geo
area_shape, area_geo = read_csv2geo()
# Plot the area
# plot_multipolygon(area_shape)

# Define the resolution and BBox
# area_bbox = BBox(bbox=area_shape.bounds, crs=CRS.WGS84)
bbox = BBox(bbox=area_geo.bounds, crs=CRS.WGS84)

# Calculate the image size based on the resolution and area
resolution = 20

width, height = bbox_to_dimensions(bbox, resolution=resolution) # Calculate the width and height of the bounding box

print('BBox is: ',bbox.__repr__())

# Calculate the number of splits needed for width and height
max_size = 2000  # Maximum size for one dimension
num_x, num_y  = int(np.ceil(width / max_size)), int(np.ceil(height / max_size))

# Use BBoxSplitter to divide the bounding box
grid_splitter = BBoxSplitter([bbox],
                            CRS.WGS84, 
                            ( num_x, num_y  )
                            )
plot_bbox_with_annotation(grid_splitter,area_shape) # Show the grid splitter

# Example usage: From April 1, 2023, to August 31, 2024, for monthly intervals
monthly_intervals = generate_intervals(2023, 4, 1, 2023, 10, 31, 'monthly')

# Example for weekly intervals
# weekly_intervals = generate_intervals(2023, 4, 1, 2023, 10, 31, 'weekly')

# Example for daily intervals
print(len(["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "CLP"]))

bboxs = grid_splitter.get_bbox_list()
for time_interval in monthly_intervals:
    save_path = f"./data/Alldata_{time_interval[0]}_to_{time_interval[1]}.mat"
    
    # check if the file exists
    if Path(save_path).exists():
        print(f"File {save_path} already exists. Skipping...")
        continue
    else:
        print(f"File {save_path} does not exist. Processing...")
    data = {}

    for i, tile_bbox in enumerate(bboxs):
        size = bbox_to_dimensions(tile_bbox, resolution=resolution)
        key = f"tile_{i}"
        print(f"Processing {key}: {tile_bbox.__repr__()}, Size: {size}")
        img = get_sar_and_optical(
                                bounds = tile_bbox, 
                                img_size = size,
                                time_interval = time_interval,
                                )
        print(f"Image shape: {img.shape}")

        if key not in data:
            data[key] = {}
        data[key]['bbox'] = tile_bbox
        data[key]['size'] = size
        data[key]['image'] = img
        # Place your SentinelHubRequest or other processing logic here
    # save data to mat file
    scipy.io.savemat(save_path, data)
