import pandas as pd
from shapely import wkt
import geopandas as gpd
import matplotlib.pyplot as plt
from .img_download import get_img_CLP
# from mpl_toolkits.basemap import Basemap  # Available here: https://github.com/matplotlib/basemap
from matplotlib.patches import Polygon as PltPolygon
from shapely.geometry import MultiLineString, MultiPolygon, Polygon, box, shape
import numpy as np
from sentinelhub import BBox
from sentinelhub import CRS
import datetime

ndvi_eval = """
//VERSION=3

function evaluatePixel(samples) {
    let val = index(samples.B08, samples.B04);
    return [val];
}

function setup() {
  return {
    input: [{
      bands: [
        "B04",
        "B08",
        "dataMask"
      ]
    }],
    output: {
      bands: 1
    }
  }
}
"""
def plot_bbox_with_annotation(grid_splitter, multipolygon):

    fig, ax = plt.subplots(figsize=(10, 10))
    for polygon in multipolygon.geoms:
        x, y = polygon.exterior.xy
        ax.plot(x, y, color='red')
        # Annotate the coordinates of each corner
        for xi, yi in zip(x, y):
            ax.annotate(f"({xi:.3f}, {yi:.3f})", (xi, yi), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='red')
        
        for interior in polygon.interiors:
            ix, iy = interior.xy
            ax.plot(ix, iy, color='blue')  # This should not execute if no interiors exist
    for bbox in grid_splitter.get_bbox_list():
        # Extract corners for plotting
        lon_min, lat_min, lon_max, lat_max = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
        lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
        lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
        ax.plot(lons, lats, color='blue')
        
        # Annotate the bounding box on the plot
        center_x = np.mean([lon_min, lon_max])
        center_y = np.mean([lat_min, lat_max])
        annotation_text = f"({lon_min:.3f}, {lat_min:.3f},\n{lon_max:.3f}, {lat_max:.3f})"
        ax.annotate(annotation_text, xy=(center_x, center_y), color='red',
                    fontsize=8, ha='center', va='center')
    ax.set_title('Grid Split Bounding Boxes')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()
# Function to plot a MultiPolygon
def plot_multipolygon(multipolygon):
    fig, ax = plt.subplots()
    for polygon in multipolygon.geoms:
        x, y = polygon.exterior.xy
        ax.plot(x, y, color='red')
        for interior in polygon.interiors:
            ix, iy = interior.xy
            ax.plot(ix, iy, color='red')  # This should not execute if no interiors exist

    ax.set_title('MultiPolygon Plot')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()
def show_splitter(splitter, alpha=0.2, area_buffer=0.2, show_legend=False):
    area_bbox = splitter.get_area_bbox()
    minx, miny, maxx, maxy = area_bbox
    lng, lat = area_bbox.middle
    w, h = maxx - minx, maxy - miny
    minx = minx - area_buffer * w
    miny = miny - area_buffer * h
    maxx = maxx + area_buffer * w
    maxy = maxy + area_buffer * h

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    base_map = Basemap(
        projection="mill",
        lat_0=lat,
        lon_0=lng,
        llcrnrlon=minx,
        llcrnrlat=miny,
        urcrnrlon=maxx,
        urcrnrlat=maxy,
        resolution="l",
        epsg=4326,
    )
    base_map.drawcoastlines(color=(0, 0, 0, 0))

    area_shape = splitter.get_area_shape()

    if isinstance(area_shape, Polygon):
        polygon_iter = [area_shape]
    elif isinstance(area_shape, MultiPolygon):
        polygon_iter = area_shape.geoms
    else:
        raise ValueError(f"Geometry of type {type(area_shape)} is not supported")

    for polygon in polygon_iter:
        if isinstance(polygon.boundary, MultiLineString):
            for linestring in polygon.boundary:
                ax.add_patch(PltPolygon(np.array(linestring), closed=True, facecolor=(0, 0, 0, 0), edgecolor="red"))
        else:
            ax.add_patch(
                PltPolygon(np.array(polygon.boundary.coords), closed=True, facecolor=(0, 0, 0, 0), edgecolor="red")
            )

    bbox_list = splitter.get_bbox_list()
    info_list = splitter.get_info_list()

    cm = plt.get_cmap("jet", len(bbox_list))
    legend_shapes = []
    for i, bbox in enumerate(bbox_list):
        wgs84_bbox = bbox.transform(CRS.WGS84).get_polygon()

        tile_color = tuple(list(cm(i))[:3] + [alpha])
        ax.add_patch(PltPolygon(np.array(wgs84_bbox), closed=True, facecolor=tile_color, edgecolor="green"))

        if show_legend:
            legend_shapes.append(plt.Rectangle((0, 0), 1, 1, fc=cm(i)))

    if show_legend:
        legend_names = []
        for info in info_list:
            legend_name = "{},{}".format(info["index_x"], info["index_y"])

            for prop in ["grid_index", "tile"]:
                if prop in info:
                    legend_name = "{},{}".format(info[prop], legend_name)

            legend_names.append(legend_name)

        plt.legend(legend_shapes, legend_names)
    plt.tight_layout()
    plt.show()
def extract_coordinates(gdf):
    coords_list = []

    for geom in gdf.geometry:
        if geom.geom_type == 'Point':
            # For Points
            coords_list.append((geom.x, geom.y))
        elif geom.geom_type in ['LineString', 'LinearRing']:
            # For LineStrings
            coords_list.append(list(geom.coords))
        elif geom.geom_type == 'Polygon':
            # For Polygons
            exterior_coords = list(geom.exterior.coords)
            interior_coords = [list(ring.coords) for ring in geom.interiors]
            coords_list.append({'exterior': exterior_coords, 'interiors': interior_coords})
        elif geom.geom_type == 'MultiPoint':
            # For MultiPoints
            coords_list.append([(point.x, point.y) for point in geom.geoms])
        elif geom.geom_type == 'MultiLineString':
            # For MultiLineStrings
            coords_list.append([list(line.coords) for line in geom.geoms])
        elif geom.geom_type == 'MultiPolygon':
            # For MultiPolygons
            coords_list.append([
                {'exterior': list(poly.exterior.coords), 
                 'interiors': [list(ring.coords) for ring in poly.interiors]}
                for poly in geom.geoms
            ])
    
    return coords_list


def generate_intervals(start_year, start_month, start_day, end_year, end_month, end_day, interval_type='weekly'):
    # Starting and ending dates
    start_date = datetime.date(start_year, start_month, start_day)
    end_date = datetime.date(end_year, end_month, end_day)
    
    # List to hold all intervals
    intervals = []
    
    # Current start of the interval
    current_start = start_date
    
    while current_start <= end_date:
        if interval_type == 'weekly':
            # Calculate the end of the current week
            current_end = current_start + datetime.timedelta(days=6)
        elif interval_type == 'monthly':
            # Calculate the end of the current month
            next_month = current_start.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
            current_end = next_month - datetime.timedelta(days=next_month.day)
        
        # If the end of the interval goes beyond the end date, adjust it
        if current_end > end_date:
            current_end = end_date
        
        # Append the interval (start and end) to the list
        intervals.append((current_start.strftime('%Y-%m-%d'), current_end.strftime('%Y-%m-%d')))
        
        # Move to the next interval start
        current_start = current_end + datetime.timedelta(days=1)
    
    return intervals


def get_sar_and_optical(bounds, img_size, time_interval):
    if isinstance(bounds, BBox):
        bbox = bounds
    else:
        bbox = BBox(bounds, crs=CRS.WGS84)
    opt_img = get_img_CLP(
        bbox, img_size, time_interval
    )
    return np.transpose(opt_img, (2, 0, 1))
