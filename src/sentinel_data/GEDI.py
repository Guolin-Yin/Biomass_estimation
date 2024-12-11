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
import h5py
import tabulate
import contextily as ctx
import numpy as np
import pyproj
from getpass import getpass
from IPython.display import HTML, display
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import orient
from pyproj import Transformer
import os
import subprocess
import matplotlib.pyplot as plt
import csv
import glob
import json
import earthaccess
import requests
import datetime as dt
import pandas as pd
import shutil
from shapely.geometry import MultiPolygon, Polygon, box, Point
import os
# Write the credentials to the .netrc file
# Get user's home directory
home_dir = os.path.expanduser('~')
netrc_path = os.path.join(home_dir, '.netrc')

# Write the credentials to the .netrc file
with open(netrc_path, 'w') as f:
    f.write("machine urs.earthdata.nasa.gov login jasonip password EarthMining_101\n")

# Set file permissions to secure the credentials using os module instead of shell command
os.chmod(netrc_path, 0o600)

doi = '10.3334/ORNLDAAC/2056'# GEDI L4A DOI
doiS2 = '10.5067/HLS/HLSS30.002' # S2 DOI

# CMR API base url
cmrurl='https://cmr.earthdata.nasa.gov/search/'

doisearch = cmrurl + 'collections.json?doi=' + doi
response = requests.get(doisearch)
response.raise_for_status()
concept_id = response.json()['feed']['entry'][0]['id']

print(concept_id)
def check_points_in_polygon(output_file, target_area_polygon, granule_url, granule_size, granule_poly):
    """
    Check if H5 file contains points within target polygon and return results
    """
    with h5py.File(output_file, 'r') as hf:
        actual_points = []
        for beam in [k for k in hf.keys() if k.startswith('BEAM')]:
            beam_data = hf[beam]
            lats = beam_data['lat_lowestmode'][:]
            lons = beam_data['lon_lowestmode'][:]
        
            for lat, lon in zip(lats, lons):
                if not np.isnan(lat) and not np.isnan(lon):
                    actual_points.append(Point(lon, lat))
        
        actual_gdf = gpd.GeoDataFrame(geometry=actual_points, crs="EPSG:4326")
        points_in_poly = actual_gdf[actual_gdf.geometry.within(target_area_polygon)]
        
        # Keep or remove based on points check
        if not points_in_poly.empty:
            return True, [granule_url, granule_size, granule_poly]
        else:
            os.remove(output_file)  # Remove if no relevant points
            return False, None

def search_gedi_granules_poly_2(concept_id, target_area_polygon, start_date, end_date, cmrurl='https://cmr.earthdata.nasa.gov/search/', data_amount=1):
    # Calculate the bounding box from the polygon for initial broad search
    min_lon, min_lat, max_lon, max_lat = target_area_polygon.bounds
    bound = (min_lon, min_lat, max_lon, max_lat)
    
    # Format dates and bounding box for CMR API
    dt_format = '%Y-%m-%dT%H:%M:%SZ'
    temporal_str = start_date.strftime(dt_format) + ',' + end_date.strftime(dt_format)
    bound_str = ','.join(map(str, bound))

    page_num = 1
    page_size = 2000
    granule_arr = []

    output_dir = 'gedi_files'
    os.makedirs(output_dir, exist_ok=True)
    
    granule_arr = []
    downloaded_files = []
    while True:
        # CMR API request
        cmr_param = {
            "collection_concept_id": concept_id,
            "page_size": page_size,
            "page_num": page_num,
            "temporal": temporal_str,
            "bounding_box[]": bound_str
        }

        granulesearch = cmrurl + 'granules.json'
        response = requests.get(granulesearch, params=cmr_param)
        response.raise_for_status()
        granules = response.json()['feed']['entry']
        
        if granules:
            for g in granules:
                granule_url = ''
                granule_size = float(g['granule_size'])

                # Step 1: Check footprint intersection
                if 'polygons' in g:
                    polygons = g['polygons']
                    multipolygons = []
                    for poly in polygons:
                        i = iter(poly[0].split(" "))
                        ltln = list(map(" ".join, zip(i, i)))
                        multipolygons.append(Polygon([[float(p.split(" ")[1]), float(p.split(" ")[0])] for p in ltln]))
                    granule_poly = MultiPolygon(multipolygons)
                    
                    try:
                        # Only proceed if footprint intersects
                        if granule_poly.intersects(target_area_polygon):
                            # Get download URL and filename
                            for links in g['links']:
                                if 'title' in links and links['title'].startswith('Download') and links['title'].endswith('.h5'):
                                    granule_url = links['href']
                                    filename = os.path.basename(granule_url)
                                    output_file = os.path.join(output_dir, filename)
                            # Check if file already exists
                                    if os.path.exists(output_file):
                                        print(f"File {filename} already exists, skipping download")
                                        continue
                            # Download file with proper name
                                    print(f"Downloading {filename}")
                                    response = requests.get(granule_url, allow_redirects=True)
                                    with open(output_file, 'wb') as f:
                                        f.write(response.content)
                                    granule_data = [granule_url, granule_size, granule_poly]              

                                    granule_arr.append(granule_data)
                                    downloaded_files.append(output_file)
                    
                    except Exception as e:
                        print(f"Error processing granule {g['id']}: {e}")
                        if 'output_file' in locals() and os.path.exists(output_file):
                            os.remove(output_file)
                        continue

            page_num += 1
        else:
            break

    # Create DataFrame
    l4adf = pd.DataFrame(granule_arr, columns=["granule_url", "granule_size", "granule_poly"])
    
    # Drop duplicates and empty geometries
    l4adf = l4adf[l4adf['granule_poly'] != '']
    l4a_granules = l4adf.drop_duplicates(subset=['granule_url'])
    
    # Save URLs for download
    l4a_granules.to_csv('granules.txt', columns=['granule_url'], index=False, header=False, mode='a')
    # Save the full DataFrame with all columns, appending to existing file
    l4adf.to_csv('gedi_granules.csv', index=False, mode='a', header=False)
    return l4adf, downloaded_files
def search_gedi_granules_poly(concept_id, target_area_polygon, start_date, end_date, cmrurl='https://cmr.earthdata.nasa.gov/search/', data_amount=1):
    # Calculate the bounding box from the polygon for initial broad search
    min_lon, min_lat, max_lon, max_lat = target_area_polygon.bounds
    bound = (min_lon, min_lat, max_lon, max_lat)
    # Format dates and bounding box for CMR API
    dt_format = '%Y-%m-%dT%H:%M:%SZ'
    temporal_str = start_date.strftime(dt_format) + ',' + end_date.strftime(dt_format)
    bound_str = ','.join(map(str, bound))

    page_num = 1
    page_size = 2000
    granule_arr = []

    while True:
        # ... existing CMR API request code ...
        cmr_param = {
            "collection_concept_id": concept_id,
            "page_size": page_size,
            "page_num": page_num,
            "temporal": temporal_str,
            "bounding_box[]": bound_str
        }

        granulesearch = cmrurl + 'granules.json'
        response = requests.get(granulesearch, params=cmr_param)
        response.raise_for_status()
        granules = response.json()['feed']['entry']
        if granules:
            for g in granules:
                granule_url = ''
                granule_poly = ''
                granule_size = float(g['granule_size'])

                # reading bounding geometries
                if 'polygons' in g:
                    polygons = g['polygons']
                    multipolygons = []
                    for poly in polygons:
                        i = iter(poly[0].split(" "))
                        ltln = list(map(" ".join, zip(i, i)))
                        multipolygons.append(Polygon([[float(p.split(" ")[1]), float(p.split(" ")[0])] for p in ltln]))
                    granule_poly = MultiPolygon(multipolygons)
                    try:
                        if granule_poly.intersects(target_area_polygon):
                            for links in g['links']:
                                if 'title' in links and links['title'].startswith('Download') \
                                    and links['title'].endswith('.h5'):
                                    granule_url = links['href']
                            
                            temp_filename = 'temp_gedi.h5'
                            response = requests.get(granule_url, allow_redirects=True)
                            with open(temp_filename, 'wb') as f:
                                f.write(response.content)
                            # Read actual measurement points from HDF5
                            with h5py.File(temp_filename, 'r') as hf:
                                actual_points = []
                                for beam in [k for k in hf.keys() if k.startswith('BEAM')]:
                                    beam_data = hf[beam]
                                    lats = beam_data['lat_lowestmode'][:]
                                    lons = beam_data['lon_lowestmode'][:]
                                
                                # Create points for each measurement
                                    for lat, lon in zip(lats, lons):
                                        if not np.isnan(lat) and not np.isnan(lon):
                                            actual_points.append(Point(lon, lat))
                                # Create GeoDataFrame of actual points
                                actual_gdf = gpd.GeoDataFrame(geometry=actual_points, crs="EPSG:4326")
                            points_in_poly = actual_gdf[actual_gdf.geometry.within(target_area_polygon)]
                            if not points_in_poly.empty:
                                granule_arr.append([granule_url, granule_size, granule_poly])
                        os.remove(temp_filename)
                    except Exception as e:
                        print(f"Error with granule {g['id']}: {e}")
                        continue

            page_num += 1
        else:
            break
    # Create DataFrame without adding bounding box
    l4adf = pd.DataFrame(granule_arr, columns=["granule_url", "granule_size", "granule_poly"])

    # Drop granules with empty geometry
    l4adf = l4adf[l4adf['granule_poly'] != '']

    # drop duplicate URLs if any
    l4a_granules = l4adf.drop_duplicates(subset=['granule_url'])
    l4a_granules.to_csv('granules.txt', columns=['granule_url'], index=False, header=False)

    # Download the files
    command = f"""
    head -n {data_amount} granules.txt | tr -d '\\r' | xargs -n 1 -I {{}} bash -c 'filename=$(basename {{}}); curl -LJO -n -c ~/.urs_cookies -b ~/.urs_cookies -o "gedi_folder/$filename" {{}}; echo $filename'
    """

    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    downloaded_files = result.stdout.splitlines()

    # print(f"Found {len(l4adf)} granules intersecting with the polygon")
    # print(f"Total file size (MB): {l4adf['granule_size'].sum()}")

    return l4adf, downloaded_files
def to_csv(downloaded_files, output_folder , file_name = None):
    hfList = []

    # read the L4A files

    hfList.append(h5py.File(downloaded_files, 'r'))

    # printing root-level groups
    list(hfList[0].keys())

    # read the METADATA group
    metadata = hfList[0]['METADATA/DatasetIdentification']
    # store attributes and descriptions in an array
    data = []
    for attr in metadata.attrs.keys():
        data.append([attr, metadata.attrs[attr]])

    # display `data` array as a table
    tbl_n = 1 # table number
    # print(f'Table {tbl_n}. Attributes and discription from `METADATA` group')
    headers = ["attribute", "description"]
    # display(HTML(tabulate.tabulate(data, headers, tablefmt='html')))

    # read the ANCILLARY group
    ancillary = []

    for hf in hfList:
        ancillary.append(hf['ANCILLARY'])

    # read model_data subgroup
    model_data = []
    for data in ancillary:
        model_data.append(data['model_data'])

    # initialize an empty dataframe
    model_data_df = pd.DataFrame()

    first_model = model_data[0]
    # loop through parameters
    for v in first_model.dtype.names:
        # exclude multidimensional variables
        if (len(first_model[v].shape) == 1):
            # copy parameters as dataframe column
            model_data_df[v] = first_model[v]
            # converting object datatype to string
            if model_data_df[v].dtype.kind=='O':
                model_data_df[v] = model_data_df[v].str.decode('utf-8')

    # print the parameters
    tbl_n += 1
    # read pft_lut subgroup
    pft_lut = ancillary[0]['pft_lut']
    headers = pft_lut.dtype.names
    # print pft class and names
    data = zip(pft_lut[headers[0]], pft_lut[headers[1]])
    # display(HTML(tabulate.tabulate(data, headers, tablefmt='html')))
    # read region_lut subgroup
    region_lut = ancillary[0]['region_lut']
    headers = region_lut.dtype.names
    # print region class and names
    data = zip(region_lut[headers[0]], region_lut[headers[1]])
    # display(HTML(tabulate.tabulate(data, headers, tablefmt='html')))
    # index of DBT_NAm predict_stratum, idx = 6
    idx = model_data_df[model_data_df['predict_stratum']=='DBT_NAm'].index.item()
    # print vcov matrix
    model_data[0]['vcov'][idx]
    ## get predictor_id, rh_index and par for idx = 6
    predictor_id = model_data[0]['predictor_id'][idx]
    rh_index = model_data[0]['rh_index'][idx]
    par = model_data[0]['par'][idx]

    # print
    print_s = f"""predictor_id: {predictor_id}
    rh_index: {rh_index}
    par: {par}"""
    # print(print_s)

    # initialize arrays
    stratum_arr, modelname_arr, fitstratum_arr, agbd_arr = [], [], [], []
    # loop the model_data_df dataframe
    for idx, row in model_data_df.iterrows():
        stratum_arr.append(model_data_df['predict_stratum'][idx])
        modelname_arr.append(model_data_df['model_name'][idx])
        fitstratum_arr.append(model_data_df['fit_stratum'][idx])
        i_0 = 0
        predictor_id = model_data[0]['predictor_id'][idx]
        rh_index = model_data[0]['rh_index'][idx]
        par = model_data[0]['par'][idx]
        model_str = 'AGBD = ' + str(par[0]) # intercept
        for i in predictor_id[predictor_id>0]:
            # use product of two RH metrics when consecutive
            # predictor_id have same values
            if (i == i_0):
                model_str += ' x RH_' + str(rh_index[i-1])
            # adding slope coefficients
            else:
                model_str += ' + ' + str(par[i]) + ' x RH_' + str(rh_index[i-1])
            i_0 = i
        # agbd model
        agbd_arr.append(model_str)

    # unique agbd models
    unique_models = list(set(agbd_arr))

    # printing agbd models by predict_stratum
    data=[]
    for model in unique_models:
        s, m, f = [], [], []
        for i, x in enumerate(agbd_arr):
            if x == model:
                s.append(stratum_arr[i])
                m.append(modelname_arr[i])
                f.append(fitstratum_arr[i])
        data.append([", ".join(s), ", ".join(list(set(m))), ", ".join(list(set(f))), model])
    tbl_n += 1
    # print(f'Table {tbl_n}. AGBD Linear Models by Prediction Stratum')
    headers = ["predict_stratum", "model_name", "fit_stratum", "AGBD model"]
    # display(HTML(tabulate.tabulate(data, headers, tablefmt='html', stralign="left")))

    data = []
    # loop through the root groups
    for v in list(hfList[0].keys()):
        if v.startswith('BEAM'):
            beam = hfList[0].get(v)
            b_beam = beam.get('beam')[0]
            channel = beam.get('channel')[0]
            data.append([v, hf[v].attrs['description'], b_beam, channel])

    # print as a table
    tbl_n += 1
    # print(f'Table {tbl_n}. GEDI Beams')
    headers = ["beam name", "description", "beam", "channel"]
    # display(HTML(tabulate.tabulate(data, headers, tablefmt='html')))

    beam_str = ['BEAM0101','BEAM0110','BEAM1000', 'BEAM1011']
    beam0110 = hf[beam_str[0]]

    data = []
    # loop over all the variables within BEAM0110 group
    for v in beam0110.keys():
        var = beam0110[v]
        source = ''
        # if the key is a subgroup assign GROUP tag
        if isinstance(var, h5py.Group):
            data.append([v, 'GROUP', 'GROUP', 'GROUP'])
        # read source, description, units attributes of each variables
        else:
            if 'source' in var.attrs.keys():
                source = var.attrs['source']
            data.append([v, var.attrs['description'], var.attrs['units'], source])

    # print all variable name and attributes as a table
    tbl_n += 1
    # print(f'Table {tbl_n}. Variables within {beam_str} group')
    headers = ["variable", "description", "units", "source"]
    data = sorted(data, key=lambda x:x[3])
    # display(HTML(tabulate.tabulate(data, headers, tablefmt='html')))

    # Folder to save individual CSV files

    os.makedirs(output_folder, exist_ok=True)

    # Initialize file counter


    for hf in hfList:
        # Temporary lists for each hf
        elev_l = []
        lat_l = []
        lon_l = []
        agbd_l = []
        error_l = []
        beam_n = []
        time_l = []
        quality_l = []

        # Loop over all base groups in each hf
        for var in list(hf.keys()):
            if var.startswith('BEAM'):
                beam = hf.get(var)
                agbd = beam.get('agbd')[:]
                error = beam.get('agbd_se')[:]
                elev = beam.get('elev_lowestmode')[:]
                lat = beam.get('lat_lowestmode')[:]
                lon = beam.get('lon_lowestmode')[:]
                time = beam.get('delta_time')[:]
                quality = beam.get('l4_quality_flag')[:]

                # Append data to temporary lists
                agbd_l.extend(agbd.tolist())
                error_l.extend(error.tolist())
                elev_l.extend(elev.tolist())
                lat_l.extend(lat.tolist())
                lon_l.extend(lon.tolist())
                time_l.extend(time.tolist())
                quality_l.extend(quality.tolist())
                n = lat.shape[0]
                beam_n.extend(np.repeat(str(var), n).tolist())

        # Create a DataFrame for the current hf
        df_hf = pd.DataFrame(list(zip(beam_n, agbd_l, error_l, elev_l, lat_l, lon_l, time_l, quality_l)),
                            columns=["beam", "agbd", "agbd_se", "elev_lowestmode", "lat_lowestmode", "lon_lowestmode", "delta_time", "l4_quality_flag"])

        # Save the DataFrame to a CSV file
        df_hf.to_csv(os.path.join(output_folder, file_name), index=False)

        # Clear the temporary DataFrame and lists to free RAM
        del df_hf, elev_l, lat_l, lon_l, agbd_l, error_l, beam_n, time_l, quality_l
        # file_counter += 1  # Increment file counter

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
    polys = []
    for _, polygon in gdf.iterrows():
        bounds = polygon.geometry.bounds  # Returns (minx, miny, maxx, maxy)
        bbox = (bounds[0], bounds[1], bounds[2], bounds[3])  # (min_lon, min_lat, max_lon, max_lat)
        poly = polygon.geometry
        polys.append(poly)
        bboxes.append(bbox)
    return bboxes, polys
def generate_weekly_date_pairs(start_date, end_date):
    """
    Generates a list of weekly date pairs (start, end) from start_date to end_date.
    Each date is in the format: datetime.datetime(year, month, day, 0, 0).

    Args:
        start_date: The starting date (datetime object).
        end_date: The ending date (datetime object).

    Returns:
        A list of tuples, where each tuple contains the start and end dates of a week.
    """
    weekly_pairs = []
    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)  # Set time to 00:00:00
    while current_date <= end_date:
        end_of_week = current_date + dt.timedelta(days=6)
        end_of_week = end_of_week.replace(hour=0, minute=0, second=0, microsecond=0)  # Set time to 00:00:00
        weekly_pairs.append((current_date, end_of_week))
        current_date += dt.timedelta(weeks=1)
    return weekly_pairs
