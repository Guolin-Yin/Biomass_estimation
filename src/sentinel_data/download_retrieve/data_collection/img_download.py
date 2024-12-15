import os
import numpy as np
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, MimeType, DataCollection, MosaickingOrder, bbox_to_dimensions, SentinelHubCatalog, Geometry

from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

from dotenv import load_dotenv


config = SHConfig()
load_dotenv()
# read your credentials here from env variables
config.sh_client_id = os.getenv('SENTI_ID')
config.sh_client_secret = os.getenv('SENTI_SECRET')


# set up credentials
client = BackendApplicationClient(client_id=os.getenv('SENTI_ID'))
oauth = OAuth2Session(client=client)

# get an authentication token
token = oauth.fetch_token(token_url='https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token',
                          client_secret=os.getenv('SENTI_SECRET'), include_client_id=True)
evalscript_allbands = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "CLP"],
            }],
            output: {
                bands: 11
            }
        };
    }
    function evaluatePixel(sample) {
        return [sample.B02,
                sample.B03,
                sample.B04,
                sample.B05,
                sample.B06,
                sample.B07,
                sample.B08,
                sample.B8A,
                sample.B09,
                sample.B11,
                sample.B12,
                sample.CLP]
    }
    """

evalscript_true_color = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B01","B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL",  "dataMask"],
            }],
            output: {
                bands: 14
            }
        };
    }
    function evaluatePixel(sample) {
        switch (sample.SCL) {
        // No Data (Missing data) (black)    
        case 0: return [sample.B01,sample.B02,sample.B03,sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B09, sample.B11, sample.B12, 0, sample.dataMask];
            
        // Cloud shadows (dark brown)
        case 3: return [sample.B01,sample.B02,sample.B03,sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B09, sample.B11, sample.B12, 1, sample.dataMask];
        
        // Cloud medium probability (grey)
        case 8: return [sample.B01,sample.B02,sample.B03,sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B09, sample.B11, sample.B12, 2, sample.dataMask];
            
        // Cloud high probability (white)
        case 9: return [sample.B01,sample.B02,sample.B03,sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B09, sample.B11, sample.B12, 2, sample.dataMask];
        
        default : return [sample.B01,sample.B02,sample.B03,sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B09, sample.B11, sample.B12, 0, sample.dataMask];
        }
    }
    """
# https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html#available-bands-and-data
"""
0 - No data
1 - Saturated / Defective
2 - Dark Area Pixels
3 - Cloud Shadows
4 - Vegetation
5 - Bare Soils
6 - Water
7 - Clouds low probability / Unclassified
8 - Clouds medium probability
9 - Clouds high probability
10 - Cirrus
11 - Snow / Ice
"""
# evalscript_hls = """
#     //VERSION=3
#     function setup() {
#         return {
#             input: [{
#                 bands: ["CoastalAerosol", "Blue", "Green", "Red", "NIR_Narrow", "SWIR1", "SWIR2", "Cirrus","dataMask"],
#                 units: "DN"  
#             }],
#             output: {
#                 bands: 9,
#                 sampleType: "FLOAT32" 
#             }
#         };
#     }
#     function evaluatePixel(sample) {
#         return [sample.CoastalAerosol,
#                 sample.Blue,
#                 sample.Green,
#                 sample.Red,
#                 sample.NIR_Narrow,
#                 sample.SWIR1,
#                 sample.SWIR2,
#                 sample.Cirrus,
#                 sample.dataMask]
#     }
#     """
    
evalscript_hls = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: [
                    "CoastalAerosol", "Blue", "Green", "Red", "NIR_Narrow", "SWIR1", "SWIR2",
                    "Cirrus", "QA", "Fmask"  // Added QA and Fmask bands for better cloud masking
                ],
                units: "REFLECTANCE"  // Changed to REFLECTANCE
            }],
            output: {
                bands: 11,  // Updated number of bands
                sampleType: "FLOAT32" 
            }
        };
    }

    function evaluatePixel(sample) {
        // Cloud masking using Fmask
        // Fmask values: 0=clear, 1=water, 2=cloud_shadow, 3=snow, 4=cloud
        let isValid = sample.Fmask == 0 || sample.Fmask == 1;  // Only keep clear and water pixels
        
        // Additional check using Cirrus band
        let isCirrus = sample.Cirrus > 0.01;  // Threshold for cirrus clouds
        
        // QA band check (specific bits for quality issues)
        let qaGood = (sample.QA & 0x8000) === 0;  // Check if high quality bit is set
        
        // Create mask (1 for good pixels, 0 for bad pixels)
        let mask = (isValid && !isCirrus && qaGood) ? 1 : 0;
        
        // Return masked reflectance values (divided by 10000 to get proper scale)
        return [
            sample.CoastalAerosol / 10000.0,
            sample.Blue / 10000.0,
            sample.Green / 10000.0,
            sample.Red / 10000.0,
            sample.NIR_Narrow / 10000.0,
            sample.SWIR1 / 10000.0,
            sample.SWIR2 / 10000.0,
            sample.Cirrus / 10000.0,
            sample.QA,
            sample.Fmask,
            mask
        ];
    }
"""
def get_img_CLP(img_bbox, img_size, time_interval):


    request = SentinelHubRequest(
        evalscript=evalscript_hls,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.HARMONIZED_LANDSAT_SENTINEL,
                time_interval=time_interval,
                mosaicking_order=MosaickingOrder.LEAST_CC,
                maxcc=0.1
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=img_bbox,
        size=img_size,
        config=config
    )

    # Get the data, including the CLM band
    return request.get_data()[0]
