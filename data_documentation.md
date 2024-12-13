# GEDI-AI Dataset Documentation




## Directory Structure

The dataset is organized into two primary directories:
```
GEDI-AI-Dataset/
├── data/
│   ├── GEDI04_A_2019110075658_O01995_02_T02063_02_002_02_V002_patch_0.tiff
│   ├── GEDI04_A_2019110075658_O01995_02_T02063_02_002_02_V002_patch_1.tiff
│   └── …
└── labels/
    ├── GEDI04_A_2019110075658_O01995_02_T02063_02_002_02_V002_patch_0_label.tiff
    ├── GEDI04_A_2019110075658_O01995_02_T02063_02_002_02_V002_patch_1_label.tiff
    └── …
```

- **data/**: Contains the multi-band satellite imagery files in TIFF format.
- **labels/**: Contains the corresponding label files in TIFF format.

---

## Data Files

### File Format
- **Format**: GeoTIFF (`.tiff`)
- **Shape**: Each data file has a shape of `(9, Height, Width)`, where:
  - **9**: Number of spectral bands
  - **Height**: Number of pixels in the vertical direction
  - **Width**: Number of pixels in the horizontal direction

### Bands
Each data file comprises 9 spectral bands, corresponding to various wavelengths captured by the satellite sensors.

---

## Label Files

### File Format
- **Format**: GeoTIFF (`.tiff`)
- **Shape**: Each label file has a shape of `(Height, Width)`, matching the spatial dimensions of the corresponding data file.

### Label Encoding
- **Value `-1`**: Denotes positions with no measurements.
- **Other Values**: Represent specific classes or categories as defined in the dataset's labeling schema.

---

## Data Bands Description

The dataset includes the following spectral bands, identical to those in the IBM dataset:

1. **Coastal Aerosol (`CoastalAerosol`)**
   - **Description**: Captures aerosol particles in coastal regions.

2. **Blue (`Blue`)**
   - **Description**: Reflects the blue portion of the electromagnetic spectrum.

3. **Green (`Green`)**
   - **Description**: Reflects the green portion of the electromagnetic spectrum.

4. **Red (`Red`)**
   - **Description**: Reflects the red portion of the electromagnetic spectrum.

5. **NIR Narrow (`NIR_Narrow`)**
   - **Description**: Captures near-infrared light with a narrow wavelength band.

6. **SWIR1 (`SWIR1`)**
   - **Description**: Short-Wave Infrared band 1.

7. **SWIR2 (`SWIR2`)**
   - **Description**: Short-Wave Infrared band 2.

8. **Cirrus (`Cirrus`)**
   - **Description**: Detects cirrus clouds and high-altitude atmospheric phenomena.

9. **Data Mask (`dataMask`)**
   - **Description**: Masking layer indicating valid data regions.
   - **Value Encoding**:
     - `1`: Valid data
     - `0`: Invalid or masked data

---

## File Naming Convention

Consistent file naming is crucial for correctly pairing data files with their corresponding labels. The naming convention follows a structured pattern:

### Data Files
data/GEDI04_A_YYYYMMDDHHHHH_OXXXX_XX_TXXXXX_XX_XXX_XX_VXXX_patch_X.tiff
- **Example**: `data/GEDI04_A_2019110075658_O01995_02_T02063_02_002_02_V002_patch_0.tiff`
- **Components**:
  - `GEDI04_A`: Dataset identifier
  - `2019110075658`: Acquisition date and unique identifier
  - `O01995`: Orbit number
  - `02_T02063`: Tile or region identifier
  - `02_002_02_V002`: Additional metadata identifiers
  - `patch_0`: Patch number within the larger tile
- **Data shape**
    - (9, Height, Width)
### Label Files
labels/GEDI04_A_YYYYMMDDHHHHH_OXXXX_XX_TXXXXX_XX_XXX_XX_VXXX_patch_X_label.tiff
- **Example**: `labels/GEDI04_A_2019110075658_O01995_02_T02063_02_002_02_V002_patch_0_label.tiff`
- **Data shape**
    - (Height, Width)
### Mapping Data to Labels

Each data file in the `data/` directory has a corresponding label file in the `labels/` directory. The correspondence is established by matching the base filename before the `_label` suffix.

- **Data File**: `data/GEDI04_A_2019110075658_O01995_02_T02063_02_002_02_V002_patch_0.tiff`
- **Label File**: `labels/GEDI04_A_2019110075658_O01995_02_T02063_02_002_02_V002_patch_0_label.tiff`


```python
import rasterio
import numpy as np

# Load data file
data_path = 'data/GEDI04_A_2019110075658_O01995_02_T02063_02_002_02_V002_patch_0.tiff'
with rasterio.open(data_path) as src:
    data = src.read()  # Shape: (9, height, width)

# Load label file
label_path = 'labels/GEDI04_A_2019110075658_O01995_02_T02063_02_002_02_V002_patch_0_label.tiff'
with rasterio.open(label_path) as src:
    label = src.read(1)  # Shape: (height, width)