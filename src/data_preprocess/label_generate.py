from pathlib import Path
import re
from data_preprocess.data_Dwonload import create_gedi_raster

image_dir = Path('./data')
for hls_image in image_dir.glob('**/*.tiff'):
    # Extract base name by removing patch number and extension
    base_name = re.sub(r'_patch_\d+\.tiff$', '', str(hls_image))
    csv_files = Path(base_name.replace('data', 'csv_labels')).with_suffix('.csv')
    output_file = Path('./labels') / f"{hls_image.stem}_label.tiff"
    if output_file.exists():
        continue
    output_file.parent.mkdir(parents=True, exist_ok=True)
    create_gedi_raster(csv_files, hls_image, output_file=output_file, if_plot=True)
