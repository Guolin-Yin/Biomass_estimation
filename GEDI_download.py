from src.sentinel_data.GEDI import search_gedi_granules_poly_2, to_csv, load_polygons, get_bboxes_from_polygons, generate_weekly_date_pairs, concept_id
from tqdm import tqdm
import datetime as dt
import os
import shutil
from pathlib import Path
import sys
from pathlib import Path
from shapely.geometry import box


# data_amount = 1000
# polygons = load_polygons('species.csv')
# bboxes, polys = get_bboxes_from_polygons(polygons)

# weekly_pairs = generate_weekly_date_pairs(dt.datetime(2022, 4, 1), dt.datetime(2022, 10, 1))

# start_date, end_date = dt.datetime(2017, 4, 1), dt.datetime(2024, 10, 1)

# bar = tqdm(polys, desc="Processing bounding boxes")
# for index, bound in enumerate(bar):
    # for start_date, end_date in weekly_pairs:
    
# bound_coords = (-8.499590926570734, 38.77199531533364, -6.395956468456142, 40.13990015731868)
# bound = box(*bound_coords)
# l4adf, downloaded_files = search_gedi_granules_poly_2(
#     concept_id, 
#     bound, 
#     start_date, 
#     end_date, 
#     data_amount=data_amount
# )

total_files = 0
downloaded_files = Path('./gedi_files').glob('**/*.h5')
for file in downloaded_files:
    # Convert to CSV directly from original location
    output_file = Path('./csv_labels') / str(Path(file).with_suffix('.csv').name)
    if output_file.exists():
        continue
    try:
        to_csv(
            file,
            output_folder="./csv_labels", 
            file_name = str(Path(file).with_suffix('.csv').name)
        )
    except Exception as e:
        print(f"Error converting to CSV: {e}")
        continue
    # total_files += len(downloaded_files)
        # bar.set_description(f"Found {total_files} files for polygon {index}/{len(polys)}")
        # bar.update()

