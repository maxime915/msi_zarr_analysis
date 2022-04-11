""

import pathlib
import re
import sys
from PIL import Image
import zarr
import shutil

from cytomine import Cytomine
from cytomine.models import Project, ImageInstance, AnnotationCollection, TermCollection, SliceInstanceCollection

import numpy as np

from connect_from_json import connect

AUTO_ERASE = True
SAVE_PROCESSED = False

destination = sys.argv[1] if len(sys.argv) > 1 else "cytomine_raw" + ["_cont", "_proc"][SAVE_PROCESSED]
dest_zarr = pathlib.Path(destination).with_suffix('.zarr')
dest_Urothelium = pathlib.Path(destination).with_suffix('.Urothelium.jpg')
dest_Stroma = pathlib.Path(destination).with_suffix('.Stroma.jpg')

for p in [dest_zarr, dest_Stroma, dest_Urothelium]:
    if p.exists():
        if AUTO_ERASE:
            if p.is_dir():
                shutil.rmtree(p)
                continue
            elif p.is_file():
                p.unlink()
                continue
        raise ValueError(f"{p=} already exists (choose another {destination=}")

project_id = 31054043
image_id = 146726078

conn = connect()

project = Project().fetch(id=project_id)
image = ImageInstance().fetch(id=image_id)


terms = TermCollection().fetch_with_filter("project", project.id)

annotations = AnnotationCollection(
    project=project.id,
    image=image.id,
    showTerm=True,
    showWKT=True
).fetch()

term_set = {terms.find_by_attribute('id', annotation.term[0]).name for annotation in annotations if annotation.term}

assert term_set == {'Urothelium', 'Stroma', 'BloodVessels'}
term_set.remove('BloodVessels')

cls_mask = {term: np.zeros((image.height, image.width), dtype=np.uint8) for term in term_set}

np_int = np.zeros((image.depth, 1, image.height, image.width))
np_len = np.zeros((1, 1, image.height, image.width))

for annotation in annotations:
    
    if not annotation.term:
        continue
    
    term_name = terms.find_by_attribute('id', annotation.term[0]).name
    
    if term_name not in term_set:
        continue
    
    for profile in annotation.profile():
        coord_x, coord_y = profile['point']
        cls_mask[term_name][coord_y, coord_x] = 255
        
        if np_len[0, 0, coord_y, coord_x] > 0:
            print(f"already full for {coord_x=}, {coord_y=}")
            if cls_mask[term_name][coord_y, coord_x] == 0:
                print(f"\tbut not for the same term (=> overlap)")

        np_int[:, 0, coord_y, coord_x] = profile['profile']
        np_len[0, 0, coord_y, coord_x] = image.depth

# save masks
Image.fromarray(cls_mask['Stroma']).save(dest_Stroma)
Image.fromarray(cls_mask['Urothelium']).save(dest_Urothelium)

# save zarr
z = zarr.open(dest_zarr, mode='w-')
z['/0'] = np_int
z['/labels/lengths/0'] = np_len

mz_label = -np.ones((image.depth,))

# get all filenames & associate mzs values
# white_list = ["HR2MSI mouse urinary bladder S096 - optical image.png", "RAW", "ten-ion-images.png"]
# pattern = re.compile(r"HR2MSI_mouse_urinary_bladder_S096_(\d+)_(\d+\.\d+)_jet_norm.png")
# for img in ImageInstanceCollection().fetch_with_filter("project", project.id):
#     path = img.originalFilename
#     res = pattern.search(path)
#     if not res:
#         if path not in white_list:
#             print(f"{path=} did not match")
#         continue
    
#     channel = int(res.group(1)) - 1
#     mz_val = float(res.group(2))
    
#     mz_label[channel] = mz_val

slices = SliceInstanceCollection().fetch_with_filter("imageinstance", image.id)

for slice in slices:
    mz_label[slice.zStack] = float(slice.zName)

    

if SAVE_PROCESSED:
    np_mzs = np.zeros((image.depth, 1, image.height, image.width))
    valid_mask = np_len[0, 0, ...] > 0
    np_mzs[:, 0, valid_mask] = np.tile(mz_label, (np.count_nonzero(valid_mask), 1)).T

    z.attrs["pims-msi"] = {"binary_mode": "processed"}
else:
    np_mzs = np.zeros((image.depth, 1, 1, 1))
    np_mzs[:, 0, 0, 0] = mz_label

    z.attrs["pims-msi"] = {"binary_mode": "continuous"}

z['/labels/mzs/0'] = np_mzs
