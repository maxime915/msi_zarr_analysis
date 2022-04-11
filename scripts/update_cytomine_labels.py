"update_cytomine_labels: ..."

import re

from cytomine.models import Project, ImageInstance, ImageInstanceCollection, SliceInstanceCollection

from connect_from_json import connect

project_id = 31054043
image_id = 146726078

conn = connect()

project = Project().fetch(id=project_id)
image = ImageInstance().fetch(id=image_id)

# map int(channel) \in {0, 1, ...} to float(mz_val)
channel_names = {}

# files that aren't slice but shouldn't trigger a warning
white_list = ["HR2MSI mouse urinary bladder S096 - optical image.png", "RAW", "ten-ion-images.png"]

pattern = re.compile(r"HR2MSI_mouse_urinary_bladder_S096_(\d+)_(\d+\.\d+)_jet_norm.png")

# get all filenames & associate mzs values
for img in ImageInstanceCollection().fetch_with_filter("project", project.id):
    path = img.originalFilename
    res = pattern.search(path)

    if not res:
        if path not in white_list:
            print(f"{path=} did not match")
        continue
    
    # cast values
    channel = int(res.group(1)) - 1
    mz_val = float(res.group(2))
    
    channel_names[channel] = mz_val

# load slice from the image to label
slice_collection = SliceInstanceCollection().fetch_with_filter("imageinstance", image.id)

for slice_instance in slice_collection:
    # don't overwrite anything
    if slice_instance.zName is not None:
        print(f"{slice_instance.zName=} {slice_instance.zStack=}")
        continue

    slice_instance.zName = str(channel_names[slice_instance.zStack])
    slice_instance.update()
