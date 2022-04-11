""


from cytomine import Cytomine
from cytomine.models import Project, ImageInstance, TermCollection, SliceInstanceCollection

from connect_from_json import connect


project_id = 31054043
image_id = 146726078

conn = connect()

project = Project().fetch(id=project_id)
image = ImageInstance().fetch(id=image_id)

terms = TermCollection().fetch_with_filter("project", project.id)

slices = SliceInstanceCollection().fetch_with_filter("imageinstance", image.id)

for slice in slices:
    print(f'{slice.zName=} {slice.zStack=}')
