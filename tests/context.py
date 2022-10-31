import sys

import pathlib

# msi_zarr_analysis/tests/context.py
sys.path.insert(0, str((pathlib.Path(__file__).parent.parent).resolve()))

# cytomine python client
sys.path.insert(0, str(pathlib.Path.home() / "cytomine-repos" / "Cytomine-python-client"))

print(f"{sys.path[:2]=!r}")