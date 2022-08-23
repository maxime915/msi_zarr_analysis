
import base64
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter

def main(path: str):
    
    with open(path, 'r', encoding="utf8") as result_file:
        results = result_file.read()

    pattern = re.compile(r".*\[root\] \[DEBUG\] : \[..\]: \d+\.\d+ ; '(.*)'")

    # how many bins for the histogram values
    bins = 5000
    selection_size = 10000
    values = []

    for match in pattern.finditer(results):
        encoded: str = match.group(1)

        # use hist to find the number in each bin        
        buffer = base64.b64decode(encoded.encode('ascii'))
        selection = np.unpackbits(np.frombuffer(buffer=buffer, dtype=np.uint8))
        
        assert selection.shape == (selection_size,)
        values.append([a.sum() for a in np.split(selection, bins)])
    
    # plot_surface to show how it evolves
    surface = np.asarray(values)
    # surface = gaussian_filter(surface, sigma=7)
    ny, nx = surface.shape
    X, Y = np.meshgrid(range(nx), range(ny))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={"projection": "3d"})

    # op1: surface plot
    # surf = ax.plot_surface(X, Y, surface, cmap=cm.coolwarm, linewidth=0, antialiased=True, rstride=1, cstride=1)
    
    # opt2: scatter plot
    surf = ax.scatter(X, Y, surface, cmap=cm.coolwarm)    

    ax.set_zlim(0, selection_size / bins)    
    fig.colorbar(surf)
    
    fig.tight_layout()
    fig.savefig("saga.png")

if __name__ == "__main__":
    assert len(sys.argv) > 1
    for arg in sys.argv[1:]:
        main(arg)