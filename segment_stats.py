import sys

import numpy as np
import zarr
from funlib.math import encode64
from skimage.measure import regionprops


def segment_stats(fragments, t):
    regions = regionprops(fragments)
    position = []
    region_size = []
    idx = []
    for props in regions:
        z0, y0, x0 = props.centroid
        position.append((t, int(z0), int(y0), int(x0)))
        ids = encode64((t, int(z0), int(y0), int(x0)))
        idx.append(ids)
        region_size.append(props.area)

    return np.array(idx), np.array(position), np.array(region_size)


if __name__ == "__main__":

    zarrfile = sys.argv[1]

    z = zarr.open(zarrfile, 'a')
    # Todo change t
    for t in range(3):
        fragments = z['fragments'][t]
        idx, position, region_size = segment_stats(fragments, t)
        z['fragment_stats/id/'+str(t)] = position
        z['fragment_stats/position/'+str(t)] = position
        z['fragment_stats/volume/'+str(t)] = region_size
