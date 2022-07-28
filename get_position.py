import re
import zarr

from skimage.measure import regionprops, regionprops_table
import sys
import numpy as np


def get_idx_position_size(fragments):
    regions = regionprops(fragments)
    position = []
    region_size = []
    hash_idx = []

    for props in regions:
        z0, y0, x0 = props.centroid
        position.append((z0, y0, x0))
        # Todo not use hash
        idx = hash((z0, y0, x0))
        hash_idx.append(idx)
        region_size.append(props.area)
    
    return np.array(hash_idx), np.array(position), np.array(region_size)


if __name__ == "__main__":

    zarrfile = sys.argv[1]

    z = zarr.open(zarrfile, 'a')
    # Todo change t
    for t in range(3):
        fragments = z['fragments'][t]
        hash_idx,position,region_size=get_idx_position_size(fragments)
        z['fragment_stats/position/'+str(t)] = position
        z['fragment_stats/size/'+str(t)] = region_size



    