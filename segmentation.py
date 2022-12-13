# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:07:34 2022

@author: wayne
"""

import configargparse
import cv2
import mahotas as mh
import numpy as np
import waterz
import zarr
from funlib.math import encode64
from skimage.measure import regionprops

from segment_stats import segment_stats
from write_to_db import create_cells_client


def get_fragments(image):
    """Apply watershed to an image to get (over-)segmentation fragments.

    Parameters
    ----------
    image: array
        Boundary prediction image.
    """
    # normalized to 255 can get better watershed output
    inputimage = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # watershed algorithm set local minima as seed
    minima = mh.regmin(inputimage)
    markers, nr_markers = mh.label(minima)
    fragments = mh.cwatershed(inputimage, markers, return_lines=False)
    fragments = fragments.astype('uint64')
    return fragments


def get_affinities(image):
    """Get affinities from boundary predictions."""
    # normalized to 0 - 1 get affinity graph
    # TODO Might not be necessary, since data seems to be 0 - 1 already
    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    # affinity graph needs 3 channel input, original is just 1
    affs = np.zeros((3,) + image.shape)
    pre_nor = NormalizeData(image)

    # Invert the picture set membranes with low affinity
    # and cells with high affinity
    for i in range(3):
        affs[i] = pre_nor*-1+1

    # make sure correct type
    aff = affs.astype('float32')
    return aff


def Parser():
    parser = configargparse.ArgParser()
    parser.add('zarrfile', help='The zarr file containing the data')
    parser.add('-c', '--config', is_config_file=True)
    parser.add('-n', '--name', help="Project Name")  # required=True
    parser.add('--host', help="MongoDB Host Name", default=None)
    # , required=True)
    parser.add('--port', help="MongoDB Port", default=None)  # required=True)
    parser.add('--user', help="Username", default=None)  # required=True)
    parser.add('--password', help="Password", default=None)  # required=True)
    parser.add('--max_volume', help="Maximum number of voxels in a cell",
               default=25000)
    parser.add('--deploy', action='store_true',
               help='Whether to connect to the database or just print results')
    return parser


if __name__ == "__main__":
    """

    Example
    -------
    import h5py
    name='silja_dataset_1_T110_predictions_fused.h5'
    f = h5py.File(name,'r+')
    key = list(f.keys())[0]
    image = f[key][3]

    Channels:
        0: Membrane channel
        1: Wide Field
        2: ZO1 protein
        3: Membrane boundary prediction
    """
    parser = Parser()
    args = parser.parse_args()

    z = zarr.open(args.zarrfile, 'a')
    # input image is the 4th channel image of data
    # for example our data is (c,t,z,y,x)
    # Usually normalized 0 - 1
    channel, time, *_ = z['Raw'].shape
    frags_t = []
    if args.deploy:
        client, cells = create_cells_client(args.name, args.host, args.port,
                                            args.user, args.password)
    else:
        cells = []

    for t in range(time):
        image = z['Raw'][3, t, :, :, :]
        print(image.shape)
        affs = get_affinities(image)
        fragments = get_fragments(image)
        thresholds = [0, 100]

        gen = waterz.agglomerate(affs, thresholds, fragments=fragments,
                                 return_merge_history=True,
                                 return_region_graph=True)
        # the initial seg is the fragments, with threshold=0
        segs, _, _ = next(gen)
        # prevent generator change data in RAM 
        seg = segs.copy()
        frags_t.append(seg)
        # Fragment values in segs starts at 1
        ids, positions, volumes = segment_stats(fragments, t)
        # positions
        # Get the merges from the second, larger threshold
        _, merges, _ = next(gen)

        def merged_position(fragments, a, b):
            """
            Merge 'a' and 'b' into 'w'

            return a center point: tuple float
            """
            merge_mask = np.zeros((fragments.shape), dtype='int')
            merge_mask[seg == a] = 1
            merge_mask[seg == b] = 1
            region = regionprops(merge_mask)
            pos_w = region[0].centroid
            return pos_w

        merge_tree = np.empty((len(merges), 3),dtype=int)
        merge_scores = np.empty((len(merges),))
        # Separately store the stats for the whole merge tree!
        merge_ids = []
        merge_positions = {}
        merge_volumes = {}
        merge_parents = {}
        for i, merge in enumerate(merges):
            # e.g. {a: 1, b: 2, c: 1, score: 0.01}
            a, b = merge['a']-1, merge['b']-1 # match the ID index 0,1,2,3,.... order
            score = merge['score']
            u, v = ids[a], ids[b]
            pos_u, pos_v = positions[a], positions[b]
            vol_u, vol_v = volumes[a], volumes[b]
            # Create the merged node
            vol_w = vol_u + vol_v
            pos_w = merged_position(fragments, a+1, b+1) # index+1 = label
            pos_w = (t, int(pos_w[0]), int(pos_w[1]), int(pos_w[2]))
            # covert to int for encode
            w = encode64(pos_w)
            # Replace values... since a,b merge into a
            positions[a] = (pos_w)
            volumes[a] = vol_w
            # Add to merge tree
            merge_tree[i] = u, v, w
            merge_scores[i] = score
            # Add to merge stats
            merge_ids.append((u, v, w))
            merge_volumes.update({u: vol_u, v: vol_v, w: vol_w})
            merge_parents.update({u: w, v: w, w: w})
            merge_positions.update({u: pos_u, v: pos_v, w: pos_w})

        for cell_id in np.unique(merge_tree):
            position = merge_positions[cell_id]
            volume = merge_volumes[cell_id]
            merge_parent = merge_parents[cell_id]
            # Bigger cells are considered "over merged"
            # mangodb can not encode numpy.int try int()
            if volume < args.max_volume:
                cells.insert_one({
                    'id': int(cell_id),
                    'score': float(score),
                    't': int(position[0]),
                    'z': int(position[1]),
                    'y': int(position[2]),
                    'x': int(position[3]),
                    'movement_vector': (0, 0, 0),
                    'merge_parent': int(merge_parent),
                    'volume': int(volume)
                })

    #if args.deploy:
        #client.close()

    # Add fragments
    z['Fragments'] = np.array(frags_t)
    # Add merge tree
    z['Merge_tree/Merge'] = merge_tree
    z['Merge_tree/Scoring'] = merge_scores