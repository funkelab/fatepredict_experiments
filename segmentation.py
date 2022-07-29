# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:07:34 2022

@author: wayne
"""
import sys

import configargparse
import cv2
import mahotas as mh
import numpy as np
import zarr
from skimage.measure import regionprops
import waterz
from segment_stats import encode64, segment_stats
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


# TODO This is basically a wrapper around waterz.agglomerate, doesn't add
# functionality, just sets threshold, could be removed
# TODO What is the scoring function choice + why?
def main(affs, thresholds, gt=None, fragments=None, aff_threshold_low=0.0001,
         aff_threshold_high=0.9999, return_merge_history=True,
         return_region_graph=True,
         scoring_function='OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
         discretize_queue=0,
         force_rebuild=False):
    '''
    ----------
    sudo apt install libboost-dev
    pip install git+https://github.com/funkey/waterz.git
    Parameters
    ----------
        affs: numpy array, float32, 4 dimensional
            The affinities as an array with affs[channel][z][y][x].
        thresholds: list of float32
            The thresholds to compute segmentations for.
            For each threshold, one
            segmentation is returned.
        gt: numpy array, uint32, 3 dimensional (optional)
            An optional ground-truth segmentation as an array with gt[z][y][x].
            If given, metrics
        fragments: numpy array, uint64, 3 dimensional (optional)
            An optional volume of fragments to use, instead of the build-in
            zwatershed.
        aff_threshold_low: float, default 0.0001
        aff_threshold_high: float, default 0.9999,
            Thresholds on the affinities for the initial segmentation step.
        return_merge_history: bool
            If set to True, the returning tuple will contain a merge history,
            relative to the previous segmentation.
        return_region_graph: bool
            If set to True, the returning tuple will contain the region graph
            for the returned segmentation.
        scoring_function: string, default
            'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>'
            A C++ type string specifying the edge scoring function to use. See
                https://github.com/funkey/waterz/blob/master/waterz/backend/MergeFunctions.hpp
            for available functions, and
                https://github.com/funkey/waterz/blob/master/waterz/backend/Operators.hpp
            for operators to combine them.
        discretize_queue: int
            If set to non-zero, a bin queue with that many bins will be used to
            approximate the priority queue for merge operations.
        force_rebuild:
            Force the rebuild of the module. Only needed for development.
    Returns
    -------
        Results are returned as tuples from a generator object, and only
        computed on-the-fly when iterated over. This way, you can ask for
        hundreds of thresholds while at any point only one segmentation is
        stored in memory.
        Depending on the given parameters, the returned values are a subset of
        the following items (in that order):
        segmentation
            The current segmentation (numpy array, uint64, 3 dimensional).
        metrics (only if ground truth was provided)
            A  dictionary with the keys 'V_Rand_split', 'V_Rand_merge',
            'V_Info_split', and 'V_Info_merge'.
        merge_history (only if return_merge_history is True)
            A list of dictionaries with keys 'a', 'b', 'c', and 'score',
            indicating that region a got merged with b into c with the given
            score.
        region_graph (only if return_region_graph is True)
            A list of dictionaries with keys 'u', 'v', and 'score', indicating
            an edge between u and v with the given score.
    '''
    #
    #  TODO how to decide the threshold?
    # thresholds = [0.2,0.4]
    for segs, merges, regions in waterz.agglomerate(affs,
                                                    thresholds,
                                                    fragments=fragments,
                                                    return_merge_history=True,
                                                    return_region_graph=True):
        yield segs, merges, regions


def Parser():
    parser = configargparse.ArgParser()
    parser.add('zarrfile', required=True,
               help='The zarr file containing the data')
    parser.add('-c', '--config', is_config_file=True)
    parser.add('-n', '--name', help="Project Name") # required=True
    parser.add('--host', help="MongoDB Host Name", default=None)  # required=True)
    parser.add('--port', help="MongoDB Port", default=None)  # required=True)
    parser.add('--user', help="Username", default=None)  # required=True)
    parser.add('--pass', help="Password", default=None)  # required=True)
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
                                            args.username, args.password)
    else:
        cells = []

    for t in range(time):
        image = z['Raw'][3, t, :, :, :]
        print(image.shape)
        affs = get_affinities(image)
        fragments = get_fragments(image)
        thresholds = [0, 100]

        gen = main(affs, thresholds=thresholds, fragments=fragments)
        # the initial seg is the fragments, with threshold=0
        segs, _, _ = next(gen)
        frags_t.append(segs)
        #Todo do we really need copy ? maybe fragments is the same?
        seg = segs.copy()
        # Fragment values in segs starts at 1
        ids, positions, volumes = segment_stats(fragments, t)
        # positions
        # Get the merges from the second, larger threshold
        _, merges, _ = next(gen)

        # DELME
        ids = np.array(ids)
        better_ids = ids[seg - 1]  # Same shape as segs, each

        def merged_position(fragments,a,b):
            #pos_w = (pos_u + pos_v) // 2
            # TODO weighted version
            # Can we weigh the center of gravity just by volume or do we need
            # the extent in all three dimensions?
            # TODO Convert position to tuple of ints
            '''
            merge 'a' and 'b' into 'w'
            '''
            merge_mask = np.zeros((fragments.shape),dtype='int')
            merge_mask[seg==a] = 1
            merge_mask[seg==b] = 1
            region = regionprops(merge_mask)
            pos_w = region[0].centroid
            return pos_w

        merge_tree = np.empty(len(merges), 3)
        merge_scores = np.empty(len(merges),)
        # Separately store the stats for the whole merge tree!
        merge_ids = []
        merge_positions = {}
        merge_volumes = {}
        merge_parents = {}
        for i, merge in enumerate(merges):
            # e.g. {a: 1, b: 2, c: 1, score: 0.01}
            a, b = merge['a']-1, merge['b']-1
            score = merge['score']
            u, v = ids[a], ids[b]
            pos_u, pos_v = positions[a], positions[b]
            vol_u, vol_v = volumes[a], volumes[b]
            # Create the merged node
            vol_w = vol_u + vol_v
            pos_w = merged_position(pos_u, pos_v, vol_u, vol_v)
            w = encode64((t, *pos_w))
            # Replace values...
            ids[a] = w
            positions[a] = pos_w
            volumes[a] = vol_w
            # Add to merge tree
            merge_tree[i] = u, v, w
            merge_scores[i] = score
            # Add to merge stats
            # TODO Make this less ugly :)
            merge_ids += [u, v, w]
            merge_volumes[u] = vol_u
            merge_volumes[v] = vol_v
            merge_volumes[w] = vol_w
            merge_parents[u] = w
            merge_parents[v] = w
            merge_parents[w] = w
            merge_positions[u] = pos_u
            merge_positions[v] = pos_v
            merge_positions[w] = pos_w

        for cell_id in merge_ids:
            position = merge_positions[cell_id]
            volume = merge_volumes[cell_id]
            merge_parent = merge_parents[cell_id]
            # Bigger cells are considered "over merged"
            if volume < args.max_volume:
                cells.append({
                    'id': cell_id,
                    'score': float(score),
                    't': position[0],
                    'z': position[1],
                    'y': position[2],
                    'x': position[3],
                    'movement_vector': tuple(0, 0, 0),
                    'merge_parent': merge_parent,
                    'volume': volume
                })

    if not args.deploy:
        print(cells)
    else:
        client.close()

    # Add fragments
    z['fragments'] = np.array(frags_t)

    # TODO Get all overlaps of fragments
    def overlap(frg_pre,frg_next):
        # frg_pre : ndarray.
        f_pre = frg_pre.flatten()
        f_next = frg_next.flatten()
        pairs, counts = np.unique(np.stack([f_pre, f_next]), axis=1, return_counts=True)
        return pairs, counts


