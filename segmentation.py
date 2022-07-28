# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:07:34 2022

@author: wayne
"""
import sys

import cv2
import argparse
from cv2 import threshold
import mahotas as mh
import numpy as np
import zarr
import waterz


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
    #thresholds = [0.2,0.4]
    for segs, merges, regions in waterz.agglomerate(affs,
                                                    thresholds,
                                                    fragments=fragments,
                                                    return_merge_history=True,
                                                    return_region_graph=True):
        yield segs, merges, regions



if __name__ == "__main__":
    """

    Example
    -------
    import h5py
    name='silja_dataset_1_T110_predictions_fused.h5'
    f = h5py.File(name,'r+')
    key = list(f.keys())[0]
    image = f[key][3]
    """
    zarrfile = sys.argv[1]

    z = zarr.open(zarrfile, 'a')

    '''
    # Todo change the parser (what are we want?)
    parser = argparse.ArgumentParser(description='Create xxxxx')
    parser.add_argument("-t", "--threshold",nargs='+',default=[], action='append', required=True, 
                        help="The thresholds to compute segmentations for (list float32).")
    args = parser.parse_args()
    '''
    '''
    Channels:
        0: Membrane channel
        1: Wide Field
        2: ZO1 protein
        3: Membrane boundary prediction
    '''
    # input image is the 4th channel image of data
    # for example our data is (c,t,z,y,x)
    # Usually normalized 0 - 1
    # TODO iterate over time
    frags_t=[]
    for t in range(3):
        image = z['Raw'][3, t, :, :, :]
        print(image.shape)
        affs = get_affinities(image)
        fragments = get_fragments(image)
        thresholds = [0]

        
        gen=main(affs,thresholds=thresholds,fragments=fragments)
        # the initial seg is the fragments
        segs,_,_  = next(gen)
        frags_t.append(segs)
        #_,merges,_  = next(gen)
        
    z['fragments'] = np.array(frags_t)
    print('------------------------')
    print(len(np.unique(np.array(frags_t))))
    print('------------------------')
    # Todo how to save list?
    #z['merge_tree'] = 
    