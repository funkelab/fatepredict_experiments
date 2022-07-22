# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:07:34 2022

@author: wayne
"""

import numpy as np
import waterz
import cv2
import mahotas as mh

#input image is the 4th channel image of data
#for example our data is (c,t,z,y,x)
t=1
image = data[3,t,:,:,:]

'''
import h5py 
name='silja_dataset_1_T110_predictions_fused.h5'
f = h5py.File(name,'r+')
key = list(f.keys())[0]
image = f[key][3]
'''

#normalized to 255 can get better wahtershed output
inputimage = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)



#whatershed algorithm set local minima as seed
minima = mh.regmin(inputimage)
markers,nr_markers = mh.label(minima)
fragments = mh.cwatershed(inputimage, markers, return_lines=False)

#normalized to 0-1 get affinity graph
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# affinity graph needs 3 channel input
affs = np.zeros((3,)+image.shape)
pre_nor = NormalizeData(image)

#Invert the picture set memberance with low affinity and cells with high affinity
for i in range(3):
  affs[i]=pre_nor*-1+1

# make sure correct type
aff = affs.astype('float32')
fragments=fragments.astype('uint64')
'''  
affs: numpy array, float32, 4 dimensional
            The affinities as an array with affs[channel][z][y][x].
        thresholds: list of float32
            The thresholds to compute segmentations for. For each threshold, one
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
        scoring_function: string, default 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>'
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
thresholds=[0.2]
for segmentations, merge_history, region_graph in waterz.agglomerate(aff, thresholds,fragments=fragments,return_merge_history = True, return_region_graph = True):
    print(merge_history)
    print(region_graph)

