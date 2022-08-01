import numpy as np
from funlib.math import decode64
from scipy import ndimage as ndi


def apply_tf(img, affine_martix, interpolate=False):
    order = 1 if interpolate else 0
    affine_img = ndi.affine_transform(img, affine_martix,
                                      order=order,
                                      mode='constant', cval=0)
    return affine_img


def overlap(frg_pre, frg_next):
    f_pre = frg_pre.flatten()
    f_next = frg_next.flatten()
    pairs, counts = np.unique(np.stack([f_pre, f_next]), axis=1,
                              return_counts=True)
    return pairs.T, counts


def dist(i_pos, k_pos):
    # TODO Define the distance between two position
    return 0


def main(frg_pre, frg_next):
    # TODO move to main script
    # track i ,j
    pairs, counts = overlap(frg_pre, frg_next)
    distances = []
    for p, c in zip(pairs, counts):
        # p: overlap pairs
        # c: region of overlap
        # i: region i in pre
        # k: region k in next (which overlap with i)
        i_pos = decode64(p[0])
        k_pos = decode64(p[1])
        distances.append(dist(i_pos, k_pos))
    return distances
