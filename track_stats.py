import zarr
from segment_stats import encode64,decode64
#Todo aff

def apply_tf(img,affine_martix,interpolate == False):
    order = 1 if interpolate == True else 0    
    affine_img= ndi.affine_transform(img, affine_martix,order=order,
                                     mode='constant', cval=0)
    return affine_img

def overlap(frg_pre,frg_next):
        # frg_pre : ndarray.
        f_pre = frg_pre.flatten()
        f_next = frg_next.flatten()
        pairs, counts = np.unique(np.stack([f_pre, f_next]), axis=1, return_counts=True)
        return pairs.T, counts


# track i ,j 

pairs,counts = overlap(frg_pre,frg_next)

for p, c in zip(pairs,counts):
    # p: overlap paires
    # c: region of overlap
    # i: region i in pre
    # k: region k in next (which overlap with i) 
    i_pos = decode64(p[0])
    k_pos = decode64(p[1])
    # Todo def dis():
    distance =  dis(i_pos,k_pos)


