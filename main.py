"""
Statistics of Patch Offsets for Image Completion - Kaiming He and Jian Sun
A Python Implementation - Pranshu Gupta and Shrija Mishra
"""

import cv2
import sys
import kdtree
import operator
import numpy as np
import config as cfg
from time import time
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def GetBoundingBox(mask):
    """
    Get Bounding Box for a Binary Mask
    Arguments: mask - a binary mask
    Returns: col_min, col_max, row_min, row_max
    """
    start = time()
    a = np.where(mask != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    if cfg.PRINT_BB_IMAGE:
        cv2.rectangle(mask, (bbox[2], bbox[0]), (bbox[3], bbox[1]), (0,0,255), 1)
        cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + cfg.BB_IMAGE_SUFFIX, mask)
    end = time()
    print "GetBoundingBox execution time: ", end - start
    return bbox

def GetSearchDomain(shape, bbox):
    """
    get a rectangle that is 3 times larger (in length) than the bounding box of the hole
    this is the region which will be used for the extracting the patches
    """
    start = time()
    col_min, col_max = max(0, 2*bbox[0] - bbox[1]), min(2*bbox[1] - bbox[0], shape[1]-1)
    row_min, row_max = max(0, 2*bbox[2] - bbox[3]), min(2*bbox[3] - bbox[2], shape[0]-1)
    end = time()
    print "GetSearchDomain execution time: ", end - start
    return col_min, col_max, row_min, row_max

def GetPatches(image, bbox, hole):
    """
    get the patches from the search region in the input image
    """
    start = time()
    indices, patches = [], []
    rows, cols = image.shape
    for i in xrange(bbox[2]+cfg.PATCH_SIZE/2, bbox[3]-cfg.PATCH_SIZE/2):
        for j in xrange(bbox[0]+cfg.PATCH_SIZE/2, bbox[1]-cfg.PATCH_SIZE/2):
            indices.append([i,j])
            patches.append(image[i-cfg.PATCH_SIZE/2:i+cfg.PATCH_SIZE/2, j-cfg.PATCH_SIZE/2:j+cfg.PATCH_SIZE/2])
    end = time()
    print "GetPatches execution time: ", end - start
    return np.array(indices), np.array(patches, dtype='int64').reshape(len(patches), cfg.PATCH_SIZE**2)

def ReduceDimension(patches):
    start = time()
    pca = PCA(n_components=24)
    reducedPatches = pca.fit_transform(patches)
    end = time()
    print "ReduceDimension execution time: ", end - start
    return reducedPatches

def GetOffsets(patches):
    start = time()
    kd = kdtree.KDTree(patches, leafsize=cfg.KDT_LEAF_SIZE, tau=cfg.TAU, deflat_factor=cfg.DEFLAT_FACTOR)
    dist, offsets = kdtree.get_annf_offsets(patches, kd.tree, cfg.DEFLAT_FACTOR, cfg.TAU)
    end = time()
    print "GetOffsets execution time: ", end - start
    return offsets

def GetKDominantOffsets(offsets, K, height, width):
    x, y = [offset[0] for offset in offsets if offset != None], [offset[1] for offset in offsets if offset != None]
    bins = [[i for i in range(np.min(x),np.max(x))],[i for i in xrange(np.min(y),np.max(y))]]
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    hist = hist.T
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, title='imshow: square bins')
    plt.imshow(hist, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    smooth_hist = ndimage.filters.gaussian_filter(hist,2**0.5)
    plt.imshow(smooth_hist, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.show()
	# Pick the best 60 local maxima
    peaks = {}
    for i in range(4,len(smooth_hist)):
        for j in range(4,len(smooth_hist[0])):
            max_in_box = np.max(smooth_hist[i-4:i+4, j-4:j+4])
            if max_in_box != smooth_hist[i][j]:
                peaks[(i-height,j-width)] = smooth_hist[i][j]
	# Sort these local maxima and return top 60
	sorted_peaks = sorted(peaks.items(), key = operator.itemgetter(1))
	sorted_peaks = sorted_peaks[-1*K:]
	peak_offsets = [p[0] for p in sorted_peaks]
	return peak_offsets 


def main(imageFile, maskFile):
    """
    Image Completion Pipeline
        1. Patch Extraction
        2. Patch Offsets
        3. Image Stacking
        4. Blending
    """
    image = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
    bb = GetBoundingBox(mask)
    bbwidth = bb[3] - bb[2]
    bbheight = bb[1] - bb[0]
    cfg.TAU = max(bbwidth, bbheight)/15
    cfg.DEFLAT_FACTOR = image.shape[1]
    sd = GetSearchDomain(image.shape, bb)
    indices, patches = GetPatches(image, sd, bb)
    reducedPatches = ReduceDimension(patches)
    offsets = GetOffsets(reducedPatches)
    kDominantOffset = GetKDominantOffsets(offsets, 60, image.shape[0], image.shape[1])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python main.py image_name mask_file_name"
        exit()
    cfg.IMAGE = sys.argv[1].split('.')[0]
    imageFile = cfg.SRC_FOLDER + sys.argv[1]
    print imageFile
    maskFile = cfg.SRC_FOLDER + sys.argv[2]
    main(imageFile, maskFile)
    