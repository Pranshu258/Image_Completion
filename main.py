"""
Statistics of Patch Offsets for Image Completion - Kaiming He and Jian Sun
A Python Implementation - Pranshu Gupta and Shrija Mishra
"""

import cv2
import sys
import numba
import scipy
import math
import numpy as np, operator
import config as cfg
from time import time
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import collections 

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
    """
    start = time()
    col_min, col_max = max(0, 2*bbox[0] - bbox[1]), min(2*bbox[1] - bbox[0], shape[1]-1)
    row_min, row_max = max(0, 2*bbox[2] - bbox[3]), min(2*bbox[3] - bbox[2], shape[0]-1)
    end = time()
    print "GetSearchDomain execution time: ", end - start
    return col_min, col_max, row_min, row_max

def GetPatches(image, bbox, hole):
    start = time()
    indices, patches = [], []
    rows, cols = image.shape
    for i in xrange(bbox[2]+cfg.PATCH_SIZE/2, bbox[3]-cfg.PATCH_SIZE/2):
        for j in xrange(bbox[0]+cfg.PATCH_SIZE/2, bbox[1]-cfg.PATCH_SIZE/2):
            if i not in xrange(hole[2]-cfg.PATCH_SIZE/2, hole[3]+cfg.PATCH_SIZE/2) and j not in xrange(hole[0]-cfg.PATCH_SIZE/2, hole[1]+cfg.PATCH_SIZE/2):
                indices.append([i,j])
                patches.append(image[i-cfg.PATCH_SIZE/2:i+cfg.PATCH_SIZE/2, j-cfg.PATCH_SIZE/2:j+cfg.PATCH_SIZE/2])
    end = time()
    print "GetPatches execution time: ", end - start
    return np.array(indices), np.array(patches, dtype='int64').reshape(len(patches), cfg.PATCH_SIZE**2)

def GetOffsets(indices, patches, tau):
    start = time()
    offsets = np.zeros((len(patches),2), dtype='uint8')
    k = 4*tau*tau+1
    kd = scipy.spatial.KDTree(patches, leafsize=10)
    distances, neighbors = kd.query(x=patches, k=k)
    for i in xrange(len(patches)):
        for j in xrange(10):
            dist = ((indices[i][0]-indices[neighbors[i][j]][0])**2 + (indices[i][1]-indices[neighbors[i][j]][1])**2)**0.5
            if dist >= tau:
                offsets[i] = [indices[neighbors[i][j]][0] - indices[i][0], indices[neighbors[i][j]][1] - indices[i][1]]
                break
    end = time()
    print "GetOffsets execution time: ", end - start
    return offsets

def ReduceDimension(patches):
    start = time()
    pca = PCA(n_components=24)
    reducedPatches = pca.fit_transform(patches)
    end = time()
    print "ReduceDimension execution time: ", end - start
    return reducedPatches

def GetOffsetStatistics(offsets):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = offsets[0,:], offsets[:,0]
    hist, xedges, yedges = np.histogram2d(x, y, bins = 10, range=[[0, 4], [0, 4]])#bins=([i for i in xrange(-150, 150)], [i for i in xrange(-100, 100)]), range=[[-150, 150], [-100, 100]])
    # xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    # xpos = xpos.flatten('F')
    # ypos = ypos.flatten('F')
    # zpos = np.zeros_like(xpos)
    # # Construct arrays with the dimensions for the 16 bars.
    # dx = 0.5 * np.ones_like(zpos)
    # dy = dx.copy()
    # dz = hist.flatten()
    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    # plt.show()
    
def GetKDominantOffsets(offsets, K, height, width):
    # offsetList = offsets.tolist()
    # print offsetList
    # counts = collections.Counter([x for sublist in offsetList for x in sublist])
    # print counts
    # new_list = sorted([x for x in counts], key=lambda x: -counts[x])
    # print new_list 
    x, y = [], []
    for point in offsets:
        x.append(point[0])
        y.append(point[1])
    # x = np.random.normal(2, 1, 100)
    # y = np.random.normal(1, 1, 100)
    # xedges = [0, 1, 3, 5]
    # yedges = [0, 2, 3, 4, 6]
    # H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    bins = [[i for i in range(np.min(x),np.max(x))],[i for i in xrange(np.min(x),np.max(x))]]
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    hist = hist.T
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(131, title='imshow: square bins')
    plt.imshow(hist, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    print hist
    #plt.hist2d(x, y, bins=bins, cmap=plt.cm.jet)

	# Remove the [0,0] matches
    #hist[height][width] = 0
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
    height,width = mask.shape
    image = cv2.resize( image, (0,0), fx=0.5, fy=0.5)
    mask = cv2.resize( mask, (0,0), fx=0.5, fy=0.5)
    bb = GetBoundingBox(mask)
    bbwidth = bb[3] - bb[2]
    bbheight = bb[1] - bb[0]
    sd = GetSearchDomain(image.shape, bb)
    indices, patches = GetPatches(image, sd, bb)
    reducedPatches = ReduceDimension(patches)
    offsets = GetOffsets(indices, reducedPatches, max(bbheight, bbwidth)/15)
    #dominantOffset = GetOffsetStatistics(offsets)
    kDominantOffset = GetKDominantOffsets(offsets, 60, height, width)
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python main.py image_name mask_file_name"
        exit()
    cfg.IMAGE = sys.argv[1].split('.')[0]
    imageFile = cfg.SRC_FOLDER + sys.argv[1]
    print imageFile
    maskFile = cfg.SRC_FOLDER + sys.argv[2]
    main(imageFile, maskFile)
    