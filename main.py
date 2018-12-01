"""
Statistics of Patch Offsets for Image Completion - Kaiming He and Jian Sun
A Python Implementation - Pranshu Gupta and Shrija Mishra
"""

import cv2
import sys
import plot
import kdtree
import energy
import operator
import numpy as np
import config as cfg
from time import time
from scipy import ndimage
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
        cv2.rectangle(mask, (bbox[2], bbox[0]), (bbox[3], bbox[1]), (255,255,255), 1)
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
            if i not in xrange(hole[2]-cfg.PATCH_SIZE/2, hole[3]+cfg.PATCH_SIZE/2) and j not in xrange(hole[0]-cfg.PATCH_SIZE/2, hole[1]+cfg.PATCH_SIZE/2):
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

def GetOffsets(patches, indices):
    start = time()
    kd = kdtree.KDTree(patches, leafsize=cfg.KDT_LEAF_SIZE, tau=cfg.TAU)
    dist, offsets = kdtree.get_annf_offsets(patches, indices, kd.tree, cfg.TAU)
    end = time()
    print "GetOffsets execution time: ", end - start
    return offsets

def GetKDominantOffsets(offsets, K, height, width):
    start = time()
    x, y = [offset[0] for offset in offsets if offset != None], [offset[1] for offset in offsets if offset != None]
    bins = [[i for i in range(np.min(x),np.max(x))], [i for i in xrange(np.min(y),np.max(y))]]
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    hist = hist.T
    # plot.PlotHistogram2D(hist, xedges, yedges)
    p, q = np.where(hist == cv2.dilate(hist, np.ones(8))) # Non Maximal Suppression
    nonMaxSuppressedHist = np.zeros(hist.shape)
    nonMaxSuppressedHist[p, q] = hist[p, q]
    # plot.PlotHistogram2D(nonMaxSuppressedHist, xedges, yedges)
    p, q = np.where(nonMaxSuppressedHist >= np.partition(nonMaxSuppressedHist.flatten(), -K)[-K])
    peakHist = np.zeros(hist.shape)
    peakHist[p, q] = nonMaxSuppressedHist[p, q]
    # plot.PlotHistogram2D(peakHist, xedges, yedges)
    peakOffsets, freq = [[xedges[j], yedges[i]] for (i, j) in zip(p, q)], nonMaxSuppressedHist[p, q].flatten()
    peakOffsets = np.array([x for _, x in sorted(zip(freq, peakOffsets), reverse=True)], dtype="int64")[:K]
    end = time()
    # plot.ScatterPlot3D(peakOffsets[:,0], peakOffsets[:,1], freq, [height, width])
    print "GetKDominantOffsets execution time: ", end - start
    return peakOffsets 

def GetOptimizedLabels(image, mask, labels):
    start = time()
    optimizer = energy.Optimizer(image, mask, labels)
    optimalLabels = optimizer.OptimizeLabelling()
    end = time()
    print "GetOptimizedLabels execution time: ", end - start
    return optimalLabels 

def CompleteImage(image, mask, offsets, optimalLabels):
    failedPoints = np.zeros(image.shape)
    completedPoints = np.zeros(image.shape)
    x, y = np.where(mask != 0)
    sites = [[i, j] for (i, j) in zip(x, y)]
    finalImg = image
    for i in xrange(len(sites)):
        j = optimalLabels[i]
        try:
            finalImg[sites[i][0], sites[i][1]] = image[sites[i][0] + offsets[j][0], sites[i][1] + offsets[j][1]]
            completedPoints[sites[i][0], sites[i][1]] = finalImg[sites[i][0], sites[i][1]]
        except:
            failedPoints[sites[i][0], sites[i][1]] = [255,255,255]
    return finalImg, failedPoints, completedPoints

def PoissonBlending(image, mask, center):
    src = cv2.imread(cfg.OUT_FOLDER + cfg.IMAGE + "_CompletedPoints.png")
    dst = cv2.imread(cfg.OUT_FOLDER + cfg.IMAGE + "_Complete.png")
    blendedImage = cv2.seamlessClone(src, dst, mask, center, cv2.MIXED_CLONE)
    return blendedImage


def main(imageFile, maskFile):
    """
    Image Completion Pipeline
        1. Patch Extraction
        2. Patch Offsets
        3. Image Stacking
        4. Blending
    """
    # image = cv2.resize(cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE), (0,0), fx=0.5, fy=0.5)
    # imageR = cv2.resize(cv2.imread(imageFile), (0,0), fx=0.5, fy=0.5)
    # mask = cv2.resize(cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE), (0,0), fx=0.5, fy=0.5)
    # bb = GetBoundingBox(mask)
    image = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
    imageR = cv2.imread(imageFile)
    mask = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
    bb = GetBoundingBox(mask)
    bbwidth = bb[3] - bb[2]
    bbheight = bb[1] - bb[0]
    cfg.TAU = max(bbwidth, bbheight)/15
    cfg.DEFLAT_FACTOR = image.shape[1]
    sd = GetSearchDomain(image.shape, bb)
    indices, patches = GetPatches(image, sd, bb)
    reducedPatches = ReduceDimension(patches)
    offsets = GetOffsets(reducedPatches, indices)
    kDominantOffset = GetKDominantOffsets(offsets, 60, image.shape[0], image.shape[1])
    optimalLabels = GetOptimizedLabels(imageR, mask, kDominantOffset)
    completedImage, failedPoints, completedPoints = CompleteImage(imageR, mask, kDominantOffset, optimalLabels)
    cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_Complete.png", completedImage)
    # cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_Failed.png", failedPoints)
    cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_CompletedPoints.png", completedPoints)
    center = (bb[2]+bbwidth/2, bb[0]+bbheight/2)
    blendedImage = PoissonBlending(imageR, mask, center)
    cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_blendedImage.png", blendedImage)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python main.py image_name mask_file_name"
        exit()
    cfg.IMAGE = sys.argv[1].split('.')[0]
    imageFile = cfg.SRC_FOLDER + sys.argv[1]
    print imageFile
    maskFile = cfg.SRC_FOLDER + sys.argv[2]
    main(imageFile, maskFile)
    