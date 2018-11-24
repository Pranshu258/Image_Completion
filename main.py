"""
Statistics of Patch Offsets for Image Completion - Kaiming He and Jian Sun
A Python Implementation - Pranshu Gupta and Shrija Mishra
"""

import cv2
import sys
import numpy as np
import config as cfg
from time import time
from sklearn.decomposition import PCA
import kdtree

def get_bounding_box(mask):
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
    print "get_bounding_box execution time: ", end - start
    return bbox

def get_search_domain(shape, bbox):
    """
    get a rectangle that is 3 times larger (in length) than the bounding box of the hole
    this is the region which will be used for the extracting the patches
    """
    start = time()
    col_min, col_max = max(0, 2*bbox[0] - bbox[1]), min(2*bbox[1] - bbox[0], shape[1]-1)
    row_min, row_max = max(0, 2*bbox[2] - bbox[3]), min(2*bbox[3] - bbox[2], shape[0]-1)
    end = time()
    print "get_search_domain execution time: ", end - start
    return col_min, col_max, row_min, row_max

def get_patches(image, bbox, hole):
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
    print "get_patches execution time: ", end - start
    return np.array(indices), np.array(patches, dtype='int64').reshape(len(patches), cfg.PATCH_SIZE**2)

def reduce_dimension(patches):
    start = time()
    pca = PCA(n_components=cfg.PCA_COMPONENTS)
    reduced_patches = pca.fit_transform(patches)
    end = time()
    print "reduce_dimension execution time: ", end - start
    return reduced_patches

def get_offsets(patches):
    start = time()
    kd = kdtree.KDTree(patches, leafsize=cfg.KDT_LEAF_SIZE, tau=cfg.TAU, deflat_factor=cfg.DEFLAT_FACTOR)
    dist, offsets = kdtree.get_annf_offsets(patches, kd.tree, cfg.DEFLAT_FACTOR, cfg.TAU)
    end = time()
    print "get_offsets execution time: ", end - start
    return

def get_offset_statistics(offsets):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = offsets[0,:], offsets[:,0]
    hist, xedges, yedges = np.histogram2d(x, y, bins = 10, range=[[0, 4], [0, 4]])
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
    bb = get_bounding_box(mask)
    bbwidth = bb[3] - bb[2]
    bbheight = bb[1] - bb[0]
    cfg.TAU = max(bbwidth, bbheight)/15
    cfg.DEFLAT_FACTOR = image.shape[1]
    sd = get_search_domain(image.shape, bb)
    indices, patches = get_patches(image, sd, bb)
    reducedPatches = reduce_dimension(patches)
    offsets = get_offsets(reducedPatches)
    # dominantOffset = get_offset_statistics(offsets)
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python main.py image_name mask_file_name"
        exit()
    cfg.IMAGE = sys.argv[1].split('.')[0]
    imageFile = cfg.SRC_FOLDER + sys.argv[1]
    maskFile = cfg.SRC_FOLDER + sys.argv[2]
    main(imageFile, maskFile)
    