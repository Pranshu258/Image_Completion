"""
Statistics of Patch Offsets for Image Completion - Kaiming He and Jian Sun
A Python Implementation - Pranshu Gupta and Shrija Mishra
"""

import cv2
import sys
import numba
import numpy as np
import config as cfg
from time import time
from sklearn.feature_extraction import image

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
    """
    start = time()
    col_min, col_max = max(0, 2*bbox[0] - bbox[1]), min(2*bbox[1] - bbox[0], shape[1]-1)
    row_min, row_max = max(0, 2*bbox[2] - bbox[3]), min(2*bbox[3] - bbox[2], shape[0]-1)
    end = time()
    print "get_search_domain execution time: ", end - start
    return col_min, col_max, row_min, row_max

def get_patches(image, bbox, hole):
    start = time()
    indices, patches = [], []
    rows, cols = image.shape
    for i in xrange(bbox[2]+cfg.PATCH_SIZE/2, bbox[3]-cfg.PATCH_SIZE/2):
        for j in xrange(bbox[0]+cfg.PATCH_SIZE/2, bbox[1]-cfg.PATCH_SIZE/2):
            if i not in xrange(hole[2]-cfg.PATCH_SIZE/2, hole[3]+cfg.PATCH_SIZE/2) and j not in xrange(hole[0]-cfg.PATCH_SIZE/2, hole[1]+cfg.PATCH_SIZE/2):
                indices.append([i,j])
                patches.append(image[i-cfg.PATCH_SIZE/2:i+cfg.PATCH_SIZE/2, j-cfg.PATCH_SIZE/2:j+cfg.PATCH_SIZE/2])
    end = time()
    print "get_patches execution time: ", end - start
    return np.array(indices), np.array(patches).reshape(len(patches), cfg.PATCH_SIZE**2)


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
    sd = get_search_domain(image.shape, bb)
    indices, patches = get_patches(image, sd, bb)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python main.py image_name mask_file_name"
        exit()
    cfg.IMAGE = sys.argv[1].split('.')[0]
    imageFile = cfg.SRC_FOLDER + sys.argv[1]
    maskFile = cfg.SRC_FOLDER + sys.argv[2]
    main(imageFile, maskFile)
    