"""
Statistics of Patch Offsets for Image Completion - Kaiming He and Jian Sun
A Python Implementation - Pranshu Gupta and Shrija Mishra
"""

import cv2
import sys
import numpy as np
import config as cfg
from sklearn.feature_extraction import image

def get_bounding_box(mask):
    """
    Get Bounding Box for a Binary Mask
    Arguments: mask - a binary mask
    Returns: col_min, col_max, row_min, row_max
    """
    a = np.where(mask != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    if cfg.PRINT_BB_IMAGE:
        cv2.rectangle(mask, (bbox[2], bbox[0]), (bbox[3], bbox[1]), (0,0,255), 1)
        cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + cfg.BB_IMAGE_SUFFIX, mask)
    return bbox

def get_search_domain(image, bbox):
    """
    get a rectangle that is 3 times larger (in length) than the bounding box of the hole
    """
    col_min, col_max = max(0, 2*bbox[0] - bbox[1]), min(2*bbox[1] - bbox[0], image.shape[1]-1)
    row_min, row_max = max(0, 2*bbox[2] - bbox[3]), min(2*bbox[3] - bbox[2], image.shape[0]-1)
    return col_min, col_max, row_min, row_max

def get_patches(images, bbox):
    region = images[bbox[2]:bbox[3], bbox[0]:bbox[1]]
    patches = image.extract_patches_2d(region, cfg.PATCH_SIZE))
    return patches

def main(imageFile, maskFile):
    """
    Image Completion Pipeline - Image and Mask 
    """
    image = cv2.imread(imageFile)
    mask = cv2.imread(maskFile)
    bb = get_bounding_box(mask)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python main.py image_name mask_file_name"
        exit()
    cfg.IMAGE = sys.argv[1].split('.')[0]
    imageFile = cfg.SRC_FOLDER + sys.argv[1]
    maskFile = cfg.SRC_FOLDER + sys.argv[2]
    main(imageFile, maskFile)
    