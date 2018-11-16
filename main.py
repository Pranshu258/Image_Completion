"""
Statistics of Patch Offsets for Image Completion - Kaiming He and Jian Sun
A Python Implementation - Pranshu Gupta and Shrija Mishra
"""

import cv2
import sys
import numpy as np
import config as cfg

def get_bounding_box(mask):
    """Get Bounding Box for a Binary Mask"""
    a = np.where(mask != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    if cfg.PRINT_BB_IMAGE:
        cv2.rectangle(mask, (bbox[2], bbox[0]), (bbox[3], bbox[1]), (0,0,255), 1)
        cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + cfg.BB_IMAGE_SUFFIX, mask)
    return bbox

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
    