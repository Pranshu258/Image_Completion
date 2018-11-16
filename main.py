"""
STATISTICS OF PATCH OFFSETS FOR IMAGE COMPLETION - KAIMING HE & JIAN SUN
"""

import cv2
import numpy as np

SRC_FOLDER = "images/source/"
OUT_FOLDER = "images/output/"
IMAGE = "7796"

def get_bounding_box(mask):
    a = np.where(mask != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

def main (imageFile, maskFile):
    image = cv2.imread(imageFile)
    mask = cv2.imread(maskFile)
    bb = get_bounding_box(mask)
    print bb

if __name__ == "__main__":
    imageFile = SRC_FOLDER + IMAGE + ".jpg"
    maskFile = SRC_FOLDER + IMAGE + ".png"
    main(imageFile, maskFile)
    