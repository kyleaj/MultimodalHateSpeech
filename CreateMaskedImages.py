import cv2
import sys
import os
import numpy as np

im_dir = sys.argv[1]
binary_mask_dir = sys.argv[2]
out_im_dir = sys.argv[3]
out_mask_dir = sys.argv[4]

for f in os.listdir(im_dir):
    im = cv2.imread(os.path.join(im_dir, f))
    binp = os.path.join(binary_mask_dir, f+"_mask.npy")
    if not(os.path.exists(binp)):
        continue
    bin_mask = np.load(binp)

    im[bin_mask>0] = 255
    cv2.imwrite(os.path.join(out_im_dir, f), im)
    cv2.imwrite(os.path.join(out_mask_dir, "mask_"+f), (bin_mask*255).astype(np.uint8))

