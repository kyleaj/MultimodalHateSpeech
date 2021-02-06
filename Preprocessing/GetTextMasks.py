import PIL
from PIL import Image
import pytesseract
import os
import sys
import cv2
import time
import numpy as np

image_dir = sys.argv[1]
out_dir = sys.argv[2]

pytesseract.pytesseract.tesseract_cmd = r'path'

start_time = time.time()

ims = os.listdir(image_dir)
im_num = len(ims)

failed = []

remaining = -1

for i, im in enumerate(ims):
    outpath = os.path.join(out_dir, im + ".npy")
    if (os.path.exists(outpath)) or not("jpg" in im.lower() or "png" in im.lower() or "jpeg" in im.lower()):
        continue

    filename = os.path.join(image_dir, im)
    input_image = Image.open(filename)

    try:
        print("Getting boxes")
        print(filename)
        boxes = pytesseract.image_to_boxes(Image.open(filename))
        print(type(boxes))
        print(boxes)
        exit(0)

        output = output[0]
        output = output.cpu().numpy()
        
        np.save(outpath, output)

        elapsed = time.time() - start_time
        speed = (i+1) / elapsed
        remaining = (im_num - i - 1) / speed
        remaining = int(remaining * 100) / 100.0
    except Exception as e:
        failed.append(e)
        failed.append(im)

    #print(str(i+1) + " / " + str(im_num) + ", about " + str(remaining) + "s left.")
    #sys.stdout.write("\033[F") # Cursor up one line