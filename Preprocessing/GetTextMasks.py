import os
import sys
import cv2
import time
import numpy as np
import easyocr

image_dir = sys.argv[1]
out_dir = sys.argv[2]

reader = easyocr.Reader(['en'])

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

    try:
        print("Getting boxes")
        print(filename)
        results = reader.readtext(filename)
        im = cv2.imread(filename)
        mask = np.zeros_like(im)
        for result in results:
            a, b, c, d, _, _ = result
            min_x = min(a[0], b[0], c[0], d[0])
            min_y = min(a[1], b[1], c[1], d[1])
            max_x = max(a[0], b[0], c[0], d[0])
            max_y = max(a[1], b[1], c[1], d[1])
            mask[min_y:max_y,min_x:max_x,:] = 1

        im = im*mask
        cv2.imwrite("masked.png", im)

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