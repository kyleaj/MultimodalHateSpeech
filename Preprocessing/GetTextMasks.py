import os
import sys
import cv2
import time
import numpy as np
import easyocr

image_dir = sys.argv[1]
out_dir = sys.argv[2]

debug_dir = None
if (len(sys.argv) == 4):
    print("Debug out on")
    debug_dir = sys.argv[3]

reader = easyocr.Reader(['en'])

start_time = time.time()

ims = os.listdir(image_dir)
im_num = len(ims)

failed = []

remaining = -1

for i, im in enumerate(ims):
    outpath = os.path.join(out_dir, im + "_mask.npy")
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
            print(result)
            a, b, c, d = result[0]
            min_x = min(a[0], b[0], c[0], d[0])
            min_y = min(a[1], b[1], c[1], d[1])
            max_x = max(a[0], b[0], c[0], d[0])
            max_y = max(a[1], b[1], c[1], d[1])
            mask[min_y:max_y,min_x:max_x,:] = 1

        if not(debug_dir is None):
            cv2.imwrite(filename + "_masked.png", im*mask)
            cv2.imwrite(filename + "_un_masked.png", im*(1-mask))
        np.save(outpath, mask)

        elapsed = time.time() - start_time
        speed = (i+1) / elapsed
        remaining = (im_num - i - 1) / speed
        remaining = int(remaining * 100) / 100.0
    except Exception as e:
        print(e)
        failed.append(e)
        failed.append(im)

    print(str(i+1) + " / " + str(im_num) + ", about " + str(remaining) + "s left.")
    sys.stdout.write("\033[F") # Cursor up one line

print()
print()
print()
print(failed)
