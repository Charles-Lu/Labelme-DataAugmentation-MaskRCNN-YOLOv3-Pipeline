import glob
from natsort import natsorted
import cv2
import os

debug = False
mode = "both"  # "binary", "resize", "both"
dir = "augmentation"

masks = natsorted(glob.glob(os.path.join(dir, "*.png")))

num = len(masks)

count = 0
for m in masks:

    result = cv2.imread(m)
    if mode == "resize" or mode == "both":
        result = cv2.resize(result, (512, 512), interpolation=cv2.INTER_AREA)
    if mode == "binary" or mode == "both":
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(result, 50, 255, cv2.THRESH_BINARY)
    cv2.imwrite(m, result)
    if count % 100 == 0:
        print(count/num)
    count += 1
