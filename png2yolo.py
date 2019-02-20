import glob
from natsort import natsorted
import os
import cv2
import numpy as np

dataset_dir = "augmentation\\train"
extension = "*.png"
num_class = 5

images = natsorted(glob.glob(os.path.join(dataset_dir, "*.jpg")))
overall_list = [images]
for mask_index in range(1, num_class + 1):
    mask_name = os.path.join(dataset_dir, str(mask_index) + extension)
    masks = natsorted(glob.glob(mask_name))
    overall_list.append(masks)


for i in range(len(images)):
    name = images[i].strip(".jpg") + ".txt"
    with open(name, "a") as f:
        annotation = []
        for j in range(len(overall_list)):
            mask = cv2.imread(overall_list[j][i])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            row = cv2.reduce(mask, 0, cv2.REDUCE_MAX).flatten()
            column = cv2.reduce(mask, 1, cv2.REDUCE_MAX).flatten()
            row_nonzero = np.nonzero(row)[0]
            column_nonzero = np.nonzero(column)[0]
            if not row_nonzero.any() or not column_nonzero.any():
                continue
            x_min = row_nonzero[0]
            x_max = row_nonzero[-1]
            y_min = column_nonzero[0]
            y_max = column_nonzero[-1]

            abs_x = (x_min + x_max) / 2
            abs_y = (y_min + y_max) / 2
            abs_w = x_max - x_min
            abs_h = y_max - y_min
            w, h = 512, 512

            bbox = [j, abs_x/w, abs_y/h, abs_w/w, abs_h/h]
            # print(bbox)
            annotation.append(" ".join(str(x) for x in bbox))

        line = "\n".join(annotation)
        f.write(line)
    if i % 100 == 0:
        print(i / len(images))

