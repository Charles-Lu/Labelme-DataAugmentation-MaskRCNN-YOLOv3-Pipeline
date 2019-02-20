import glob
from natsort import natsorted
import os
import numpy.random as random

dataset_dir = "augmentation"
num_class = 5
extension = "*.png"
val = 0.2

val_dir = os.path.join(dataset_dir, "val")
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

images = natsorted(glob.glob(os.path.join(dataset_dir, "*.jpg")))
overall_list = [images]
for mask_index in range(1, num_class + 1):
    mask_name = os.path.join(dataset_dir, str(mask_index) + extension)
    masks = natsorted(glob.glob(mask_name))
    overall_list.append(masks)

for index in range(len(images)):
    if random.choice([0,1], p=[1-val, val]) == 1:
        for num in range(len(overall_list)):
            img = overall_list[num][index]
            img_name = os.path.basename(img)
            os.rename(img, os.path.join(val_dir, img_name))
