import Augmentor
import numpy as np
from PIL import Image
import glob
from natsort import natsorted
import os
import random
import matplotlib.pyplot as plt
import cv2

debug = False  # print debug info or not
count = 100  # num of images to generate
batch = 50  # size of a single batch
begin = 0  # indicate the current index, use only if you are continuing a crashed generation
image_dir = "image"
mask_dir = "mask"
extension = "*.png"
out_dir = "augmentation"  # out directory
background_file = "indoorCVPR_09/*/*.jpg"
mask_list = ["rune_success", "rune_shoot", "rune_target", "rune_center", "rune_blank"]
mean = 0  # the mean of gaussian noise

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# this function is used to lower the brightness of background image
def decrease_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = value
    v[v < lim] = 0
    v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# import img and corresponding masks in natural order
images = natsorted(glob.glob(os.path.join(image_dir, extension)))
overall_list = [images]
for mask_name in mask_list:
    mask_path = os.path.join(mask_dir, mask_name, extension)
    masks = natsorted(glob.glob(mask_path))
    print(mask_name + ":" + str(len(masks)))
    overall_list.append(masks)

# TODO: Edit according to class number
collated_images_and_masks = list(zip(overall_list[0], overall_list[1], overall_list[2], overall_list[3], overall_list[4], overall_list[5]))
images = [[np.asarray(Image.open(y)) for y in x] for x in collated_images_and_masks]
print("image and masks loaded!")

if debug:
    print(collated_images_and_masks)

# import background
bg_list = []
for bg in glob.glob(background_file):
    bg_list.append(bg)
print("background list load: " + str(len(bg_list)))

# define augmentor
p = Augmentor.DataPipeline(images)
p.zoom(1, 0.3, 1.0)
p.zoom_random(1, .9)
p.skew(1, .8)
p.flip_random(1)
p.random_distortion(.3, 10, 10, 7)
p.random_color(1, .3, 1.2)
p.random_contrast(1, .5, 1.5)
p.random_brightness(1, 0.5, 1.5)
p.shear(.5, 15, 15)
# p.random_erasing(.75, 0.25)
p.rotate_random_90(1)
p.rotate(1, max_left_rotation=15, max_right_rotation=15)

# begin generation
for i in range(begin//batch, count//batch):
    print(str(i) + " st batch begin")
    augmented_images = p.sample(batch)
    print(str(i) + " st batch sampled")

    if debug:
        r_index = random.randint(0, len(augmented_images) - 1)
        f, axarr = plt.subplots(1, 3, figsize=(20, 15))
        axarr[0].imshow(augmented_images[r_index][0])
        axarr[1].imshow(augmented_images[r_index][1], cmap="gray")
        axarr[2].imshow(augmented_images[r_index][2], cmap="gray")
        plt.show()

    # export alpha (transparency) from original png for better merging
    for j in range(batch):
        alpha = None
        if augmented_images[j][0].shape[2] == 4:
            _, _, _, a = cv2.split(augmented_images[j][0])
            alpha = []
            for ii, x in enumerate(a):
                alpha.append([])
                for jj, y in enumerate(x):
                    alpha[ii].append([y, y, y])
            alpha = np.asarray(alpha)
            # print(alpha)

        bg_path = random.choice(bg_list)  # choose a background randomly
        background = cv2.imread(bg_path)

        # sometimes cv2 return null... handle exception
        if background is None:
            print(str(i*batch+j) + " st background is None")
            continue

        # resize background to the size of mask
        background = cv2.resize(background, (1000, 1000), interpolation=cv2.INTER_CUBIC)
        background = decrease_brightness(background, value=np.random.randint(30,100))  # randomly reduce the brightness of bg
        original = cv2.cvtColor(augmented_images[j][0], cv2.COLOR_RGB2BGR)

        # if no alpha data, create on from threshholding
        if alpha is None:
            # alpha blend image to background
            orig_grey = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            orig_grey[orig_grey > 0] = 255  # get blending mask

            # perform eroding on mask to avoid black edge around image
            k = random.randint(3, 6)  # random erode size
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.erode(orig_grey, kernel)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            alpha = mask.astype(float)
        else:
            alpha = alpha.astype(float)

        foreground = original.astype(float)
        background = background.astype(float)

        # alpha blending
        alpha = alpha / 255
        # print(alpha)
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)
        outImage = cv2.addWeighted(foreground, 1, background, 1, 0)

        # resize output image to 512, 512 due to limitation of computing power
        outImage = outImage / 255
        outImage = cv2.resize(outImage, (512, 512), interpolation=cv2.INTER_AREA)

        # sometimes alpha blend causes overflow; skip if overflow happens
        result_eval = np.where(outImage.flatten() < 0.04)
        # print(len(result_eval[0]))
        if len(result_eval[0]) > 0.5 * 512 * 512 * 3:
            print(str(i*batch+j) + " st image overflow")
            continue

        outImage = outImage * 255
        # transform image back to 8 bit int
        outImage = np.uint8(outImage)
        outImage = np.int16(outImage)

        upper_bound = 255 - outImage
        lower_bound = 0 - outImage

        # adding gaussian noise
        row, col, ch = outImage.shape
        sigma = np.random.randint(40)
        gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.int8)
        gauss = np.clip(gauss, lower_bound, upper_bound)
        outImage = np.add(outImage, gauss, casting='unsafe')
        outImage = np.uint8(outImage)

        if debug:
            cv2.imshow("output", outImage)
            cv2.waitKey(0)

        # save image and corresponding masks
        cv2.imwrite(os.path.join(out_dir, str(0) + "_" + str(i*batch+j)) + ".jpg", outImage)
        for index in range(1, len(mask_list) + 1):
            path = os.path.join(out_dir, str(index) + "_" + str(i*batch+j) + ".png")
            cv2.imwrite(path, augmented_images[j][index])

        print(str(i*batch+j) + " st image success")
    print(str(i) + " st batch finish")
