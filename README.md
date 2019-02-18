# Labelme to Data-Augmentation to Mask-RCNN Pipeline

## Introduction

In this repository, we present a pipeline that augments datasets of vary limited samples (eg. 1 sample for each class) to a larger dataset (eg. 5000 samples for each class) that Mask R-CNN can directly read.  
  
Original image:  
<img src="examples/vlcsnap-2019-01-22-14h11m20s882.png" width="300">  
Augumented image:  
<img src="examples/0_568.jpg" width="300">  
  
Note that this method should only be applied on very simple objects. You cannot expect dataset with 1 sample per class to work on complex environment.  

## Pipeline

1. Use [Labelme](https://github.com/wkentaro/labelme/) libarary to generate annotated json file for each sample image. Please use only polygon or circle to create the mask.
<img src="examples/labelme.png" width="300">

2. Use `labelme2png.py` to create png binary mask for each class. It only suports polygon and circle for now. The script takes the json file from last step as input.  
<img src="examples/vlcsnap-2019-01-22-14h11m20s882_rune_success.png" width="100"> <img src="examples/vlcsnap-2019-01-22-14h11m20s882_rune_target.png" width="100"> ...

3. Use `data_augmentation.py` to augment the sample images with their masks.

TO BE COMPLETE
