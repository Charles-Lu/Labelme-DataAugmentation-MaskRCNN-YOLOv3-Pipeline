# Labelme-to-Binary-PNG-Mask
A short script to convert labelme pic and json file to binary PNG masks by classes. Only support polygon and circle yet.

Input: 
  path to directory of png and corresponding json file
Output: 
  for filename in dir:
    for label in all_labels:
        filename_label.png in the format of binary mask
        
Refer to https://github.com/wkentaro/labelme for more infomation.
