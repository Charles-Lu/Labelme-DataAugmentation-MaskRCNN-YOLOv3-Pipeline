# Labelme-to-Binary-PNG-Mask
A short script to convert labelme pic and json file to binary PNG masks by classes. Only support polygon and circle yet.\
\
Input: \
&nbsp;&nbsp;&nbsp;&nbsp;path to directory of png and corresponding json file\
Output: \
&nbsp;&nbsp;&nbsp;&nbsp;for filename in dir:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for label in all_labels:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;filename_label.png in the format of binary mask\
        \
Refer to https://github.com/wkentaro/labelme for more infomation.
