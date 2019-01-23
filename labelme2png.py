import glob
import os
import random
import json
# from shapely import geometry
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import math

# save mask in "8_bit_binary" or "1_bit_binary", 
# 8-bit mode has better compatibility in general
mode = "8_bit_binary"
in_dir = "data\\raw"
out_dir = "data\\mask"

if mode == "1_bit_binary":
    pil_mode = "1"
    pil_fill = 1
elif mode == "8_bit_binary":
    pil_mode = "L"
    pil_fill = 255


for file in glob.glob(os.path.join(in_dir, "*.png")):
    im = Image.open(file)
    width, height = im.size
    
    # Find png and corresponding json file by name
    general_path = file.strip(".png")
    general_name = os.path.basename(general_path)
    json_path = general_path + ".json"
    json_data = open(json_path).read()
    data = json.loads(json_data)
    
    shapes = data["shapes"]
    masks = {}
    for shape in shapes:
        label = shape["label"]
        points = shape["points"]
        shape_type = shape["shape_type"]
        
        # create new mask if the class appear for first time
        if label not in masks:
            masks[label] = Image.new(pil_mode, (width, height))
        draw = ImageDraw.Draw(masks[label])
        
        # TODO: Support more format. Only support polygon and circle yet
        if shape_type == "polygon":
            p = [(p[0], p[1]) for p in points]
            draw.polygon(p, fill=pil_fill)
        elif shape_type == "circle":
            center = points[0]
            another = points[1]
            radius = math.hypot(another[0] - center[0], another[1] - center[1])
            draw.ellipse((center[0] - radius, center[1] - radius,
                          center[0] + radius, center[1] + radius), fill=pil_fill)

    for label, mask in masks.items():
        mask_name = "_".join([general_name, label])
        mask_path = os.path.join(out_dir, mask_name)
        mask.save(mask_path + ".png", "PNG")
