import glob, os
with open("train.txt", "w") as f:
    os.chdir("./augmentation/train/")
    for file in glob.glob("*.jpg"):
        f.write(os.path.abspath(file) + "\n")
