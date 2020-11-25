from PIL import Image
import glob
import cv2 as cv
from shutil import copyfile

path = glob.glob("./images/*.png")

for img in path:
    alist = img[9:]
    if alist[0] != 't':
        pass
    else:
        print(alist)
        if '-' in alist:
            copyfile(img, "./train_A/" + alist)
            print(True)
        else:
            copyfile(img, "./train_B/" + alist)