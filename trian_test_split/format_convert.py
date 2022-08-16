import cv2
from PIL import Image
import os
# import cfg

filelist = os.listdir('../data/images')


def other2jpg(path):
    img = Image.open(path)

    img = img.convert("RGB")
    img.save(path, "JPEG", quality=80, optimize=True, progressive=True)
    print("convert successful")


for filename in filelist:
    paths = os.path.join('../data/images', filename)
    img_data = Image.open(paths)


    # print(img_data.format)
    if img_data.format != 'JPEG':
        print(filename + ' ' + img_data.format)  # WEBP PNG
        other2jpg(paths)

    # print(img_data.mode)
    if img_data.mode != 'RGB':
        print(filename + ' ' + img_data.format)
