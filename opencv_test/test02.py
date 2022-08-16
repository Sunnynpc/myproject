import cv2
import numpy as np

img = cv2.imread('test.png')
# 高斯滤波
# retval = cv2.GaussianBlur(img,(9,9),0)
# 中值滤波
retval = cv2.medianBlur(img, 9)
# cv2.namedWindow('retval', cv2.WINDOW_NORMAL)
cv2.imwrite('./retval.png',retval)