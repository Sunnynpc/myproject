import cv2
import numpy as np

img = cv2.imread('retval.png')

kernel = np.ones(3,np.uint8)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

img = cv2.erode(img,None,iterations = 5)
# img = cv2.dilate(img,kernel,iterations = 3)
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


cv2.imwrite('./result.png',img)
# cv2.namedWindow('result',cv2.WINDOW_NORMAL)
# cv2.imshow('result',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()