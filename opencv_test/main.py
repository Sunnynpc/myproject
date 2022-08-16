import cv2
import numpy as np
from skimage import exposure

gray = cv2.imread('retval.png',cv2.IMREAD_GRAYSCALE)
# 二值图
# ret,img1 = cv2.threshold(img,180,255,cv2.THRESH_BINARY_INV)
# ret,img2 = cv2.threshold(img,180,255,cv2.THRESH_BINARY)
# img1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,2)
# img2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,2)
# cv2.imwrite('./threshold1.png',np.float32(img1))
# cv2.imwrite('./threshold2.png',np.float32(img2))


# 抠出菌落范围
# circle = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 4000, param1=40, param2=20, minRadius=1500, maxRadius=1600)
#
# circle = np.int0(circle[0])
# x, y, radius = circle[0]
# radius = radius-50
# # print(circle)
# ry = cv2.imread('retval.png')
# w,h = gray.shape
# for i in range(w):
#     for j in range(h):
#         if np.sqrt(np.square(i-y)+np.square(j-x))>radius:
#             ry[i,j]=0
# cv2.imwrite('./ry.png',ry)
# cv2.imshow('ry',ry)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


ry = cv2.imread('./ry.png',cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3,3),np.uint8)
# ry = exposure.adjust_gamma(ry, 10.2)
# ry_g = cv2.cvtColor(ry,cv2.COLOR_BGR2GRAY)
# ry_g[ry_g>200] = 0
# ry_g[ry_g<50] = 0


ry_g = cv2.adaptiveThreshold(ry,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,2)
ry_g = cv2.medianBlur(ry_g ,3)
ry_g = cv2.dilate(ry_g,kernel,iterations = 2)

cv2.imwrite('./thresh1.png',ry_g)
temp = cv2.imread('test.png',cv2.IMREAD_COLOR)
contours,hierarchy= cv2.findContours(ry_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ry_contours = contours
# dddd = cv2.drawContours(temp,ry_contours,-1,(0,0,255),2)
# cv2.imwrite("./count.png",dddd)
# cv2.imshow('ry_canny',temp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cclist = []
for i in ry_contours:
    # print(i.shape[0])
    if i.shape[0] < 85 and i.shape[0] > 5:
        cclist.append(i)
rry = temp.copy()
print(len(cclist))
dddd11 = cv2.drawContours(rry, cclist, -1, (255, 0, 255), 2)
cv2.imwrite("./count.png",dddd11)
cv2.namedWindow('ry_canny',cv2.WINDOW_NORMAL)
cv2.imshow('ry_canny',dddd11)
cv2.waitKey(0)
cv2.destroyAllWindows()


