# 菌落识别
import cv2
import numpy as np

img = cv2.imread('1.png')  # shape(3648, 5472, 3)
# cv2.namedWindow('pic1', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('pic1', img.shape[1] // 5, img.shape[0] // 5)
# cv2.imshow('pic1', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 中值滤波（去除噪点）
img2 = cv2.medianBlur(img, 9)
# cv2.imwrite('blur.png', img2)


# 霍夫圆运算（寻找范围）
img3 = cv2.imread('2.png')
gray = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)

# circle = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 4000, param1=40, param2=20, minRadius=1500, maxRadius=1600)
# [2867.5 1731.5 1545.5]
# [2868.5 1732.5 1703.4]

# if not circle is None:
#     circle = np.uint16(np.around(circle))
#     for i in circle[0, :]:
#         cv2.circle(img3, (i[0], i[1]), i[2], (0, 255, 0), 4)
# cv2.imwrite('3.png', img3)


# 自适应二值化（识别杂质）
ret, thresh_const = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 2)

# cv2.imwrite('thresh_test.png', np.hstack([thresh_const, thresh_adaptive]))

# 矩阵运算叠加原图（识别结果）
# 圆的坐标和半径
x, y, r = 2868.5, 1732.5, 1450


# 计算像素坐标是否在圆范围，返回布朗值
def judge(i, j):
    distance = np.sqrt((y - i) ** 2 + (x - j) ** 2)
    return distance < r


# 圆边界计算
border = np.fromfunction(judge, thresh_adaptive.shape)
# 叠加检测结果矩阵
border = np.where(thresh_adaptive > 0, border, False)

b, g, r = cv2.split(img2)
b = np.where(border, 255, b)
g = np.where(border, 0, g)
r = np.where(border, 255, r)

# 合并三个通道得到最终图
img_res = cv2.merge([b, g, r])
cv2.imwrite('result.png', img_res)
