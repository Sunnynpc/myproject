import cv2
import numpy as np

# x = np.uint8([250])
# y = np.uint8([10])
#
# print(cv2.add(x,y))
# print(cv2.subtract(y,x))
img1 = cv2.imread("images/1.jpg")
img2 = cv2.imread("images/6.jpg")

# img = cv2.add(img1,img2)
# img = cv2.subtract(img1,img2)
img = cv2.addWeighted(img1,0.7,img2,0.3,0)
cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()