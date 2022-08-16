import cv2
import numpy as np

img = cv2.imread("images/5.jpg")

# dst = cv2.blur(img,(5,5))
# dst = cv2.GaussianBlur(img,(5,5),0)
dst = cv2.medianBlur(img, 5)
cv2.imshow("img",img)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()