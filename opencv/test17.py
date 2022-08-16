import cv2
import numpy as np

img = cv2.imread("images/1.jpg")
kernel = np.array([[1,1,0],[1,0,-1],[0,-1,-1]],np.float32)
dst = cv2.filter2D(img,-1,kernel)
cv2.imshow("img",img)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()