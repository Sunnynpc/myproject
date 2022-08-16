import cv2
import numpy as np

src = cv2.imread("images/1.jpg")

h,w,c = src.shape
print(w,h)
# M = np.array([[1.,0.,50.],[0.,1.,50.]])
# M = np.array([[1.2,0.,0.],[0.,0.8,0.]])
# M = np.array([[1.,0.8,0.],[0,1,0]])
M = cv2.getRotationMatrix2D((w/2,h/2),45,0.7)
dst = cv2.warpAffine(src,M,(w,h))

cv2.imshow("src",src)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()