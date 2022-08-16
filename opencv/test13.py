import cv2
import numpy as np

src = cv2.imread("images/25.jpg")

pts1 = np.float32([[25,30], [179, 25], [12, 188], [189, 190]])
pts2 = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(src,M,(200,200))
cv2.imshow("src",src)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()