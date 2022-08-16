import cv2

src = cv2.imread("images/1.jpg")

src[...,0] = 0
src[...,2] = 0
cv2.imshow("src",src)
cv2.waitKey(0)
cv2.destroyAllWindows()