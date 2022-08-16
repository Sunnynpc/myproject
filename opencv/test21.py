import cv2

img = cv2.imread("images/1.jpg")
canny = cv2.Canny(img,30,150)

cv2.imshow("src",img)
cv2.imshow("dst",canny)
cv2.waitKey(0)
cv2.destroyAllWindows()