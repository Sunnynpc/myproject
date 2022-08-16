import cv2

img = cv2.imread("images/25.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dst1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,33,0)
# dst2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,33,0)
dst2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,33,0)

cv2.imshow("src",img)
cv2.imshow("gray",gray)
cv2.imshow("dst1",dst1)
cv2.imshow("dst2",dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()