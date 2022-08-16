import cv2

img = cv2.imread("images/25.jpg")
cv2.imshow("src",img)

img = cv2.convertScaleAbs(img,alpha=6,beta=0)
img = cv2.GaussianBlur(img,(5,5),0)
canny = cv2.Canny(img,50,150)
cv2.imshow("abs",img)
cv2.imshow("canny",canny)
cv2.waitKey(0)
cv2.destroyAllWindows()