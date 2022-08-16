import cv2

src = cv2.imread("images/1.jpg")

gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
_,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

contours, _ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(contours)
print(len(contours))
# cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(src,contours,-1,(0,0,255),3)
cv2.imshow("src",src)
cv2.waitKey(0)
cv2.destroyAllWindows()