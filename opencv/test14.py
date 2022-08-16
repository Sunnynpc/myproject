import cv2

src = cv2.imread("images/7.jpg",0)

_,binary = cv2.threshold(src,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
# dst = cv2.dilate(src,kernel)
dst = cv2.erode(src, kernel)
cv2.imshow("src",binary)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

