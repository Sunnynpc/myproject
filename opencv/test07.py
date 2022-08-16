import cv2

img = cv2.imread("images/1.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(ret)
cv2.imshow("src",img)
cv2.imshow("gray",gray)
cv2.imshow("binary",binary)
cv2.waitKey(0)
cv2.destroyAllWindows()