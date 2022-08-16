import cv2

img1 = cv2.imread("images/1.jpg")
img2 = cv2.imread("images/6.jpg")

gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# _,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
_,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)


mask_inv = cv2.bitwise_not(binary)
binary = cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)
bt_add = cv2.bitwise_and(img1,binary)
bt_or = cv2.bitwise_or(img1,binary)
bt_xor = cv2.bitwise_xor(img1,img2)


cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.imshow("gray",gray)
cv2.imshow("mask_inv",mask_inv)
cv2.imshow("bt_add",bt_add)
cv2.imshow("bt_or",bt_or)
cv2.imshow("bt_xor",bt_xor)
cv2.waitKey(0)
cv2.destroyAllWindows()