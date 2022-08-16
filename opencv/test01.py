import cv2

img = cv2.imread("images/1.jpg")
# print(img)
# print(type(img))
# print(img.shape)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('images/2.jpg')
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()