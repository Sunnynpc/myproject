import cv2

src = cv2.imread("images/1.jpg")
h,w,c = src.shape
# dst = cv2.resize(src,(w*2,h*2))
# dst = cv2.transpose(src)
# print(src.shape)
# print(dst.shape)
dst = cv2.flip(src,1)
cv2.imshow("src",src)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()