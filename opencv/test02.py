import cv2
import numpy as np

img_data = np.zeros((300,300,3),np.uint8)
print(img_data.shape)
# img_data[:,:,0] = 255
img_data[...,0] = 255
print(img_data)
print(img_data.dtype)
cv2.imshow("img",img_data)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("images/save.jpg",img_data)

img_data2 = np.zeros((300,300,3),np.uint8)
cv2.imwrite('images/save1.jpg',img_data2)
