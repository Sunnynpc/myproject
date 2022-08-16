import cv2
from PIL import Image
import numpy as np

# img = Image.open("images/1.jpg")
# img = np.array(img,dtype=np.uint8)
# img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
# cv2.imshow("img",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

src = cv2.imread("images/1.jpg")
# dst = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
dst = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
cv2.imshow("src",src)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()