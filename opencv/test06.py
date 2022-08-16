import cv2
import numpy as np

img = cv2.imread("images/1.jpg")
# cv2.line(img,(100,30),(210,180),color=(0,0,255),thickness=2)
# cv2.circle(img,(50,50),30,color=(0,0,255),thickness=2)
# cv2.ellipse(img,(100,100),(100,50),0,0,360,(255,0,0),2)
# cv2.rectangle(img,(100,30),(210,80),(0,0,255),2)

pts = np.array([[10, 5], [50, 10], [70, 20], [20, 30]], np.int32)
cv2.polylines(img,[pts],True,(0,255,0),2)

cv2.putText(img,"hello boy",(30,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,lineType=cv2.LINE_AA)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
