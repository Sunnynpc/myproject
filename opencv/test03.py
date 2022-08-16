import cv2

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
# cap = cv2.VideoCapture(r"F:\上课视频\20220401_151133.mp4")
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()