import cv2 as cv
from rescale import rescaleFrame, changeRes

capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    frame = changeRes(100,100)
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
